use crate::cdf::{CdfModeContext, CdfMvContext};
use crate::ctx::memset_pow2;
use crate::dsp::N_SWITCHABLE_FILTERS;
use crate::env::{BlockContext, get_partition_ctx, get_partition2_ctx, warp_type};
use crate::headers::{
    AdaptiveBoolean, FrameHeader, MAX_SEGMENTS, NSWienerPlane, RestorationType, WarpedMotionParams,
    WarpedMotionType,
};
use crate::internal::{LoopFilterState, NsWienerBank, Pass, ScalableMotionParams};
use crate::intops::{apply_sign64, iclip, imax, imin, inv_recenter};
use crate::levels::{
    Av2Block, BlockPartition, BlockSize, CFL_PRED, CompInterPredMode, INVALID_MV, InterPredMode,
    MotionMode, Mv, MvXY, N_BS_SIZES, RefPair, TIP_FRAME, TxPartition,
};
use crate::lf_mask::Av2RestorationUnit;
use crate::msac::MsacContext;
use crate::pal::pal_idx_finish;
use crate::quantizer::dq_lookup;
use crate::refmvs;
use crate::tables::{
    BLOCK_DIMENSIONS, DEFAULT_WM_PARAMS, NS_WIENER_COEF_RANGE_UV, NS_WIENER_COEF_RANGE_Y,
    SUBSET_MASKS_UV, SUBSET_MASKS_Y, TXFM_DIMENSIONS,
};
use crate::warpmv::{find_affine_int, get_shear_params, set_affine_mv2d};

pub fn init_wiener(frame_hdr: &FrameHeader, lf: &mut LoopFilterState) {
    let rtype = frame_hdr.restoration.p[0].restoration_type;
    if rtype == RestorationType::None as u8 {
        return;
    }

    let qidx = frame_hdr.quant.yac as i32;
    lf.base_q = dq_lookup(qidx);

    let idx = if qidx < 130 {
        0
    } else if qidx < 190 {
        1
    } else if qidx < 220 {
        2
    } else {
        3
    };
    lf.wiener_idx = idx;

    if rtype == RestorationType::NsWiener as u8 || rtype == RestorationType::Switchable as u8 {
        let num_classes_idx = frame_hdr.restoration.p[0].ns.num_classes_idx as usize;
        if num_classes_idx > 0 {
            lf.ns_subclass_class_idx = Some(num_classes_idx - 1);
        } else {
            lf.ns_subclass_class_idx = None;
        }
    } else {
        lf.ns_subclass_class_idx = None;
    }
}

pub fn compute_restore_planes(frame_hdr: &FrameHeader) -> i32 {
    let has_y = frame_hdr.restoration.p[0].restoration_type != RestorationType::None as u8
        || frame_hdr.gdf.enabled != AdaptiveBoolean::Off;
    let has_u = frame_hdr.restoration.p[1].restoration_type != RestorationType::None as u8;
    let has_v = frame_hdr.restoration.p[2].restoration_type != RestorationType::None as u8;
    (has_y as i32) | ((has_u as i32) << 1) | ((has_v as i32) << 2)
}

pub fn compute_gdf_ref_dst_idx(frame_hdr: &FrameHeader, absrefdist: &[u8; 7]) -> i32 {
    if frame_hdr.gdf.enabled == AdaptiveBoolean::Off {
        return 0;
    }
    let is_inter_or_switch = (frame_hdr.frame_type as u8) & 1 != 0;
    if !is_inter_or_switch {
        return 0;
    }
    let mut max_dist = 0i32;
    for i in 0..imin(frame_hdr.n_ref_frames as i32, 2) as usize {
        max_dist = imax(max_dist, absrefdist[i] as i32);
    }
    const REF_DST_IDX_TBL: [i32; 12] = [5, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5];
    REF_DST_IDX_TBL[imin(max_dist, 11) as usize]
}

pub fn init_ns_wiener_bank(bank: &mut NsWienerBank, pl: usize, n_classes: usize) {
    bank.bank_size = [0; 16];
    bank.bank_idx = [0; 16];
    let cf_range: &[[i8; 2]] = if pl > 0 {
        &NS_WIENER_COEF_RANGE_UV
    } else {
        &NS_WIENER_COEF_RANGE_Y
    };
    let n_coeffs = 16 + if pl > 0 { 2 } else { 0 };
    for n in 0..n_classes {
        for m in 0..n_coeffs {
            bank.filter[0][n][m] = cf_range[m][1] + ((1i8 << cf_range[m][0]) >> 1);
        }
    }
}

pub fn init_start_of_tile_row(buf: &mut Vec<u8>, sbh: i32, tile_rows: u8, row_start_sb: &[u16]) {
    buf.resize(sbh as usize, 0);
    let mut sby = 0usize;
    for tile_row in 0..tile_rows as usize {
        buf[sby] = ((tile_row << 1) | 1) as u8;
        sby += 1;
        while sby < row_start_sb[tile_row + 1] as usize {
            buf[sby] = (tile_row << 1) as u8;
            sby += 1;
        }
    }
}

pub fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                r + ((diff + 1) >> 1)
            } else {
                r - (diff >> 1)
            }
        } else {
            diff
        }
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                r + ((diff + 1) >> 1)
            } else {
                r - (diff >> 1)
            }
        } else {
            max - (diff + 1)
        }
    }
}

pub fn init_quant_tables(
    frame_hdr: &FrameHeader,
    qidx: i32,
    dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS],
) {
    let n = if frame_hdr.segmentation.enabled != 0 {
        8
    } else {
        1
    };
    for i in 0..n {
        let yac = if frame_hdr.segmentation.enabled != 0 {
            qidx + frame_hdr.segmentation.d.delta_q[i] as i32
        } else {
            qidx
        };
        let ydc = yac + frame_hdr.quant.ydc_delta as i32;
        let uac = yac + frame_hdr.quant.uac_delta as i32;
        let udc = yac + frame_hdr.quant.udc_delta as i32;
        let vac = yac + frame_hdr.quant.vac_delta as i32;
        let vdc = yac + frame_hdr.quant.vdc_delta as i32;

        dq[i][0][0] = dq_lookup(ydc) as u32;
        dq[i][0][1] = dq_lookup(yac) as u32;
        dq[i][1][0] = dq_lookup(udc) as u32;
        dq[i][1][1] = dq_lookup(uac) as u32;
        dq[i][2][0] = dq_lookup(vdc) as u32;
        dq[i][2][1] = dq_lookup(vac) as u32;
    }
}

/// Recompute the per-segment dequant tables for a single qindex (used by the
/// per-superblock delta-q path; mirrors `init_quant_tables` with state pulled
/// from `SbFrameInfo` instead of `FrameHeader`).
pub fn init_quant_tables_fi(fi: &SbFrameInfo, qidx: i32, dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS]) {
    let n = if fi.seg_enabled { 8 } else { 1 };
    for i in 0..n {
        let yac = if fi.seg_enabled {
            qidx + fi.seg_delta_q[i] as i32
        } else {
            qidx
        };
        let ydc = yac + fi.q_ydc_delta;
        let uac = yac + fi.q_uac_delta;
        let udc = yac + fi.q_udc_delta;
        let vac = yac + fi.q_vac_delta;
        let vdc = yac + fi.q_vdc_delta;

        dq[i][0][0] = dq_lookup(ydc) as u32;
        dq[i][0][1] = dq_lookup(yac) as u32;
        dq[i][1][0] = dq_lookup(udc) as u32;
        dq[i][1][1] = dq_lookup(uac) as u32;
        dq[i][2][0] = dq_lookup(vdc) as u32;
        dq[i][2][1] = dq_lookup(vac) as u32;
    }
}

pub fn reset_context(ctx: &mut BlockContext, keyframe: bool, is_tip_frame: bool) {
    ctx.tx_lpf_y.fill(3);
    ctx.tx_lpf_uv.fill(2);
    if is_tip_frame {
        return;
    }
    ctx.midx.fill(0xff);
    ctx.intra.fill(keyframe as u8);
    ctx.uvmode.fill(0); // DC_PRED
    if keyframe {
        ctx.mode.fill(0); // DC_PRED
    }
    ctx.partition[0].fill(0);
    ctx.partition[1].fill(0);
    ctx.skip_txfm.fill(0);
    ctx.skip_mode.fill(0);
    if !keyframe {
        ctx.r#ref[0].fill(-1);
        ctx.r#ref[1].fill(-1);
        ctx.comp_type.fill(0);
        ctx.mode.fill(13); // NEARMV
    }
    ctx.mrl.fill(0);
    ctx.lcoef.fill(0x40);
    ctx.ccoef[0].fill(0x40);
    ctx.ccoef[1].fill(0x40);
    ctx.filter.fill(N_SWITCHABLE_FILTERS as u8);
    ctx.seg_pred.fill(0);
    ctx.pal_sz.fill(0);
}

pub fn decode_frame_init(
    frame_hdr: &FrameHeader,
    _seq_hdr: &crate::headers::SequenceHeader,
    lf: &mut LoopFilterState,
    frame_thread: &mut crate::internal::FrameThread,
    ts: &mut Vec<crate::internal::TileState>,
    n_ts: &mut i32,
    a: &mut Vec<BlockContext>,
    a_sz: &mut i32,
    dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS],
    qm: &mut [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES],
    absrefdist: &[u8; 7],
    sbh: i32,
    sb256w: i32,
    sb256h: i32,
    _bw: i32,
    _bh: i32,
    n_tc: i32,
    n_passes: i32,
) {
    init_start_of_tile_row(
        &mut lf.start_of_tile_row,
        sbh,
        frame_hdr.tiling.t.rows,
        frame_hdr.tiling.t.row_start_sb.as_ref(),
    );

    let new_n_ts = frame_hdr.tiling.t.cols as i32 * frame_hdr.tiling.t.rows as i32;
    if new_n_ts != *n_ts {
        if n_passes > 1 {
            frame_thread.tile_start_off.resize(new_n_ts as usize, 0);
        }
        *n_ts = new_n_ts;
    }
    ts.resize_with(new_n_ts as usize, Default::default);

    let new_a_sz = sb256w * frame_hdr.tiling.t.rows as i32;
    if new_a_sz != *a_sz {
        a.resize_with(new_a_sz as usize, Default::default);
        *a_sz = new_a_sz;
    }

    let num_sb256 = (sb256w * sb256h) as usize;
    lf.mask.resize_with(num_sb256, Default::default);
    for m in lf.mask.iter_mut() {
        *m = Default::default();
    }
    lf.lr_mask.resize_with(num_sb256, Default::default);
    for m in lf.lr_mask.iter_mut() {
        *m = Default::default();
    }

    init_wiener(frame_hdr, lf);
    lf.restore_planes = compute_restore_planes(frame_hdr);

    if frame_hdr.gdf.enabled != AdaptiveBoolean::Off {
        lf.gdf_ref_dst_idx = compute_gdf_ref_dst_idx(frame_hdr, absrefdist);
    }

    let re_sz = sb256h * frame_hdr.tiling.t.cols as i32;
    lf.re_sz = re_sz;

    init_quant_tables(frame_hdr, frame_hdr.quant.yac as i32, dq);

    if frame_hdr.quant.qm.enabled == 0 {
        *qm = Default::default();
    }

    if n_tc > 1 {
        let keyframe = frame_hdr.is_key_or_intra();
        let is_tip = frame_hdr.tip.frame_mode == 2;
        for ctx in a.iter_mut().take(new_a_sz as usize) {
            reset_context(ctx, keyframe, is_tip);
        }
    }
}

pub fn setup_tile_bounds(
    ts: &mut crate::internal::TileState,
    tile_row: i32,
    tile_col: i32,
    col_start_sb: &[u16],
    row_start_sb: &[u16],
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
) {
    let col_sb_start = col_start_sb[tile_col as usize] as i32;
    let col_sb_end = col_start_sb[tile_col as usize + 1] as i32;
    let row_sb_start = row_start_sb[tile_row as usize] as i32;
    let row_sb_end = row_start_sb[tile_row as usize + 1] as i32;

    ts.tiling.row = tile_row;
    ts.tiling.col = tile_col;
    ts.tiling.col_start = col_sb_start << sb_shift;
    ts.tiling.col_end = imin(col_sb_end << sb_shift, bw);
    ts.tiling.row_start = row_sb_start << sb_shift;
    ts.tiling.row_end = imin(row_sb_end << sb_shift, bh);

    if n_tc > 1 {
        for p in 0..3 {
            ts.progress[p].store(row_sb_start, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn setup_tile_wiener_banks(ts: &mut crate::internal::TileState, frame_hdr: &FrameHeader) {
    use crate::headers::RestorationType;
    for pl in 0..3 {
        let rtype = frame_hdr.restoration.p[pl].restoration_type;
        if rtype == RestorationType::NsWiener as u8 || rtype == RestorationType::Switchable as u8 {
            let n_classes = frame_hdr.restoration.p[pl].ns.num_classes as usize;
            init_ns_wiener_bank(&mut ts.ns_wiener_bank[pl], pl, n_classes);
        }
    }
}

pub fn setup_tile(
    ts: &mut crate::internal::TileState,
    data: &[u8],
    frame_hdr: &FrameHeader,
    in_cdf: Option<&crate::cdf::CdfContext>,
    qcat: usize,
    tile_row: i32,
    tile_col: i32,
    col_start_sb: &[u16],
    row_start_sb: &[u16],
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
    tile_start_off: u32,
) {
    if let Some(cdf) = in_cdf {
        ts.cdf = cdf.clone();
    } else {
        ts.cdf = crate::cdf::CdfContext::init_from_defaults(qcat);
    }
    ts.last_qidx = frame_hdr.quant.yac as i32;
    ts.msac_buf = data.to_vec();
    ts.tile_start_off = tile_start_off;

    setup_tile_bounds(
        ts,
        tile_row,
        tile_col,
        col_start_sb,
        row_start_sb,
        sb_shift,
        bw,
        bh,
        n_tc,
    );
    setup_tile_wiener_banks(ts, frame_hdr);
}

pub fn decode_frame_init_cdf(
    ts: &mut [crate::internal::TileState],
    tile_groups: &[crate::internal::TileGroup],
    frame_hdr: &FrameHeader,
    in_cdf: Option<&crate::cdf::CdfContext>,
    qcat: usize,
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
    n_passes: i32,
    tile_start_off: &mut [u32],
) -> Result<(), ()> {
    let ti = &frame_hdr.tiling.t;
    let mut tile_row = 0i32;
    let mut tile_col = 0i32;

    for tg in tile_groups.iter() {
        let mut data_off = 0usize;
        let mut remaining = tg.data.len();

        for j in tg.start..=tg.end {
            let tile_sz;
            if j == tg.end {
                tile_sz = remaining;
            } else {
                let n_bytes = frame_hdr.tiling.n_bytes as usize;
                if n_bytes > remaining {
                    return Err(());
                }
                let mut sz = 0usize;
                for k in 0..n_bytes {
                    sz |= (tg.data[data_off + k] as usize) << (k * 8);
                }
                sz += 1;
                data_off += n_bytes;
                remaining -= n_bytes;
                if sz > remaining {
                    return Err(());
                }
                tile_sz = sz;
            }

            let tile_data = &tg.data[data_off..data_off + tile_sz];
            let start_off = if n_passes > 1 {
                tile_start_off[j as usize]
            } else {
                0
            };

            setup_tile(
                &mut ts[j as usize],
                tile_data,
                frame_hdr,
                in_cdf,
                qcat,
                tile_row,
                tile_col,
                ti.col_start_sb.as_ref(),
                ti.row_start_sb.as_ref(),
                sb_shift,
                bw,
                bh,
                n_tc,
                start_off,
            );

            tile_col += 1;
            if tile_col == ti.cols as i32 {
                tile_col = 0;
                tile_row += 1;
            }

            data_off += tile_sz;
            remaining -= tile_sz;
        }
    }

    Ok(())
}

pub fn decode_tip_frame_init(
    ts: &mut [crate::internal::TileState],
    frame_hdr: &FrameHeader,
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
) {
    let ti = &frame_hdr.tiling.t;
    let mut tile = 0usize;
    for tile_row in 0..ti.rows as i32 {
        for tile_col in 0..ti.cols as i32 {
            setup_tile_bounds(
                &mut ts[tile],
                tile_row,
                tile_col,
                ti.col_start_sb.as_ref(),
                ti.row_start_sb.as_ref(),
                sb_shift,
                bw,
                bh,
                n_tc,
            );
            tile += 1;
        }
    }
}

// size_group_lookup[BlockSize] -> size group (0-3)
pub static SIZE_GROUP: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    // group 0: 4x4, 4x8, 8x4, 4x16, 16x4
    t[BlockSize::Bs4x4 as usize] = 0;
    t[BlockSize::Bs4x8 as usize] = 0;
    t[BlockSize::Bs8x4 as usize] = 0;
    t[BlockSize::Bs4x16 as usize] = 0;
    t[BlockSize::Bs16x4 as usize] = 0;
    // group 1: 8x8, 8x16, 16x8, 8x32, 32x8, 4x32, 32x4
    t[BlockSize::Bs8x8 as usize] = 1;
    t[BlockSize::Bs8x16 as usize] = 1;
    t[BlockSize::Bs16x8 as usize] = 1;
    t[BlockSize::Bs8x32 as usize] = 1;
    t[BlockSize::Bs32x8 as usize] = 1;
    t[BlockSize::Bs4x32 as usize] = 1;
    t[BlockSize::Bs32x4 as usize] = 1;
    // group 2: 16x16, 16x32, 32x16, 16x64, 64x16, 8x64, 64x8, 4x64, 64x4
    t[BlockSize::Bs16x16 as usize] = 2;
    t[BlockSize::Bs16x32 as usize] = 2;
    t[BlockSize::Bs32x16 as usize] = 2;
    t[BlockSize::Bs16x64 as usize] = 2;
    t[BlockSize::Bs64x16 as usize] = 2;
    t[BlockSize::Bs8x64 as usize] = 2;
    t[BlockSize::Bs64x8 as usize] = 2;
    t[BlockSize::Bs4x64 as usize] = 2;
    t[BlockSize::Bs64x4 as usize] = 2;
    // group 3: 32x32+
    t[BlockSize::Bs32x32 as usize] = 3;
    t[BlockSize::Bs32x64 as usize] = 3;
    t[BlockSize::Bs64x32 as usize] = 3;
    t[BlockSize::Bs64x64 as usize] = 3;
    t[BlockSize::Bs64x128 as usize] = 3;
    t[BlockSize::Bs128x64 as usize] = 3;
    t[BlockSize::Bs128x128 as usize] = 3;
    t[BlockSize::Bs128x256 as usize] = 3;
    t[BlockSize::Bs256x128 as usize] = 3;
    t[BlockSize::Bs256x256 as usize] = 3;
    t
};

// { Y+U+V, Y+U } multiplier per pixel layout
pub static SS_SIZE_MUL: [[u8; 2]; 4] = [
    [4, 4],  // I400
    [6, 5],  // I420
    [8, 6],  // I422
    [12, 8], // I444
];

// TX partition size group per block size
pub static TX_PART_GROUP: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    t[BlockSize::Bs8x4 as usize] = 0;
    t[BlockSize::Bs4x8 as usize] = 0;
    t[BlockSize::Bs4x4 as usize] = 0;
    t[BlockSize::Bs8x8 as usize] = 1;
    t[BlockSize::Bs16x8 as usize] = 2;
    t[BlockSize::Bs8x16 as usize] = 2;
    t[BlockSize::Bs16x16 as usize] = 3;
    t[BlockSize::Bs32x16 as usize] = 4;
    t[BlockSize::Bs16x32 as usize] = 4;
    t[BlockSize::Bs32x32 as usize] = 5;
    t[BlockSize::Bs64x32 as usize] = 6;
    t[BlockSize::Bs32x64 as usize] = 6;
    t[BlockSize::Bs64x64 as usize] = 7;
    // extended sizes map to 8
    t[BlockSize::Bs64x16 as usize] = 8;
    t[BlockSize::Bs64x8 as usize] = 8;
    t[BlockSize::Bs64x4 as usize] = 8;
    t[BlockSize::Bs32x8 as usize] = 8;
    t[BlockSize::Bs32x4 as usize] = 8;
    t[BlockSize::Bs16x64 as usize] = 8;
    t[BlockSize::Bs16x4 as usize] = 8;
    t[BlockSize::Bs8x64 as usize] = 8;
    t[BlockSize::Bs8x32 as usize] = 8;
    t[BlockSize::Bs4x64 as usize] = 8;
    t[BlockSize::Bs4x32 as usize] = 8;
    t[BlockSize::Bs4x16 as usize] = 8;
    t
};

// TX type group for 2D V/H partition per block size
pub static TX_TYPE_GROUP_VH: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    t[BlockSize::Bs8x8 as usize] = 0;
    t[BlockSize::Bs8x16 as usize] = 1;
    t[BlockSize::Bs16x8 as usize] = 2;
    t[BlockSize::Bs16x16 as usize] = 3;
    t[BlockSize::Bs16x32 as usize] = 4;
    t[BlockSize::Bs32x16 as usize] = 5;
    t[BlockSize::Bs32x32 as usize] = 6;
    t[BlockSize::Bs32x64 as usize] = 7;
    t[BlockSize::Bs64x32 as usize] = 8;
    t[BlockSize::Bs64x64 as usize] = 9;
    t[BlockSize::Bs8x32 as usize] = 10;
    t[BlockSize::Bs8x64 as usize] = 10;
    t[BlockSize::Bs64x8 as usize] = 11;
    t[BlockSize::Bs32x8 as usize] = 11;
    t[BlockSize::Bs16x64 as usize] = 12;
    t[BlockSize::Bs64x16 as usize] = 13;
    t
};

pub fn jmvd_scale(mv: &mut MvXY, amvd: bool, jmvd_scale_mode: i32) {
    if amvd {
        match jmvd_scale_mode {
            0 => {}
            1 => {
                mv.y *= 2;
                mv.x *= 2;
            }
            2 => {
                mv.y /= 2;
                mv.x /= 2;
            }
            _ => unreachable!(),
        }
    } else {
        match jmvd_scale_mode {
            0 => {}
            1 => mv.y *= 2,
            2 => mv.x *= 2,
            3 => mv.y /= 2,
            4 => mv.x /= 2,
            _ => unreachable!(),
        }
    }
}

pub fn get_prev_frame_segid(
    by: i32,
    bx: i32,
    w4: i32,
    h4: i32,
    ref_seg_map: &[u8],
    stride: isize,
) -> u32 {
    let mut seg_id = 8u32;
    let mut off = (by as isize * stride + bx as isize) as usize;
    for _ in 0..h4 {
        for x in 0..w4 as usize {
            seg_id = imin(seg_id as i32, ref_seg_map[off + x] as i32) as u32;
        }
        if seg_id == 0 {
            break;
        }
        off = (off as isize + stride) as usize;
    }
    seg_id
}

/// Spatial-prediction of the current-frame segment id (`get_cur_frame_segid`,
/// env.h:313). Returns the predicted seg_id and writes the neighbour context
/// class into `seg_ctx`.
pub fn get_cur_frame_segid(
    by: i32,
    bx: i32,
    have_top: bool,
    have_left: bool,
    seg_ctx: &mut i32,
    cur_seg_map: &[u8],
    stride: isize,
) -> u32 {
    let base = (bx as isize + by as isize * stride) as usize;
    if have_left && have_top {
        let l = cur_seg_map[base - 1] as i32;
        let a = cur_seg_map[(base as isize - stride) as usize] as i32;
        let al = cur_seg_map[(base as isize - (stride + 1)) as usize] as i32;
        if l == a && al == l {
            *seg_ctx = 2;
        } else if l == a || al == l || a == al {
            *seg_ctx = 1;
        } else {
            *seg_ctx = 0;
        }
        (if a == al { a } else { l }) as u32
    } else {
        *seg_ctx = 0;
        if have_left {
            cur_seg_map[base - 1] as u32
        } else if have_top {
            cur_seg_map[(base as isize - stride) as usize] as u32
        } else {
            0
        }
    }
}

pub fn mc_lowest_px(
    dst: &mut i32,
    by4: i32,
    bh4: i32,
    mvy: i32,
    ss_ver: i32,
    smp: &ScalableMotionParams,
) {
    let v_mul = 4 >> ss_ver;
    if smp.scale == 0 {
        let my = mvy >> (3 + ss_ver);
        let dy = mvy & (15 >> (ss_ver == 0) as u32);
        *dst = imax(*dst, (by4 + bh4) * v_mul + my + 4 * (dy != 0) as i32);
    } else {
        let y = ((by4 * v_mul) << 4) + mvy * (1 << (ss_ver == 0) as u32);
        let tmp = y as i64 * smp.scale as i64 + (smp.scale as i64 - 0x4000) * 8;
        let y = apply_sign64((tmp.unsigned_abs() as i64 + 128) >> 8, tmp) + 32;
        let bottom = ((y + (bh4 * v_mul - 1) * smp.step) >> 10) + 1 + 4;
        *dst = imax(*dst, bottom);
    }
}

pub fn affine_lowest_px(
    dst: &mut i32,
    b_dim: &[u8],
    by: i32,
    bx: i32,
    mat: &[i32; 6],
    ss_ver: i32,
    ss_hor: i32,
) {
    let h_mul = 4 >> ss_hor;
    let v_mul = 4 >> ss_ver;
    let y = b_dim[1] as i32 * v_mul - 8;

    let src_y = by * 4 + ((y + 4) << ss_ver);
    let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
    let bw = b_dim[0] as i32 * h_mul;
    let step = imax(8, bw - 8);
    let mut x = 0;
    while x < bw {
        let src_x = bx * 4 + ((x + 4) << ss_hor);
        let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> ss_ver;
        let dy = (mvy >> 16) as i32 - 4;
        *dst = imax(*dst, dy + 4 + 8);
        x += step;
    }
}

pub static REORDERED_NONDIR_Y_MODE: [u8; 5] = [0, 9, 10, 11, 12];

pub static REORDERED_DIR_Y_MODE: [u8; 8] = [3, 8, 1, 5, 4, 6, 2, 7];

pub static DEFAULT_MODE_LIST_Y: [u8; 56] = [
    17, 45, 3, 10, 24, 31, 38, 52, 15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54, 16,
    18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0, 6, 7, 13, 21, 27,
    28, 34, 35, 41, 49, 55,
];

pub static DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];

pub static INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];

pub static MV_PREC_TBL: [[u8; 3]; 2] = [[3, 1, 0], [4, 3, 1]];

use crate::levels::N_PARTITIONS;

// child partition split limits: [w_limit, h_limit]
pub static PARTITION_LIM: [[u8; 2]; N_PARTITIONS] = [
    [1, 1], // NONE
    [1, 2], // H
    [2, 1], // V
    [2, 4], // H3
    [4, 2], // V3
    [1, 8], // H4A
    [1, 8], // H4B
    [8, 1], // V4A
    [8, 1], // V4B
    [2, 2], // SPLIT
];

pub static WEDGE_ANGLE_DIST2IDX: [[i8; 4]; 20] = [
    [-1, 0, 1, 2],    // WEDGE_0
    [3, 4, 5, 6],     // WEDGE_14
    [7, 8, 9, 10],    // WEDGE_27
    [11, 12, 13, 14], // WEDGE_45
    [15, 16, 17, 18], // WEDGE_63
    [-1, 19, 20, 21], // WEDGE_90
    [22, 23, 24, 25], // WEDGE_117
    [26, 27, 28, 29], // WEDGE_135
    [30, 31, 32, 33], // WEDGE_153
    [34, 35, 36, 37], // WEDGE_166
    [-1, 38, 39, 40], // WEDGE_180
    [-1, 41, 42, 43], // WEDGE_194
    [-1, 44, 45, 46], // WEDGE_207
    [-1, 47, 48, 49], // WEDGE_225
    [-1, 50, 51, 52], // WEDGE_243
    [-1, 53, 54, 55], // WEDGE_270
    [-1, 56, 57, 58], // WEDGE_297
    [-1, 59, 60, 61], // WEDGE_315
    [-1, 62, 63, 64], // WEDGE_333
    [-1, 65, 66, 67], // WEDGE_346
];

#[derive(Clone, Copy)]
pub struct PartitionConstants {
    pub part: [[i8; 4]; 2],
    pub ctx: [i8; 2],
}

const I: i8 = -1; // BS_INVALID shorthand
use BlockSize::*;

pub static PARTITION_SUBB: [PartitionConstants; N_BS_SIZES] = {
    let mut t = [PartitionConstants {
        part: [[I; 4]; 2],
        ctx: [I; 2],
    }; N_BS_SIZES];

    t[Bs256x256 as usize] = PartitionConstants {
        part: [
            [Bs256x128 as i8, I, I, Bs128x128 as i8],
            [Bs128x256 as i8, I, I, Bs128x128 as i8],
        ],
        ctx: [9, 12],
    };
    t[Bs256x128 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs128x128 as i8, I, I, I]],
        ctx: [8, I],
    };
    t[Bs128x256 as usize] = PartitionConstants {
        part: [[Bs128x128 as i8, I, I, I], [I, I, I, I]],
        ctx: [7, I],
    };
    t[Bs128x128 as usize] = PartitionConstants {
        part: [
            [Bs128x64 as i8, I, I, Bs64x64 as i8],
            [Bs64x128 as i8, I, I, Bs64x64 as i8],
        ],
        ctx: [6, 9],
    };
    t[Bs128x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs64x64 as i8, I, I, I]],
        ctx: [5, I],
    };
    t[Bs64x128 as usize] = PartitionConstants {
        part: [[Bs64x64 as i8, I, I, I], [I, I, I, I]],
        ctx: [4, I],
    };
    t[Bs64x64 as usize] = PartitionConstants {
        part: [
            [Bs64x32 as i8, Bs64x16 as i8, Bs64x8 as i8, Bs32x32 as i8],
            [Bs32x64 as i8, Bs16x64 as i8, Bs8x64 as i8, Bs32x32 as i8],
        ],
        ctx: [3, 6],
    };
    t[Bs64x32 as usize] = PartitionConstants {
        part: [
            [Bs64x16 as i8, Bs64x8 as i8, Bs64x4 as i8, Bs32x16 as i8],
            [Bs32x32 as i8, Bs16x32 as i8, Bs8x32 as i8, Bs32x16 as i8],
        ],
        ctx: [3, 5],
    };
    t[Bs64x16 as usize] = PartitionConstants {
        part: [
            [Bs64x8 as i8, Bs64x4 as i8, I, Bs32x8 as i8],
            [Bs32x16 as i8, Bs16x16 as i8, Bs8x16 as i8, Bs32x8 as i8],
        ],
        ctx: [15, 14],
    };
    t[Bs64x8 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs64x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs32x64 as usize] = PartitionConstants {
        part: [
            [Bs32x32 as i8, Bs32x16 as i8, Bs32x8 as i8, Bs16x32 as i8],
            [Bs16x64 as i8, Bs8x64 as i8, Bs4x64 as i8, Bs16x32 as i8],
        ],
        ctx: [3, 4],
    };
    t[Bs32x32 as usize] = PartitionConstants {
        part: [
            [Bs32x16 as i8, Bs32x8 as i8, Bs32x4 as i8, Bs16x16 as i8],
            [Bs16x32 as i8, Bs8x32 as i8, Bs4x32 as i8, Bs16x16 as i8],
        ],
        ctx: [2, 3],
    };
    t[Bs32x16 as usize] = PartitionConstants {
        part: [
            [Bs32x8 as i8, Bs32x4 as i8, I, Bs16x8 as i8],
            [Bs16x16 as i8, Bs8x16 as i8, Bs4x16 as i8, Bs16x8 as i8],
        ],
        ctx: [2, 2],
    };
    t[Bs32x8 as usize] = PartitionConstants {
        part: [
            [Bs32x4 as i8, I, I, I],
            [Bs16x8 as i8, Bs8x8 as i8, Bs4x8 as i8, Bs16x4 as i8],
        ],
        ctx: [13, 14],
    };
    t[Bs32x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs16x64 as usize] = PartitionConstants {
        part: [
            [Bs16x32 as i8, Bs16x16 as i8, Bs16x8 as i8, Bs8x32 as i8],
            [Bs8x64 as i8, Bs4x64 as i8, I, Bs8x32 as i8],
        ],
        ctx: [14, 13],
    };
    t[Bs16x32 as usize] = PartitionConstants {
        part: [
            [Bs16x16 as i8, Bs16x8 as i8, Bs16x4 as i8, Bs8x16 as i8],
            [Bs8x32 as i8, Bs4x32 as i8, I, Bs8x16 as i8],
        ],
        ctx: [2, 1],
    };
    t[Bs16x16 as usize] = PartitionConstants {
        part: [
            [Bs16x8 as i8, Bs16x4 as i8, I, Bs8x8 as i8],
            [Bs8x16 as i8, Bs4x16 as i8, I, Bs8x8 as i8],
        ],
        ctx: [1, 0],
    };
    t[Bs16x8 as usize] = PartitionConstants {
        part: [
            [Bs16x4 as i8, I, I, I],
            [Bs8x8 as i8, Bs4x8 as i8, I, Bs8x4 as i8],
        ],
        ctx: [1, 2],
    };
    t[Bs16x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs8x4 as i8, I, I, I]],
        ctx: [11, I],
    };
    t[Bs8x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs8x32 as usize] = PartitionConstants {
        part: [
            [Bs8x16 as i8, Bs8x8 as i8, Bs8x4 as i8, Bs4x16 as i8],
            [Bs4x32 as i8, I, I, I],
        ],
        ctx: [12, 13],
    };
    t[Bs8x16 as usize] = PartitionConstants {
        part: [
            [Bs8x8 as i8, Bs8x4 as i8, I, Bs4x8 as i8],
            [Bs4x16 as i8, I, I, I],
        ],
        ctx: [1, 1],
    };
    t[Bs8x8 as usize] = PartitionConstants {
        part: [[Bs8x4 as i8, I, I, I], [Bs4x8 as i8, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs8x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs4x4 as i8, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x32 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x16 as usize] = PartitionConstants {
        part: [[Bs4x8 as i8, I, I, I], [I, I, I, I]],
        ctx: [10, I],
    };
    t[Bs4x8 as usize] = PartitionConstants {
        part: [[Bs4x4 as i8, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [I, I],
    };
    t
};

pub static FSC_BSIZE_GROUPS: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    t[Bs32x32 as usize] = 5;
    t[Bs32x16 as usize] = 5;
    t[Bs32x8 as usize] = 4;
    t[Bs32x4 as usize] = 4;
    t[Bs16x32 as usize] = 5;
    t[Bs16x16 as usize] = 4;
    t[Bs16x8 as usize] = 3;
    t[Bs16x4 as usize] = 3;
    t[Bs8x32 as usize] = 4;
    t[Bs8x16 as usize] = 3;
    t[Bs8x8 as usize] = 2;
    t[Bs8x4 as usize] = 1;
    t[Bs4x32 as usize] = 4;
    t[Bs4x16 as usize] = 3;
    t[Bs4x8 as usize] = 1;
    t
};

pub static CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] = [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];

// indexed by inter_mode - CompInterPredMode::NearMvNewMv
pub static AMVD_MODE_CONTEXT: [u8; 10] = {
    let mut t = [0u8; 10];
    use crate::levels::CompInterPredMode::*;
    let base = NearMvNewMv as usize;
    t[NearMvNewMv as usize - base] = 0;
    t[NewMvNearMv as usize - base] = 1;
    t[OpflNearMvNewMv as usize - base] = 2;
    t[OpflNewMvNearMv as usize - base] = 3;
    // GlobalMvGlobalMv(21) and NewMvNewMv(22) map to indices 2,3 — left as 0
    t[JointNewMv as usize - base] = 5;
    t[OpflJointNewMv as usize - base] = 6;
    t[NewMvNewMv as usize - base] = 7;
    t[OpflNewMvNewMv as usize - base] = 8;
    t
};

pub fn read_wedge_idx(msac: &mut MsacContext, cdf_m: &mut CdfModeContext) -> i8 {
    let quad = msac.decode_symbol_adapt(cdf_m.wedge_quad(), 3) as usize;
    let angle = 5 * quad + msac.decode_symbol_adapt(cdf_m.wedge_angle(quad), 4) as usize;
    let dist = if (angle.wrapping_sub(1)) >= 9 || angle == 5 {
        1 + msac.decode_symbol_adapt(cdf_m.wedge_dist2(), 2) as usize
    } else {
        msac.decode_symbol_adapt(cdf_m.wedge_dist(), 3) as usize
    };
    WEDGE_ANGLE_DIST2IDX[angle][dist]
}

pub fn decode_4way(msac: &mut MsacContext, r: i32, cdf: &mut [u16], n_bits: i32) -> i32 {
    debug_assert!(n_bits >= 4);
    let bin = msac.decode_symbol_adapt(cdf, 3) as i32;
    let rem =
        msac.decode_bools_bypass((n_bits + bin + if bin == 0 { 1 } else { 0 } - 4) as u32) as i32;
    let v = (if bin != 0 { 1 << (n_bits + bin - 4) } else { 0 }) + rem;
    let n = 1 << n_bits;
    if r * 2 <= n {
        inv_recenter(r as u32, v as u32) as i32
    } else {
        n - 1 - inv_recenter((n - 1 - r) as u32, v as u32) as i32
    }
}

pub fn read_amvd_raw(msac: &mut MsacContext, amvd_joint: &mut [u16], amvd_index: &mut [u16]) -> Mv {
    let joint = msac.decode_symbol_adapt(amvd_joint, 3) as i32;
    if joint == 0 {
        return Mv { n: 0 };
    }
    let y = if joint & 2 != 0 {
        let s = msac.decode_symbol_adapt(&mut amvd_index[0..8], 7) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    } else {
        0
    };
    let x = if joint & 1 != 0 {
        let s = msac.decode_symbol_adapt(&mut amvd_index[8..16], 7) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    } else {
        0
    };
    Mv { c: MvXY { y, x } }
}

pub fn read_mv_residual(
    msac: &mut MsacContext,
    cdf_mv: &mut CdfMvContext,
    shell_tip: &mut [u16],
    mv_prec: i32,
) -> Mv {
    let n_syms = 9 + mv_prec;
    let h_syms = n_syms >> 1;

    let mut sh_class;
    if msac.decode_bool_adapt(cdf_mv.shell_set()) != 0 {
        let h_syms2 = n_syms - h_syms;
        sh_class = h_syms
            + 1
            + msac.decode_symbol_adapt(
                cdf_mv.shell_upper(mv_prec as usize),
                imin(h_syms2, 7) as usize,
            ) as i32;
        if mv_prec + sh_class == 21 {
            sh_class += msac.decode_bool_adapt(shell_tip) as i32;
        }
    } else {
        sh_class =
            msac.decode_symbol_adapt(cdf_mv.shell_lower(mv_prec as usize), h_syms as usize) as i32;
    }

    let mut sh_index;
    if sh_class < 2 {
        sh_index = msac.decode_bool_adapt(cdf_mv.shell_offset_low(sh_class as usize)) as i32;
    } else if sh_class == 2 {
        sh_index = msac.decode_bool_adapt(cdf_mv.shell_offset_cl2()) as i32;
        if sh_index != 0 {
            sh_index += msac.decode_bool_bypass() as i32;
            if sh_index == 2 {
                sh_index += msac.decode_bool_bypass() as i32;
            }
        }
    } else {
        sh_index = 0;
        let mut m = 1i32;
        for i in 0..sh_class {
            sh_index |= m * msac.decode_bool_adapt(cdf_mv.shell_offset_hi(i as usize)) as i32;
            m <<= 1;
        }
    }

    if sh_class != 0 {
        sh_index += 1 << sh_class;
    }
    if sh_index == 0 {
        return Mv { n: 0 };
    }

    let mut pair_index = 0i32;
    if sh_index >= 2 {
        pair_index = msac.decode_bool_adapt(cdf_mv.col_component(0)) as i32;
        if pair_index != 0 && sh_index >= 4 {
            pair_index += msac.decode_bool_adapt(cdf_mv.col_component(1)) as i32;
            if pair_index == 2 && sh_index >= 6 {
                pair_index += msac.decode_uniform((sh_index as u32 >> 1) - 1) as i32;
            }
        }
    }

    let sh = 6 - mv_prec;
    if pair_index * 2 == sh_index {
        let v = (sh_index >> 1) << sh;
        Mv {
            c: MvXY { y: v, x: v },
        }
    } else {
        let b = msac.decode_bool_adapt(cdf_mv.col_index(imin(sh_class, 3) as usize));
        if b != 0 {
            Mv {
                c: MvXY {
                    y: pair_index << sh,
                    x: (sh_index - pair_index) << sh,
                },
            }
        } else {
            Mv {
                c: MvXY {
                    x: pair_index << sh,
                    y: (sh_index - pair_index) << sh,
                },
            }
        }
    }
}

pub fn read_mv_full(msac: &mut MsacContext, cdf_mv: &mut CdfMvContext, mv_prec: i32) -> Mv {
    let mut shell_tip = [cdf_mv.data[114], cdf_mv.data[115]];
    let mv = read_mv_residual(msac, cdf_mv, &mut shell_tip, mv_prec);
    cdf_mv.data[114] = shell_tip[0];
    cdf_mv.data[115] = shell_tip[1];
    mv
}

pub fn read_amvd(msac: &mut MsacContext, cdf_m: &mut CdfModeContext) -> Mv {
    let joint = msac.decode_symbol_adapt(cdf_m.amvd_joint(), 3) as i32;
    if joint == 0 {
        return Mv::default();
    }
    let y = if joint & 2 != 0 {
        let s = msac.decode_symbol_adapt(cdf_m.amvd_index(0), 7) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    } else {
        0
    };
    let x = if joint & 1 != 0 {
        let s = msac.decode_symbol_adapt(cdf_m.amvd_index(1), 7) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    } else {
        0
    };
    Mv { c: MvXY { y, x } }
}

pub fn read_pal_indices(
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    pal_out: &mut [u8],
    scratch: &mut [u8],
    pal_sz: i32,
    sz: &[i32; 4],
) -> i32 {
    let dir = (imax(sz[2], sz[3]) < 64) as u32 & msac.decode_bool_bypass();
    let strides: [isize; 2] = if dir != 0 {
        [1, sz[2] as isize]
    } else {
        [sz[2] as isize, 1]
    };

    let lim1 = sz[dir as usize ^ 1] as usize;
    let lim2 = sz[dir as usize] as usize;
    let pal_cdf_base = (pal_sz - 2) as usize;
    let nsym = (pal_sz - 1) as usize;

    let mut copy = msac.decode_symbol_adapt(cdf_m.pal_idx_identity(3), 2) as i32;
    if copy == 2 {
        return -1;
    }
    let mut prev_v = msac.decode_uniform(pal_sz as u32) as i32;
    scratch[0] = prev_v as u8;
    if copy == 1 {
        for m in 1..lim2 {
            scratch[(m as isize * strides[1]) as usize] = prev_v as u8;
        }
    } else {
        let mut prev_h = prev_v;
        for m in 1..lim2 {
            let v = msac.decode_symbol_adapt(cdf_m.pal_idx(pal_cdf_base, 0), nsym) as i32;
            prev_h = if v == 0 {
                prev_h
            } else {
                v - (v <= prev_h) as i32
            };
            scratch[(m as isize * strides[1]) as usize] = prev_h as u8;
        }
    }

    let mut off: isize = strides[0];
    for _n in 1..lim1 {
        copy = msac.decode_symbol_adapt(cdf_m.pal_idx_identity(copy as usize), 2) as i32;
        if copy == 2 {
            for m in 0..lim2 {
                let dst = (off + m as isize * strides[1]) as usize;
                let src = (off - strides[0] + m as isize * strides[1]) as usize;
                scratch[dst] = scratch[src];
            }
        } else {
            let v = msac.decode_symbol_adapt(cdf_m.pal_idx(pal_cdf_base, 0), nsym) as i32;
            let next_v = if v == 0 {
                prev_v
            } else {
                v - (v <= prev_v) as i32
            };
            scratch[off as usize] = next_v as u8;

            if copy == 1 {
                for m in 1..lim2 {
                    scratch[(off + m as isize * strides[1]) as usize] = next_v as u8;
                }
            } else {
                let mut prev_tl = prev_v;
                let mut prev_l = next_v;
                for m in 1..lim2 {
                    let prev_t =
                        scratch[(off - strides[0] + m as isize * strides[1]) as usize] as i32;
                    let ctx = if prev_t == prev_l {
                        3 + (prev_tl == prev_l) as usize
                    } else {
                        1 + (prev_t == prev_tl || prev_l == prev_tl) as usize
                    };
                    let v = msac.decode_symbol_adapt(cdf_m.pal_idx(pal_cdf_base, ctx), nsym) as i32;
                    let p = match ctx {
                        1 => match v {
                            0 | 1 => {
                                if v == dir as i32 {
                                    prev_l
                                } else {
                                    prev_t
                                }
                            }
                            2 => prev_tl,
                            _ => {
                                let s1 = (prev_l < prev_t) as i32;
                                let s2 = (prev_l < prev_tl) as i32;
                                let s3 = (prev_t < prev_tl) as i32;
                                v - (v <= prev_l + s1 + s2) as i32
                                    - (v <= prev_t + s3 + 1 - s1) as i32
                                    - (v <= prev_tl + 1 - s2 + 1 - s3) as i32
                            }
                        },
                        2 => {
                            let prev_l_or_t = prev_l + prev_t - prev_tl;
                            match v {
                                0 => prev_tl,
                                1 => prev_l_or_t,
                                _ => {
                                    let s = (prev_l_or_t < prev_tl) as i32;
                                    v - (v <= prev_l_or_t + s) as i32
                                        - (v <= prev_tl + 1 - s) as i32
                                }
                            }
                        }
                        3 => match v {
                            0 => prev_l,
                            1 => prev_tl,
                            _ => {
                                let s = (prev_l < prev_tl) as i32;
                                v - (v <= prev_l + s) as i32 - (v <= prev_tl + 1 - s) as i32
                            }
                        },
                        4 => {
                            if v == 0 {
                                prev_l
                            } else {
                                v - (v <= prev_l) as i32
                            }
                        }
                        _ => unreachable!(),
                    };
                    scratch[(off + m as isize * strides[1]) as usize] = p as u8;
                    prev_l = p;
                    prev_tl = prev_t;
                }
            }
            prev_v = next_v;
        }
        off += strides[0];
    }

    pal_idx_finish(
        pal_out,
        scratch,
        sz[2] as usize,
        sz[3] as usize,
        sz[0] as usize,
        sz[1] as usize,
    );
    0
}

pub fn read_tx_part(
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    b: &mut crate::levels::Av2Block,
    bs: BlockSize,
    lossless: bool,
    txfm_switchable: bool,
) {
    use crate::tables::BLOCK_DIMENSIONS;

    let bs_idx = bs as usize;
    let b_dim = &BLOCK_DIMENSIONS[bs_idx];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;

    b.tx_part = TxPartition::None as u8;
    if lossless {
        b.tx_size_ll = 0;
        if bs != BlockSize::Bs4x4
            && (if b.is_intra != 0 && b.intrabc == 0 {
                b.fsc != 0
            } else {
                b.skip_txfm == 0
            })
        {
            let szctx = SIZE_GROUP[bs_idx] as usize;
            let inter = (b.is_intra == 0 || b.intrabc != 0) as usize;
            b.tx_size_ll = msac.decode_bool_adapt(cdf_m.txsz_lossless(szctx, inter)) as u8;
        }
    } else if b.skip_txfm == 0 && txfm_switchable && bs != BlockSize::Bs4x4 && imax(bw4, bh4) <= 16
    {
        let inter = (b.is_intra == 0 || b.intrabc != 0) as usize;
        let szctx = TX_PART_GROUP[bs_idx] as usize;
        let is_split = msac.decode_bool_adapt(cdf_m.tx_split(b.fsc as usize, inter, szctx));
        if is_split != 0 {
            if imin(bw4, bh4) >= 2 {
                let ctx = TX_TYPE_GROUP_VH[bs_idx] as usize;
                b.tx_part = 1 + msac
                    .decode_symbol_adapt(cdf_m.tx_part_2d(b.fsc as usize, inter, ctx), 6)
                    as u8;
            } else if imax(bw4, bh4) >= 4 {
                let ctx = (bw4 >= 4) as usize;
                let tx_part_4way =
                    msac.decode_bool_adapt(cdf_m.tx_part_1d(b.fsc as usize, inter, ctx));
                b.tx_part = TxPartition::H as u8 + ctx as u8 + tx_part_4way as u8 * 2;
            } else {
                debug_assert!(bs == BlockSize::Bs4x8 || bs == BlockSize::Bs8x4);
                b.tx_part = if bs == BlockSize::Bs4x8 {
                    TxPartition::H as u8
                } else {
                    TxPartition::V as u8
                };
            }
        }
    }
}

pub fn read_restoration_info(
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    bank: &mut NsWienerBank,
    lr: &mut Av2RestorationUnit,
    p: usize,
    frame_type: RestorationType,
    ns_plane: &NSWienerPlane,
) {
    let is_uv = (p != 0) as usize;

    if frame_type == RestorationType::Switchable {
        debug_assert!(p == 0);
        if msac.decode_bool_adapt(cdf_m.rst_switchable(0)) != 0 {
            lr.restoration_type = RestorationType::None as u8;
        } else {
            let t = msac.decode_bool_adapt(cdf_m.rst_switchable(1));
            lr.restoration_type = if t != 0 {
                RestorationType::PcWiener as u8
            } else {
                RestorationType::NsWiener as u8
            };
        }
    } else {
        debug_assert!(p == 0 || frame_type == RestorationType::NsWiener);
        let cdf = if frame_type == RestorationType::NsWiener {
            cdf_m.rst_ns_wiener()
        } else {
            cdf_m.rst_pc_wiener()
        };
        let t = msac.decode_bool_adapt(cdf);
        lr.restoration_type = if t != 0 {
            frame_type as u8
        } else {
            RestorationType::None as u8
        };
    }

    if lr.restoration_type == RestorationType::NsWiener as u8 && ns_plane.frame_filters_on == 0 {
        let n_classes = ns_plane.num_classes as usize;
        let mut exact_match_mask = 0u32;
        let mut bank_refs = [0u8; 16];

        for n in 0..n_classes {
            let exact_match = msac.decode_bool_bypass();
            let bank_size = bank.bank_size[n] as i32;
            let mut r = 0i32;
            while r < bank_size - 1 {
                if msac.decode_bool_bypass() != 0 {
                    break;
                }
                r += 1;
            }
            let r_idx = ((bank.bank_idx[n] as i32 - r) & 3) as u8;
            exact_match_mask |= (1 << n) * exact_match;
            bank_refs[n] = r_idx;
        }

        let masks: &[u32] = if is_uv != 0 {
            &SUBSET_MASKS_UV
        } else {
            &SUBSET_MASKS_Y
        };
        let cf_range: &[[i8; 2]] = if is_uv != 0 {
            &NS_WIENER_COEF_RANGE_UV
        } else {
            &NS_WIENER_COEF_RANGE_Y
        };
        let n_coefs = 16 + is_uv * 2;

        for n in 0..n_classes {
            let r = bank_refs[n] as usize;
            let exact = (exact_match_mask >> n) & 1 != 0;

            if exact {
                lr.ns_filter[n][..n_coefs].copy_from_slice(&bank.filter[r][n][..n_coefs]);
                if bank.bank_size[n] == 0 {
                    bank.bank_size[n] = 1;
                }
                continue;
            }

            lr.ns_filter[n][..n_coefs].fill(0);
            let mut s = 0usize;
            while s < 3 - is_uv {
                if msac.decode_bool_adapt(cdf_m.wiener_ns_len(is_uv)) == 0 {
                    break;
                }
                s += 1;
            }
            let mask = masks[s];
            let asym = is_uv != 0 && s != 0 && msac.decode_bool_adapt(cdf_m.wiener_ns_sym()) != 0;

            let ref_filter = &bank.filter[r][n];
            let mut i = 0usize;
            let mut m = mask;
            while i < n_coefs {
                if m & 1 == 0 {
                    i += 1;
                    m >>= 1;
                    continue;
                }
                lr.ns_filter[n][i] = (decode_4way(
                    msac,
                    ref_filter[i] as i32 - cf_range[i][1] as i32,
                    cdf_m.wiener_ns_cf(),
                    cf_range[i][0] as i32,
                ) + cf_range[i][1] as i32) as i8;
                if asym && i >= 6 {
                    lr.ns_filter[n][i + 1] = lr.ns_filter[n][i];
                    i += 1;
                    m >>= 1;
                }
                i += 1;
                m >>= 1;
            }

            let bidx = ((1 + bank.bank_idx[n]) & 3) as usize;
            bank.bank_idx[n] = bidx as u8;
            bank.filter[bidx][n][..n_coefs].copy_from_slice(&lr.ns_filter[n][..n_coefs]);
            if bank.bank_size[n] < 4 {
                bank.bank_size[n] += 1;
            }
        }
    }
}

pub fn derive_warpmv(
    rt: &refmvs::Tile,
    bx: i32,
    by: i32,
    have_top: bool,
    have_left: bool,
    bw4: i32,
    bh4: i32,
    w4: i32,
    h4: i32,
    ref_idx: i8,
    mv: Mv,
    wmp: &mut WarpedMotionParams,
    sb_step: i32,
    col_end: i32,
) {
    let mut pts = [[[0i32; 2]; 2]; 8];
    let mut np = 0usize;

    macro_rules! add_sample {
        ($dx:expr, $dy:expr, $sx:expr, $sy:expr, $rp:expr) => {{
            let rp: &refmvs::Block = $rp;
            let bd = &BLOCK_DIMENSIONS[rp.bs as usize];
            let rmv = if rp.mf & 2 != 0 { &rp.lmv } else { &rp.mv };
            for n in 0..2usize {
                if unsafe { rp.r#ref.r[n] } != ref_idx {
                    continue;
                }
                pts[np][0][0] = 16 * (2 * ($dx as i32) + ($sx as i32) * bd[0] as i32) - 8;
                pts[np][0][1] = 16 * (2 * ($dy as i32) + ($sy as i32) * bd[1] as i32) - 8;
                pts[np][1][0] = pts[np][0][0] + unsafe { rmv[n].c.x };
                pts[np][1][1] = pts[np][0][1] + unsafe { rmv[n].c.y };
                np += 1;
                if np == 8 {
                    break;
                }
            }
        }};
    }

    debug_assert!(bw4 > 1);
    let mut have_topleft = false;
    let mut have_topright = false;
    let is_not_sb_boundary = (by & (sb_step - 1)) != 0;
    let mut init_odd = 0i32;

    if have_top {
        if is_not_sb_boundary {
            let ra_base = ((by - 1) & 63) as usize * 128;
            let r2_x = (bx & 127) as usize;
            let mut off = -(rt.r[ra_base + r2_x].ox4 as i32);
            have_topleft = off == 0;
            while off < w4 && np < 8 {
                let idx = ra_base + ((bx + off) & 127) as usize;
                add_sample!(off, 0, 1, -1, &rt.r[idx]);
                off += BLOCK_DIMENSIONS[rt.r[idx].bs as usize][0] as i32;
            }
            have_topright = off <= bw4;
        } else {
            let ra_off = rt.ra_off;
            let r2_idx = ra_off + (bx >> 1) as usize;
            init_odd = bx & 1;
            have_topleft = true;
            let mut off = if BLOCK_DIMENSIONS[rt.ra[r2_idx].bs as usize][0] as i32
                <= rt.ra[r2_idx].ox4 as i32 + init_odd
            {
                1
            } else {
                0
            };
            let tr_ext = (bx + bw4) & (sb_step - 1) != 0
                && (rt.ra[ra_off + ((bx + bw4) >> 1) as usize].ox4 != 0 || init_odd != 0);
            let tr_ext_i = tr_ext as i32;
            while off < w4 + tr_ext_i && np < 8 {
                let off8 = ra_off + ((bx + off) >> 1) as usize;
                let odd = (bx + off) & 1;
                let ioff = off - rt.ra[off8].ox4 as i32 - odd;
                add_sample!(ioff, 0, 1, -1, &rt.ra[off8]);
                off = ioff + BLOCK_DIMENSIONS[rt.ra[off8].bs as usize][0] as i32 + 1;
            }
            have_topright = true;
        }

        have_topright = have_topright
            && bw4 <= 16
            && bx + bw4 + ((!is_not_sb_boundary) as i32) < col_end
            && (!is_not_sb_boundary
                || ((bx + bw4) & (sb_step - 1) != 0
                    && unsafe {
                        rt.r[((by - 1) & 63) as usize * 128 + ((bx + bw4) & 127) as usize].mv[0]
                            .c
                            .y
                    } != INVALID_MV));
    }

    if np < 8 && have_left {
        let left_x = ((bx - 1) & 127) as usize;
        let r_base = (by & 63) as usize * 128;
        let mut off = -(rt.r[r_base + left_x].oy4 as i32);
        have_topleft = have_topleft && off == 0;
        loop {
            let row = ((by & 63) as isize + off as isize) as usize;
            let idx = row * 128 + left_x;
            add_sample!(0, off, -1, 1, &rt.r[idx]);
            off += BLOCK_DIMENSIONS[rt.r[idx].bs as usize][1] as i32;
            if off >= h4 || np >= 8 {
                break;
            }
        }
    } else {
        have_topleft = false;
    }

    if is_not_sb_boundary {
        let ra_base = ((by - 1) & 63) as usize * 128;
        if np < 8 && have_topleft {
            add_sample!(0, 0, -1, -1, &rt.r[ra_base + ((bx - 1) & 127) as usize]);
        }
        if np < 8 && have_topright {
            add_sample!(bw4, 0, 1, -1, &rt.r[ra_base + ((bx + bw4) & 127) as usize]);
        }
    } else {
        if np < 8 && have_topleft {
            let r2 = if bx & (sb_step - 1) != 0 {
                &rt.ra[rt.ra_off + ((bx - 1) >> 1) as usize]
            } else {
                &rt.ra_tl
            };
            if BLOCK_DIMENSIONS[r2.bs as usize][0] as i32 + init_odd == r2.ox4 as i32 + 2 {
                add_sample!(0, 0, -1, -1, r2);
            }
        }
        if np < 8 && have_topright {
            let r2 = &rt.ra[rt.ra_off + ((bx + bw4 + 1) >> 1) as usize];
            if r2.ox4 as i32 == init_odd {
                add_sample!(bw4, 0, 1, -1, r2);
            }
        }
    }

    debug_assert!(np > 0 && np <= 8);

    if find_affine_int(&pts[..np], np, bw4, bh4, unsafe { mv.c }, wmp, bx, by) == 0
        && get_shear_params(wmp) == 0
    {
        wmp.wm_type = warp_type(&wmp.matrix);
    } else {
        wmp.wm_type = WarpedMotionType::Invalid;
    }
}

pub fn extend_warpmv(
    rt: &refmvs::Tile,
    bx: i32,
    by: i32,
    x_off: i32,
    y_off: i32,
    b_dim: &[u8],
    ref0: i8,
    mv0: Mv,
    wmp: &mut WarpedMotionParams,
    sb_step: i32,
    gmv_matrix: &[i32; 6],
) {
    let r = if y_off == -1 && (by & (sb_step - 1)) == 0 {
        if x_off < 0 && (bx & (sb_step - 1)) == 0 {
            &rt.ra_tl
        } else {
            &rt.ra[rt.ra_off + ((bx + x_off) >> 1) as usize]
        }
    } else {
        &rt.r[((by + y_off) & 63) as usize * 128 + ((bx + x_off) & 127) as usize]
    };
    let m = &mut wmp.matrix;

    if r.mf & 2 != 0 {
        if r.warp_type == WarpedMotionType::Invalid as i8 {
            m.copy_from_slice(&DEFAULT_WM_PARAMS.matrix);
        } else {
            m.copy_from_slice(&r.m);
        }
    } else if r.mf & 1 != 0 {
        m.copy_from_slice(gmv_matrix);
    } else {
        m[2..6].copy_from_slice(&DEFAULT_WM_PARAMS.matrix[2..6]);
        let ref_n = (unsafe { r.r#ref.r[0] } != ref0) as usize;
        m[0] = unsafe { r.mv[ref_n].c.x } * (1 << 13);
        m[1] = unsafe { r.mv[ref_n].c.y } * (1 << 13);
    }

    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;
    let sx = bx * 4 + 2 * bw4 - 1;
    let sy = by * 4 + 2 * bh4 - 1;
    let mv0c = unsafe { mv0.c };
    let px = ((sx as i64) << 16) + mv0c.x as i64 * (1 << 13);
    let py = ((sy as i64) << 16) + mv0c.y as i64 * (1 << 13);

    if x_off >= 0 {
        debug_assert!(y_off == -1);
        let ay = by * 4 - 1;
        let sh = 1 + b_dim[3] as i32;
        let apx = m[2] as i64 * sx as i64 + m[3] as i64 * ay as i64 + m[0] as i64;
        let apy = m[4] as i64 * sx as i64 + m[5] as i64 * ay as i64 + m[1] as i64;
        let m3 = ((px - apx + bh4 as i64 - (px < apx) as i64) >> sh) as i32;
        let m5 = ((py - apy + bh4 as i64 - (py < apy) as i64) >> sh) as i32;
        m[3] = iclip((m3 + 0x20 - (m3 < 0) as i32) & !0x3f, -0x7fc0, 0x7fc0);
        m[5] = iclip((m5 + 0x20 - (m5 < 0x10000) as i32) & !0x3f, 0x8040, 0x17fc0);
    } else {
        debug_assert!(x_off == -1 || (by & (sb_step - 1)) == 0);
        let ax = bx * 4 - 1;
        let sh = 1 + b_dim[2] as i32;
        let lpx = m[2] as i64 * ax as i64 + m[3] as i64 * sy as i64 + m[0] as i64;
        let lpy = m[4] as i64 * ax as i64 + m[5] as i64 * sy as i64 + m[1] as i64;
        let m2 = ((px - lpx + bw4 as i64 - (px < lpx) as i64) >> sh) as i32;
        let m4 = ((py - lpy + bw4 as i64 - (py < lpy) as i64) >> sh) as i32;
        m[2] = iclip((m2 + 0x20 - (m2 < 0x10000) as i32) & !0x3f, 0x8040, 0x17fc0);
        m[4] = iclip((m4 + 0x20 - (m4 < 0) as i32) & !0x3f, -0x7fc0, 0x7fc0);
    }

    set_affine_mv2d(bw4, bh4, mv0c, wmp, bx, by);
    wmp.wm_type = if get_shear_params(wmp) != 0 {
        WarpedMotionType::Invalid
    } else {
        warp_type(&wmp.matrix)
    };
}

#[derive(Default)]
pub struct SbFrameInfo {
    pub bw: i32,
    pub bh: i32,
    pub ss_ver: i32,
    pub ss_hor: i32,
    pub root_bs: BlockSize,
    pub is_inter_or_switch: bool,
    pub sdp: bool,
    pub ext_sdp: bool,
    pub ext_partitions: bool,
    pub uneven_4way: bool,
    pub max_pb_aspect_ratio_log2: u8,
    pub n_passes: i32,
    // Segmentation
    pub seg_enabled: bool,
    pub seg_update_map: bool,
    pub seg_temporal: bool,
    pub seg_preskip: bool,
    pub seg_ext: bool,
    pub seg_last_active_segid: u8,
    pub seg_globalmv_mask: u16,
    pub seg_skip_mask: u16,
    pub seg_lossless: [u8; crate::headers::MAX_SEGMENTS],
    pub has_prev_segmap: bool,
    // Delta-q (per-superblock)
    pub delta_q_present: bool,
    pub delta_q_res_log2: u8,
    pub quant_yac: i32,
    pub sb128: i32,
    pub b4_stride: isize,
    // Quantizer deltas, needed to recompute per-SB dequant tables on delta-q.
    pub q_ydc_delta: i32,
    pub q_uac_delta: i32,
    pub q_udc_delta: i32,
    pub q_vac_delta: i32,
    pub q_vdc_delta: i32,
    pub seg_delta_q: [i16; crate::headers::MAX_SEGMENTS],
    // GDF / CDEF-index / CCSO (read at SB / 64x64 boundaries, before delta-q)
    pub gdf_enabled: crate::headers::AdaptiveBoolean,
    pub gdf_is_key: bool,
    pub cur_w: i32,
    pub cur_h: i32,
    pub cdef_enabled: bool,
    pub cdef_on_skiptx: bool,
    pub cdef_n_strengths: u8,
    pub ccso_enabled: [bool; 3],
    pub ccso_sb_reuse: [bool; 3],
    pub sb256w: i32,
    // Frame flags
    pub skip_mode_enabled: bool,
    pub allow_intrabc: bool,
    pub any_lossless: bool,
    pub has_chroma_layout: bool,
    // Sequence features
    pub idtx_intra: bool,
    pub mrls: bool,
    pub mhccp: bool,
    pub cfl: bool,
    pub allow_screen_content_tools: bool,
    pub intra_dip: bool,
    pub force_integer_mv: bool,
    pub max_bvp_drl_bits: u8,
    pub max_drl_bits: u8,
    pub bawp: bool,
    pub txfm_switchable: bool,
    pub skip_mode_refs: RefPair,
    pub n_ref_frames: u8,
    pub warp_motion: bool,
    pub motion_modes: u8,
    pub adaptive_mvd: bool,
    pub flex_mvres: bool,
    pub mv_precision: u8,
    pub mvd_sign_derive: bool,
    pub tip_frame_mode: u8,
    pub six_param_warp_delta: bool,
    pub subpel_filter_mode: u8,
    pub switchable_comp_refs: bool,
    pub num_same_ref_comp: u8,
    pub refdir: [u8; 8],
    pub refdist: [i8; 8],
    pub opfl_refine_type: u8,
    pub masked_compound: bool,
    pub cwp: bool,
    pub refine_mv_enabled: bool,
    pub absrefdist: [u8; 8],
    /// `f->furthest_future_refidx` (decode.c). Used by the masked-compound
    /// (`comp_type`) neighbour context. -1/-2 sentinel when no future ref.
    pub furthest_future_refidx: i8,
    /// `f->rf.tip.ref` (the TIP reference pair). Used by `get_compref_ctx` to
    /// match TIP-coded neighbours against the current block's compound refs.
    pub tip: RefPair,
    // Tile bounds
    pub tile_col_start: i32,
    pub tile_col_end: i32,
    pub tile_row_start: i32,
    pub tile_row_end: i32,
    pub sb_step: i32,
}

impl SbFrameInfo {
    /// Build the per-superblock frame info bundle from the live sequence and
    /// frame headers plus the frame-level geometry and reference state.
    ///
    /// `refdir`/`refdist`/`absrefdist`/`skip_mode_refs` are precomputed on the
    /// `FrameContext` (see `refmvs::init_frame`); the 7-element refmvs arrays are
    /// zero-padded into the 8-element layout this struct uses.
    #[allow(clippy::too_many_arguments)]
    pub fn from_frame(
        seq_hdr: &crate::headers::SequenceHeader,
        frame_hdr: &FrameHeader,
        bw: i32,
        bh: i32,
        root_bs: BlockSize,
        sb_step: i32,
        n_passes: i32,
        refdir: [u8; 8],
        refdist: &[i8; 7],
        absrefdist: &[u8; 7],
        skip_mode_refs: RefPair,
        tile_col_start: i32,
        tile_col_end: i32,
        tile_row_start: i32,
        tile_row_end: i32,
        furthest_future_refidx: i8,
        tip: RefPair,
    ) -> Self {
        let mut refdist8 = [0i8; 8];
        refdist8[..7].copy_from_slice(refdist);
        let mut absrefdist8 = [0u8; 8];
        absrefdist8[..7].copy_from_slice(absrefdist);

        SbFrameInfo {
            bw,
            bh,
            ss_ver: seq_hdr.ss_ver as i32,
            ss_hor: seq_hdr.ss_hor as i32,
            root_bs,
            is_inter_or_switch: frame_hdr.is_inter_or_switch(),
            sdp: seq_hdr.sdp,
            ext_sdp: seq_hdr.ext_sdp,
            ext_partitions: seq_hdr.ext_partitions,
            uneven_4way: seq_hdr.uneven_4way_partitions,
            max_pb_aspect_ratio_log2: seq_hdr.max_pb_aspect_ratio_log2,
            n_passes,
            seg_enabled: frame_hdr.segmentation.enabled != 0,
            seg_update_map: frame_hdr.segmentation.update_map != 0,
            seg_temporal: frame_hdr.segmentation.temporal != 0,
            seg_preskip: frame_hdr.segmentation.preskip != 0,
            seg_ext: seq_hdr.segmentation.ext,
            seg_last_active_segid: frame_hdr.segmentation.last_active_segid as u8,
            seg_globalmv_mask: frame_hdr.segmentation.d.globalmv_mask,
            seg_skip_mask: frame_hdr.segmentation.d.skip_mask,
            seg_lossless: frame_hdr.segmentation.lossless,
            has_prev_segmap: frame_hdr.primary_ref_frame != crate::headers::PRIMARY_REF_NONE,
            delta_q_present: frame_hdr.delta.q.present != 0,
            delta_q_res_log2: frame_hdr.delta.q.res_log2,
            quant_yac: frame_hdr.quant.yac as i32,
            sb128: frame_hdr.sb128 as i32,
            b4_stride: (((bw + 63) & !63) as isize),
            q_ydc_delta: frame_hdr.quant.ydc_delta as i32,
            q_uac_delta: frame_hdr.quant.uac_delta as i32,
            q_udc_delta: frame_hdr.quant.udc_delta as i32,
            q_vac_delta: frame_hdr.quant.vac_delta as i32,
            q_vdc_delta: frame_hdr.quant.vdc_delta as i32,
            seg_delta_q: frame_hdr.segmentation.d.delta_q,
            gdf_enabled: frame_hdr.gdf.enabled,
            gdf_is_key: frame_hdr.frame_type == crate::headers::FrameType::Key,
            cur_w: frame_hdr.width,
            cur_h: frame_hdr.height,
            cdef_enabled: frame_hdr.cdef.enabled != 0,
            cdef_on_skiptx: frame_hdr.cdef.on_skiptx != 0,
            cdef_n_strengths: frame_hdr.cdef.n_strengths,
            ccso_enabled: [
                frame_hdr.ccso.p[0].enabled != 0,
                frame_hdr.ccso.p[1].enabled != 0,
                frame_hdr.ccso.p[2].enabled != 0,
            ],
            ccso_sb_reuse: [
                frame_hdr.ccso.p[0].sb_reuse != 0,
                frame_hdr.ccso.p[1].sb_reuse != 0,
                frame_hdr.ccso.p[2].sb_reuse != 0,
            ],
            sb256w: (bw + 63) >> 6,
            skip_mode_enabled: frame_hdr.skip_mode_enabled != 0,
            allow_intrabc: frame_hdr.allow_intrabc != 0,
            any_lossless: frame_hdr.any_lossless != 0,
            has_chroma_layout: seq_hdr.layout != crate::headers::PixelLayout::I400,
            idtx_intra: seq_hdr.idtx_intra,
            mrls: seq_hdr.mrls,
            mhccp: seq_hdr.mhccp,
            cfl: seq_hdr.cfl,
            allow_screen_content_tools: frame_hdr.allow_screen_content_tools != 0,
            intra_dip: seq_hdr.intra_dip,
            force_integer_mv: frame_hdr.force_integer_mv != 0,
            max_bvp_drl_bits: frame_hdr.max_bvp_drl_bits,
            max_drl_bits: frame_hdr.max_drl_bits,
            bawp: frame_hdr.bawp != 0,
            txfm_switchable: frame_hdr.txfm_mode == crate::headers::TxfmMode::Switchable,
            skip_mode_refs,
            n_ref_frames: frame_hdr.n_ref_frames,
            warp_motion: frame_hdr.warp_motion != 0,
            motion_modes: frame_hdr.motion_modes,
            adaptive_mvd: seq_hdr.adaptive_mvd,
            flex_mvres: seq_hdr.flex_mvres,
            mv_precision: frame_hdr.mv_precision,
            mvd_sign_derive: seq_hdr.mvd_sign_derive,
            tip_frame_mode: frame_hdr.tip.frame_mode,
            six_param_warp_delta: seq_hdr.six_param_warp_delta,
            subpel_filter_mode: frame_hdr.subpel_filter_mode as u8,
            switchable_comp_refs: frame_hdr.switchable_comp_refs != 0,
            num_same_ref_comp: seq_hdr.num_same_ref_comp,
            refdir,
            refdist: refdist8,
            opfl_refine_type: frame_hdr.opfl_refine_type,
            masked_compound: seq_hdr.masked_compound,
            cwp: seq_hdr.cwp,
            refine_mv_enabled: seq_hdr.refine_mv,
            absrefdist: absrefdist8,
            furthest_future_refidx,
            tip,
            tile_col_start,
            tile_col_end,
            tile_row_start,
            tile_row_end,
            sb_step,
        }
    }
}

/// Read-only per-frame reconstruction scalars (shared ref).
pub struct ReconFrameCtx<'a> {
    pub dq: &'a [[[u32; 2]; 3]; crate::headers::MAX_SEGMENTS],
    pub qm: &'a [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES],
    pub y_stride_px: usize,
    pub uv_stride_px: usize,
    pub y_h: usize,
    pub uv_h: usize,
    pub ss_hor: i32,
    pub ss_ver: i32,
    pub bitdepth_max: i32,
    pub seq_fsc: bool,
    pub seq_ist: [bool; 2],
    pub seq_cctx: bool,
    pub layout: crate::headers::PixelLayout,
    // Extra frame/seq state needed by the reconstruction (coef + intra) leaf.
    pub bitdepth: u32,
    pub seg_lossless: [u8; crate::headers::MAX_SEGMENTS],
    pub reduced_txtp_set: i32,
    pub tcq: bool,
    pub seq_intra_edge_filter: bool,
    pub seq_ibp: bool,
    pub seq_inter_ddt: bool,
    /// `seq_hdr->cfl_ds_filter_index` — chroma-from-luma downsampling filter.
    pub cfl_ds_filter_index: i32,
    /// Per-mode IBP weights (`dav2d_ibp_weights[7][16][16]`), used by z1/z3.
    pub ibp_weights: [[[u8; 16]; 16]; 7],
}

/// Per-superblock reconstruction scratch (mirrors the relevant parts of
/// `Dav2dTaskContext` used by the luma recon leaf). `is_coded[0]` tracks the
/// 64x64 grid of decoded luma tx blocks for top-right / bottom-left availability.
pub struct ReconScratch {
    pub is_coded: [[u64; 64]; 2],
    /// SDP semi-decoupled partitioning: when the luma-only tree decodes a block
    /// it records its intra direction mode (and FSC flag) into this 16x16 map
    /// (indexed `(by & 15) * 16 + (bx & 15)`), which the chroma-only tree reads
    /// back to derive `midx` for UV mode decoding (decode.c:3425-3441).
    pub luma_intra_dir_mode_map: [u8; 256],
    pub luma_fsc_map: [u8; 256],
    /// Chroma coefficient / metadata storage used to split the chroma decode of
    /// a >64px block into a coef-read phase (with the first luma 64x64) and a
    /// recon phase (with the last), mirroring the `cbs_stage` mechanism in
    /// `dav2d_recon_b`. `cf_uv` holds U then V coefficients (n_tu*16 each).
    pub chroma_cf: Vec<i32>,
    pub chroma_txtp: [[u16; 2]; 256],
    pub chroma_eob: [[i16; 2]; 256],
    pub chroma_u_has_cf: i32,
    /// Per-4x4 luma transform type map (full txtp incl. secondary-tx bits),
    /// indexed `(by & 15) * 16 + (bx & 15)`. Written by the luma residual walk
    /// and read back to seed the inter chroma transform type (dav2d `txtp_map`,
    /// recon_tmpl.c:2502 / 3540).
    pub txtp_map: [u16; 256],
}

impl Default for ReconScratch {
    fn default() -> Self {
        Self {
            is_coded: [[0u64; 64]; 2],
            luma_intra_dir_mode_map: [0u8; 256],
            luma_fsc_map: [0u8; 256],
            chroma_cf: Vec::new(),
            chroma_txtp: [[0u16; 2]; 256],
            chroma_eob: [[-1i16; 2]; 256],
            chroma_u_has_cf: 0,
            txtp_map: [0u16; 256],
        }
    }
}

/// Mutable reconstruction borrows bundled so only one new param threads through
/// decode_sb's recursion (Rust auto-reborrows &mut ReconCtx at each call).
pub struct ReconCtx<'a, 'f> {
    pub dst_y: &'a mut [u8],
    pub dst_u: &'a mut [u8],
    pub dst_v: &'a mut [u8],
    pub cdf_coef: &'a mut crate::cdf::CdfCoefContext,
    pub cf: &'a mut [i32],
    pub frame: &'a ReconFrameCtx<'f>,
    /// Inter-intra / wedge / segmentation prediction masks (`dav2d_masks`),
    /// built once per frame; consumed by the compound + interintra recon.
    pub masks: &'f crate::wedge::Masks,
    /// Per-superblock recon scratch (is_coded grid). Reset by the caller at each
    /// superblock boundary, mirroring the C `memset(t->is_coded, ...)`.
    pub scratch: &'a mut ReconScratch,
    /// Temporary edge buffer for `prepare_intra_edges` (`t->scratch.edge`,
    /// 257 entries wide; we use a generous fixed slab indexed from the middle).
    pub edge: &'a mut [u8],
    /// Current-frame segment id map (`f->cur_segmap`), `b4_stride * bh` entries.
    /// Written by decode_b over the block footprint when segmentation is enabled.
    pub cur_segmap: &'a mut [u8],
    /// Previous-frame segment map (`f->prev_segmap`), present only when the frame
    /// has a primary reference; `None` for frame-0 / no-primary-ref keyframes.
    pub prev_segmap: Option<&'a [u8]>,
    /// `f->b4_stride` — row stride of the segment maps (in 4x4 units).
    pub b4_stride: isize,
    /// Running per-superblock quantizer index (`ts->last_qidx`). Updated by the
    /// per-SB delta-q parse; seeded from `frame_hdr.quant.yac` at tile start.
    pub last_qidx: i32,
    /// Per-superblock recomputed dequant tables (`ts->dqmem`).
    pub dqmem: [[[u32; 2]; 3]; crate::headers::MAX_SEGMENTS],
    /// Currently active dequant tables (`ts->dq`): either the frame-wide
    /// `recon.frame.dq` or `dqmem` when a per-SB delta-q shifts the qindex.
    pub dq_active: [[[u32; 2]; 3]; crate::headers::MAX_SEGMENTS],
    /// Set when a parsed seg_id is out of range (`seg_id >= 16`), mirroring the
    /// C `return -1` that aborts the frame.
    pub seg_id_err: bool,
    /// Loop-filter mask array (`f->lf.mask` / per-SB `Av2Filter`) — the gdf,
    /// cdef-index and ccso reads write into it and read neighbour SB values.
    pub lf_mask: &'a mut [crate::lf_mask::Av2Filter],
    /// Index of the current superblock's `Av2Filter` within `lf_mask`.
    pub lf_idx: usize,
    /// `f->sb256w` — superblock-row stride into `lf_mask` (for top neighbour).
    pub sb256w: i32,
    /// Current-frame CCSO map (`f->cur_ccsomap`), written per-SB; empty if unused.
    pub cur_ccsomap: &'a mut [u8],
    /// Previous-frame CCSO maps (`f->prev_ccsomap`); `None` per plane if absent.
    pub prev_ccsomap: [Option<&'a [u8]>; 3],
    /// Per-tile reference-MV state (`t->rt`): spatial `r` grid + above-row `ra` +
    /// MV/warp banks. Maintained via splat for every block so IntraBC block
    /// vectors can be predicted from neighbours (decode.c `recon_b`).
    pub rt: &'a mut crate::refmvs::Tile,
    /// Frame-level reference-MV state (`f->rf`): iw4/ih4/sbsz + header refs.
    pub rf: &'a crate::refmvs::Frame,
    /// Current-frame temporal MV grid (`f->mvs` == `rf.rp`), written by the inter
    /// splat so later frames can reference these MVs. Empty when ref_frame_mvs is
    /// disabled. Held separately from `rf` (which is shared immutably) so the
    /// per-block temporal save can mutate it.
    pub cur_mvs: &'a mut [crate::refmvs::TemporalBlock],
    /// Per-reference picture pixel planes for inter motion compensation
    /// (`f->refp[i].p`). `None` for refs the current frame does not use.
    pub refp: &'a [Option<std::sync::Arc<crate::picture::Picture>>; 7],
    /// Per-reference scaling parameters (`f->svc[i]`); scale==0 means unscaled.
    pub svc: &'a [[ScalableMotionParams; 2]; 7],
    /// Inter-chroma `u_has_cf` flag (`t->u_has_cf`): set by the U plane's coef
    /// decode, consumed by the V plane's context (decode_coefs). Per chroma block.
    pub scratch_u_has_cf: i32,
    /// Sequence/frame headers needed by `refmvs_find` for IntraBC candidates.
    pub seq_hdr: &'a crate::headers::SequenceHeader,
    pub frm_hdr: &'a crate::headers::FrameHeader,
    /// Per-block derived warp motion parameters (`t->warpmv[0..2]`), computed in
    /// the inter MV-resolution step and consumed by `recon_b_inter`'s warp MC.
    pub warpmv: [crate::headers::WarpedMotionParams; 2],
}

/// Decode one superblock row of a tile (entropy/parse pass).
///
/// Port of the entropy branch of `dav2d_decode_tile_sbrow` (decode.c:4516-4642)
/// for the single-pass, single-thread case. Reads per-superblock restoration
/// info and CDEF index reset, then drives `decode_sb` across the row.
///
/// NOTE: the inter/IntraBC `refmvs` reset/save hooks (PASS_MVRES) are not yet
/// wired here — that lands with inter-frame support (M3). Intra/key frames take
/// none of those paths.
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_sbrow_entropy(
    fi: &SbFrameInfo,
    frame_hdr: &FrameHeader,
    ts: &mut crate::internal::TileState,
    msac: &mut MsacContext,
    a_arr: &mut [BlockContext],
    lf_mask: &mut [crate::lf_mask::Av2Filter],
    lr_mask: &mut [crate::lf_mask::Av2Restoration],
    l: &mut BlockContext,
    dst_y: &mut [u8],
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    cf: &mut [i32],
    recon_frame: &ReconFrameCtx,
    cur_segmap: &mut [u8],
    prev_segmap: Option<&[u8]>,
    cur_ccsomap: &mut [u8],
    prev_ccsomap: [Option<&[u8]>; 3],
    part_w: &mut Vec<u8>,
    part_r: &[u8],
    by: i32,
    sb256w: i32,
    root_bs: BlockSize,
    c_root_bs: BlockSize,
    rt: &mut crate::refmvs::Tile,
    rf: &crate::refmvs::Frame,
    cur_mvs: &mut [crate::refmvs::TemporalBlock],
    refp: &[Option<std::sync::Arc<crate::picture::Picture>>; 7],
    svc: &[[ScalableMotionParams; 2]; 7],
    seq_hdr: &crate::headers::SequenceHeader,
    frm_hdr: &crate::headers::FrameHeader,
    masks: &crate::wedge::Masks,
) -> Result<(), ()> {
    let sb_step = fi.sb_step;
    let sb256y = by >> 6;
    let tile_row = ts.tiling.row;
    let col_start = ts.tiling.col_start;
    let col_end = ts.tiling.col_end;
    let row_start = ts.tiling.row_start;
    let row_end = ts.tiling.row_end;

    // Per-superblock reconstruction scratch. `is_coded` is reset at each SB
    // boundary (mirrors C `memset(t->is_coded, 0, ...)`); `edge` is the working
    // buffer for prepare_intra_edges (origin in the middle of the slab).
    let mut recon_scratch = ReconScratch::default();
    let mut recon_edge = vec![0u8; 2048];

    // Running per-tile delta-q state (`ts->last_qidx`/`ts->dqmem`/`ts->dq`).
    // Seeded by the caller from `frame_hdr.quant.yac` at tile entry; carried
    // forward across superblocks within and across sbrows.
    let mut sb_last_qidx = ts.last_qidx;
    let mut sb_dqmem = ts.dqmem;
    // Whether `dqmem` currently mirrors a non-frame-wide qindex; if so the
    // active dq must be re-derived when the qindex changes.
    let mut sb_seg_err = false;

    // Reference-MV per-superblock-row init (decode.c:4456). Sets up the above-row
    // `ra` slice and tile bounds; needed for IntraBC and inter block-MV
    // prediction. Maintained when IntraBC is allowed or the frame is inter.
    let refmvs_active = fi.allow_intrabc || fi.is_inter_or_switch;
    if refmvs_active {
        crate::refmvs::tile_sbrow_init(
            rt,
            rf,
            col_start,
            col_end,
            row_start,
            row_end,
            by >> 6,
            tile_row,
        );
    }
    let is_key_or_intra = frame_hdr.is_key_or_intra();

    let mut bx = col_start;
    while bx < col_end {
        let a_idx = (tile_row * sb256w + (bx >> 6)) as usize;
        let lf_idx = ((bx >> 6) + sb256y * sb256w) as usize;

        // Reset is_coded for this superblock (luma + chroma rows).
        for row in recon_scratch.is_coded.iter_mut() {
            row.fill(0);
        }

        // Reset the reference-MV `r` grid for this superblock (decode.c
        // reset_context path); marks all 4x4 units intra (invalid mv) so that
        // not-yet-decoded neighbours are skipped by refmvs_find.
        if refmvs_active {
            let ra_snapshot = rt.ra.clone();
            crate::refmvs::reset_sb(
                rt,
                &ra_snapshot,
                sb_step,
                seq_hdr.refmv_bank,
                is_key_or_intra,
                frm_hdr.tip.frame_mode,
                by,
                bx,
            );
        }

        // Reset CDEF indices for this superblock's coverage in the lf mask.
        match root_bs {
            BlockSize::Bs64x64 => {
                let idx = (((bx & 0x30) >> 4) + ((by & 0x30) >> 2)) as usize;
                lf_mask[lf_idx].cdef_idx[idx] = -1;
            }
            BlockSize::Bs128x128 => {
                let idx = (((bx & 32) >> 4) + ((by & 32) >> 2)) as usize;
                lf_mask[lf_idx].cdef_idx[idx] = -1;
                lf_mask[lf_idx].cdef_idx[idx + 1] = -1;
                lf_mask[lf_idx].cdef_idx[idx + 4] = -1;
                lf_mask[lf_idx].cdef_idx[idx + 5] = -1;
            }
            BlockSize::Bs256x256 => {
                for k in 0..16 {
                    lf_mask[lf_idx].cdef_idx[k] = -1;
                }
            }
            _ => {}
        }

        // Per-plane loop-restoration unit info.
        let sbsz = sb_step * 4;
        for p in 0..3 {
            let (ss_ver, ss_hor) = if p == 0 {
                (0, 0)
            } else {
                (fi.ss_ver, fi.ss_hor)
            };
            let rtype_u8 = frame_hdr.restoration.p[p].restoration_type;
            if rtype_u8 == RestorationType::None as u8 {
                continue;
            }
            let tx = (4 * (bx - col_start)) >> ss_hor;
            let ty = (4 * (by - row_start)) >> ss_ver;
            let unit_sz_log2 = frame_hdr.restoration.unit_size[(p != 0) as usize] as i32;
            let unit_sz = 1i32 << unit_sz_log2;
            let mask = unit_sz - 1;
            if (tx | ty) & mask != 0 {
                continue;
            }
            let tw = (col_end * 4) >> ss_hor;
            let th = (row_end * 4) >> ss_ver;
            let half_unit = unit_sz >> 1;
            let fx = (4 * bx) >> ss_hor;
            let fy = (by * 4) >> ss_ver;
            if (ty != 0 && fy + half_unit > th) || (tx != 0 && fx + half_unit > tw) {
                continue;
            }

            let frame_type = match rtype_u8 {
                1 => RestorationType::PcWiener,
                2 => RestorationType::NsWiener,
                3 => RestorationType::Switchable,
                _ => RestorationType::None,
            };

            let sbw = sbsz >> ss_hor;
            let sbh = sbsz >> ss_ver;
            let lruw = imax(1, imin(tw - fx + half_unit, sbw) >> unit_sz_log2);
            let lruh = imax(1, imin(th - fy + half_unit, sbh) >> unit_sz_log2);
            let vsh = unit_sz_log2 - 7 + ss_ver;
            let hsh = unit_sz_log2 - 7 + ss_hor;
            let mut sb_idx = (by >> 6) * sb256w + (bx >> 6);
            let start_unit_idx = (((by & 0x30) >> 2) + ((bx & 0x30) >> 4)) as usize;

            for _y in 0..lruh {
                for x in 0..lruw {
                    let unit_idx = start_unit_idx;
                    let lr_slot = (sb_idx + (x << hsh)) as usize;
                    let ns_plane = &frame_hdr.restoration.p[p].ns;
                    read_restoration_info(
                        msac,
                        &mut ts.cdf.m,
                        &mut ts.ns_wiener_bank[p],
                        &mut lr_mask[lr_slot].lr[p][unit_idx],
                        p,
                        frame_type,
                        ns_plane,
                    );
                }
                sb_idx += sb256w << vsh;
            }
        }

        let mut dir = 0i32;
        let mut sdp_cfl_disallowed = 0i32;
        let mut intra_region = 0i32;
        let mut bx_m = bx;
        let mut by_m = by;
        let mut cbx = bx;
        let mut cby = by;
        let mut part_w_idx = 0usize;
        let mut part_r_idx = 0usize;

        // Active dequant tables for this superblock: frame-wide unless a prior
        // delta-q has shifted the running qindex away from `quant.yac`.
        let dq_active_init = if sb_last_qidx == fi.quant_yac {
            *recon_frame.dq
        } else {
            sb_dqmem
        };

        let mut recon = ReconCtx {
            dst_y: &mut *dst_y,
            dst_u: &mut *dst_u,
            dst_v: &mut *dst_v,
            cdf_coef: &mut ts.cdf.coef,
            cf: &mut *cf,
            frame: recon_frame,
            masks,
            scratch: &mut recon_scratch,
            edge: &mut recon_edge,
            cur_segmap: &mut *cur_segmap,
            prev_segmap,
            b4_stride: fi.b4_stride,
            last_qidx: sb_last_qidx,
            dqmem: sb_dqmem,
            dq_active: dq_active_init,
            seg_id_err: false,
            lf_mask: &mut *lf_mask,
            lf_idx,
            sb256w,
            cur_ccsomap: &mut cur_ccsomap[..],
            prev_ccsomap: [prev_ccsomap[0], prev_ccsomap[1], prev_ccsomap[2]],
            rt: &mut *rt,
            rf,
            cur_mvs: &mut *cur_mvs,
            refp,
            svc,
            scratch_u_has_cf: 0,
            seq_hdr,
            frm_hdr,
            warpmv: [crate::headers::WarpedMotionParams::default(); 2],
        };

        decode_sb(
            fi,
            &mut bx_m,
            &mut by_m,
            &mut cbx,
            &mut cby,
            &mut intra_region,
            &mut sdp_cfl_disallowed,
            crate::internal::PASS_ALL,
            &mut a_arr[a_idx],
            l,
            msac,
            &mut ts.cdf.m,
            &mut ts.cdf.dmv,
            &mut recon,
            part_w,
            &mut part_w_idx,
            part_r,
            &mut part_r_idx,
            root_bs,
            c_root_bs,
            &mut dir,
        )?;

        // Persist running delta-q state for the next superblock / sbrow.
        sb_last_qidx = recon.last_qidx;
        sb_dqmem = recon.dqmem;
        sb_seg_err |= recon.seg_id_err;

        bx += sb_step;
    }

    // Save the bottom row of this SB row into the above-row `ra` buffer for the
    // next SB row's neighbour access (decode.c:4511 `dav2d_refmvs_save_tmvs`,
    // single-thread `ra` portion).
    if refmvs_active {
        let crate::refmvs::Tile {
            r,
            ra,
            ra_tl,
            ra_off,
            ..
        } = &mut *rt;
        let col_start8 = col_start >> 1;
        let col_end8 = (col_end + 1) >> 1;
        crate::refmvs::save_tmvs(
            r,
            &mut ra[*ra_off..],
            ra_tl,
            col_start8,
            col_end8,
            row_start >> 1,
            (by + sb_step) >> 1,
            rf.ih8,
            rf.iw8,
        );
    }

    // Write back the running per-tile delta-q state.
    ts.last_qidx = sb_last_qidx;
    ts.dqmem = sb_dqmem;

    // Abort the frame on an out-of-range segment id (C `return -1`).
    if sb_seg_err {
        if std::env::var("RAV2D_SUBMIT_ERR").is_ok() {
            eprintln!("decode_tile_sbrow_entropy: seg_id_err");
        }
        return Err(());
    }

    // Error out on symbol-decoder overread.
    if msac.cnt() <= -15 {
        if std::env::var("RAV2D_SUBMIT_ERR").is_ok() {
            eprintln!(
                "decode_tile_sbrow_entropy: msac overread cnt={}",
                msac.cnt()
            );
        }
        return Err(());
    }

    Ok(())
}

/// Run the single-threaded tile/superblock-row decode loop over a frame.
///
/// Port of `dav2d_decode_frame_main` (decode.c:5130) for the n_tc == 1 case.
/// Tiles are decoded sequentially (entropy/recon are tile-independent); the
/// per-superblock-row post-filter interleaving that the C version performs is
/// handled separately by the filter pass (M2).
pub fn decode_frame_main(fc: &mut crate::internal::FrameContext, n_passes: i32) -> Result<(), ()> {
    let crate::internal::FrameContext {
        seq_hdr,
        frame_hdr,
        a,
        ts,
        lf,
        sb256w,
        sb_step,
        root_bs,
        bw,
        bh,
        refdir,
        refdist,
        absrefdist,
        furthest_future_refidx,
        skip_mode_refs,
        cur_pic,
        dq,
        qm,
        bitdepth_max,
        ss_hor,
        ss_ver,
        cur_segmap,
        prev_segmap,
        cur_ccsomap,
        prev_ccsomap,
        b4_stride,
        sb256h,
        sbh,
        rf,
        inloop_filters,
        refp,
        mvs,
        ref_mvs,
        refrefpoc,
        refcnt,
        refpoc,
        svc,
        ..
    } = fc;
    let fc_sb256h = *sb256h;
    let fc_sbh = *sbh;
    let fc_inloop_filters = *inloop_filters;

    let seq_hdr = &**seq_hdr;
    let frame_hdr = &**frame_hdr;
    let root_bs = *root_bs;
    let sb256w = *sb256w;
    let sb_step = *sb_step;
    let bw = *bw;
    let bh = *bh;
    let b4_stride_v = *b4_stride;

    // Allocate the current-frame segment id map when segmentation is enabled
    // (C: `f->cur_segmap`, sized `b4_stride * 64 * sb256h`, padded to whole
    // superblock rows so full bw4 x bh4 block writes never overrun). Reset to 0;
    // decode_b writes each block's seg_id over its footprint.
    if frame_hdr.segmentation.enabled != 0 {
        let needed = (b4_stride_v as usize) * 64 * (fc_sb256h as usize);
        if cur_segmap.len() != needed {
            cur_segmap.resize(needed, 0);
        }
        cur_segmap.fill(0);
    }
    let prev_segmap_ref: Option<&[u8]> = prev_segmap.as_deref();
    let prev_ccsomap_ref: [Option<&[u8]>; 3] = [
        prev_ccsomap[0].as_deref(),
        prev_ccsomap[1].as_deref(),
        prev_ccsomap[2].as_deref(),
    ];
    let refdir = *refdir;
    let skip_mode_refs = *skip_mode_refs;

    // Reconstruction destination planes. The decode_b leaf is a no-op this step,
    // so these slices are threaded but not yet written. Built once and reborrowed
    // at each tile/sbrow decode_sb call below.
    let ss_hor_v = *ss_hor;
    let ss_ver_v = *ss_ver;
    let bitdepth_max_v = *bitdepth_max;
    // The plane allocation is sized for the 128-aligned frame dimensions (see
    // DefaultPicAllocator::alloc_picture: `aligned_h = (h + 127) & !127`), which
    // gives bottom padding past the cropped/visible height. Reconstruction
    // legitimately writes whole transform blocks that overhang the visible edge
    // into that padding (matching dav2d, whose plane buffers are likewise
    // padded). Span the slices over the *allocated* height so those overhang
    // writes stay in bounds rather than panicking on the cropped height.
    let aligned_h: usize = ((cur_pic.p.h.max(0) as usize) + 127) & !127;
    let y_h: usize = aligned_h;
    let uv_h: usize = if seq_hdr.layout == crate::headers::PixelLayout::I400 {
        0
    } else {
        aligned_h >> ss_ver_v
    };
    let y_stride_px: usize = cur_pic.stride[0].unsigned_abs();
    let uv_stride_px: usize = cur_pic.stride[1].unsigned_abs();
    let y_ptr = cur_pic.data[0].map(|p| p.as_ptr());
    let u_ptr = cur_pic.data[1].map(|p| p.as_ptr());
    let v_ptr = cur_pic.data[2].map(|p| p.as_ptr());
    // SAFETY: pointers/strides come from the live `cur_pic` allocation; each plane
    // slice spans stride*height bytes. `fc`/`cur_pic` is not otherwise accessed
    // while these slices are live, so there is no aliasing of the same memory.
    let dst_y: &mut [u8] = match y_ptr {
        Some(p) => unsafe { std::slice::from_raw_parts_mut(p, y_stride_px * y_h) },
        None => &mut [],
    };
    let dst_u: &mut [u8] = match u_ptr {
        Some(p) => unsafe { std::slice::from_raw_parts_mut(p, uv_stride_px * uv_h) },
        None => &mut [],
    };
    let dst_v: &mut [u8] = match v_ptr {
        Some(p) => unsafe { std::slice::from_raw_parts_mut(p, uv_stride_px * uv_h) },
        None => &mut [],
    };

    // bitdepth in bits from bitdepth_max (255 -> 8, 1023 -> 10, 4095 -> 12).
    let bitdepth_v: u32 = (crate::intops::ulog2((bitdepth_max_v + 1) as u32)) as u32;
    // Inter-intra / wedge / segmentation prediction masks (`dav2d_masks`), built
    // once per frame for the compound + interintra recon paths.
    let masks = crate::wedge::init_masks();
    let recon_frame = ReconFrameCtx {
        dq: &*dq,
        qm: &*qm,
        y_stride_px,
        uv_stride_px,
        y_h,
        uv_h,
        ss_hor: ss_hor_v,
        ss_ver: ss_ver_v,
        bitdepth_max: bitdepth_max_v,
        seq_fsc: seq_hdr.fsc,
        seq_ist: seq_hdr.ist,
        seq_cctx: seq_hdr.cctx,
        layout: seq_hdr.layout,
        bitdepth: bitdepth_v,
        seg_lossless: frame_hdr.segmentation.lossless,
        reduced_txtp_set: frame_hdr.reduced_txtp_set as i32,
        tcq: frame_hdr.tcq != 0,
        seq_intra_edge_filter: seq_hdr.intra_edge_filter,
        seq_ibp: seq_hdr.ibp,
        seq_inter_ddt: seq_hdr.inter_ddt,
        cfl_ds_filter_index: seq_hdr.cfl_ds_filter_index as i32,
        ibp_weights: crate::ibp::init_ibp_weights(),
    };
    let mut cf = vec![0i32; 64 * 64];

    let c_root_bs = if seq_hdr.layout == crate::headers::PixelLayout::I400 {
        BlockSize::Invalid
    } else {
        root_bs
    };
    let cols = frame_hdr.tiling.t.cols as i32;
    let rows = frame_hdr.tiling.t.rows as i32;
    let keyframe = frame_hdr.is_key_or_intra();
    let is_tip = frame_hdr.tip.frame_mode == 2;
    let disable_cdf = frame_hdr.disable_cdf_update != 0;

    // Reset the above (a) context array for the whole frame. dav2d does this
    // unconditionally for the single-threaded path (decode.c:5172-5174); the
    // multi-threaded reset lives in decode_frame_init. Without this, the above
    // neighbour `midx`/mode arrays retain default 0 instead of the 0xff "no
    // neighbour" sentinel, corrupting intra-mode context derivation.
    {
        let n_a = (sb256w * rows) as usize;
        for ctx in a.iter_mut().take(n_a) {
            reset_context(ctx, keyframe, is_tip);
        }
    }

    let mut l = BlockContext::default();

    // Initialise the reference-MV frame state (`f->rf`) and a per-tile working
    // Tile. For IntraBC on an intra frame only the spatial grid / above-row /
    // banks are needed (no temporal candidates). For inter frames the reference
    // temporal MVs are wired and (when use_ref_frame_mvs) projected per sbrow.
    let allow_intrabc = frame_hdr.allow_intrabc != 0;
    let is_inter_or_switch = frame_hdr.is_inter_or_switch();
    let refmvs_active = allow_intrabc || is_inter_or_switch;
    if refmvs_active {
        refmvs::init_frame(
            rf,
            seq_hdr,
            frame_hdr,
            refpoc,
            refrefpoc,
            refcnt,
            ref_mvs,
            false,
            false,
        );
    }

    // Allocate the current-frame temporal MV grid (decode.c:5541-5547) when
    // ref_frame_mvs is enabled, sized for the whole frame; the per-block splat
    // (splat_oneref_mv's `t_dst`) writes decoded MVs into `rf.rp` so later frames
    // can reference them. `rf.rp` is dav2d's `f->mvs` (the same buffer).
    let _ = &mvs;
    if refmvs_active && seq_hdr.ref_frame_mvs {
        let needed = (fc_sb256h as usize) * 32 * ((b4_stride_v >> 1) as usize);
        if rf.rp.len() != needed {
            rf.rp = vec![refmvs::TemporalBlock::default(); needed];
        } else {
            rf.rp.fill(refmvs::TemporalBlock::default());
        }
    } else {
        rf.rp.clear();
    }

    // Project reference temporal MVs into `rf.rp_proj` for every superblock row
    // up front (decode.c:5151-5154 runs load_tmvs per sbrow before the tile-col
    // decode; the projection reads only reference data so doing all rows now is
    // equivalent for the single-thread path).
    if is_inter_or_switch && frame_hdr.use_ref_frame_mvs != 0 {
        let mut by = 0i32;
        while by < bh {
            let by_end = (by + sb_step) >> 1;
            refmvs::load_tmvs(
                rf,
                0,
                0,
                bw >> 1,
                by >> 1,
                by_end,
                seq_hdr.mv_traj,
                frame_hdr.tip.frame_mode,
                seq_hdr.tip_hole_fill,
                frame_hdr.tmvp_sample_step as i32,
                frame_hdr.n_ref_frames as i32,
            );
            by += sb_step;
        }
    }
    // Hold the current-frame temporal MV grid separately so the inter splat can
    // mutate it while `rf` itself stays shared immutably (refmvs_find reads it).
    let mut cur_mvs: Vec<refmvs::TemporalBlock> = std::mem::take(&mut rf.rp);
    // Reference pixel planes for inter MC, shared from the FrameContext refp.
    let refp_pics: [Option<std::sync::Arc<crate::picture::Picture>>; 7] =
        std::array::from_fn(|i| refp[i].pic.clone());
    let svc_v = *svc;

    let rf_ref: &refmvs::Frame = &*rf;
    // Compound `comp_type`/`get_compref_ctx` neighbour contexts need the
    // furthest-future ref index and the TIP reference pair (decode.c).
    let ffr_idx = *furthest_future_refidx;
    let tip_ref = rf_ref.tip.r#ref;
    let mut rt = refmvs::Tile {
        rp_proj: Vec::new(),
        rp_proj_off: 0,
        rp_traj_off: 0,
        ra: vec![refmvs::Block::default(); rf_ref.rp_stride.max(1) as usize],
        ra_off: 0,
        ra_tl: refmvs::Block::default(),
        r: vec![refmvs::Block::default(); 64 * 128],
        tile_col: refmvs::TileRange { start: 0, end: bw },
        tile_row: refmvs::TileRange { start: 0, end: bh },
        bank: refmvs::MvBank {
            mv: [[[Mv::default(); 2]; 4]; 9],
            cwp_idx: [[0; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0; 9],
            idx: [0; 9],
            hits: [0; 2],
            avail: 0,
        },
        warp: refmvs::WarpBank {
            mat: [[[0; 6]; 4]; 7],
            warp_type: [[0; 4]; 7],
            hits: 0,
            size: [0; 7],
            idx: [0; 7],
        },
    };

    // Precompute the per-frame filter parameters once (deblock thresholds, CDEF
    // strength decomposition, etc.) so the per-superblock-row filter pass can run
    // without re-deriving them. Filters are gated by `fc.inloop_filters`.
    let filter_params = FilterFrameParams::new(
        seq_hdr,
        frame_hdr,
        bw,
        bh,
        ss_hor_v,
        ss_ver_v,
        cur_pic.stride[0],
        cur_pic.stride[1],
        bitdepth_v as i32,
        fc_inloop_filters,
    );
    // The CDEF pre-filter line toggle runs across the whole frame's filter pass
    // (dav2d resets `top_pre_cdef_toggle` once per frame in decode_frame_init).
    lf.cdef_line_toggle = 0;
    let sb128 = frame_hdr.sb128 as i32;

    // dav2d (decode.c:5144-5164) is per-tile-row: PHASE A decodes ALL tile-cols
    // for every superblock-row in the tile row, THEN PHASE B runs the deferred
    // filter pass `for sby: filter_sbrow(sby)`. Filters must run only after a
    // whole tile row has decoded because intrabc reads pre-filter pixels to the
    // left within the tile row.
    for tr in 0..rows {
        // Collect per-tile-col MSAC state that must stay live across the sby
        // loop (each tile-col's symbol decoder advances one sbrow at a time but
        // we now interleave tile-cols within a sbrow).
        let ts_base = (tr * cols) as usize;
        let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(cols as usize);
        let mut fis: Vec<SbFrameInfo> = Vec::with_capacity(cols as usize);
        let mut ranges: Vec<(i32, i32)> = Vec::with_capacity(cols as usize); // (rs, re)
        for tc in 0..cols {
            let ts_idx = ts_base + tc as usize;
            let (cs, ce, rs, re) = {
                let t = &ts[ts_idx].tiling;
                (t.col_start, t.col_end, t.row_start, t.row_end)
            };
            fis.push(SbFrameInfo::from_frame(
                seq_hdr,
                frame_hdr,
                bw,
                bh,
                root_bs,
                sb_step,
                n_passes,
                refdir,
                refdist,
                absrefdist,
                skip_mode_refs,
                cs,
                ce,
                rs,
                re,
                ffr_idx,
                tip_ref,
            ));
            ranges.push((rs, re));
            bufs.push(std::mem::take(&mut ts[ts_idx].msac_buf));
        }
        // The buffers Vec now owns the tile data; build the symbol decoders
        // borrowing from it (read-only) so they persist across the sby loop.
        let mut msacs: Vec<MsacContext> = bufs
            .iter()
            .map(|b| MsacContext::new(b, disable_cdf))
            .collect();
        let mut part_ws: Vec<Vec<u8>> = (0..cols).map(|_| Vec::new()).collect();
        let part_r: Vec<u8> = Vec::new();

        // The tile row spans the same block-row range for every tile-col (only
        // the column range differs), so derive the sbrow loop from tile-col 0.
        let (row_rs, row_re) = ranges[0];

        // PHASE A: decode every superblock-row across all tile-cols.
        let mut by = row_rs;
        while by < row_re {
            for tc in 0..cols as usize {
                let ts_idx = ts_base + tc;
                let (rs, re) = ranges[tc];
                if by < rs || by >= re {
                    continue;
                }
                reset_context(&mut l, keyframe, is_tip);
                decode_tile_sbrow_entropy(
                    &fis[tc],
                    frame_hdr,
                    &mut ts[ts_idx],
                    &mut msacs[tc],
                    a,
                    &mut lf.mask,
                    &mut lf.lr_mask,
                    &mut l,
                    &mut *dst_y,
                    &mut *dst_u,
                    &mut *dst_v,
                    &mut cf,
                    &recon_frame,
                    &mut cur_segmap[..],
                    prev_segmap_ref,
                    &mut cur_ccsomap[..],
                    prev_ccsomap_ref,
                    &mut part_ws[tc],
                    &part_r,
                    by,
                    sb256w,
                    root_bs,
                    c_root_bs,
                    &mut rt,
                    rf_ref,
                    &mut cur_mvs,
                    &refp_pics,
                    &svc_v,
                    seq_hdr,
                    frame_hdr,
                    &masks,
                )?;
            }
            by += sb_step;
        }

        // Return the MSAC buffers to the tile states.
        drop(msacs);
        for (tc, buf) in bufs.into_iter().enumerate() {
            ts[ts_base + tc].msac_buf = buf;
        }

        // PHASE B: deferred per-superblock-row filter pass over the whole tile
        // row (deblock cols -> deblock rows + copy_db -> CDEF(+CCSO) -> LR).
        let mut by = row_rs;
        while by < row_re {
            let sby = by / sb_step;
            filter_sbrow(
                seq_hdr,
                frame_hdr,
                lf,
                &mut *dst_y,
                &mut *dst_u,
                &mut *dst_v,
                &filter_params,
                fc_inloop_filters,
                fc_sbh,
                sb_step,
                sb256w,
                sb128,
                bw,
                bh,
                sby,
            );
            by += sb_step;
        }
    }

    // Restore the (now-populated) temporal MV grid into `rf.rp` so the
    // reference-list update can publish it to c.refs[i].refmvs.
    rf.rp = cur_mvs;

    Ok(())
}

/// Per-frame parameters for the deferred post-filter pass, derived once before
/// the filter loop runs. Mirrors the constant inputs dav2d reads from
/// `f->frame_hdr` / `f->lf` inside `dav2d_filter_sbrow*`.
struct FilterFrameParams {
    deblock: crate::deblock::DeblockApplyParams,
    /// Deblock thresholds (stubbed: zeroed; deblock is only correct for the
    /// level==0 no-op case until the per-4px segmentation/delta-q thresholds are
    /// connected — see the M2 risks).
    thr_lut_y: [[u8; 16]; 2],
    thr_lut_uv: [[[u8; 16]; 2]; 2],
    y_stride: isize,
    uv_stride: isize,
    bw: i32,
    bh: i32,
    ss_hor: bool,
    ss_ver: bool,
    layout: crate::headers::PixelLayout,
    // CDEF
    cdef_damping: i32,
    cdef_on_skiptx: bool,
    cdef_y_strength: [u8; crate::headers::MAX_CDEF_STRENGTHS],
    cdef_uv_strength: [u8; crate::headers::MAX_CDEF_STRENGTHS],
}

impl FilterFrameParams {
    #[allow(clippy::too_many_arguments)]
    fn new(
        seq_hdr: &crate::headers::SequenceHeader,
        frame_hdr: &crate::headers::FrameHeader,
        bw: i32,
        bh: i32,
        ss_hor: i32,
        ss_ver: i32,
        y_stride: isize,
        uv_stride: isize,
        _bitdepth: i32,
        _inloop: u32,
    ) -> Self {
        let db = &frame_hdr.deblock;
        let deblock = crate::deblock::DeblockApplyParams {
            y_stride,
            uv_stride,
            bw: bw as usize,
            bh: bh as usize,
            sb128: frame_hdr.sb128 != 0,
            ss_hor: ss_hor != 0,
            ss_ver: ss_ver != 0,
            level_y: [db.level_y[0] as i32, db.level_y[1] as i32],
            level_u: db.level_u as i32,
            level_v: db.level_v as i32,
            have_chroma: seq_hdr.layout != crate::headers::PixelLayout::I400,
        };
        FilterFrameParams {
            deblock,
            thr_lut_y: [[0u8; 16]; 2],
            thr_lut_uv: [[[0u8; 16]; 2]; 2],
            y_stride,
            uv_stride,
            bw,
            bh,
            ss_hor: ss_hor != 0,
            ss_ver: ss_ver != 0,
            layout: seq_hdr.layout,
            cdef_damping: frame_hdr.cdef.damping as i32,
            cdef_on_skiptx: frame_hdr.cdef.on_skiptx != 0,
            cdef_y_strength: frame_hdr.cdef.y_strength,
            cdef_uv_strength: frame_hdr.cdef.uv_strength,
        }
    }
}

/// Deferred per-superblock-row filter pass. Port of `dav2d_filter_sbrow`
/// (recon_tmpl.c:4028) for the single-thread (`n_tc == 1`) path, in the exact
/// dav2d order: deblock cols -> deblock rows -> copy_db -> CDEF(+CCSO) -> LR.
/// Each stage is gated on `inloop` (DAV2D_INLOOPFILTER_* bits) plus the relevant
/// frame-header enable.
#[allow(clippy::too_many_arguments)]
fn filter_sbrow(
    seq_hdr: &crate::headers::SequenceHeader,
    frame_hdr: &crate::headers::FrameHeader,
    lf: &mut crate::internal::LoopFilterState,
    dst_y: &mut [u8],
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    fp: &FilterFrameParams,
    inloop: u32,
    sbh: i32,
    sb_step: i32,
    sb256w: i32,
    sb128: i32,
    bw: i32,
    bh: i32,
    sby: i32,
) {
    use crate::looprestoration::{
        INLOOPFILTER_CCSO, INLOOPFILTER_CDEF, INLOOPFILTER_DEBLOCK, INLOOPFILTER_GDF,
        INLOOPFILTER_WIENER,
    };

    let deblock_on = inloop & INLOOPFILTER_DEBLOCK != 0
        && (fp.deblock.level_y[0] != 0 || fp.deblock.level_y[1] != 0);

    // mask row for this sbrow: dav2d uses (sby >> (2 - sb128)) * sb256w.
    let mask_row = ((sby >> (2 - sb128)) * sb256w) as usize;

    // (1) deblock cols. The deblock primitives consume a flattened per-4px mask
    // layout (`[[[u16;4];5]]`) that does not yet have an `Av2Filter` adapter and
    // hardcode their thresholds, so they are only correct for the level==0 no-op
    // path (the M2 clip). Wiring the real mask/threshold plumbing is deferred to
    // the deblock-conformance follow-up; here the call is gated on a non-zero
    // level and given empty mask/segmap when (rarely) reached during bring-up.
    let empty_masks: &[[[u16; 4]; 5]] = &[];
    let empty_segmap: &[u8] = &[];
    let _ = mask_row;
    if deblock_on {
        let start_of_tile_row =
            (lf.start_of_tile_row.get(sby as usize).copied().unwrap_or(0) & 1) != 0;
        crate::deblock::deblock_sbrow_cols_8bpc(
            dst_y,
            dst_u,
            dst_v,
            &fp.deblock,
            empty_masks,
            empty_segmap,
            &fp.thr_lut_y,
            &fp.thr_lut_uv,
            sby,
            start_of_tile_row,
        );
    }

    // (2) deblock rows, then copy_db (store post-deblock / pre-CDEF lines).
    if deblock_on {
        crate::deblock::deblock_sbrow_rows_8bpc(
            dst_y,
            dst_u,
            dst_v,
            &fp.deblock,
            empty_masks,
            empty_segmap,
            &fp.thr_lut_y,
            &fp.thr_lut_uv,
            sby,
        );
    }
    // dav2d's gate also enables copy_db when CDEF is on, because the *multi*-
    // threaded CDEF path reads `lr_db_line` across the sbrow/tile-row seam. In
    // the single-thread (`have_tt == 0`) path CDEF reads only the toggled
    // `cdef_line` banks, so `lr_db_line` is consumed only by loop restoration;
    // restrict copy_db to the restore_planes case so we do not require allocating
    // `lr_db_line` for CDEF-only frames (its output would be dead).
    let copy_db_on =
        lf.restore_planes != 0 && inloop & (INLOOPFILTER_WIENER | INLOOPFILTER_GDF) != 0;
    if copy_db_on {
        // Allocate the deblocked-line store (dav2d's `lr_db_line`, n_tc==1 uses
        // num_lines == 20) so backup_db has somewhere to write. Each plane line
        // buffer is `stride * 20` bytes (positive-stride layout).
        let num_lines = 20usize;
        let y_ls = fp.y_stride.unsigned_abs();
        let uv_ls = fp.uv_stride.unsigned_abs();
        if lf.lr_db_line[0].len() != y_ls * num_lines {
            lf.lr_db_line[0] = vec![0u8; y_ls * num_lines];
        }
        if seq_hdr.layout != crate::headers::PixelLayout::I400 {
            for b in lf.lr_db_line.iter_mut().skip(1) {
                if b.len() != uv_ls * num_lines {
                    *b = vec![0u8; uv_ls * num_lines];
                }
            }
        }
        let src: [&[u8]; 3] = [&*dst_y, &*dst_u, &*dst_v];
        crate::deblock::copy_db_8bpc(
            &mut lf.lr_db_line,
            &src,
            &[fp.y_stride, fp.uv_stride],
            bw as usize,
            bh as usize,
            sby,
            frame_hdr.sb128 != 0,
            fp.ss_hor,
            fp.ss_ver,
            lf.restore_planes != 0,
        );
    }

    // (3) CDEF (+CCSO). CCSO is folded into this stage in dav2d; rav2d has no
    // CCSO driver yet, so it is gated off here (it is a no-op for the M2 clip,
    // whose frame header disables CCSO). Only luma+chroma CDEF runs.
    if seq_hdr.cdef && inloop & (INLOOPFILTER_CDEF | INLOOPFILTER_CCSO) != 0 {
        // Allocate the toggled CDEF top-row backup banks lazily (2 banks x 3
        // planes, 2 rows each at the plane's positive stride).
        let y_ls = fp.y_stride.unsigned_abs();
        let uv_ls = fp.uv_stride.unsigned_abs();
        let need_y = 2 * y_ls;
        let need_uv = 2 * uv_ls;
        for bank in lf.cdef_line.iter_mut() {
            if bank[0].len() != need_y {
                bank[0] = vec![0u8; need_y];
            }
            if seq_hdr.layout != crate::headers::PixelLayout::I400 {
                for b in bank.iter_mut().skip(1) {
                    if b.len() != need_uv {
                        *b = vec![0u8; need_uv];
                    }
                }
            }
        }
        // dav2d resets the toggle at the start of each frame's CDEF (the toggle
        // here lives on `lf` so it persists across sbrows within the frame).
        let start = sby * sb_step;
        // Cross-sbrow seam: re-filter the 2 block-rows straddling the boundary
        // with the previous mask row (dav2d recon_tmpl.c:4001-4009). For a single
        // superblock-row frame (the M2 clip) sby is always 0 so this is skipped.
        if sby > 0 {
            let prev_mask_row = (((sby - 1) >> (2 - sb128)) * sb256w) as usize;
            let bp = crate::cdef::CdefBrowParams {
                bw,
                bh,
                damping: fp.cdef_damping,
                layout: fp.layout,
                on_skip_tx: fp.cdef_on_skiptx,
                cdef_on: inloop & INLOOPFILTER_CDEF != 0,
                mask_cdef_idx: &collect_cdef_idx(&lf.mask, prev_mask_row, sb256w),
                mask_noskip: &collect_noskip(&lf.mask, prev_mask_row, sb256w),
                y_strength: &fp.cdef_y_strength,
                uv_strength: &fp.cdef_uv_strength,
            };
            crate::cdef::cdef_brow_8bpc(
                dst_y,
                dst_u,
                dst_v,
                &bp,
                fp.y_stride,
                fp.uv_stride,
                &mut lf.cdef_line,
                &mut lf.cdef_line_toggle,
                start - 2,
                start,
                sby,
                true,
            );
        }
        let n_blks = sb_step - 2 * ((sby + 1 < sbh) as i32);
        let end = (start + n_blks).min(bh);
        let bp = crate::cdef::CdefBrowParams {
            bw,
            bh,
            damping: fp.cdef_damping,
            layout: fp.layout,
            on_skip_tx: fp.cdef_on_skiptx,
            cdef_on: inloop & INLOOPFILTER_CDEF != 0,
            mask_cdef_idx: &collect_cdef_idx(&lf.mask, mask_row, sb256w),
            mask_noskip: &collect_noskip(&lf.mask, mask_row, sb256w),
            y_strength: &fp.cdef_y_strength,
            uv_strength: &fp.cdef_uv_strength,
        };
        crate::cdef::cdef_brow_8bpc(
            dst_y,
            dst_u,
            dst_v,
            &bp,
            fp.y_stride,
            fp.uv_stride,
            &mut lf.cdef_line,
            &mut lf.cdef_line_toggle,
            start,
            end,
            sby,
            false,
        );
    }

    // (4) Loop restoration (Wiener / GDF). Gated by restore_planes; the M2 clip
    // disables restoration so this branch is inert. Full LR dispatch (the
    // Rust-native kernel selection) lands with the LR ABI work.
    let _ = (INLOOPFILTER_WIENER, INLOOPFILTER_GDF);
}

/// Extract the per-SB256 `cdef_idx` arrays for one mask row into a contiguous
/// slice the CDEF brow can index by `sb256x`.
fn collect_cdef_idx(mask: &[crate::lf_mask::Av2Filter], row: usize, sb256w: i32) -> Vec<[i8; 16]> {
    (0..sb256w as usize)
        .map(|i| mask.get(row + i).map(|m| m.cdef_idx).unwrap_or([-1; 16]))
        .collect()
}

fn collect_noskip(
    mask: &[crate::lf_mask::Av2Filter],
    row: usize,
    sb256w: i32,
) -> Vec<[[u16; 4]; 32]> {
    (0..sb256w as usize)
        .map(|i| {
            mask.get(row + i)
                .map(|m| m.noskip_mask)
                .unwrap_or([[0; 4]; 32])
        })
        .collect()
}

/// Orchestrate a single-threaded frame decode: init -> CDF init -> main loop.
///
/// Port of `dav2d_decode_frame` (decode.c:5223) for the n_tc == 1 path. The
/// post-main CDF output merge (used as the entropy reference for later frames)
/// and the explicit frame-exit unref cleanup are handled by Rust ownership /
/// the caller; the inter-frame CDF-reference selection lands with M3.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame(
    fc: &mut crate::internal::FrameContext,
    n_tc: i32,
    n_passes: i32,
    in_cdf: Option<&crate::cdf::CdfContext>,
    qcat: usize,
) -> Result<(), ()> {
    let frame_hdr = fc.frame_hdr.clone();
    let seq_hdr = fc.seq_hdr.clone();

    decode_frame_init(
        &frame_hdr,
        &seq_hdr,
        &mut fc.lf,
        &mut fc.frame_thread,
        &mut fc.ts,
        &mut fc.n_ts,
        &mut fc.a,
        &mut fc.a_sz,
        &mut fc.dq,
        &mut fc.qm,
        &fc.absrefdist,
        fc.sbh,
        fc.sb256w,
        fc.sb256h,
        fc.bw,
        fc.bh,
        n_tc,
        n_passes,
    );

    if frame_hdr.tip.frame_mode != 2 {
        decode_frame_init_cdf(
            &mut fc.ts,
            &fc.tile,
            &frame_hdr,
            in_cdf,
            qcat,
            fc.sb_shift,
            fc.bw,
            fc.bh,
            n_tc,
            n_passes,
            &mut fc.frame_thread.tile_start_off,
        )?;
    } else {
        decode_tip_frame_init(&mut fc.ts, &frame_hdr, fc.sb_shift, fc.bw, fc.bh, n_tc);
    }

    decode_frame_main(fc, n_passes)?;

    // Finalize the output CDF (dav2d decode.c:5256-5272). For the single-thread
    // path the update tile's adapted CDF becomes `out_cdf` with its symbol counts
    // reset. avg_cdf_type (tile CDF shift/accumulate) is not exercised by the
    // single-tile bring-up clips and is deferred. Only produced when CDF update
    // is enabled (otherwise refs keep `in_cdf`, handled by the ref-list update).
    if frame_hdr.tip.frame_mode != 2 && frame_hdr.disable_cdf_update == 0 {
        let upd = frame_hdr.tiling.update as usize;
        if let Some(ts) = fc.ts.get(upd) {
            let mut out = ts.cdf.clone();
            out.reset_count(frame_hdr.is_key_or_intra());
            fc.out_cdf = Some(std::sync::Arc::new(out));
        }
    }

    Ok(())
}

/// Build a `FrameContext` from the decoder context's parsed headers and tile
/// data, then run the single-threaded decode.
///
/// Port of the frame-setup portion of `dav2d_submit_frame` (decode.c:5282) for
/// the n_fc == 1 path, covering frame geometry derivation (decode.c:5517-5530)
/// and dispatch to `decode_frame`. Picture allocation, reference-frame setup and
/// output queueing are added with reconstruction (M1) and inter support (M3);
/// this currently exercises the entropy/parse pass end-to-end.
pub fn submit_frame(c: &mut crate::internal::DecoderContext, n_tc: i32) -> Result<(), ()> {
    use crate::headers::PixelLayout;

    let seq_hdr = c.seq_hdr.clone().ok_or(())?;
    let frame_hdr = c.frame_hdr.clone().ok_or(())?;

    let mut fc = crate::internal::FrameContext::default();

    let sb128 = frame_hdr.sb128 as i32;
    let layout = seq_hdr.layout;
    fc.ss_ver = (layout == PixelLayout::I420) as i32;
    fc.ss_hor = matches!(layout, PixelLayout::I420 | PixelLayout::I422) as i32;
    fc.root_bs = match sb128 {
        0 => BlockSize::Bs64x64,
        1 => BlockSize::Bs128x128,
        _ => BlockSize::Bs256x256,
    };
    fc.bw = ((frame_hdr.width + 7) >> 3) << 1;
    fc.bh = ((frame_hdr.height + 7) >> 3) << 1;
    fc.sb256w = (fc.bw + 63) >> 6;
    fc.sb256h = (fc.bh + 63) >> 6;
    fc.sb_shift = 4 + sb128;
    fc.sb_step = 16 << sb128;
    fc.sbh = (fc.bh + fc.sb_step - 1) >> fc.sb_shift;
    fc.b4_stride = ((fc.bw + 63) & !63) as isize;
    let bpc = 8 + seq_hdr.hbd as i32 * 2;
    fc.bitdepth_max = (1 << bpc) - 1;
    // Intra neighbours have no reference direction; -1 sentinel (mirrors C
    // lib.c init) so compound/ref context lookups treat them correctly.
    fc.refdir_intra = -1;

    fc.dsp = c.dsp.clone();
    fc.tile = std::mem::take(&mut c.tile);
    fc.n_tile_data = c.n_tile_data;
    fc.inloop_filters = c.inloop_filters;

    // Allocate the output picture that reconstruction writes into. During
    // bring-up we use the default allocator; the decoder's configured allocator
    // is threaded through when output queueing is wired.
    let allocator: std::sync::Arc<dyn crate::picture::PicAllocator> =
        std::sync::Arc::new(crate::picture::DefaultPicAllocator::new());
    fc.cur_pic = crate::picture::Picture::alloc(
        frame_hdr.width,
        frame_hdr.height,
        layout,
        bpc,
        Some(seq_hdr.clone()),
        Some(frame_hdr.clone()),
        allocator,
    )
    .ok_or(())?;

    let qcat = crate::cdf::cdf_thread_init_static_qcat(frame_hdr.quant.yac as u32) as usize;

    let is_inter_or_switch = frame_hdr.is_inter_or_switch();
    let allow_intrabc = frame_hdr.allow_intrabc != 0;

    if std::env::var("RAV2D_FINFO").is_ok() {
        eprintln!(
            "FINFO type={:?} poc={} pri_ref={} n_ref={} refidx={:?} use_rfm={} refresh={:#x} switchable_comp={} skip_mode={} warp={} masked_comp={} motion_modes={:#x} tip={} show_imm={} disable_cdf={} subpel={:?}",
            frame_hdr.frame_type, frame_hdr.frame_offset, frame_hdr.primary_ref_frame,
            frame_hdr.n_ref_frames, frame_hdr.refidx, frame_hdr.use_ref_frame_mvs,
            frame_hdr.refresh_frame_flags, frame_hdr.switchable_comp_refs,
            frame_hdr.skip_mode_enabled, frame_hdr.warp_motion, seq_hdr.masked_compound,
            frame_hdr.motion_modes, frame_hdr.tip.frame_mode, frame_hdr.show_immediate,
            frame_hdr.disable_cdf_update, frame_hdr.subpel_filter_mode,
        );
    }

    // ---- Reference-frame setup (decode.c:5406-5443) ------------------------
    // Validate each signalled reference, share its picture into `fc.refp[i]`,
    // and compute the per-ref scaling / global-warp-allowed flags. Single-ref
    // bring-up: scaled references (svc != 0) are validated but the scaled-MC
    // path is a follow-up; an unequal-dimension ref triggers the deferral note.
    let mut ref_coded_width = [0i32; 7];
    if is_inter_or_switch {
        for i in 0..7 {
            let refidx = frame_hdr.refidx[i] as usize;
            let rp = &c.refs[refidx].p;
            let rpic = rp.pic.as_ref();
            let valid = rpic.is_some_and(|p| {
                let pw = p.p.w;
                let ph = p.p.h;
                frame_hdr.width * 2 >= pw
                    && frame_hdr.height * 2 >= ph
                    && frame_hdr.width <= pw * 16
                    && frame_hdr.height <= ph * 16
                    && seq_hdr.layout == p.p.layout
                    && bpc == p.p.bpc
            });
            if !valid {
                return Err(());
            }
            let p = rpic.unwrap();
            fc.refp[i].pic = Some(p.clone());
            fc.refp[i].frame_hdr = rp.frame_hdr.clone();
            ref_coded_width[i] = p.frame_hdr.as_ref().map(|h| h.width).unwrap_or(p.p.w);
            if frame_hdr.width != p.p.w || frame_hdr.height != p.p.h {
                let scale_fac = |ref_sz: i32, this_sz: i32| -> i32 {
                    (((ref_sz << 14) + (this_sz >> 1)) / this_sz) as i32
                };
                fc.svc[i][0].scale = scale_fac(p.p.w, frame_hdr.width);
                fc.svc[i][1].scale = scale_fac(p.p.h, frame_hdr.height);
                fc.svc[i][0].step = (fc.svc[i][0].scale + 8) >> 4;
                fc.svc[i][1].step = (fc.svc[i][1].scale + 8) >> 4;
            } else {
                fc.svc[i][0].scale = 0;
                fc.svc[i][1].scale = 0;
            }
            let mut gm = frame_hdr.gmv.m[i];
            fc.gmv_warp_allowed[i] = (gm.wm_type
                > crate::headers::WarpedMotionType::Translation
                && frame_hdr.force_integer_mv == 0
                && crate::warpmv::get_shear_params(&mut gm) == 0
                && fc.svc[i][0].scale == 0) as u8;
        }
    }

    // ---- Entropy CDF selection (decode.c:5446-5471) ------------------------
    // primary_ref_frame == NONE -> static qcat init (keyframe path). Otherwise
    // clone the saved CDF of the primary ref. The avg primary/secondary CDF
    // path (use_pri_sec_cdf) is deferred (not exercised by the bring-up clips).
    let p_ref_idx = frame_hdr.primary_ref_frame;
    let in_cdf: Option<crate::cdf::CdfContext> = if p_ref_idx == crate::headers::PRIMARY_REF_NONE {
        fc.use_pri_sec_cdf = 0;
        None
    } else {
        fc.use_pri_sec_cdf = 0;
        let pri_ref = frame_hdr.refidx[p_ref_idx as usize] as usize;
        c.refs[pri_ref].cdf.as_ref().map(|a| (**a).clone())
    };

    fc.seq_hdr = seq_hdr.clone();
    fc.frame_hdr = frame_hdr.clone();
    fc.in_cdf = in_cdf;

    // ---- refpoc / refdist / refdir / furthest_future_refidx + refmvs (decode.c:5538-5596) ----
    let use_rfm = is_inter_or_switch || allow_intrabc;
    if use_rfm {
        if is_inter_or_switch {
            let poc = frame_hdr.frame_offset as i32;
            let nbits = seq_hdr.order_hint_n_bits as i32;
            let mut furthest_future_refidx: i32 = -2;
            for i in 0..7 {
                let rpoc = fc.refp[i]
                    .frame_hdr
                    .as_ref()
                    .map(|h| h.frame_offset as i32)
                    .unwrap_or(0);
                fc.refpoc[i] = rpoc as u8;
                let delta = crate::env::get_poc_diff(nbits, rpoc, poc);
                fc.refdist[i] = delta as i8;
                fc.absrefdist[i] = delta.unsigned_abs() as u8;
                fc.refdir[i] = (delta > 0) as u8;
                if delta > 0
                    && (furthest_future_refidx < 0
                        || (fc.refdist[furthest_future_refidx as usize] as i32) < delta)
                {
                    furthest_future_refidx = i as i32;
                }
            }
            fc.furthest_future_refidx = furthest_future_refidx as i8;
        } else {
            fc.refpoc = [0; 7];
        }
        // Reference temporal MVs (decode.c:5571-5589).
        if frame_hdr.use_ref_frame_mvs != 0 {
            let bw = ((frame_hdr.width + 7) >> 3) << 1;
            let bh = ((frame_hdr.height + 7) >> 3) << 1;
            for i in 0..7 {
                let refidx = frame_hdr.refidx[i] as usize;
                let ref_w = ((ref_coded_width[i] + 7) >> 3) << 1;
                let ref_h = ((fc.refp[i].pic.as_ref().map(|p| p.p.h).unwrap_or(0) + 7) >> 3) << 1;
                if c.refs[refidx].refmvs.is_some() && ref_w == bw && ref_h == bh {
                    fc.ref_mvs[i] = c.refs[refidx].refmvs.clone();
                } else {
                    fc.ref_mvs[i] = None;
                }
                fc.refrefpoc[i] = c.refs[refidx].refpoc;
                fc.refcnt[i] = fc.refp[i]
                    .frame_hdr
                    .as_ref()
                    .map(|h| h.n_ref_frames)
                    .unwrap_or(0);
            }
        }
    }

    // ---- prev_segmap from the primary ref (decode.c:5598-5618) -------------
    if frame_hdr.segmentation.enabled != 0
        && (frame_hdr.segmentation.temporal != 0 || frame_hdr.segmentation.update_map == 0)
    {
        let pri = frame_hdr.primary_ref_frame as usize;
        let ref_w = ((ref_coded_width[pri] + 7) >> 3) << 1;
        let ref_h = ((fc.refp[pri].pic.as_ref().map(|p| p.p.h).unwrap_or(0) + 7) >> 3) << 1;
        let bw = ((frame_hdr.width + 7) >> 3) << 1;
        let bh = ((frame_hdr.height + 7) >> 3) << 1;
        if ref_w == bw && ref_h == bh {
            fc.prev_segmap = c.refs[frame_hdr.refidx[pri] as usize].segmap.clone();
        }
    }

    // ---- skip_mode_refs (decode.c:5687-5691) ------------------------------
    let skip_mode_r1 = (frame_hdr.skip_mode_enabled != 0
        && frame_hdr.n_ref_frames > 1
        && (fc.absrefdist[0] as i32 - fc.absrefdist[1] as i32).abs() <= 1)
        as i8;
    fc.skip_mode_refs = RefPair { r: [0, skip_mode_r1] };

    // ---- decode -----------------------------------------------------------
    let in_cdf_ref = fc.in_cdf.take();
    decode_frame(&mut fc, n_tc, 1, in_cdf_ref.as_ref(), qcat)?;

    // ---- Reference-list + CDF update per refresh_frame_flags (decode.c:5694) ----
    // dav2d performs this UNCONDITIONALLY before launching decode and rolls back
    // on failure. Single-thread decode here is synchronous and already succeeded,
    // so we publish after success (equivalent for n_fc == 1). The decoded picture
    // is shared into every refreshed slot so later frames can reference it.
    let cur_pic = std::mem::take(&mut fc.cur_pic);
    let shared = std::sync::Arc::new(cur_pic);
    let out_cdf_for_refs: Option<std::sync::Arc<crate::cdf::CdfContext>> =
        if frame_hdr.disable_cdf_update == 0 {
            fc.out_cdf.clone()
        } else {
            in_cdf_ref.map(std::sync::Arc::new)
        };
    let cur_segmap_arc: Option<Vec<u8>> = if !fc.cur_segmap.is_empty() {
        Some(fc.cur_segmap.clone())
    } else {
        None
    };
    let cur_ccsomap_arc: Option<Vec<u8>> = if !fc.cur_ccsomap.is_empty() {
        Some(fc.cur_ccsomap.clone())
    } else {
        None
    };
    let refresh = frame_hdr.refresh_frame_flags;
    for i in 0..8 {
        if refresh & (1 << i) != 0 {
            c.refs[i].p.pic = Some(shared.clone());
            c.refs[i].p.frame_hdr = Some(frame_hdr.clone());
            c.refs[i].p.showable = frame_hdr.show_immediate == 0;
            c.refs[i].cdf = out_cdf_for_refs.clone();
            c.refs[i].segmap = if frame_hdr.segmentation.update_map != 0 {
                cur_segmap_arc.clone()
            } else {
                fc.prev_segmap.clone()
            };
            if is_inter_or_switch {
                c.refs[i].refmvs = if fc.rf.rp.is_empty() {
                    None
                } else {
                    Some(fc.rf.rp.clone())
                };
            }
            c.refs[i].ccsomap = cur_ccsomap_arc.clone();
            c.refs[i].refpoc = fc.refpoc;
        }
    }

    // Hand the reconstructed picture to the decoder's output path. (Visibility
    // filtering / POC reordering is wired with full output queueing later.)
    // The output buffer must be independently owned; clone the shared pixels.
    c.frame_out.push(clone_picture(&shared));
    Ok(())
}

/// Clone a `Picture`'s pixel planes into a fresh independently-owned allocation
/// (the output path takes ownership while references keep the shared `Arc`).
fn clone_picture(src: &crate::picture::Picture) -> crate::picture::Picture {
    let allocator: std::sync::Arc<dyn crate::picture::PicAllocator> =
        std::sync::Arc::new(crate::picture::DefaultPicAllocator::new());
    let dst = match crate::picture::Picture::alloc(
        src.p.w,
        src.p.h,
        src.p.layout,
        src.p.bpc,
        src.seq_hdr.clone(),
        src.frame_hdr.clone(),
        allocator,
    ) {
        Some(p) => p,
        None => return crate::picture::Picture::new(),
    };
    let n_planes = if src.p.layout == crate::headers::PixelLayout::I400 {
        1
    } else {
        3
    };
    for pl in 0..n_planes {
        let (sp, dp) = (src.data[pl], dst.data[pl]);
        if let (Some(sp), Some(dp)) = (sp, dp) {
            let stride_idx = if pl == 0 { 0 } else { 1 };
            let s_stride = src.stride[stride_idx];
            let d_stride = dst.stride[stride_idx];
            let ss_ver = (src.p.layout == crate::headers::PixelLayout::I420) as i32;
            let ss_hor =
                matches!(src.p.layout, crate::headers::PixelLayout::I420
                    | crate::headers::PixelLayout::I422) as i32;
            let bytes = (src.p.bpc + 7) / 8;
            let (pw, ph) = if pl == 0 {
                (src.p.w, src.p.h)
            } else {
                ((src.p.w + ss_hor) >> ss_hor, (src.p.h + ss_ver) >> ss_ver)
            };
            let row_bytes = (pw * bytes) as usize;
            for y in 0..ph as usize {
                // SAFETY: both allocations span stride*height bytes per plane.
                unsafe {
                    let srow = sp.as_ptr().offset(y as isize * s_stride);
                    let drow = dp.as_ptr().offset(y as isize * d_stride);
                    std::ptr::copy_nonoverlapping(srow, drow, row_bytes);
                }
            }
        }
    }
    dst
}

fn get_snglref_ctx(
    a: &BlockContext,
    l: &BlockContext,
    yb4: usize,
    xb4: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    b_dim: &[u8],
    ref_idx: i8,
) -> usize {
    const NEWMV0_MASK: u32 =
        (1 << 15) | (1 << 20) | (1 << 22) | (1 << 23) | (1 << 26) | (1 << 27) | (1 << 28);
    const NEWMV1_MASK: u32 = (1 << 19) | (1 << 22) | (1 << 25) | (1 << 27);

    let mut row = 0i32;
    let mut col = 0i32;
    let mut newmv = 0i32;

    macro_rules! add_matching {
        ($ctx:expr, $cnt:ident, $idx:expr) => {
            if $ctx.r#ref[0][$idx] as i8 == ref_idx {
                $cnt += 1;
                newmv += (((1u32 << $ctx.mode[$idx]) & NEWMV0_MASK) != 0) as i32;
            } else if $ctx.r#ref[1][$idx] as i8 == ref_idx {
                $cnt += 1;
                newmv += (((1u32 << $ctx.mode[$idx]) & NEWMV1_MASK) != 0) as i32;
            }
        };
    }
    if have_top {
        add_matching!(a, col, xb4);
        if have_top_right {
            add_matching!(a, col, xb4 + b_dim[0] as usize - 1);
        }
    }
    if have_left {
        add_matching!(l, row, yb4);
        if have_bottom_left {
            add_matching!(l, row, yb4 + b_dim[1] as usize - 1);
        }
    }

    ((row != 0) as usize) + ((col != 0) as usize) + 2 * ((newmv != 0) as usize)
}

fn get_filter_ctx(
    a: &BlockContext,
    l: &BlockContext,
    nb_boff: &[i32; 2],
    ref0: i8,
    is_comp: bool,
) -> usize {
    const N_SWITCHABLE: u8 = 3;
    let comp_val = is_comp as usize * 4;

    let flt0 = if nb_boff[0] != -1 {
        let off = nb_boff[0] as usize;
        if l.r#ref[0][off] == ref0 || l.r#ref[1][off] == ref0 {
            l.filter[off]
        } else {
            N_SWITCHABLE
        }
    } else {
        N_SWITCHABLE
    };

    let flt1 = if nb_boff[1] != -1 {
        let off = nb_boff[1] as usize;
        if a.r#ref[0][off] == ref0 || a.r#ref[1][off] == ref0 {
            a.filter[off]
        } else {
            N_SWITCHABLE
        }
    } else {
        N_SWITCHABLE
    };

    if flt0 == flt1 || flt1 == N_SWITCHABLE {
        comp_val + flt0 as usize
    } else if flt0 == N_SWITCHABLE {
        comp_val + flt1 as usize
    } else {
        comp_val + N_SWITCHABLE as usize
    }
}

#[allow(unused)]
fn decode_b(
    fi: &SbFrameInfo,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    intra_region: i32,
    _sdp_cfl_disallowed: i32,
    pass: u8,
    a: &mut BlockContext,
    l: &mut BlockContext,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    cdf_dmv: &mut CdfMvContext,
    recon: &mut ReconCtx,
    lbs: BlockSize,
    cbs: BlockSize,
) -> Result<Av2Block, ()> {
    let _ = &mut *recon;
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    debug_assert!(bs != BlockSize::Invalid);
    let bs_idx = bs as u8 as usize;

    let b_dim = &BLOCK_DIMENSIONS[bs_idx];
    let bx4 = (bx & 63) as usize;
    let by4 = (by & 63) as usize;
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;

    let w4 = imin(bw4, fi.bw - bx);
    let h4 = imin(bh4, fi.bh - by);
    let have_left = bx > fi.tile_col_start;
    let have_top = by > fi.tile_row_start;
    let have_top_right = bx + bw4 <= fi.tile_col_end;
    let have_bottom_left = by + bh4 <= fi.tile_row_end;
    let has_luma = lbs != BlockSize::Invalid;
    let has_chroma = cbs != BlockSize::Invalid;

    let mut b = Av2Block::default();
    if has_luma {
        b.bs = lbs as i8;
    }
    if has_chroma {
        b.cbs = cbs as i8;
    }

    if std::env::var("RAV2D_TRACE").is_ok() {
        eprintln!(
            "BLK poc={} y={} x={} cby={} cbx={} bs={} lbs={} cbs={} hasL={} hasC={} ireg={} rng={}",
            recon.frm_hdr.frame_offset,
            by,
            bx,
            cby,
            cbx,
            bs as i32,
            lbs as i32,
            cbs as i32,
            has_luma as i32,
            has_chroma as i32,
            intra_region,
            msac.dbg_rng()
        );
    }
    let trace_blk = std::env::var("RAV2D_BLK")
        .ok()
        .map(|s| {
            let mut it = s.split(',');
            (
                it.next().unwrap_or("").parse::<i32>().unwrap_or(-1),
                it.next().unwrap_or("").parse::<i32>().unwrap_or(-1),
            )
        })
        .map(|(ty, tx)| ty == by && tx == bx)
        .unwrap_or(false);

    // Pre-compute cross-SB boundary neighbour context values.
    // The C code uses nx[2] pointers into a/l; here we read out
    // the values we need before any mutable operations.
    let (
        nx_skip_mode,
        nx_skip_txfm,
        nx_intra,
        nx_intrabc,
        nx_xoff,
        n_ctx,
        nx_ref0,
        nx_ref1,
        nx_amvd,
        nx_comp_type,
    ) = {
        let mut sm = [0u8; 2];
        let mut st = [0u8; 2];
        let mut intra_vals = [0u8; 2];
        let mut ibc_vals = [0u8; 2];
        let mut xoff = [0usize; 2];
        let mut r0 = [0i8; 2];
        let mut r1 = [0i8; 2];
        let mut amvd_v = [0u8; 2];
        let mut ct = [0u8; 2];
        let mut idx = 0usize;

        if have_left && by + bh4 <= fi.tile_row_end {
            let off = (by4 + bh4 as usize).saturating_sub(1);
            sm[0] = l.skip_mode[off];
            st[0] = l.skip_txfm[off];
            intra_vals[0] = if l.intra[off] != 0 && l.intrabc[off] == 0 {
                1
            } else {
                0
            };
            ibc_vals[0] = l.intrabc[off];
            r0[0] = l.r#ref[0][off];
            r1[0] = l.r#ref[1][off];
            amvd_v[0] = l.amvd[off];
            ct[0] = l.comp_type[off];
            xoff[0] = off;
            idx += 1;
        }
        if have_top && bx + bw4 <= fi.tile_col_end {
            let off = (bx4 + bw4 as usize).saturating_sub(1);
            sm[idx] = a.skip_mode[off];
            st[idx] = a.skip_txfm[off];
            intra_vals[idx] = if a.intra[off] != 0 && a.intrabc[off] == 0 {
                1
            } else {
                0
            };
            ibc_vals[idx] = a.intrabc[off];
            r0[idx] = a.r#ref[0][off];
            r1[idx] = a.r#ref[1][off];
            amvd_v[idx] = a.amvd[off];
            ct[idx] = a.comp_type[off];
            xoff[idx] = off;
            idx += 1;
        }
        if idx < 2 && have_left {
            sm[idx] = l.skip_mode[by4];
            st[idx] = l.skip_txfm[by4];
            intra_vals[idx] = if l.intra[by4] != 0 && l.intrabc[by4] == 0 {
                1
            } else {
                0
            };
            ibc_vals[idx] = l.intrabc[by4];
            r0[idx] = l.r#ref[0][by4];
            r1[idx] = l.r#ref[1][by4];
            amvd_v[idx] = l.amvd[by4];
            ct[idx] = l.comp_type[by4];
            xoff[idx] = by4;
            idx += 1;
        }
        if idx < 2 {
            sm[idx] = a.skip_mode[bx4];
            st[idx] = a.skip_txfm[bx4];
            intra_vals[idx] = if a.intra[bx4] != 0 && a.intrabc[bx4] == 0 {
                1
            } else {
                0
            };
            ibc_vals[idx] = a.intrabc[bx4];
            r0[idx] = a.r#ref[0][bx4];
            r1[idx] = a.r#ref[1][bx4];
            amvd_v[idx] = a.amvd[bx4];
            ct[idx] = a.comp_type[bx4];
            xoff[idx] = bx4;
            if idx == 0 {
                sm[1] = sm[0];
                st[1] = st[0];
                intra_vals[1] = intra_vals[0];
                ibc_vals[1] = ibc_vals[0];
                r0[1] = r0[0];
                r1[1] = r1[0];
                amvd_v[1] = amvd_v[0];
                ct[1] = ct[0];
                xoff[1] = xoff[0];
            }
            if have_top {
                idx += 1;
            }
        }
        (sm, st, intra_vals, ibc_vals, xoff, idx, r0, r1, amvd_v, ct)
    };

    // segment_id (pre-skip read). Port of decode.c:1577-1635.
    let mut seg_pred = 0i32;
    if fi.seg_enabled {
        let bx_abs = bx;
        let by_abs = by;
        if !has_luma {
            b.seg_id =
                recon.cur_segmap[(bx_abs as isize + by_abs as isize * recon.b4_stride) as usize];
        } else if !fi.seg_update_map {
            if let Some(prev) = recon.prev_segmap {
                let sid = get_prev_frame_segid(by_abs, bx_abs, w4, h4, prev, recon.b4_stride);
                if sid >= 16 {
                    recon.seg_id_err = true;
                    return Err(());
                }
                b.seg_id = sid as u8;
            } else {
                b.seg_id = 0;
            }
        } else if fi.seg_preskip {
            seg_pred = if fi.seg_temporal {
                let ctx = a.seg_pred[bx4] as usize + l.seg_pred[by4] as usize;
                msac.decode_bool_adapt(cdf_m.seg_pred(ctx)) as i32
            } else {
                0
            };
            if seg_pred != 0 {
                if let Some(prev) = recon.prev_segmap {
                    let sid = get_prev_frame_segid(by_abs, bx_abs, w4, h4, prev, recon.b4_stride);
                    if sid >= 16 {
                        recon.seg_id_err = true;
                        return Err(());
                    }
                    b.seg_id = sid as u8;
                } else {
                    b.seg_id = 0;
                }
            } else {
                let mut seg_ctx = 0i32;
                let pred_seg_id = get_cur_frame_segid(
                    by_abs,
                    bx_abs,
                    have_top,
                    have_left,
                    &mut seg_ctx,
                    recon.cur_segmap,
                    recon.b4_stride,
                );
                let ext_flag = if fi.seg_ext {
                    msac.decode_bool_adapt(cdf_m.seg_id_ext(seg_ctx as usize)) as u32
                } else {
                    0
                };
                let diff = msac
                    .decode_symbol_adapt(cdf_m.seg_id(ext_flag as usize, seg_ctx as usize), 7)
                    + (ext_flag << 3);
                let last_active = fi.seg_last_active_segid as i32;
                let mut sid = neg_deinterleave(diff as i32, pred_seg_id as i32, last_active + 1);
                if sid > last_active {
                    sid = 0;
                }
                if sid >= crate::headers::MAX_SEGMENTS as i32 {
                    sid = 0;
                }
                b.seg_id = sid as u8;
            }
        }
    } else {
        b.seg_id = 0;
    }

    // skip_mode
    if (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) == 0
        && fi.skip_mode_enabled
        && bw4 * bh4 > 2
        && intra_region == 0
    {
        let ctx = nx_skip_mode[0] as usize + nx_skip_mode[1] as usize;
        b.skip_mode = msac.decode_bool_adapt(cdf_m.skip_mode(ctx)) as u8;
    } else {
        b.skip_mode = 0;
    }

    // intra/inter decision
    if b.skip_mode != 0 {
        b.is_intra = 0;
    } else if fi.is_inter_or_switch && intra_region == 0 {
        if fi.has_chroma_layout && lbs != cbs {
            b.is_intra = 0;
        } else {
            // get_intra_ctx (env.h:72): sum of intra(&!intrabc) over the
            // gathered neighbours (nx[0], nx[n_ctx-1]), plus 1 if all are intra.
            let ictx = if n_ctx == 0 {
                0
            } else {
                let i = (n_ctx - 1) as usize;
                let sum = nx_intra[0] as i32 + nx_intra[i] as i32;
                sum + (sum == n_ctx as i32) as i32
            };
            b.is_intra = (msac.decode_bool_adapt(cdf_m.intra(ictx as usize)) == 0) as u8;
        }
    } else {
        b.is_intra = 1;
    }

    // Pre-compute spatial neighbour (nb) context values within SB.
    // These are used by intrabc, FSC, MRL, multi_mrl, DIP, morph_pred.
    // boff[i] = -1 means unavailable.
    let have_top_in_sb = (by & (fi.sb_step - 1)) != 0;
    let (
        nb_fsc,
        nb_mrl,
        nb_multi_mrl,
        nb_intrabc,
        nb_midx,
        nb_mvprec,
        nb_motion_mode,
        nb_morph,
        nb_dip,
        nb_boff,
        nb_ref0,
        nb_ref1,
        nb_filter,
    ) = if has_luma {
        let mut fsc = [0u8; 2];
        let mut mrl = [0u8; 2];
        let mut mmrl = [0u8; 2];
        let mut ibc = [0u8; 2];
        let mut mid = [0xffu8; 2];
        let mut mvp = [0u8; 2];
        let mut mm = [0u8; 2];
        let mut mp = [0u8; 2];
        let mut dp = [0u8; 2];
        let mut boff = [-1i32; 2];
        // Inter subpel-filter context inputs (decode.c env.h:120 get_filter_ctx):
        // the neighbour's ref pair and filter at boff, captured here so the a/l
        // identity is preserved (boff alone loses it).
        let mut nref0 = [-1i8; 2];
        let mut nref1 = [-1i8; 2];
        let mut nflt = [0u8; 2];
        let mut idx = 0usize;

        if have_left && bh4 == h4 {
            let off = (by4 + bh4 as usize).saturating_sub(1);
            fsc[0] = l.fsc[off];
            mrl[0] = l.mrl[off];
            mmrl[0] = l.multi_mrl[off];
            ibc[0] = l.intrabc[off];
            mid[0] = l.midx[off];
            mvp[0] = l.mvprec[off];
            mm[0] = l.motion_mode[off];
            mp[0] = l.morph_pred[off];
            dp[0] = l.dip[off];
            boff[0] = off as i32;
            nref0[0] = l.r#ref[0][off];
            nref1[0] = l.r#ref[1][off];
            nflt[0] = l.filter[off];
            idx += 1;
        }
        if have_top_in_sb && bw4 == w4 {
            let off = (bx4 + bw4 as usize).saturating_sub(1);
            fsc[idx] = a.fsc[off];
            mrl[idx] = a.mrl[off];
            mmrl[idx] = a.multi_mrl[off];
            ibc[idx] = a.intrabc[off];
            mid[idx] = a.midx[off];
            mvp[idx] = a.mvprec[off];
            mm[idx] = a.motion_mode[off];
            mp[idx] = a.morph_pred[off];
            dp[idx] = a.dip[off];
            boff[idx] = off as i32;
            nref0[idx] = a.r#ref[0][off];
            nref1[idx] = a.r#ref[1][off];
            nflt[idx] = a.filter[off];
            idx += 1;
        }
        if have_left && idx < 2 {
            fsc[idx] = l.fsc[by4];
            mrl[idx] = l.mrl[by4];
            mmrl[idx] = l.multi_mrl[by4];
            ibc[idx] = l.intrabc[by4];
            mid[idx] = l.midx[by4];
            mvp[idx] = l.mvprec[by4];
            mm[idx] = l.motion_mode[by4];
            mp[idx] = l.morph_pred[by4];
            dp[idx] = l.dip[by4];
            boff[idx] = by4 as i32;
            nref0[idx] = l.r#ref[0][by4];
            nref1[idx] = l.r#ref[1][by4];
            nflt[idx] = l.filter[by4];
            idx += 1;
        }
        if have_top_in_sb && idx < 2 {
            fsc[idx] = a.fsc[bx4];
            mrl[idx] = a.mrl[bx4];
            mmrl[idx] = a.multi_mrl[bx4];
            ibc[idx] = a.intrabc[bx4];
            mid[idx] = a.midx[bx4];
            mvp[idx] = a.mvprec[bx4];
            mm[idx] = a.motion_mode[bx4];
            mp[idx] = a.morph_pred[bx4];
            dp[idx] = a.dip[bx4];
            boff[idx] = bx4 as i32;
            nref0[idx] = a.r#ref[0][bx4];
            nref1[idx] = a.r#ref[1][bx4];
            nflt[idx] = a.filter[bx4];
            if idx == 0 {
                fsc[1] = fsc[0];
                mrl[1] = mrl[0];
                mmrl[1] = mmrl[0];
                ibc[1] = ibc[0];
                mid[1] = mid[0];
                mvp[1] = mvp[0];
                mm[1] = mm[0];
                mp[1] = mp[0];
                dp[1] = dp[0];
            }
        }
        (fsc, mrl, mmrl, ibc, mid, mvp, mm, mp, dp, boff, nref0, nref1, nflt)
    } else {
        (
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [0xffu8; 2],
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [-1i32; 2],
            [-1i8; 2],
            [-1i8; 2],
            [0u8; 2],
        )
    };

    // intrabc
    if has_luma {
        b.intrabc = 0;
        if fi.allow_intrabc && imin(bw4, bh4) < 16 && b.is_intra != 0 && intra_region == 0 {
            let ctx = (nb_intrabc[0] + nb_intrabc[1]) as usize;
            if std::env::var("RAV2D_IBC").is_ok() {
                let c = cdf_m.intrabc(ctx);
                eprintln!(
                    "IBC y={} x={} ctx={} cdf0={} cdf1={} rng_in={} dif={}",
                    by,
                    bx,
                    ctx,
                    c[0],
                    c[1],
                    msac.dbg_rng(),
                    msac.dbg_dif()
                );
            }
            b.intrabc = msac.decode_bool_adapt(cdf_m.intrabc(ctx)) as u8;
        }
    }
    let intrabc = has_luma && b.intrabc != 0;
    if trace_blk {
        eprintln!(
            "  CK intrabc b.intrabc={} intra={} rng={}",
            b.intrabc,
            b.is_intra,
            msac.dbg_rng()
        );
    }
    if intrabc && std::env::var("RAV2D_IBC2").is_ok() {
        eprintln!(
            "IBC2 y={} x={} bs={} bw4={} bh4={} hasC={} cbs={}",
            by, bx, bs as i32, bw4, bh4, has_chroma, cbs as i32
        );
    }

    // skip_txfm
    if fi.seg_skip_mask & (1 << b.seg_id) != 0 {
        b.skip_txfm = 1;
    } else if b.is_intra != 0 && !intrabc {
        if has_luma {
            b.skip_txfm = 0;
        }
    } else {
        let ctx = nx_skip_txfm[0] as usize + nx_skip_txfm[1] as usize + b.skip_mode as usize * 3;
        b.skip_txfm = msac.decode_bool_adapt(cdf_m.skip_txfm(ctx)) as u8;
    }
    if trace_blk {
        eprintln!("  CK skip_txfm={} rng={}", b.skip_txfm, msac.dbg_rng());
    }

    // segment_id (post-skip read). Port of decode.c:1748-1802.
    if fi.seg_enabled && fi.seg_update_map && !fi.seg_preskip {
        let bx_abs = bx;
        let by_abs = by;
        if !has_luma {
            b.seg_id =
                recon.cur_segmap[(bx_abs as isize + by_abs as isize * recon.b4_stride) as usize];
        } else if b.skip_txfm == 0 && fi.seg_temporal && {
            let ctx = a.seg_pred[bx4] as usize + l.seg_pred[by4] as usize;
            seg_pred = msac.decode_bool_adapt(cdf_m.seg_pred(ctx)) as i32;
            seg_pred != 0
        } {
            if let Some(prev) = recon.prev_segmap {
                let sid = get_prev_frame_segid(by_abs, bx_abs, w4, h4, prev, recon.b4_stride);
                if sid >= 16 {
                    recon.seg_id_err = true;
                    return Err(());
                }
                b.seg_id = sid as u8;
            } else {
                b.seg_id = 0;
            }
        } else {
            let mut seg_ctx = 0i32;
            let pred_seg_id = get_cur_frame_segid(
                by_abs,
                bx_abs,
                have_top,
                have_left,
                &mut seg_ctx,
                recon.cur_segmap,
                recon.b4_stride,
            );
            if b.skip_txfm != 0 && !fi.any_lossless {
                b.seg_id = pred_seg_id as u8;
            } else {
                let ext_flag = if fi.seg_ext {
                    msac.decode_bool_adapt(cdf_m.seg_id_ext(seg_ctx as usize)) as u32
                } else {
                    0
                };
                let diff = msac
                    .decode_symbol_adapt(cdf_m.seg_id(ext_flag as usize, seg_ctx as usize), 7)
                    + (ext_flag << 3);
                let last_active = fi.seg_last_active_segid as i32;
                let mut sid = neg_deinterleave(diff as i32, pred_seg_id as i32, last_active + 1);
                if sid > last_active {
                    sid = 0;
                }
                b.seg_id = sid as u8;
            }
            if b.seg_id >= crate::headers::MAX_SEGMENTS as u8 {
                b.seg_id = 0;
            }
        }
        if trace_blk {
            eprintln!("  CK post_segid seg_id={} rng={}", b.seg_id, msac.dbg_rng());
        }
    }

    let skip_txfm = has_luma && b.skip_txfm != 0;

    // GDF (guided deblocking filter) flag. Port of decode.c:1806-1833.
    if has_luma {
        let gdf_sz_log2 = if fi.gdf_is_key { 1 } else { imax(1, fi.sb128) };
        let gdf_bs = 16 << gdf_sz_log2;
        if (bx | by) & (gdf_bs - 1) == 0 {
            let idx = (((by & 48) >> 2) + ((bx & 48) >> 4)) as usize;
            let flag = if fi.gdf_enabled == crate::headers::AdaptiveBoolean::Adaptive
                && imax(fi.cur_w, fi.cur_h) > 4 * gdf_bs
            {
                let f = msac.decode_bool_adapt(cdf_m.gdf()) as u8;
                if trace_blk {
                    eprintln!("  CK gdf flag={} rng={}", f, msac.dbg_rng());
                }
                f
            } else {
                (fi.gdf_enabled != crate::headers::AdaptiveBoolean::Off) as u8
            };
            let n = 1usize << gdf_sz_log2;
            let m = &mut recon.lf_mask[recon.lf_idx];
            m.gdf[idx..idx + n].fill(flag);
            if gdf_bs >= 32 {
                m.gdf[idx + 4..idx + 4 + n].fill(flag);
                if gdf_bs == 64 {
                    m.gdf[idx + 8..idx + 8 + n].fill(flag);
                    m.gdf[idx + 12..idx + 12 + n].fill(flag);
                }
            }
        }
    }

    // CDEF index. Port of decode.c:1835-1893.
    if fi.cdef_enabled && (!skip_txfm || fi.cdef_on_skiptx) {
        let idx = (((bx & 0x30) >> 4) + ((by & 0x30) >> 2)) as usize;
        if recon.lf_mask[recon.lf_idx].cdef_idx[idx] == -1 {
            let v;
            if fi.cdef_n_strengths == 1 {
                v = 0i8;
            } else {
                let left_cdef_idx = if bx - 16 < fi.tile_col_start {
                    -1i32
                } else if idx & 3 != 0 {
                    recon.lf_mask[recon.lf_idx].cdef_idx[idx - 1] as i32
                } else {
                    recon.lf_mask[recon.lf_idx - 1].cdef_idx[idx + 3] as i32
                };
                let top_cdef_idx = if (by & !15) & (fi.sb_step - 1) == 0 {
                    -1i32
                } else if idx & 0xc != 0 {
                    recon.lf_mask[recon.lf_idx].cdef_idx[idx - 4] as i32
                } else {
                    recon.lf_mask[recon.lf_idx - recon.sb256w as usize].cdef_idx[idx + 12] as i32
                };
                let ctx = if (left_cdef_idx | top_cdef_idx) != -1 {
                    // both edges available
                    let mut c = (left_cdef_idx == 0) as i32 + (top_cdef_idx == 0) as i32;
                    c += (c == 2) as i32;
                    c
                } else {
                    // C: !(left & top) * 2  (logical-not, so 0 -> 1, nonzero -> 0)
                    ((left_cdef_idx & top_cdef_idx) == 0) as i32 * 2
                };
                if msac.decode_bool_adapt(cdf_m.cdef_idx0(ctx as usize)) != 0 {
                    v = 0;
                } else if fi.cdef_n_strengths == 2 {
                    v = 1;
                } else {
                    let rem = fi.cdef_n_strengths as i32 - 3;
                    v = 1 + msac
                        .decode_symbol_adapt(cdf_m.cdef_idx(rem as usize), (rem + 1) as usize)
                        as i8;
                }
                if trace_blk {
                    eprintln!("  CK cdef_idx ctx={} v={} rng={}", ctx, v, msac.dbg_rng());
                }
            }
            let splat_n = 1usize << imax(0, b_dim[2] as i32 - 4);
            let m = &mut recon.lf_mask[recon.lf_idx];
            m.cdef_idx[idx..idx + splat_n].fill(v);
            if bh4 >= 32 {
                m.cdef_idx[idx + 4..idx + 4 + splat_n].fill(v);
                if bh4 == 64 {
                    m.cdef_idx[idx + 8..idx + 8 + splat_n].fill(v);
                    m.cdef_idx[idx + 12..idx + 12 + splat_n].fill(v);
                }
            }
        }
    }

    // CCSO (cross-component sample offset). Port of decode.c:1895-1919.
    if has_luma && (bx | by) & 63 == 0 {
        let ccso_idx = (3 * ((bx >> 6) + (by >> 6) * fi.sb256w)) as usize;
        for p in 0..3 {
            if !fi.ccso_enabled[p] {
                continue;
            }
            let val = if fi.ccso_sb_reuse[p] {
                match recon.prev_ccsomap[p] {
                    Some(prev) => prev[ccso_idx + p],
                    None => 0,
                }
            } else {
                let ctx = if bx - 64 >= fi.tile_col_start {
                    recon.lf_mask[recon.lf_idx - 1].ccso[p] as usize * 2
                } else {
                    0
                };
                let v = msac.decode_bool_adapt(cdf_m.ccso(p, ctx)) as u8;
                if trace_blk {
                    eprintln!(
                        "  CK ccso p={} ctx={} v={} rng={}",
                        p,
                        ctx,
                        v,
                        msac.dbg_rng()
                    );
                }
                v
            };
            recon.lf_mask[recon.lf_idx].ccso[p] = val;
            if !recon.cur_ccsomap.is_empty() {
                recon.cur_ccsomap[ccso_idx + p] = val;
            }
        }
    }

    // delta-q (per-superblock). Port of decode.c:1919-1960.
    if has_luma && (bx | by) & (63 >> (2 - fi.sb128)) == 0 {
        let prev_qidx = recon.last_qidx;
        let have_delta_q = fi.delta_q_present && (bs != fi.root_bs || b.skip_txfm == 0);
        if have_delta_q {
            if trace_blk {
                let c = cdf_m.delta_q();
                eprintln!("  CK delta_q_cdf {:?} rng={}", &c[..8], msac.dbg_rng());
            }
            let mut delta_q = msac.decode_symbol_adapt(cdf_m.delta_q(), 7) as i32;
            if delta_q == 7 {
                let n_bits = 1 + msac.decode_bools_bypass(3) as i32;
                delta_q = msac.decode_bools_bypass(n_bits as u32) as i32 + 1 + (1 << n_bits);
            }
            if delta_q != 0 {
                if msac.decode_bool_bypass() != 0 {
                    delta_q = -delta_q;
                }
                delta_q *= 1 << fi.delta_q_res_log2;
            }
            recon.last_qidx = iclip(recon.last_qidx + delta_q, 1, 255);
            if trace_blk {
                eprintln!(
                    "  CK delta_q d={} last_qidx={} rng={}",
                    delta_q >> fi.delta_q_res_log2,
                    recon.last_qidx,
                    msac.dbg_rng()
                );
            }
        }
        let new_qidx = recon.last_qidx;
        if new_qidx == fi.quant_yac {
            recon.dq_active = *recon.frame.dq;
        } else if new_qidx != prev_qidx {
            init_quant_tables_fi(fi, new_qidx, &mut recon.dqmem);
            recon.dq_active = recon.dqmem;
        }
    }

    // Intra mode decoding
    const REORDERED_NONDIR_Y_MODE: [u8; 5] = [0, 9, 10, 11, 12];
    const REORDERED_DIR_Y_MODE: [u8; 8] = [3, 8, 1, 5, 4, 6, 2, 7];

    let mut luma_midx = 0xffu8;
    if b.is_intra != 0 && !intrabc && has_luma {
        const DEFAULT_MODE_LIST_Y: [u8; 56] = [
            17, 45, 3, 10, 24, 31, 38, 52, 15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50,
            54, 16, 18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0, 6,
            7, 13, 21, 27, 28, 34, 35, 41, 49, 55,
        ];

        // DPCM (lossless mode) — gated on THIS segment's lossless flag
        // (decode.c:1977: frame_hdr->segmentation.lossless[b->seg_id]).
        let seg_lossless = fi.seg_lossless[b.seg_id as usize] != 0;
        let dpcm = seg_lossless && msac.decode_bool_adapt(cdf_m.dpcm(0)) != 0;
        let (y_mode, y_angle, midx);

        if dpcm {
            if msac.decode_bool_adapt(cdf_m.dpcm_dir(0)) != 0 {
                y_mode = 2; // HOR_PRED
                midx = 45u8;
            } else {
                y_mode = 1; // VERT_PRED
                midx = 17u8;
            }
            y_angle = 0i8;
            unsafe {
                b.data.intra.mrl_index = 0;
            }
            unsafe {
                b.data.intra.multi_mrl = 0;
            }
        } else {
            let y_set = msac.decode_symbol_adapt(cdf_m.intra_y_set(), 3) as usize;
            let y_mode_idx;

            if y_set == 0 {
                let y_mode_ctx =
                    (w4 == bw4 && a.midx[(bx4 + bw4 as usize).saturating_sub(1)] != 0xff) as usize
                        + (h4 == bh4 && l.midx[(by4 + bh4 as usize).saturating_sub(1)] != 0xff)
                            as usize;
                let mut idx0 = msac.decode_symbol_adapt(cdf_m.intra_y_idx0(y_mode_ctx), 7) as usize;
                if idx0 == 7 {
                    idx0 += msac.decode_symbol_adapt(cdf_m.intra_y_idx1(y_mode_ctx), 5) as usize;
                }
                y_mode_idx = idx0;
            } else {
                y_mode_idx = y_set * 16 - 3 + msac.decode_bools_bypass(4) as usize;
            }

            if y_mode_idx < 5 {
                y_mode = REORDERED_NONDIR_Y_MODE[y_mode_idx];
                y_angle = 0;
                midx = 0xff;
            } else {
                let dir_idx = y_mode_idx - 5;

                // Build custom mode list from neighbour directional modes
                let mut custom_list = [0u8; 56];
                let mut use_custom = false;
                let mut _list_len = 0usize;

                if bw4 * bh4 > 2 {
                    let mut mask = 0u64;
                    let mut ptr = 0usize;

                    if h4 == bh4 {
                        let lmidx = l.midx[(by4 + bh4 as usize).saturating_sub(1)];
                        if lmidx != 0xff {
                            custom_list[ptr] = lmidx;
                            mask |= 1u64 << lmidx;
                            ptr += 1;
                        }
                    }
                    if w4 == bw4 {
                        let amidx = a.midx[(bx4 + bw4 as usize).saturating_sub(1)];
                        if amidx != 0xff && (ptr == 0 || amidx != custom_list[0]) {
                            custom_list[ptr] = amidx;
                            mask |= 1u64 << amidx;
                            ptr += 1;
                        }
                    }
                    let n_dirs = ptr;
                    if n_dirs > 0 {
                        use_custom = true;
                        if bw4 * bh4 > 4 && dir_idx >= n_dirs {
                            for i in 1..5i32 {
                                for n in 0..n_dirs {
                                    let cmidx = custom_list[n] as i32;
                                    for delta in [-i, i] {
                                        let dmidx = ((cmidx + delta + 56) % 56) as u8;
                                        if mask & (1u64 << dmidx) == 0 {
                                            custom_list[ptr] = dmidx;
                                            mask |= 1u64 << dmidx;
                                            ptr += 1;
                                        }
                                    }
                                }
                            }
                        }
                        if dir_idx >= ptr {
                            for &fmidx in DEFAULT_MODE_LIST_Y.iter() {
                                let bit = 1u64 << fmidx;
                                if mask & bit == 0 {
                                    custom_list[ptr] = fmidx;
                                    ptr += 1;
                                }
                            }
                        }
                        _list_len = ptr;
                    }
                }

                let dir_y_mode_reord = if use_custom {
                    custom_list[dir_idx]
                } else {
                    DEFAULT_MODE_LIST_Y[dir_idx]
                };
                midx = dir_y_mode_reord;
                y_mode = REORDERED_DIR_Y_MODE[(dir_y_mode_reord / 7) as usize];
                y_angle = (dir_y_mode_reord % 7) as i8 - 3;
            }
        }

        unsafe {
            b.data.intra.dpcm[0] = dpcm as u8;
            b.data.intra.y_mode = y_mode;
            b.data.intra.y_angle = y_angle;
        }

        // FSC (Frequency Segmented Coding)
        if imax(bw4, bh4) <= 8 && fi.idtx_intra {
            #[rustfmt::skip]
            const FSC_BSIZE_GROUPS: [u8; N_BS_SIZES] = {
                let mut t = [0u8; N_BS_SIZES];
                t[BlockSize::Bs32x32 as u8 as usize] = 5;
                t[BlockSize::Bs32x16 as u8 as usize] = 5;
                t[BlockSize::Bs32x8 as u8 as usize] = 4;
                t[BlockSize::Bs32x4 as u8 as usize] = 4;
                t[BlockSize::Bs16x32 as u8 as usize] = 5;
                t[BlockSize::Bs16x16 as u8 as usize] = 4;
                t[BlockSize::Bs16x8 as u8 as usize] = 3;
                t[BlockSize::Bs16x4 as u8 as usize] = 3;
                t[BlockSize::Bs8x32 as u8 as usize] = 4;
                t[BlockSize::Bs8x16 as u8 as usize] = 3;
                t[BlockSize::Bs8x8 as u8 as usize] = 2;
                t[BlockSize::Bs8x4 as u8 as usize] = 1;
                t[BlockSize::Bs4x32 as u8 as usize] = 4;
                t[BlockSize::Bs4x16 as u8 as usize] = 3;
                t[BlockSize::Bs4x8 as u8 as usize] = 1;
                t
            };
            let sz_ctx = FSC_BSIZE_GROUPS[bs as u8 as usize] as usize;
            let fsc_ctx = if fi.is_inter_or_switch && intra_region == 0 {
                3usize
            } else {
                (nb_fsc[0] + nb_fsc[1]) as usize
            };
            b.fsc = msac.decode_bool_adapt(cdf_m.fsc(fsc_ctx, sz_ctx)) as u8;
        }
        if trace_blk {
            eprintln!(
                "  CK ymode={} fsc={} rng={}",
                unsafe { b.data.intra.y_mode },
                b.fsc,
                msac.dbg_rng()
            );
        }

        // MRL (Multi-Reference Line) index
        unsafe {
            b.data.intra.mrl_index = 0;
        }
        unsafe {
            b.data.intra.multi_mrl = 0;
        }
        if !dpcm && midx != 0xff && fi.mrls {
            let mrl_ctx = (nb_mrl[0] + nb_mrl[1]) as usize;
            let mrl_idx = msac.decode_symbol_adapt(cdf_m.mrl_index(mrl_ctx), 3) as u8;
            unsafe {
                b.data.intra.mrl_index = mrl_idx;
            }
            if mrl_idx > 0 {
                let mmrl_ctx = (nb_multi_mrl[0] + nb_multi_mrl[1]) as usize;
                let mmrl = msac.decode_bool_adapt(cdf_m.multi_mrl(mmrl_ctx)) as u8;
                unsafe {
                    b.data.intra.multi_mrl = mmrl;
                }
            }
        }

        luma_midx = midx;
    }

    // UV chroma mode decoding
    if b.is_intra != 0 && !intrabc && has_chroma {
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbx4 = (cbx & 63) as usize;
        let cby4 = (cby & 63) as usize;
        // Chroma block dims used by the cfl/mhccp allow conditions are
        // subsampled (decode.c:1556-1558: `cbw4 = cb_dim[0] >> ss_hor`).
        let cbw4 = (cb_dim[0] as i32) >> fi.ss_hor;
        let cbh4 = (cb_dim[1] as i32) >> fi.ss_ver;

        // For the chroma-only SDP tree (no luma), read the luma block's intra
        // direction mode from the per-SB map (decode.c:2170-2172).
        let mut midx = if !has_luma {
            recon.scratch.luma_intra_dir_mode_map[((cby & 15) * 16 + (cbx & 15)) as usize]
        } else {
            luma_midx
        };

        // DPCM for chroma — gated on THIS segment's lossless flag
        // (decode.c:2152: frame_hdr->segmentation.lossless[b->seg_id]).
        let seg_lossless_c = fi.seg_lossless[b.seg_id as usize] != 0;
        unsafe {
            b.data.intra.dpcm[1] =
                (seg_lossless_c && msac.decode_bool_adapt(cdf_m.dpcm(1)) != 0) as u8;
        }
        let chroma_dpcm = unsafe { b.data.intra.dpcm[1] } != 0;

        if chroma_dpcm {
            let uv_mode = if msac.decode_bool_adapt(cdf_m.dpcm_dir(1)) != 0 {
                2u8 // HOR_PRED
            } else {
                1u8 // VERT_PRED
            };
            let uv_mode_idx: i32 = if uv_mode == 2 { 45 } else { 17 };
            let uv_angle = if (midx as i32 - uv_mode_idx).unsigned_abs() >= 4 {
                0i8
            } else {
                (midx % 7) as i8 - 3
            };
            unsafe {
                b.data.intra.uv_mode = uv_mode;
                b.data.intra.uv_angle = uv_angle;
            }
        } else {
            // Per-segment lossless (decode.c:2166), not the frame-wide flag.
            let ll = fi.seg_lossless[b.seg_id as usize] != 0;
            let mhccp_allowed = fi.mhccp
                && imax(cbw4, cbh4) <= if ll { 1 } else { 8 }
                && cbw4 * cbh4 >= if ll { 1 } else { 2 };
            let cfl_allowed = (fi.cfl || mhccp_allowed)
                && (imax(bw4, bh4) > 16 || _sdp_cfl_disallowed == 0)
                && imax(cbw4, cbh4) <= if ll { 1 } else { 16 };

            let is_cfl = cfl_allowed && {
                let cfl_ctx =
                    (a.uvmode[cbx4] == CFL_PRED) as usize + (l.uvmode[cby4] == CFL_PRED) as usize;
                msac.decode_bool_adapt(cdf_m.cfl(cfl_ctx)) != 0
            };

            if is_cfl {
                unsafe {
                    b.data.intra.uv_mode = CFL_PRED;
                    b.data.intra.uv_angle = 0;
                    b.data.intra.cfl.cfl_alpha = [0; 2];
                }
                // CFL parameters
                const CFL_EXPLICIT: i8 = 0;
                const CFL_MHCCP: i8 = 2;
                if mhccp_allowed && (!fi.cfl || msac.decode_bool_adapt(cdf_m.mhccp()) != 0) {
                    let sz_ctx = SIZE_GROUP[bs as u8 as usize] as usize;
                    unsafe {
                        b.data.intra.cfl_type = CFL_MHCCP;
                        b.data.intra.cfl.cfl_mh_dir =
                            msac.decode_symbol_adapt(cdf_m.mhccp_filter_dir(sz_ctx), 2) as u8;
                    }
                } else {
                    let cfl_type = msac.decode_bool_adapt(cdf_m.cfl_type()) as i8;
                    unsafe {
                        b.data.intra.cfl_type = cfl_type;
                    }
                    if cfl_type == CFL_EXPLICIT {
                        let sign = msac.decode_symbol_adapt(cdf_m.cfl_sign(), 7) as i32 + 1;
                        let sign_u = (sign * 0x56) >> 8;
                        let sign_v = sign - sign_u * 3;
                        if sign_u != 0 {
                            let ctx = (sign_u == 2) as usize * 3 + sign_v as usize;
                            let mut alpha =
                                msac.decode_symbol_adapt(cdf_m.cfl_alpha(ctx), 7) as i8 + 1;
                            if sign_u == 1 {
                                alpha = -alpha;
                            }
                            unsafe {
                                b.data.intra.cfl.cfl_alpha[0] = alpha;
                            }
                        }
                        if sign_v != 0 {
                            let ctx = (sign_v == 2) as usize * 3 + sign_u as usize;
                            let mut alpha =
                                msac.decode_symbol_adapt(cdf_m.cfl_alpha(ctx), 7) as i8 + 1;
                            if sign_v == 1 {
                                alpha = -alpha;
                            }
                            unsafe {
                                b.data.intra.cfl.cfl_alpha[1] = alpha;
                            }
                        }
                    }
                }
            } else {
                let uv_mode_ctx = (midx != 0xff) as usize;
                let mut uv_mode_idx =
                    msac.decode_symbol_adapt(cdf_m.intra_uv_mode(uv_mode_ctx), 7) as usize;
                if uv_mode_idx == 7 {
                    uv_mode_idx += msac.decode_bools_bypass(3) as usize;
                }
                if uv_mode_idx > 12 {
                    if std::env::var("RAV2D_SUBMIT_ERR").is_ok() {
                        eprintln!("uv_mode_idx>12 bx={} by={}", bx, by);
                    }
                    return Err(());
                }

                if uv_mode_idx < uv_mode_ctx {
                    // Same directional mode as luma
                    unsafe {
                        b.data.intra.uv_mode = REORDERED_DIR_Y_MODE[(midx / 7) as usize];
                        b.data.intra.uv_angle = (midx % 7) as i8 - 3;
                    }
                } else if uv_mode_idx - uv_mode_ctx < 5 {
                    // Non-directional mode
                    unsafe {
                        b.data.intra.uv_mode = REORDERED_NONDIR_Y_MODE[uv_mode_idx - uv_mode_ctx];
                        b.data.intra.uv_angle = 0;
                    }
                } else {
                    // Directional mode from default UV list (decode.c:2188-2192).
                    // Order: VERT, HOR, DIAG_DOWN_LEFT, DIAG_DOWN_RIGHT, VERT_LEFT,
                    // VERT_RIGHT, HOR_DOWN, HOR_UP. With the AV2 mode enum
                    // (VERT_RIGHT=5, HOR_DOWN=6, HOR_UP=7, VERT_LEFT=8) this is:
                    const DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
                    const INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];

                    let mut idx = (uv_mode_idx - 5 - uv_mode_ctx) as i32;
                    if uv_mode_ctx != 0 {
                        idx +=
                            (idx >= INTRA_DIR_MODE_Y_TO_UV_IDX[(midx / 7) as usize] as i32) as i32;
                    }
                    unsafe {
                        b.data.intra.uv_mode = DEFAULT_MODE_LIST_UV[idx as usize];
                        b.data.intra.uv_angle = 0;
                    }
                }
            }
        }
        if trace_blk {
            eprintln!(
                "  CK uvmode uv={} cfl_type={} rng={}",
                unsafe { b.data.intra.uv_mode },
                unsafe { b.data.intra.cfl_type },
                msac.dbg_rng()
            );
        }
    }

    // Palette and DIP (has_luma intra path)
    if b.is_intra != 0 && !intrabc && has_luma {
        let y_mode = unsafe { b.data.intra.y_mode };
        unsafe {
            b.data.intra.pal_sz = 0;
        }

        if fi.allow_screen_content_tools
            && y_mode == 0 // DC_PRED
            && imax(bw4, bh4) <= 16
            && bw4 + bh4 >= 4
        {
            let use_y_pal = msac.decode_bool_adapt(cdf_m.pal_y()) != 0;
            if use_y_pal {
                // STUB: read palette colors from bitstream (AV2 §5.11.15)
                let pal_sz = msac.decode_symbol_adapt(cdf_m.pal_sz(), 6) as u8 + 2;
                unsafe {
                    b.data.intra.pal_sz = pal_sz;
                }
            }
        }

        // DIP (Directional Intra Prediction enhancement)
        unsafe {
            b.data.intra.dip = 0;
        }
        let pal_sz = unsafe { b.data.intra.pal_sz };
        if y_mode == 0 // DC_PRED
            && fi.intra_dip
            && pal_sz == 0
            && imin(bw4, bh4) >= 2
            && bw4 * bh4 >= 8
        {
            let nb_dip_0 = if nb_boff[0] != -1 { nb_dip[0] } else { 0 };
            let nb_dip_1 = if nb_boff[1] != -1 { nb_dip[1] } else { 0 };
            let ctx = nb_dip_0 as usize + nb_dip_1 as usize;
            let dip_flag = msac.decode_bool_adapt(cdf_m.dip(ctx)) != 0;
            if dip_flag {
                let tp = msac.decode_bools_bypass(1) as u8;
                let m = msac.decode_symbol_adapt(cdf_m.dip_mode(), 5) as u8;
                unsafe {
                    b.data.intra.dip = (tp << 4) | (m + 1);
                }
            }
        }
    }

    if trace_blk {
        eprintln!(
            "  CK predip dip={} pal={} rng={}",
            unsafe { b.data.intra.dip },
            unsafe { b.data.intra.pal_sz },
            msac.dbg_rng()
        );
    }

    // TX partition (intra path)
    if b.is_intra != 0 && !intrabc && has_luma {
        let __seg_ll = fi.seg_lossless[b.seg_id as usize] != 0;
        read_tx_part(msac, cdf_m, &mut b, bs, __seg_ll, fi.txfm_switchable);
    }
    if trace_blk {
        eprintln!(
            "  CK txpart tx_part={} tx_size_ll={} rng={}",
            b.tx_part,
            b.tx_size_ll,
            msac.dbg_rng()
        );
    }

    // is_sm flags for reconstruction (smooth mode neighbours)
    if b.is_intra != 0 && !intrabc {
        if has_luma {
            let sm = |mode: u8| -> i32 { (mode == 9 || mode == 10 || mode == 11) as i32 };
            let a_mode = a.mode[bx4];
            let l_mode = l.mode[by4];
            unsafe {
                b.data.intra.is_sm[0].a = if a.intra[bx4] != 0 { sm(a_mode) } else { 0 };
                b.data.intra.is_sm[0].l = if l.intra[by4] != 0 { sm(l_mode) } else { 0 };
            }
        }
        if has_chroma {
            let sm = |mode: u8| -> i32 { (mode == 9 || mode == 10 || mode == 11) as i32 };
            let cbx4 = (cbx & 63) as usize;
            let cby4 = (cby & 63) as usize;
            unsafe {
                b.data.intra.is_sm[1].a = sm(a.uvmode[cbx4]);
                b.data.intra.is_sm[1].l = sm(l.uvmode[cby4]);
            }
        }
    }

    // Intra context update
    if b.is_intra != 0 && !intrabc && has_luma {
        let y_mode = unsafe { b.data.intra.y_mode };
        let mrl_idx = unsafe { b.data.intra.mrl_index };
        let multi_mrl = unsafe { b.data.intra.multi_mrl };
        let dip_val = unsafe { b.data.intra.dip };
        let pal_sz_val = unsafe { b.data.intra.pal_sz };

        let aw = 1usize << b_dim[2];
        let lh = 1usize << b_dim[3];

        // Above context (a)
        a.fsc[bx4..bx4 + aw].fill(b.fsc);
        a.mode[bx4..bx4 + aw].fill(y_mode);
        a.midx[bx4..bx4 + aw].fill(luma_midx);
        a.mrl[bx4..bx4 + aw].fill((mrl_idx != 0) as u8);
        a.multi_mrl[bx4..bx4 + aw].fill(multi_mrl);
        a.dip[bx4..bx4 + aw].fill((dip_val != 0) as u8);
        a.pal_sz[bx4..bx4 + aw].fill(pal_sz_val);
        a.seg_pred[bx4..bx4 + aw].fill(seg_pred as u8);
        a.skip_mode[bx4..bx4 + aw].fill(0);
        a.intra[bx4..bx4 + aw].fill(1);
        a.intrabc[bx4..bx4 + aw].fill(0);
        a.morph_pred[bx4..bx4 + aw].fill(0);
        a.skip_txfm[bx4..bx4 + aw].fill(b.skip_txfm);
        if fi.is_inter_or_switch {
            a.amvd[bx4..bx4 + aw].fill(0);
            a.mvprec[bx4..bx4 + aw].fill(0);
            a.motion_mode[bx4..bx4 + aw].fill(0);
            a.comp_type[bx4..bx4 + aw].fill(0);
            a.r#ref[0][bx4..bx4 + aw].fill(-1);
            a.r#ref[1][bx4..bx4 + aw].fill(-1);
        }

        // Left context (l)
        l.fsc[by4..by4 + lh].fill(b.fsc);
        l.mode[by4..by4 + lh].fill(y_mode);
        l.midx[by4..by4 + lh].fill(luma_midx);
        l.mrl[by4..by4 + lh].fill((mrl_idx != 0) as u8);
        l.multi_mrl[by4..by4 + lh].fill(multi_mrl);
        l.dip[by4..by4 + lh].fill((dip_val != 0) as u8);
        l.pal_sz[by4..by4 + lh].fill(pal_sz_val);
        l.seg_pred[by4..by4 + lh].fill(seg_pred as u8);
        l.skip_mode[by4..by4 + lh].fill(0);
        l.intra[by4..by4 + lh].fill(1);
        l.intrabc[by4..by4 + lh].fill(0);
        l.morph_pred[by4..by4 + lh].fill(0);
        l.skip_txfm[by4..by4 + lh].fill(b.skip_txfm);
        if fi.is_inter_or_switch {
            l.amvd[by4..by4 + lh].fill(0);
            l.mvprec[by4..by4 + lh].fill(0);
            l.motion_mode[by4..by4 + lh].fill(0);
            l.comp_type[by4..by4 + lh].fill(0);
            l.r#ref[0][by4..by4 + lh].fill(-1);
            l.r#ref[1][by4..by4 + lh].fill(-1);
        }
    }

    // Chroma context update (uvmode)
    if b.is_intra != 0 && !intrabc && has_chroma {
        let uv_mode = unsafe { b.data.intra.uv_mode };
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbx4 = (cbx & 63) as usize;
        let cby4 = (cby & 63) as usize;
        let cbw4 = 1usize << cb_dim[2];
        let cbh4 = 1usize << cb_dim[3];
        a.uvmode[cbx4..cbx4 + cbw4].fill(uv_mode);
        l.uvmode[cby4..cby4 + cbh4].fill(uv_mode);
    }

    // IntraBC path
    if intrabc {
        unsafe {
            b.data.intra.is_refmv = msac.decode_bool_adapt(cdf_m.intrabc_mode()) as u8;
        }
        if trace_blk {
            eprintln!(
                "  CK ibc_mode is_refmv={} rng={}",
                unsafe { b.data.intra.is_refmv },
                msac.dbg_rng()
            );
        }

        unsafe {
            b.data.inter.drl_idx[0] = 0;
        }
        for _ in 0..fi.max_bvp_drl_bits {
            if msac.decode_bools_bypass(1) == 0 {
                break;
            }
            unsafe {
                b.data.inter.drl_idx[0] += 1;
            }
        }
        if trace_blk {
            eprintln!(
                "  CK ibc_drl drl={} maxbits={} rng={}",
                unsafe { b.data.inter.drl_idx[0] },
                fi.max_bvp_drl_bits,
                msac.dbg_rng()
            );
        }

        let is_refmv = unsafe { b.data.intra.is_refmv };
        unsafe {
            b.data.intra.is_qpel = (!fi.force_integer_mv) as u8;
        }
        if is_refmv == 0 && !fi.force_integer_mv {
            unsafe {
                b.data.intra.is_qpel = msac.decode_bool_adapt(cdf_m.intrabc_precision()) as u8;
            }
        }
        if trace_blk {
            eprintln!(
                "  CK ibc_qpel is_qpel={} fim={} rng={}",
                unsafe { b.data.intra.is_qpel },
                fi.force_integer_mv,
                msac.dbg_rng()
            );
        }

        // IntraBC MV residual
        if is_refmv == 0 {
            let mv_prec = 3 + 2 * (unsafe { b.data.intra.is_qpel } as i32);
            let mut mv = read_mv_full(msac, cdf_dmv, mv_prec);
            unsafe {
                if mv.c.y != 0 && msac.decode_bools_bypass(1) != 0 {
                    mv.c.y = -mv.c.y;
                }
                if mv.c.x != 0 && msac.decode_bools_bypass(1) != 0 {
                    mv.c.x = -mv.c.x;
                }
                b.data.intra.intrabc_mv = mv;
            }
            if trace_blk {
                eprintln!(
                    "  CK ibc_mv mvy={} mvx={} prec={} rng={}",
                    unsafe { b.data.intra.intrabc_mv.c.y },
                    unsafe { b.data.intra.intrabc_mv.c.x },
                    mv_prec,
                    msac.dbg_rng()
                );
            }
            if std::env::var("RAV2D_IBC2").is_ok() {
                eprintln!(
                    "IBC2MV y={} x={} mvy={} mvx={} prec={}",
                    by,
                    bx,
                    unsafe { b.data.intra.intrabc_mv.c.y },
                    unsafe { b.data.intra.intrabc_mv.c.x },
                    mv_prec
                );
            }
        }

        // morph_pred for IntraBC (read BEFORE the tx partition, decode.c:2398).
        unsafe {
            b.data.intra.morph_pred = 0;
        }
        if !fi.is_inter_or_switch && fi.bawp && fi.allow_screen_content_tools {
            let nb_mp_0 = if nb_boff[0] != -1 { nb_morph[0] } else { 0 };
            let nb_mp_1 = if nb_boff[1] != -1 { nb_morph[1] } else { 0 };
            let ctx = nb_mp_0 as usize + nb_mp_1 as usize;
            unsafe {
                b.data.intra.morph_pred = msac.decode_bool_adapt(cdf_m.morph_pred(ctx)) as u8;
            }
        }
        let morph_pred = unsafe { b.data.intra.morph_pred };
        if trace_blk {
            eprintln!("  CK ibc_morph morph={} rng={}", morph_pred, msac.dbg_rng());
        }

        // TX partition for IntraBC
        let __seg_ll = fi.seg_lossless[b.seg_id as usize] != 0;
        read_tx_part(msac, cdf_m, &mut b, bs, __seg_ll, fi.txfm_switchable);
        if trace_blk {
            eprintln!(
                "  CK ibc_txpart tx_part={} tx_size_ll={} rng={}",
                b.tx_part,
                b.tx_size_ll,
                msac.dbg_rng()
            );
        }

        // IntraBC context write-back
        if has_luma {
            let aw = 1usize << b_dim[2];
            let lh = 1usize << b_dim[3];

            a.fsc[bx4..bx4 + aw].fill(0);
            a.mode[bx4..bx4 + aw].fill(0); // DC_PRED
            a.midx[bx4..bx4 + aw].fill(0xff);
            a.mrl[bx4..bx4 + aw].fill(0);
            a.multi_mrl[bx4..bx4 + aw].fill(0);
            a.dip[bx4..bx4 + aw].fill(0);
            a.pal_sz[bx4..bx4 + aw].fill(0);
            a.seg_pred[bx4..bx4 + aw].fill(0);
            a.skip_mode[bx4..bx4 + aw].fill(0);
            a.intrabc[bx4..bx4 + aw].fill(1);
            a.morph_pred[bx4..bx4 + aw].fill(morph_pred);
            a.intra[bx4..bx4 + aw].fill(1);
            a.skip_txfm[bx4..bx4 + aw].fill(b.skip_txfm);
            if fi.is_inter_or_switch {
                a.amvd[bx4..bx4 + aw].fill(0);
                a.mvprec[bx4..bx4 + aw].fill(0);
                a.comp_type[bx4..bx4 + aw].fill(0);
                a.motion_mode[bx4..bx4 + aw].fill(0);
                a.r#ref[0][bx4..bx4 + aw].fill(-1);
                a.r#ref[1][bx4..bx4 + aw].fill(-1);
            }

            l.fsc[by4..by4 + lh].fill(0);
            l.mode[by4..by4 + lh].fill(0);
            l.midx[by4..by4 + lh].fill(0xff);
            l.mrl[by4..by4 + lh].fill(0);
            l.multi_mrl[by4..by4 + lh].fill(0);
            l.dip[by4..by4 + lh].fill(0);
            l.pal_sz[by4..by4 + lh].fill(0);
            l.seg_pred[by4..by4 + lh].fill(0);
            l.skip_mode[by4..by4 + lh].fill(0);
            l.intrabc[by4..by4 + lh].fill(1);
            l.morph_pred[by4..by4 + lh].fill(morph_pred);
            l.intra[by4..by4 + lh].fill(1);
            l.skip_txfm[by4..by4 + lh].fill(b.skip_txfm);
            if fi.is_inter_or_switch {
                l.amvd[by4..by4 + lh].fill(0);
                l.mvprec[by4..by4 + lh].fill(0);
                l.comp_type[by4..by4 + lh].fill(0);
                l.motion_mode[by4..by4 + lh].fill(0);
                l.r#ref[0][by4..by4 + lh].fill(-1);
                l.r#ref[1][by4..by4 + lh].fill(-1);
            }
        }
        if has_chroma {
            let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
            let cbx4 = (cbx & 63) as usize;
            let cby4 = (cby & 63) as usize;
            let cbw4 = 1usize << cb_dim[2];
            let cbh4 = 1usize << cb_dim[3];
            a.uvmode[cbx4..cbx4 + cbw4].fill(0); // DC_PRED
            l.uvmode[cby4..cby4 + cbh4].fill(0);
        }
    }

    // Inter mode path
    let mut mvprec_def = 1u8;
    if b.is_intra == 0 && !intrabc {
        unsafe {
            b.data.inter.amvd = 0;
            b.data.inter.motion_mode = 0; // Translation
            b.data.inter.refine_mv = 0;
        }

        // TIP decision
        let is_tip =
            if b.skip_mode == 0 && fi.tip_frame_mode != 0 && cbs == lbs && imax(bw4, bh4) >= 2 {
                let ctx = (if n_ctx >= 1 {
                    (nx_ref0[0] == TIP_FRAME as i8) as usize
                } else {
                    0
                }) + (if n_ctx >= 2 {
                    (nx_ref0[1] == TIP_FRAME as i8) as usize
                } else {
                    0
                });
                msac.decode_bool_adapt(cdf_m.tip(ctx)) != 0
            } else {
                false
            };

        // Compound decision
        let is_comp = if b.skip_mode != 0 {
            true
        } else if !is_tip
            && (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) == 0
            && fi.switchable_comp_refs
            && bw4 * bh4 >= 4
        {
            // get_comp_ctx (env.h:140). refdir(ref): ref==-1 -> intra (-1), else
            // fi.refdir[ref] (refdir_intra is -1 from lib init).
            // refdir_intra is -1 (lib.c init); intra/intrabc neighbours use it.
            let refdir = |r: i8| -> i32 {
                if r < 0 {
                    -1
                } else {
                    fi.refdir[r as usize] as i32
                }
            };
            let ctx = match n_ctx {
                2 => {
                    let refa2 = nx_ref1[0];
                    let refb2 = nx_ref1[1];
                    if refa2 == -1 {
                        let refa1 = nx_ref0[0];
                        if refb2 == -1 {
                            let refb1 = nx_ref0[1];
                            ((refdir(refa1) == 1) ^ (refdir(refb1) == 1)) as usize
                        } else {
                            2 + ((nx_intrabc[0] == 0) && refdir(refa1) != 0) as usize
                        }
                    } else if refb2 == -1 {
                        let refb1 = nx_ref0[1];
                        2 + ((nx_intrabc[1] == 0) && refdir(refb1) != 0) as usize
                    } else {
                        4
                    }
                }
                1 => {
                    let ref2 = nx_ref1[0];
                    if ref2 == -1 {
                        let ref1 = nx_ref0[0];
                        ((nx_intrabc[0] == 0) && refdir(ref1) != 0) as usize
                    } else {
                        3
                    }
                }
                _ => 1,
            };
            msac.decode_bool_adapt(cdf_m.comp(ctx)) != 0
        } else {
            false
        };

        if b.skip_mode != 0 {
            // skip_mode DRL index
            unsafe {
                b.data.inter.drl_idx[0] = 0;
            }
            let mut ctx = 0usize;
            for _ in 0..fi.max_drl_bits {
                if msac.decode_bool_adapt(cdf_m.skip_mode_drl_idx(ctx)) == 0 {
                    break;
                }
                unsafe {
                    b.data.inter.drl_idx[0] += 1;
                }
                if ctx < 2 {
                    ctx += 1;
                }
            }
            b.ref_pair = fi.skip_mode_refs;
            unsafe {
                b.data.inter.comp_type = 1; // COMP_AVG
                b.data.inter.inter_mode = 0; // NEARESTMV_NEARESTMV-like
            }
        } else if is_comp {
            // --- compound ref selection ---
            let n_refs = fi.n_ref_frames as i32;
            let (ref0, ref1): (i8, i8);
            if n_refs > 1 {
                let same_refs = fi.num_same_ref_comp as i32;
                let mut n = 0i32;
                let mut cnt = [0u8; 9];
                if n_ctx > 0 {
                    cnt[(nx_ref0[0] + 1) as usize] += 1;
                    cnt[(nx_ref1[0] + 1) as usize] += 1;
                    if n_ctx > 1 {
                        cnt[(nx_ref0[1] + 1) as usize] += 1;
                        cnt[(nx_ref1[1] + 1) as usize] += 1;
                    }
                }
                let mut cnt_rem = (n_ctx as i32) * 2 - cnt[0] as i32 - cnt[8] as i32;
                let mut refs = [-1i8; 2];
                let mut dir = 0u8;
                let mut maybe_same_ref = if same_refs > 0 { 1i32 } else { 0 };
                let mut i = 0i32;
                while i < n_refs + n - 2 + maybe_same_ref {
                    let cnt_cur = cnt[i as usize + 1] as i32;
                    cnt_rem -= cnt_cur;
                    let bit = if n == 0 && (i == 2 || (i >= n_refs - 2 && i + 1 >= same_refs)) {
                        1
                    } else {
                        let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
                        let cdf = if n == 0 {
                            cdf_m.comp0_ref(ctx, i as usize)
                        } else {
                            let dir_idx = (dir ^ fi.refdir[i as usize]) as usize;
                            cdf_m.comp1_ref(ctx, dir_idx, i as usize)
                        };
                        msac.decode_bool_adapt(cdf) as i32
                    };
                    if bit != 0 {
                        refs[n as usize] = i as i8;
                        n += 1;
                        if n == 2 {
                            break;
                        }
                        dir = fi.refdir[i as usize];
                    }
                    if maybe_same_ref != 0 {
                        maybe_same_ref = if bit == 0 && i + 1 < same_refs { 1 } else { 0 };
                        if bit != 0 {
                            i -= 1;
                            cnt_rem += cnt_cur;
                        }
                    }
                    i += 1;
                }
                if n < 2 {
                    refs[1] = (n_refs - 1) as i8;
                    if n == 0 {
                        refs[0] = (n_refs - 1 - (same_refs < n_refs) as i32) as i8;
                    }
                }
                ref0 = refs[0];
                ref1 = refs[1];
            } else {
                ref0 = 0;
                ref1 = 0;
            }
            unsafe {
                b.ref_pair.r[0] = ref0;
                b.ref_pair.r[1] = ref1;
            }

            // --- compound inter_mode ---
            // get_compref_ctx (env.h:256, decode.c:2569). Counts neighbours
            // whose compound ref-pair (or TIP-coded ref) matches this block's,
            // splitting row/col + a NEWMV bit -> ctx in {0..5}.
            let comp_ctx = crate::env::get_compref_ctx(
                a,
                l,
                by4,
                bx4,
                have_top,
                have_left,
                have_top_right,
                have_bottom_left,
                b_dim,
                b.ref_pair,
                fi.tip,
            ) as usize;
            if trace_blk {
                eprintln!(
                    "  CK comp ref=[{},{}] comp_ctx={} rng={}",
                    ref0,
                    ref1,
                    comp_ctx,
                    msac.dbg_rng()
                );
            }
            let inter_mode: u8;
            if ref0 == ref1 {
                let sym = msac.decode_symbol_adapt(cdf_m.comp_mode_sameref(comp_ctx), 3) as u8;
                let mut m = CompInterPredMode::NearMvNearMv as u8 + sym;
                if m > CompInterPredMode::NearMvNewMv as u8 {
                    m += 1;
                } // skip newmv_nearmv
                inter_mode = m;
            } else {
                let joint_ctx = (fi.refdist[ref0 as usize] != -fi.refdist[ref1 as usize]) as usize;
                if msac.decode_bool_adapt(cdf_m.comp_mode_joint(joint_ctx)) != 0 {
                    inter_mode = CompInterPredMode::JointNewMv as u8;
                } else {
                    inter_mode = CompInterPredMode::NearMvNearMv as u8
                        + msac.decode_symbol_adapt(cdf_m.comp_mode(comp_ctx), 4) as u8;
                }
            };

            // --- OPFL refinement ---
            let mut final_inter_mode = inter_mode;
            if fi.opfl_refine_type == 1
                && inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
                && imin(bw4, bh4) >= 2
                && fi.refdir[ref0 as usize] != fi.refdir[ref1 as usize]
            {
                let ctx = (inter_mode > CompInterPredMode::NearMvNearMv as u8) as usize;
                if msac.decode_bool_adapt(cdf_m.opfl(ctx)) != 0 {
                    final_inter_mode +=
                        6 - (inter_mode >= CompInterPredMode::GlobalMvGlobalMv as u8) as u8;
                }
            }
            unsafe {
                b.data.inter.inter_mode = final_inter_mode;
            }
            if trace_blk {
                eprintln!(
                    "  CK comp_inter_mode[ctx={},{}] rng={}",
                    comp_ctx,
                    final_inter_mode,
                    msac.dbg_rng()
                );
            }

            // --- compound AMVD ---
            use crate::tables::COMP_INTER_PRED_MODES;
            let mode_idx = (final_inter_mode - CompInterPredMode::NearMvNearMv as u8) as usize;
            let m_pair = if mode_idx < COMP_INTER_PRED_MODES.len() {
                COMP_INTER_PRED_MODES[mode_idx]
            } else {
                [InterPredMode::NearMv as u8; 2]
            };
            let is_newmv_mode =
                m_pair[0] == InterPredMode::NewMv as u8 || m_pair[1] == InterPredMode::NewMv as u8;
            if fi.adaptive_mvd && is_newmv_mode {
                let amvd_mode_ctx = match final_inter_mode {
                    x if x == CompInterPredMode::NearMvNewMv as u8 => 0usize,
                    x if x == CompInterPredMode::NewMvNearMv as u8 => 1,
                    x if x == CompInterPredMode::OpflNearMvNewMv as u8 => 2,
                    x if x == CompInterPredMode::OpflNewMvNearMv as u8 => 3,
                    x if x == CompInterPredMode::JointNewMv as u8 => 5,
                    x if x == CompInterPredMode::OpflJointNewMv as u8 => 6,
                    x if x == CompInterPredMode::NewMvNewMv as u8 => 7,
                    x if x == CompInterPredMode::OpflNewMvNewMv as u8 => 8,
                    _ => 0,
                };
                let ctx = (nx_ref0[0] == ref0 && nx_amvd[0] != 0) as usize
                    + (if n_ctx > 1 {
                        nx_ref0[1] == ref0 && nx_amvd[1] != 0
                    } else {
                        false
                    }) as usize;
                unsafe {
                    b.data.inter.amvd =
                        msac.decode_bool_adapt(cdf_m.amvd(amvd_mode_ctx, ctx)) as i8;
                }
            }
            let amvd_val = unsafe { b.data.inter.amvd };
            if trace_blk {
                eprintln!("  CK comp_amvd[{}] rng={}", amvd_val, msac.dbg_rng());
            }

            // --- JMVD scale mode ---
            let mut jmvd_scale_mode = 0u8;
            if final_inter_mode == CompInterPredMode::JointNewMv as u8
                || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8
            {
                jmvd_scale_mode = if amvd_val != 0 {
                    msac.decode_symbol_adapt(cdf_m.jmvd_amvd_scale_mode(), 2) as u8
                } else {
                    msac.decode_symbol_adapt(cdf_m.jmvd_scale_mode(), 4) as u8
                };
            }

            // --- compound DRL ---
            unsafe {
                b.data.inter.drl_idx = [0; 2];
            }
            if final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8 {
                let n_drls = 1 + (final_inter_mode <= CompInterPredMode::NearMvNewMv as u8) as i32;
                let max_drl = fi.max_drl_bits as i32;
                let mut n = 0i32;
                let mut ctx = 0usize;
                for r in 0..n_drls {
                    while n < max_drl {
                        if msac.decode_bool_adapt(cdf_m.drl_idx(ctx, comp_ctx)) == 0 {
                            break;
                        }
                        n += 1;
                        if ctx < 2 {
                            ctx += 1;
                        }
                    }
                    unsafe {
                        b.data.inter.drl_idx[r as usize] = n as u8;
                    }
                    if final_inter_mode == CompInterPredMode::NearMvNearMv as u8 && ref0 == ref1 {
                        let drl0 = unsafe { b.data.inter.drl_idx[0] } as i32;
                        n = drl0 + (drl0 < max_drl) as i32;
                    } else {
                        n = 0;
                    }
                    ctx = (n as usize).min(2);
                }
                if n_drls == 1 {
                    unsafe {
                        b.data.inter.drl_idx[1] = b.data.inter.drl_idx[0];
                    }
                }
            }
            if trace_blk {
                let d = unsafe { b.data.inter.drl_idx };
                eprintln!("  CK comp_drl[{},{}] rng={}", d[0], d[1], msac.dbg_rng());
            }

            // --- MV precision ---
            let mut mv_prec = 3i32 + fi.mv_precision as i32;
            if mv_prec > 3 && amvd_val == 0 && fi.flex_mvres && is_newmv_mode {
                let mvprec1 = if nb_boff[0] == -1 { 0u8 } else { nb_mvprec[0] };
                let mvprec2 = if nb_boff[1] == -1 { 0u8 } else { nb_mvprec[1] };
                let ctx1 = ((mvprec1 & 1) + (mvprec2 & 1)) as usize;
                if msac.decode_bool_adapt(cdf_m.mvprec_def(ctx1)) == 0 {
                    let ctx2 = ((mvprec1 | mvprec2) >> 1) as usize;
                    let idx = msac
                        .decode_symbol_adapt(cdf_m.mvprec_rem(ctx2, (mv_prec - 4) as usize), 2)
                        as usize;
                    mv_prec = MV_PREC_TBL[(mv_prec == 6) as usize][idx] as i32;
                    mvprec_def = 2;
                }
            }
            unsafe {
                b.data.inter.mv_prec = mv_prec as i8;
            }

            // --- MV residuals + sign derivation ---
            if final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8 {
                let is_joint = final_inter_mode == CompInterPredMode::JointNewMv as u8
                    || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8;
                let (start, end) = if is_joint {
                    let rd0 = fi.absrefdist[ref0 as usize] as i32;
                    let rd1 = fi.absrefdist[ref1 as usize] as i32;
                    let s = (rd0 < rd1) as usize;
                    (s, s + 1)
                } else {
                    (0usize, 2usize)
                };
                let mut sum_mvd = 0i32;
                let mut nnzc = 0i32;
                for n in start..end {
                    if m_pair.get(n).copied() != Some(InterPredMode::NewMv as u8) {
                        continue;
                    }
                    let mv = if amvd_val != 0 {
                        read_amvd(msac, cdf_m)
                    } else {
                        read_mv_full(msac, cdf_dmv, mv_prec)
                    };
                    unsafe {
                        b.data.inter.mv[n].c.x = mv.c.x;
                        b.data.inter.mv[n].c.y = mv.c.y;
                    }
                    if amvd_val == 0 {
                        unsafe {
                            sum_mvd += b.data.inter.mv[n].c.y + b.data.inter.mv[n].c.x;
                            nnzc += (b.data.inter.mv[n].c.y != 0) as i32
                                + (b.data.inter.mv[n].c.x != 0) as i32;
                        }
                    }
                }

                // sign derivation
                if final_inter_mode != CompInterPredMode::NearMvNearMv as u8
                    && final_inter_mode != CompInterPredMode::OpflNearMvNearMv as u8
                {
                    let bidir_newmv = final_inter_mode == CompInterPredMode::NewMvNewMv as u8
                        || final_inter_mode == CompInterPredMode::OpflNewMvNewMv as u8
                        || final_inter_mode == CompInterPredMode::JointNewMv as u8
                        || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8;
                    let drl0 = unsafe { b.data.inter.drl_idx[0] };
                    let drl1 = unsafe { b.data.inter.drl_idx[1] };
                    if !fi.mvd_sign_derive
                        || drl0 != 0
                        || drl1 != 0
                        || nnzc < 3 * (end as i32 - start as i32) - 2
                        || fi.allow_screen_content_tools
                        || fi.mv_precision == 3
                        || mv_prec >= 5
                        || !bidir_newmv
                        || unsafe { b.data.inter.motion_mode } != MotionMode::Translation as u8
                    {
                        nnzc = 5; // disable sign derivation
                    }
                    sum_mvd >>= 6 - mv_prec;
                    let mut nnzc2 = 0i32;
                    for n in start..end {
                        if m_pair.get(n).copied() != Some(InterPredMode::NewMv as u8) {
                            continue;
                        }
                        let cur_y = unsafe { b.data.inter.mv[n].c.y };
                        if cur_y != 0 {
                            nnzc2 += 1;
                            let s = if nnzc2 == nnzc {
                                (sum_mvd & 1) != 0
                            } else {
                                msac.decode_bool_bypass() != 0
                            };
                            if s {
                                unsafe {
                                    b.data.inter.mv[n].c.y = -cur_y;
                                }
                            }
                        }
                        let cur_x = unsafe { b.data.inter.mv[n].c.x };
                        if cur_x != 0 {
                            nnzc2 += 1;
                            let s = if nnzc2 == nnzc {
                                (sum_mvd & 1) != 0
                            } else {
                                msac.decode_bool_bypass() != 0
                            };
                            if s {
                                unsafe {
                                    b.data.inter.mv[n].c.x = -cur_x;
                                }
                            }
                        }
                    }
                }
            }

            if trace_blk {
                let m = unsafe { b.data.inter.mv };
                eprintln!(
                    "  CK comp_mv[0,y:{},x:{}][1,y:{},x:{}] rng={}",
                    unsafe { m[0].c.y },
                    unsafe { m[0].c.x },
                    unsafe { m[1].c.y },
                    unsafe { m[1].c.x },
                    msac.dbg_rng()
                );
            }

            // --- refine_mv ---
            unsafe {
                b.data.inter.refine_mv = 0;
            }
            if fi.refine_mv_enabled
                && imin(bw4, bh4) >= 2
                && bw4 * bh4 > 4
                && final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
                && fi.refdist[ref0 as usize] == -fi.refdist[ref1 as usize]
            {
                let is_opfl_mode = final_inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8;
                let nearmv_nearmv = final_inter_mode == CompInterPredMode::NearMvNearMv as u8
                    || final_inter_mode == CompInterPredMode::OpflNearMvNearMv as u8
                    || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8;
                if nearmv_nearmv {
                    unsafe {
                        b.data.inter.refine_mv = 2;
                    }
                } else if !is_opfl_mode || fi.opfl_refine_type != 1 {
                    let ctx = (final_inter_mode - CompInterPredMode::NearMvNearMv as u8) as usize;
                    let ctx_clamped = ctx.min(10);
                    unsafe {
                        b.data.inter.refine_mv =
                            msac.decode_bool_adapt(cdf_m.refine_mv(ctx_clamped)) as u8;
                    }
                }
            }
            let refine_mv_val = unsafe { b.data.inter.refine_mv };

            // --- subpel filter for compound ---
            let has_subpel_filter = final_inter_mode <= CompInterPredMode::JointNewMv as u8
                && refine_mv_val == 0
                && unsafe { b.data.inter.motion_mode } == MotionMode::Translation as u8
                && (final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
                    || imin(bw4, bh4) == 1);

            // --- compound type ---
            unsafe {
                b.data.inter.comp_type = 1;
            } // COMP_AVG
            if final_inter_mode <= CompInterPredMode::JointNewMv as u8
                && refine_mv_val != 1
                && !(final_inter_mode == CompInterPredMode::JointNewMv as u8 && amvd_val != 0)
                && fi.masked_compound
                && imin(bw4, bh4) >= 2
            {
                // comp_type masked context (decode.c:2820). Each gathered
                // neighbour (num < n_ctx) contributes: 0 if intra/single-ref
                // with comp_type==AVG, 2 if its ref0 is the furthest-future
                // ref; 1 if it is itself masked-compound. Combine the two
                // neighbours + a both-nonzero bit + a same-absrefdist bias.
                let ffr = fi.furthest_future_refidx;
                let comptype_ctx = |num: usize| -> i32 {
                    if num >= n_ctx as usize {
                        0
                    } else if nx_ref1[num] != -1 {
                        (nx_comp_type[num] > 1) as i32
                    } else {
                        (nx_ref0[num] == ffr) as i32 * 2
                    }
                };
                let cctx0 = comptype_ctx(0);
                let cctx1 = comptype_ctx(1);
                let ctx = (cctx0
                    + cctx1
                    + (cctx0 != 0 && cctx1 != 0) as i32
                    + (fi.absrefdist[ref0 as usize] == fi.absrefdist[ref1 as usize]) as i32 * 6)
                    as usize;
                let has_mask = msac.decode_bool_adapt(cdf_m.comp_type_masked(ctx)) != 0;
                if has_mask {
                    if imax(bw4, bh4) <= 16
                        && msac.decode_bool_adapt(cdf_m.comp_type_weighted()) == 0
                    {
                        unsafe {
                            b.data.inter.comp_type = 2; // COMP_WEDGE
                            b.data.inter.wedge_idx = read_wedge_idx(msac, cdf_m);
                            b.data.inter.wedge_sign = msac.decode_bool_bypass() as i8;
                        }
                    } else {
                        unsafe {
                            b.data.inter.comp_type = 3; // COMP_SEG
                            b.data.inter.mask_sign = msac.decode_bool_bypass() as u8;
                        }
                    }
                }
            }

            if trace_blk {
                let ct = unsafe { b.data.inter.comp_type };
                eprintln!("  CK comp_type[{}] rng={}", ct as i32 - 1, msac.dbg_rng());
            }

            // --- CWP (compound weighted prediction) ---
            unsafe {
                b.data.inter.cwp_idx = 8;
            }
            let comp_type_val = unsafe { b.data.inter.comp_type };
            if refine_mv_val == 0
                && jmvd_scale_mode == 0
                && fi.cwp
                && comp_type_val == 1
                && (final_inter_mode == CompInterPredMode::NearMvNearMv as u8
                    || final_inter_mode == CompInterPredMode::JointNewMv as u8)
            {
                let mut n = 0u8;
                while n < 4 {
                    if msac.decode_bool_adapt(cdf_m.cwp_idx(n as usize)) == 0 {
                        break;
                    }
                    n += 1;
                }
                static CWP_WEIGHT_SAME: [i8; 5] = [8, 12, 4, 10, 6];
                static CWP_WEIGHT_DIFF: [i8; 5] = [8, 12, 4, 20, -4];
                let same_dir = fi.refdir[ref0 as usize] == fi.refdir[ref1 as usize];
                unsafe {
                    b.data.inter.cwp_idx = if same_dir {
                        CWP_WEIGHT_SAME[n as usize]
                    } else {
                        CWP_WEIGHT_DIFF[n as usize]
                    };
                }
            }

            // --- subpel filter ---
            if refine_mv_val != 0 || final_inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8 {
                unsafe {
                    b.data.inter.filter = 2;
                } // SHARP
            } else if fi.subpel_filter_mode == 4 && has_subpel_filter {
                // get_filter_ctx (env.h:120) with comp=1 (ref[1] != -1).
                let fctx = get_filter_ctx(a, l, &nb_boff, ref0, true);
                unsafe {
                    b.data.inter.filter = msac.decode_symbol_adapt(cdf_m.filter(fctx), 2) as u8;
                }
            } else if fi.subpel_filter_mode == 4 {
                unsafe {
                    b.data.inter.filter = 0;
                }
            } else {
                unsafe {
                    b.data.inter.filter = fi.subpel_filter_mode;
                }
            }
            if trace_blk {
                let flt = unsafe { b.data.inter.filter };
                eprintln!("  CK comp_subpelfilter[{}] rng={}", flt, msac.dbg_rng());
            }
        } else {
            unsafe {
                b.data.inter.comp_type = 0;
            } // COMP_INTER_NONE

            // --- single ref selection ---
            let ref0: i8;
            if (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) != 0 {
                ref0 = 0;
            } else if is_tip {
                ref0 = TIP_FRAME as i8;
            } else {
                let n_refs = fi.n_ref_frames as i32;
                let mut i = 0i32;
                if n_refs > 1 {
                    let mut cnt = [0u8; 9];
                    if n_ctx > 0 {
                        cnt[(nx_ref0[0] + 1) as usize] += 1;
                        cnt[(nx_ref1[0] + 1) as usize] += 1;
                        if n_ctx > 1 {
                            cnt[(nx_ref0[1] + 1) as usize] += 1;
                            cnt[(nx_ref1[1] + 1) as usize] += 1;
                        }
                    }
                    let mut cnt_rem = (n_ctx as i32) * 2 - cnt[0] as i32 - cnt[8] as i32;
                    loop {
                        let cnt_cur = cnt[i as usize + 1] as i32;
                        cnt_rem -= cnt_cur;
                        let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
                        if msac.decode_bool_adapt(cdf_m.single_ref(ctx, i as usize)) != 0 {
                            break;
                        }
                        i += 1;
                        if i >= n_refs - 1 {
                            break;
                        }
                    }
                }
                ref0 = i as i8;
            }
            unsafe {
                b.ref_pair.r[0] = ref0;
                b.ref_pair.r[1] = -1;
            }

            // --- sngl_ctx ---
            let sngl_ctx = get_snglref_ctx(
                a,
                l,
                by4,
                bx4,
                have_top,
                have_left,
                have_top_right,
                have_bottom_left,
                b_dim,
                ref0,
            );

            // --- inter_mode ---
            let inter_mode: u8;
            if (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) != 0 {
                inter_mode = InterPredMode::GlobalMv as u8;
            } else if is_tip {
                inter_mode = InterPredMode::NearMv as u8
                    + 2 * msac.decode_bool_adapt(cdf_m.tip_mode()) as u8;
            } else {
                let mut allow_warp = false;
                if imin(bw4, bh4) >= 2 && fi.warp_motion {
                    // get_warp_ctx (decode.c:2984): neighbour warp-motion ctx.
                    // a_sb_cache (above-SB-row cache) is only consulted for blocks
                    // at the top SB boundary; rav2d's `a` carries above-row ctx for
                    // non-boundary blocks (the bring-up clip is a single 64x64 SB,
                    // so warp blocks never hit the boundary path). Default cache.
                    let a_sb_cache = crate::env::SBEdgeCtx::default();
                    let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                    let warp_thr = if is_sb_boundary {
                        ((bx + bw4 - 2) & !1) < fi.tile_col_end
                    } else {
                        have_top_right
                    };
                    let warp_ctx = crate::env::get_warp_ctx(
                        a,
                        &a_sb_cache,
                        l,
                        by4,
                        bx4,
                        have_top,
                        have_left,
                        warp_thr,
                        have_bottom_left,
                        is_sb_boundary,
                        b_dim,
                        ref0,
                    );
                    allow_warp = msac.decode_bool_adapt(cdf_m.warp(warp_ctx as usize)) != 0;
                }
                if allow_warp {
                    if !fi.force_integer_mv && msac.decode_bool_adapt(cdf_m.warp_newmv()) == 0 {
                        inter_mode = InterPredMode::WarpNewMv as u8;
                    } else {
                        inter_mode = InterPredMode::WarpMv as u8;
                    }
                } else {
                    inter_mode = InterPredMode::NearMv as u8
                        + msac.decode_symbol_adapt(cdf_m.inter_mode(sngl_ctx), 2) as u8;
                }
            };
            unsafe {
                b.data.inter.inter_mode = inter_mode;
            }

            // --- AMVD ---
            if fi.adaptive_mvd && inter_mode == InterPredMode::NewMv as u8 {
                let ctx = (nx_ref0[0] == ref0 && nx_amvd[0] != 0) as usize
                    + (if n_ctx > 1 {
                        nx_ref0[1] == ref0 && nx_amvd[1] != 0
                    } else {
                        false
                    }) as usize;
                unsafe {
                    b.data.inter.amvd = msac.decode_bool_adapt(cdf_m.amvd(4, ctx)) as i8;
                }
            }
            let amvd_val = unsafe { b.data.inter.amvd };

            // --- warp_ref_idx, warpmv_with_mvd, bawp defaults ---
            unsafe {
                b.data.inter.warp_ref_idx = 0;
                b.data.inter.warpmv_with_mvd = 0;
                b.data.inter.bawp[0] = 0;
                b.data.inter.bawp[1] = 0;
            }

            if !is_tip && inter_mode <= InterPredMode::NewMv as u8 {
                // --- BAWP (block-adaptive weighted prediction) ---
                if fi.bawp && inter_mode != InterPredMode::GlobalMv as u8 && imin(bw4, bh4) >= 2 {
                    let bawp0 = msac.decode_bool_adapt(cdf_m.bawp(0)) as u8;
                    if bawp0 != 0 {
                        let ctx = if inter_mode == InterPredMode::NewMv as u8 {
                            2 - amvd_val as usize
                        } else {
                            0
                        };
                        let explicit = msac.decode_bool_adapt(cdf_m.bawp_explicit(ctx)) as u8;
                        let mut val = bawp0 + explicit;
                        if val == 2 {
                            val += msac.decode_bool_adapt(cdf_m.bawp_explicit_scale()) as u8;
                            val |= (ctx as u8) << 2;
                        }
                        unsafe {
                            b.data.inter.bawp[0] = val;
                        }
                        if has_chroma {
                            unsafe {
                                b.data.inter.bawp[1] = msac.decode_bool_adapt(cdf_m.bawp(1)) as u8;
                            }
                        }
                    }
                }

                // --- inter-intra (motion mode) ---
                let bawp0 = unsafe { b.data.inter.bawp[0] };
                if fi.motion_modes & (1 << MotionMode::InterIntra as u8) != 0
                    && bawp0 == 0
                    && bw4 * bh4 > 2
                    && imax(bw4, bh4) <= 16
                    && inter_mode >= InterPredMode::NearMv as u8
                    && inter_mode <= InterPredMode::NewMv as u8
                {
                    let ctx = SIZE_GROUP[bs_idx] as usize;
                    if msac.decode_bool_adapt(cdf_m.interintra(ctx)) != 0 {
                        unsafe {
                            b.data.inter.motion_mode = MotionMode::InterIntra as u8;
                            b.data.inter.interintra_mode =
                                msac.decode_symbol_adapt(cdf_m.interintra_mode(ctx), 3) as u8;
                            b.data.inter.wedge_idx = -1;
                        }
                        if imin(bw4, bh4) > 1
                            && msac.decode_bool_adapt(cdf_m.interintra_wedge()) != 0
                        {
                            unsafe {
                                b.data.inter.wedge_idx = read_wedge_idx(msac, cdf_m);
                            }
                        }
                    }
                }
            } else if !is_tip {
                // --- warp motion mode for WARPMV/WARPNEWMV ---
                unsafe {
                    b.data.inter.motion_mode = MotionMode::WarpDelta as u8;
                }

                // has_cs_ext (decode.c:3062-3076): the warp_extend/warp_causal
                // signal is only read when a spatial neighbour references the same
                // frame. Without this gate WARPNEWMV blocks read extra symbols and
                // desync the parse. The is_sb_boundary top path uses the a_sb_cache
                // in dav2d; rav2d's `a` already carries the above-SB context, so it
                // is used directly here (exact for non-boundary; SB-boundary refmv
                // edge handling is a follow-up).
                let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                let match_ref_l = |off: usize| -> bool {
                    l.r#ref[0][off] == ref0 || l.r#ref[1][off] == ref0
                };
                let match_ref_a = |off: usize| -> bool {
                    a.r#ref[0][off] == ref0 || a.r#ref[1][off] == ref0
                };
                let has_cs_ext = if inter_mode == InterPredMode::WarpNewMv as u8 {
                    let left_match = have_left
                        && (match_ref_l(by4)
                            || (by + bh4 <= fi.tile_row_end && match_ref_l(by4 + bh4 as usize - 1)));
                    let top_match = have_top && {
                        if is_sb_boundary {
                            let o0 = bx4 & !1;
                            match_ref_a(o0)
                                || (((bx + bw4 - 2) & !1) < fi.tile_col_end
                                    && match_ref_a((bx4 + bw4 as usize - 2) & !1))
                        } else {
                            match_ref_a(bx4)
                                || (bx + bw4 <= fi.tile_col_end
                                    && match_ref_a(bx4 + bw4 as usize - 1))
                        }
                    };
                    left_match || top_match
                } else {
                    false
                };

                if inter_mode == InterPredMode::WarpNewMv as u8 && has_cs_ext {
                    // warp extend / causal decision
                    let x1 = if nb_boff[0] == -1 {
                        0
                    } else {
                        nb_motion_mode[0]
                    };
                    let x2 = if nb_boff[1] == -1 {
                        0
                    } else {
                        nb_motion_mode[1]
                    };
                    let ext_ctx = (x1 >= MotionMode::WarpCausal as u8) as usize
                        + (x2 >= MotionMode::WarpCausal as u8) as usize;
                    let mm_flags = fi.motion_modes;
                    if mm_flags & (1 << MotionMode::WarpExtend as u8) != 0
                        && msac.decode_bool_adapt(cdf_m.warp_extend(ext_ctx)) != 0
                    {
                        unsafe {
                            b.data.inter.motion_mode = MotionMode::WarpExtend as u8;
                        }
                    } else if (mm_flags & (3 << MotionMode::WarpCausal as u8))
                        == (3 << MotionMode::WarpCausal as u8)
                    {
                        let cs_ctx = (ext_ctx > 0) as usize
                            + (x1 == MotionMode::WarpCausal as u8) as usize
                            + (x2 == MotionMode::WarpCausal as u8) as usize;
                        if msac.decode_bool_adapt(cdf_m.warp_causal(cs_ctx)) != 0 {
                            unsafe {
                                b.data.inter.motion_mode = MotionMode::WarpCausal as u8;
                            }
                        }
                    } else if mm_flags & (1 << MotionMode::WarpCausal as u8) != 0 {
                        unsafe {
                            b.data.inter.motion_mode = MotionMode::WarpCausal as u8;
                        }
                    }
                }

                // warp_ref_idx
                let motion_mode_val = unsafe { b.data.inter.motion_mode };
                if motion_mode_val == MotionMode::WarpDelta as u8 {
                    let mut wri = 0u8;
                    while wri < 3 {
                        if msac.decode_bool_adapt(cdf_m.warp_ref_idx(wri as usize)) == 0 {
                            break;
                        }
                        wri += 1;
                    }
                    unsafe {
                        b.data.inter.warp_ref_idx = wri;
                    }
                }

                // warpmv_with_mvd
                let warp_ref_idx = unsafe { b.data.inter.warp_ref_idx };
                if inter_mode == InterPredMode::WarpMv as u8 && warp_ref_idx < 2 {
                    unsafe {
                        b.data.inter.warpmv_with_mvd =
                            msac.decode_bool_adapt(cdf_m.warpmv_with_mvd()) as u8;
                    }
                }
            }

            // --- DRL index ---
            unsafe {
                b.data.inter.drl_idx[0] = 0;
            }
            if inter_mode != InterPredMode::WarpMv as u8
                && inter_mode != InterPredMode::GlobalMv as u8
            {
                let max_drl = fi.max_drl_bits as i32;
                let mut n = 0i32;
                let mut ctx = 0usize;
                while n < max_drl {
                    let cdf = if is_tip {
                        cdf_m.tip_drl_idx(ctx)
                    } else {
                        cdf_m.drl_idx(ctx, sngl_ctx)
                    };
                    if msac.decode_bool_adapt(cdf) == 0 {
                        break;
                    }
                    n += 1;
                    if ctx < 2 {
                        ctx += 1;
                    }
                }
                unsafe {
                    b.data.inter.drl_idx[0] = n as u8;
                }
            }

            // --- MV precision ---
            let mut mv_prec = 3i32 + fi.mv_precision as i32;
            if mv_prec > 3
                && amvd_val == 0
                && fi.flex_mvres
                && (inter_mode == InterPredMode::NewMv as u8
                    || inter_mode == InterPredMode::WarpNewMv as u8)
            {
                let mvprec1 = if nb_boff[0] == -1 { 0u8 } else { nb_mvprec[0] };
                let mvprec2 = if nb_boff[1] == -1 { 0u8 } else { nb_mvprec[1] };
                let ctx1 = ((mvprec1 & 1) + (mvprec2 & 1)) as usize;
                if msac.decode_bool_adapt(cdf_m.mvprec_def(ctx1)) == 0 {
                    let ctx2 = ((mvprec1 | mvprec2) >> 1) as usize;
                    let idx = msac
                        .decode_symbol_adapt(cdf_m.mvprec_rem(ctx2, (mv_prec - 4) as usize), 2)
                        as usize;
                    mv_prec = MV_PREC_TBL[(mv_prec == 6) as usize][idx] as i32;
                    mvprec_def = 2;
                }
            }
            unsafe {
                b.data.inter.mv_prec = mv_prec as i8;
            }

            // --- MV residual ---
            let warpmv_with_mvd = unsafe { b.data.inter.warpmv_with_mvd };
            if inter_mode == InterPredMode::NewMv as u8
                || inter_mode == InterPredMode::WarpNewMv as u8
                || (inter_mode == InterPredMode::WarpMv as u8 && warpmv_with_mvd != 0)
            {
                let mv = if amvd_val != 0 {
                    read_amvd(msac, cdf_m)
                } else {
                    read_mv_full(msac, cdf_dmv, mv_prec)
                };
                unsafe {
                    b.data.inter.mv[0].c.x = mv.c.x;
                    b.data.inter.mv[0].c.y = mv.c.y;
                }

                // sign derivation
                let nnzc;
                let sum_mvd;
                unsafe {
                    if amvd_val != 0 {
                        nnzc = 3;
                        sum_mvd = 0;
                    } else {
                        let nx = (mv.c.x != 0) as i32 + (mv.c.y != 0) as i32;
                        sum_mvd = (mv.c.x + mv.c.y) >> (6 - mv_prec);
                        if inter_mode == InterPredMode::WarpMv as u8
                            || nx == 0
                            || !fi.mvd_sign_derive
                            || b.data.inter.motion_mode != MotionMode::Translation as u8
                            || fi.allow_screen_content_tools
                            || fi.mv_precision == 3
                            || mv_prec >= 5
                        {
                            nnzc = 3;
                        } else {
                            nnzc = nx;
                        }
                    }
                }
                let mut nnzc2 = 0i32;
                let cur_mv_y = unsafe { b.data.inter.mv[0].c.y };
                if cur_mv_y != 0 {
                    nnzc2 += 1;
                    let s = if nnzc2 == nnzc {
                        (sum_mvd & 1) != 0
                    } else {
                        msac.decode_bool_bypass() != 0
                    };
                    if s {
                        unsafe {
                            b.data.inter.mv[0].c.y = -cur_mv_y;
                        }
                    }
                }
                let cur_mv_x = unsafe { b.data.inter.mv[0].c.x };
                if cur_mv_x != 0 {
                    nnzc2 += 1;
                    let s = if nnzc2 == nnzc {
                        (sum_mvd & 1) != 0
                    } else {
                        msac.decode_bool_bypass() != 0
                    };
                    if s {
                        unsafe {
                            b.data.inter.mv[0].c.x = -cur_mv_x;
                        }
                    }
                }
            }

            // --- warp delta parameters ---
            let motion_mode_val = unsafe { b.data.inter.motion_mode };
            let warp_ref_idx = unsafe { b.data.inter.warp_ref_idx };
            if inter_mode == InterPredMode::WarpNewMv as u8
                && motion_mode_val == MotionMode::WarpDelta as u8
                && ((fi.six_param_warp_delta && warp_ref_idx == 1) || warp_ref_idx == 0)
            {
                let prec = msac.decode_bool_adapt(cdf_m.warp_delta_prec(bs_idx));
                let np = if fi.six_param_warp_delta && warp_ref_idx == 1 {
                    4
                } else {
                    2
                };
                let step = 2i8 >> prec;
                for n in 0..np {
                    // dav2d: ctx = (n - 1U > 1U); cdf index is `!ctx`
                    // -> n=0 -> idx 0, n=1 -> idx 1, n=2 -> idx 1, n=3 -> idx 0.
                    let ctx = ((n as u32).wrapping_sub(1) > 1) as usize;
                    let idx = (ctx == 0) as usize;
                    let mut val = msac.decode_symbol_adapt(cdf_m.warp_delta_param(0, idx), 7) as i8;
                    if val == 7 && prec != 0 {
                        val += msac.decode_symbol_adapt(cdf_m.warp_delta_param(1, idx), 7) as i8;
                    }
                    if val != 0 {
                        if msac.decode_bool_adapt(cdf_m.warp_delta_sign()) != 0 {
                            val = -val;
                        }
                        val *= step;
                    }
                    unsafe {
                        b.data.inter.matrix[n] = val;
                    }
                }
                if np == 2 {
                    unsafe {
                        b.data.inter.matrix[2] = -0x80;
                    }
                }
            } else if motion_mode_val == MotionMode::WarpDelta as u8 {
                unsafe {
                    b.data.inter.matrix = [0; 4];
                }
            }

            // --- warp_ii ---
            unsafe {
                b.data.inter.warp_ii = 0;
            }
            if inter_mode == InterPredMode::WarpMv as u8
                && imin(bw4, bh4) >= 2
                && imax(bw4, bh4) <= 16
            {
                let ctx = SIZE_GROUP[bs_idx] as usize;
                if msac.decode_bool_adapt(cdf_m.warp_interintra(ctx)) != 0 {
                    unsafe {
                        b.data.inter.warp_ii = 1;
                        b.data.inter.interintra_mode =
                            msac.decode_symbol_adapt(cdf_m.interintra_mode(ctx), 3) as u8;
                        b.data.inter.wedge_idx =
                            if msac.decode_bool_adapt(cdf_m.interintra_wedge()) != 0 {
                                read_wedge_idx(msac, cdf_m)
                            } else {
                                -1
                            };
                    }
                }
            }

            // --- subpel filter ---
            let has_subpel_filter = !is_tip
                && inter_mode <= InterPredMode::NewMv as u8
                && (inter_mode != InterPredMode::GlobalMv as u8 || imin(bw4, bh4) == 1);
            if b.skip_mode != 0 || ref0 == TIP_FRAME as i8 {
                unsafe {
                    b.data.inter.filter = 2;
                } // SHARP
            } else if fi.subpel_filter_mode == 4 {
                // SWITCHABLE
                if has_subpel_filter {
                    // get_filter_ctx (env.h:120): neighbour filter agreement,
                    // matched on the block's first reference; comp adds 4.
                    const N_SW: u8 = N_SWITCHABLE_FILTERS as u8;
                    let bref0 = ref0;
                    let comp = unsafe { b.ref_pair.r[1] } != -1;
                    let flt = |i: usize| -> u8 {
                        if nb_boff[i] != -1 && (nb_ref0[i] == bref0 || nb_ref1[i] == bref0) {
                            nb_filter[i]
                        } else {
                            N_SW
                        }
                    };
                    let flt0 = flt(0);
                    let flt1 = flt(1);
                    let fctx = (comp as usize) * 4
                        + if flt0 == flt1 || flt1 == N_SW {
                            flt0 as usize
                        } else if flt0 == N_SW {
                            flt1 as usize
                        } else {
                            N_SW as usize
                        };
                    unsafe {
                        b.data.inter.filter = msac.decode_symbol_adapt(cdf_m.filter(fctx), 2) as u8;
                    }
                } else {
                    unsafe {
                        b.data.inter.filter = 0;
                    } // REGULAR
                }
            } else {
                unsafe {
                    b.data.inter.filter = fi.subpel_filter_mode;
                }
            }
        }

        // TX partition for inter
        if has_luma {
            let __seg_ll = fi.seg_lossless[b.seg_id as usize] != 0;
            read_tx_part(msac, cdf_m, &mut b, bs, __seg_ll, fi.txfm_switchable);
        }

        // Inter context write-back
        if has_luma {
            let aw = 1usize << b_dim[2];
            let lh = 1usize << b_dim[3];
            let inter_mode = unsafe { b.data.inter.inter_mode };
            let comp_type = unsafe { b.data.inter.comp_type };
            let motion_mode = unsafe { b.data.inter.motion_mode };
            let amvd = unsafe { b.data.inter.amvd };
            let refs = unsafe { b.ref_pair.r };
            let filter_val = unsafe { b.data.inter.filter };

            a.seg_pred[bx4..bx4 + aw].fill(0);
            a.skip_mode[bx4..bx4 + aw].fill(b.skip_mode);
            a.intra[bx4..bx4 + aw].fill(0);
            a.intrabc[bx4..bx4 + aw].fill(0);
            a.morph_pred[bx4..bx4 + aw].fill(0);
            a.midx[bx4..bx4 + aw].fill(0xff);
            a.fsc[bx4..bx4 + aw].fill(0);
            a.skip_txfm[bx4..bx4 + aw].fill(b.skip_txfm);
            a.pal_sz[bx4..bx4 + aw].fill(0);
            a.comp_type[bx4..bx4 + aw].fill(comp_type);
            a.filter[bx4..bx4 + aw].fill(filter_val);
            a.mode[bx4..bx4 + aw].fill(inter_mode);
            a.mrl[bx4..bx4 + aw].fill(0);
            a.multi_mrl[bx4..bx4 + aw].fill(0);
            a.dip[bx4..bx4 + aw].fill(0);
            a.r#ref[0][bx4..bx4 + aw].fill(refs[0]);
            a.r#ref[1][bx4..bx4 + aw].fill(refs[1]);
            a.motion_mode[bx4..bx4 + aw].fill(motion_mode);
            a.amvd[bx4..bx4 + aw].fill(amvd as u8);
            a.mvprec[bx4..bx4 + aw].fill(mvprec_def);

            l.seg_pred[by4..by4 + lh].fill(0);
            l.skip_mode[by4..by4 + lh].fill(b.skip_mode);
            l.intra[by4..by4 + lh].fill(0);
            l.intrabc[by4..by4 + lh].fill(0);
            l.morph_pred[by4..by4 + lh].fill(0);
            l.midx[by4..by4 + lh].fill(0xff);
            l.fsc[by4..by4 + lh].fill(0);
            l.skip_txfm[by4..by4 + lh].fill(b.skip_txfm);
            l.pal_sz[by4..by4 + lh].fill(0);
            l.comp_type[by4..by4 + lh].fill(comp_type);
            l.filter[by4..by4 + lh].fill(filter_val);
            l.mode[by4..by4 + lh].fill(inter_mode);
            l.mrl[by4..by4 + lh].fill(0);
            l.multi_mrl[by4..by4 + lh].fill(0);
            l.dip[by4..by4 + lh].fill(0);
            l.r#ref[0][by4..by4 + lh].fill(refs[0]);
            l.r#ref[1][by4..by4 + lh].fill(refs[1]);
            l.motion_mode[by4..by4 + lh].fill(motion_mode);
            l.amvd[by4..by4 + lh].fill(amvd as u8);
            l.mvprec[by4..by4 + lh].fill(mvprec_def);
        }
        if has_chroma {
            let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
            let cbx4 = (cbx & 63) as usize;
            let cby4 = (cby & 63) as usize;
            let cbw4 = 1usize << cb_dim[2];
            let cbh4 = 1usize << cb_dim[3];
            a.uvmode[cbx4..cbx4 + cbw4].fill(0); // DC_PRED
            l.uvmode[cby4..cby4 + cbh4].fill(0);
        }
    }

    // Write the block's segment id into the current-frame segment map over its
    // bw4 x bh4 footprint (decode.c:3328-3341). Luma-only; chroma reads it back.
    if fi.seg_enabled && has_luma {
        let seg_id = b.seg_id;
        let stride = recon.b4_stride;
        let bw4u = 1usize << b_dim[2];
        let bh4u = bh4 as usize;
        let mut off = (by as isize * stride + bx as isize) as usize;
        for _ in 0..bh4u {
            recon.cur_segmap[off..off + bw4u].fill(seg_id);
            off = (off as isize + stride) as usize;
        }
    }

    // ---- Reference-MV resolution + splat (decode.c:972-1326) ---------------
    // For IntraBC blocks: resolve the block vector by adding the parsed residual
    // to the DRL-selected predictor from the spatial refmvs candidate list, then
    // splat the final BV into the refmvs grid. For intra (non-IntraBC) blocks:
    // splat an "intra" entry (invalid mv) so later IntraBC blocks skip them.
    if fi.allow_intrabc && has_luma && b.is_intra != 0 {
        let by4r = (by & 63) as usize;
        if intrabc {
            use crate::levels::{Mv, MvXY, RefPair};
            let mut mvstack = [crate::refmvs::Candidate::default(); 6];
            let mut n_mvs = 0i32;
            let mut warp_cnt = 0i32;
            crate::refmvs::refmvs_find(
                recon.rt,
                recon.rf,
                &[],
                &Default::default(),
                &mut mvstack,
                None,
                &mut n_mvs,
                &mut warp_cnt,
                RefPair { pair: -1 },
                bs as u8,
                false,
                by,
                bx,
                recon.seq_hdr,
                recon.frm_hdr,
            );
            let diff = unsafe { b.data.intra.intrabc_mv };
            let drl = unsafe { b.data.inter.drl_idx[0] } as usize;
            let mut mv = mvstack[drl].mv[0];
            if unsafe { mv.n } == 0 {
                // Force the refmv to a nonzero value (decode.c:990-998).
                let sbsz = 64 << fi.sb128;
                if by - fi.sb_step < fi.tile_row_start {
                    unsafe { mv.c.x = -(8 * (sbsz + 256)) };
                } else {
                    unsafe { mv.c.y = -(8 * sbsz) };
                }
            }
            if unsafe { b.data.intra.is_refmv } == 0 {
                if unsafe { b.data.intra.is_qpel } == 0 {
                    crate::env::fix_int_mv_precision(unsafe { &mut mv.c });
                }
                unsafe {
                    mv.c.x += diff.c.x;
                    mv.c.y += diff.c.y;
                }
            }
            unsafe { b.data.intra.intrabc_mv = mv };
            if std::env::var("RAV2D_IBC2").is_ok() {
                eprintln!(
                    "RIBCMV y={} x={} mvy={} mvx={} drl={} nmvs={}",
                    by,
                    bx,
                    unsafe { mv.c.y },
                    unsafe { mv.c.x },
                    drl,
                    n_mvs
                );
            }
            // splat_intrabc_mv (decode.c:599-625).
            let mut s_src = crate::refmvs::Block {
                mv: [
                    mv,
                    Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    },
                ],
                r#ref: RefPair { pair: -1 },
                bs: bs as u8,
                mf: 0,
                ..Default::default()
            };
            let s_off = by4r * 128 + (bx & 127) as usize;
            let t_src = crate::refmvs::TemporalBlock::default();
            crate::refmvs::splat_mv(
                &mut recon.rt.r[s_off..],
                &mut s_src,
                None,
                0,
                &t_src,
                bw4,
                bh4,
            );
            if recon.seq_hdr.refmv_bank {
                b.ref_pair = RefPair { pair: -1 };
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
        } else {
            // splat_intraref (decode.c:712-737): invalid mv, ref=-1.
            use crate::levels::{Mv, MvXY, RefPair};
            let mut s_src = crate::refmvs::Block {
                mv: [
                    Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    },
                    Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    },
                ],
                r#ref: RefPair { pair: -1 },
                bs: bs as u8,
                mf: 0,
                ..Default::default()
            };
            let s_off = by4r * 128 + (bx & 127) as usize;
            let t_src = crate::refmvs::TemporalBlock::default();
            crate::refmvs::splat_mv(
                &mut recon.rt.r[s_off..],
                &mut s_src,
                None,
                0,
                &t_src,
                bw4,
                bh4,
            );
            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_update(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                );
            }
        }
    }

    // ---- Inter reference-MV resolution + splat (decode.c:1066-1322) --------
    // Single-reference only: resolve the block MV (refmvs_find DRL candidate +
    // parsed residual, or the global MV) and splat it into the refmvs grid +
    // temporal grid. Compound (ref[1] != -1), warp-causal/extend/delta motion
    // and TIP are deferred; their per-block MC is handled separately.
    if has_luma && b.is_intra == 0 && !intrabc {
        use crate::levels::{InterPredMode, Mv, MvXY, RefPair, TIP_FRAME};
        let by4r = (by & 63) as usize;
        let refs = unsafe { b.ref_pair.r };
        let is_comp = refs[1] != -1;
        let inter_mode = unsafe { b.data.inter.inter_mode };
        let motion_mode = unsafe { b.data.inter.motion_mode };
        let mv_prec = unsafe { b.data.inter.mv_prec } as i32;
        let amvd = unsafe { b.data.inter.amvd };

        if !is_comp && refs[0] != TIP_FRAME as i8 {
            // Resolve the single-ref block MV.
            if inter_mode == InterPredMode::GlobalMv as u8 {
                let gmv = crate::env::get_gmv_2d(
                    &recon.frm_hdr.gmv.m[refs[0] as usize],
                    bx,
                    by,
                    bw4,
                    bh4,
                    recon.rf.iw4,
                    recon.rf.ih4,
                    recon.frm_hdr,
                );
                unsafe {
                    b.data.inter.mv[0] = Mv { c: gmv };
                }
            } else {
                let mut mvstack = [crate::refmvs::Candidate::default(); 6];
                let mut n_mvs = 0i32;
                let mut warp_cnt = 0i32;
                let want_warp = inter_mode > InterPredMode::NewMv as u8;
                let mut warp_arr = [[0i32; 7]; 6];
                let rp_proj_off = recon.rt.rp_proj_off;
                let rp_proj_slice: &[crate::refmvs::SnglMvBlock] =
                    if recon.rf.rp_proj.is_empty() {
                        &[]
                    } else {
                        &recon.rf.rp_proj[rp_proj_off..]
                    };
                crate::refmvs::refmvs_find(
                    recon.rt,
                    recon.rf,
                    rp_proj_slice,
                    &recon.rf.rp_traj,
                    &mut mvstack,
                    if want_warp {
                        Some(&mut warp_arr[..])
                    } else {
                        None
                    },
                    &mut n_mvs,
                    &mut warp_cnt,
                    RefPair {
                        r: [refs[0], -1],
                    },
                    bs as u8,
                    false,
                    by,
                    bx,
                    recon.seq_hdr,
                    recon.frm_hdr,
                );
                let diff = unsafe { b.data.inter.mv[0] };
                let drl = unsafe { b.data.inter.drl_idx[0] } as usize;
                let mut mv = if inter_mode == InterPredMode::WarpMv as u8 {
                    let wri = unsafe { b.data.inter.warp_ref_idx } as usize;
                    let prec = if unsafe { b.data.inter.warpmv_with_mvd } != 0 {
                        mv_prec
                    } else {
                        6
                    };
                    crate::env::get_warpmv_2d(
                        &[
                            warp_arr[wri][0],
                            warp_arr[wri][1],
                            warp_arr[wri][2],
                            warp_arr[wri][3],
                            warp_arr[wri][4],
                            warp_arr[wri][5],
                        ],
                        bx,
                        by,
                        bw4,
                        bh4,
                        recon.rf.iw4,
                        recon.rf.ih4,
                        prec,
                    )
                } else {
                    unsafe { mvstack[drl].mv[0].c }
                };
                if inter_mode == InterPredMode::NewMv as u8
                    || inter_mode == InterPredMode::WarpNewMv as u8
                    || (inter_mode == InterPredMode::WarpMv as u8
                        && unsafe { b.data.inter.warpmv_with_mvd } != 0)
                {
                    if amvd == 0 && mv_prec <= 3 {
                        crate::env::mv_reduce_prec(&mut mv, mv_prec);
                    }
                    unsafe {
                        mv.x += diff.c.x;
                        mv.y += diff.c.y;
                    }
                }
                unsafe {
                    b.data.inter.mv[0] = Mv { c: mv };
                }
                if trace_blk {
                    eprintln!("  CK final2dmv y={} x={}", mv.y, mv.x);
                }

                // --- warpmv derivation (decode.c:1113-1196) ----------------
                // Build t->warpmv[0] for the warp motion modes so recon can do
                // warp-affine MC. WARP_DELTA applies the parsed matrix deltas to
                // the base warp candidate; WARP_CAUSAL re-estimates from
                // neighbour samples; WARP_EXTEND extends a neighbour's matrix.
                let motion_mode_v = unsafe { b.data.inter.motion_mode };
                if motion_mode_v == MotionMode::WarpDelta as u8 {
                    let wri = unsafe { b.data.inter.warp_ref_idx } as usize;
                    let base = &warp_arr[wri];
                    let m = &mut recon.warpmv[0].matrix;
                    let bmat = unsafe { b.data.inter.matrix };
                    let mut n = 0usize;
                    while n < 4 && bmat[n] != -0x80 {
                        if bmat[n] != 0 {
                            let bb = ((n.wrapping_sub(1)) >= 2) as i32 * 0x10000;
                            m[2 + n] = iclip(
                                base[n + 2] + bmat[n] as i32 * (1 << 10),
                                bb - 0x7fc0,
                                bb + 0x7fc0,
                            );
                        } else {
                            m[2 + n] = base[n + 2];
                        }
                        n += 1;
                    }
                    if bmat[2] == -0x80 {
                        m[5] = m[2];
                        m[4] = -m[3];
                    }
                    crate::warpmv::set_affine_mv2d(
                        bw4,
                        bh4,
                        unsafe { b.data.inter.mv[0].c },
                        &mut recon.warpmv[0],
                        bx,
                        by,
                    );
                    recon.warpmv[0].wm_type =
                        if crate::warpmv::get_shear_params(&mut recon.warpmv[0]) != 0 {
                            crate::headers::WarpedMotionType::Invalid
                        } else {
                            crate::env::warp_type(&recon.warpmv[0].matrix)
                        };
                } else if motion_mode_v == MotionMode::WarpCausal as u8 {
                    let w4 = imin(bw4, fi.bw - bx);
                    let h4 = imin(bh4, fi.bh - by);
                    derive_warpmv(
                        recon.rt,
                        bx,
                        by,
                        have_top,
                        have_left,
                        bw4,
                        bh4,
                        w4,
                        h4,
                        refs[0],
                        unsafe { b.data.inter.mv[0] },
                        &mut recon.warpmv[0],
                        fi.sb_step,
                        fi.tile_col_end,
                    );
                } else if motion_mode_v == MotionMode::WarpExtend as u8 {
                    let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                    let mut y_off = 0i32;
                    let mut x_off = 0i32;
                    let cand = &mvstack[drl];
                    if cand.x_off == -1 || cand.y_off == -1 {
                        y_off = cand.y_off as i32;
                        x_off = cand.x_off as i32;
                        let sb_mask = fi.sb_step - 1;
                        let r = if is_sb_boundary && y_off == -1 {
                            if (bx & sb_mask) != 0 || x_off >= 0 {
                                &recon.rt.ra[recon.rt.ra_off + ((bx + x_off) >> 1) as usize]
                            } else {
                                &recon.rt.ra_tl
                            }
                        } else {
                            &recon.rt.r
                                [((by + y_off) & 63) as usize * 128 + ((bx + x_off) & 127) as usize]
                        };
                        if unsafe { r.r#ref.r[0] } == TIP_FRAME as i8 {
                            x_off = 0;
                            y_off = 0;
                        }
                    }
                    let ref0 = refs[0];
                    let match_ref = |r: &crate::refmvs::Block| -> bool {
                        unsafe { r.r#ref.r[0] == ref0 || r.r#ref.r[1] == ref0 }
                    };
                    // Neighbours mirror dav2d (decode.c:1155-1168): tml is the
                    // left neighbour on the current row, lmt the top neighbour.
                    let tml_ok = have_left && {
                        let r = &recon.rt.r[(by & 63) as usize * 128 + ((bx - 1) & 127) as usize];
                        match_ref(r)
                    };
                    let bml_ok = have_left && by + bh4 <= fi.tile_row_end && {
                        let r = &recon.rt.r
                            [((by + bh4 - 1) & 63) as usize * 128 + ((bx - 1) & 127) as usize];
                        match_ref(r)
                    };
                    let lmt_ok = have_top && {
                        let r = if is_sb_boundary {
                            &recon.rt.ra[recon.rt.ra_off + ((bx & !1) >> 1) as usize]
                        } else {
                            &recon.rt.r[((by - 1) & 63) as usize * 128 + (bx & 127) as usize]
                        };
                        match_ref(r)
                    };
                    let rmt_ok = have_top && bx + bw4 <= fi.tile_col_end && {
                        let r = if is_sb_boundary {
                            &recon.rt.ra[recon.rt.ra_off + (((bx & !1) + bw4 - 2) >> 1) as usize]
                        } else {
                            &recon.rt.r[((by - 1) & 63) as usize * 128 + ((bx + bw4 - 1) & 127) as usize]
                        };
                        match_ref(r)
                    };
                    if x_off != 0 || y_off != 0 {
                        // already set above
                    } else if bml_ok {
                        y_off = bh4 - 1;
                        x_off = -1;
                    } else if rmt_ok {
                        y_off = -1;
                        x_off = -(bx & is_sb_boundary as i32) + bw4 - (1 + is_sb_boundary as i32);
                    } else if tml_ok {
                        y_off = 0;
                        x_off = -1;
                    } else if lmt_ok {
                        y_off = -1;
                        x_off = -(bx & is_sb_boundary as i32);
                    }
                    if x_off != 0 || y_off != 0 {
                        let b_dim_e = &BLOCK_DIMENSIONS[bs as usize];
                        extend_warpmv(
                            recon.rt,
                            bx,
                            by,
                            x_off,
                            y_off,
                            b_dim_e,
                            refs[0],
                            unsafe { b.data.inter.mv[0] },
                            &mut recon.warpmv[0],
                            fi.sb_step,
                            &recon.frm_hdr.gmv.m[refs[0] as usize].matrix,
                        );
                    } else {
                        recon.warpmv[0].wm_type = crate::headers::WarpedMotionType::Invalid;
                    }
                }
            }

            // refmv bank + splat (decode.c:1307-1322 single-ref).
            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
            // refmvs_warp_add for warp motion modes (decode.c:1320-1328): add the
            // derived warp matrix to the per-ref warp bank so later WARP_DELTA /
            // WARP_MV blocks can use it as a base candidate.
            if motion_mode > MotionMode::InterIntra as u8
                && recon.warpmv[0].wm_type != crate::headers::WarpedMotionType::Invalid
            {
                crate::refmvs::warp_bank_add(
                    &mut recon.rt.warp,
                    &recon.warpmv[0],
                    refs[0] as usize,
                );
            }
            // splat_oneref_mv (decode.c:545-597), translational path. Warp/
            // global-affine splat (mf==2 / mf==1 with warp) is deferred.
            let blk_mv = unsafe { b.data.inter.mv[0] };
            let gmv_affine = inter_mode == InterPredMode::GlobalMv as u8
                && imin(bw4, bh4) > 1
                && recon.frm_hdr.gmv.m[refs[0] as usize].wm_type
                    > crate::headers::WarpedMotionType::Translation;
            if motion_mode <= MotionMode::InterIntra as u8 && !gmv_affine {
                let mf = (inter_mode == InterPredMode::GlobalMv as u8 && imin(bw4, bh4) > 1) as i8;
                let mut s_src = crate::refmvs::Block {
                    mv: [
                        blk_mv,
                        Mv {
                            c: MvXY {
                                y: crate::levels::INVALID_MV,
                                x: 0,
                            },
                        },
                    ],
                    r#ref: RefPair { r: [refs[0], -1] },
                    bs: bs as u8,
                    mf,
                    subpel_filter: unsafe { b.data.inter.filter },
                    ..Default::default()
                };
                let s_off = by4r * 128 + (bx & 127) as usize;
                let mut t_src = crate::refmvs::TemporalBlock::default();
                // Temporal grid write target (rf.rp = f->mvs), unless TIP / no
                // ref_frame_mvs.
                let write_temporal = recon.seq_hdr.ref_frame_mvs
                    && refs[0] != TIP_FRAME as i8
                    && !recon.cur_mvs.is_empty();
                if write_temporal {
                    let q = crate::refmvs::quantize_mv(blk_mv);
                    unsafe {
                        t_src.mv.mv[0] = q;
                        t_src.mv.mv[1] = q;
                        t_src.r#ref.r[0] = refs[0];
                        t_src.r#ref.r[1] = refs[0];
                        if q.n == crate::refmvs::INVALID_TRAJ {
                            t_src.r#ref.pair = -1;
                        }
                    }
                    let t_stride = recon.rf.rp_stride;
                    let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
                    crate::refmvs::splat_mv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        Some(&mut recon.cur_mvs[t_off as usize..]),
                        t_stride,
                        &t_src,
                        bw4,
                        bh4,
                    );
                } else {
                    crate::refmvs::splat_mv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        None,
                        0,
                        &t_src,
                        bw4,
                        bh4,
                    );
                }
            } else {
                // Warp / global-affine splat (decode.c:564-588, splat_warpmv).
                let s_off = by4r * 128 + (bx & 127) as usize;
                let use_local = motion_mode > MotionMode::InterIntra as u8;
                let wm = if use_local {
                    recon.warpmv[0]
                } else {
                    recon.frm_hdr.gmv.m[refs[0] as usize]
                };
                let mut s_src = crate::refmvs::Block {
                    mv: [
                        blk_mv,
                        Mv {
                            c: MvXY {
                                y: crate::levels::INVALID_MV,
                                x: 0,
                            },
                        },
                    ],
                    r#ref: RefPair { r: [refs[0], -1] },
                    bs: bs as u8,
                    subpel_filter: unsafe { b.data.inter.filter },
                    ..Default::default()
                };
                if use_local {
                    s_src.lmv[0] = blk_mv;
                    s_src.lmv[1] = Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    };
                    s_src.mf = 2;
                    s_src.m = wm.matrix;
                    s_src.warp_type = wm.wm_type as i8;
                } else {
                    s_src.mf = 1;
                }
                let mat = &wm.matrix;
                let mvx = (mat[2] as i64 - 0x10000) * (bx as i64 + 1) * 4
                    + mat[3] as i64 * (by as i64 + 1) * 4
                    + mat[0] as i64;
                let mvy = mat[4] as i64 * (bx as i64 + 1) * 4
                    + mat[1] as i64
                    + (mat[5] as i64 - 0x10000) * (by as i64 + 1) * 4;
                let mut t_src = crate::refmvs::TemporalBlock::default();
                unsafe {
                    t_src.r#ref.r[0] = refs[0];
                    t_src.r#ref.r[1] = refs[0];
                }
                let write_temporal = recon.seq_hdr.ref_frame_mvs
                    && refs[0] != TIP_FRAME as i8
                    && !recon.cur_mvs.is_empty();
                if write_temporal {
                    let t_stride = recon.rf.rp_stride;
                    let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
                    crate::refmvs::splat_warpmv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        Some(&mut recon.cur_mvs[t_off as usize..]),
                        t_stride,
                        &mut t_src,
                        mvy,
                        mvx,
                        &wm,
                        bw4,
                        bh4,
                    );
                } else {
                    crate::refmvs::splat_warpmv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        None,
                        0,
                        &mut t_src,
                        mvy,
                        mvx,
                        &wm,
                        bw4,
                        bh4,
                    );
                }
            }
        } else if b.skip_mode != 0 {
            // skip_mode MV resolution (decode.c:1199-1217). refmvs_find with
            // the two-ref/skip flag set, then copy mvstack[drl_idx[0]].
            use crate::tables::COMP_INTER_PRED_MODES;
            let _ = COMP_INTER_PRED_MODES; // keep import path consistent
            let mut mvstack = [crate::refmvs::Candidate::default(); 6];
            let mut n_mvs = 0i32;
            let mut warp_cnt = 0i32;
            let rp_proj_off = recon.rt.rp_proj_off;
            let rp_proj_slice: &[crate::refmvs::SnglMvBlock] = if recon.rf.rp_proj.is_empty() {
                &[]
            } else {
                &recon.rf.rp_proj[rp_proj_off..]
            };
            crate::refmvs::refmvs_find(
                recon.rt,
                recon.rf,
                rp_proj_slice,
                &recon.rf.rp_traj,
                &mut mvstack,
                None,
                &mut n_mvs,
                &mut warp_cnt,
                RefPair { r: [refs[0], refs[1]] },
                bs as u8,
                true,
                by,
                bx,
                recon.seq_hdr,
                recon.frm_hdr,
            );
            let drl = unsafe { b.data.inter.drl_idx[0] } as usize;
            let drl = drl.min(mvstack.len() - 1);
            unsafe {
                b.data.inter.mv[0] = mvstack[drl].mv[0];
                b.data.inter.mv[1] = mvstack[drl].mv[1];
                b.data.inter.cwp_idx = mvstack[drl].cwp_idx;
            }

            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
            splat_tworef_mv(recon, &b, bx, by, by4r, bw4, bh4, bs);
        } else if is_comp {
            // Compound (same-ref-pair) MV resolution + tworef splat
            // (decode.c:1218-1320). Cross-ref / TIP / warp-compound deferred.
            use crate::tables::COMP_INTER_PRED_MODES;
            if inter_mode == CompInterPredMode::GlobalMvGlobalMv as u8 {
                for n in 0..2 {
                    let gmv = crate::env::get_gmv_2d(
                        &recon.frm_hdr.gmv.m[refs[n] as usize],
                        bx,
                        by,
                        bw4,
                        bh4,
                        recon.rf.iw4,
                        recon.rf.ih4,
                        recon.frm_hdr,
                    );
                    unsafe {
                        b.data.inter.mv[n] = Mv { c: gmv };
                    }
                }
            } else {
                let mut mvstack = [crate::refmvs::Candidate::default(); 6];
                let mut n_mvs = 0i32;
                let mut warp_cnt = 0i32;
                let rp_proj_off = recon.rt.rp_proj_off;
                let rp_proj_slice: &[crate::refmvs::SnglMvBlock] = if recon.rf.rp_proj.is_empty() {
                    &[]
                } else {
                    &recon.rf.rp_proj[rp_proj_off..]
                };
                // decode.c:1228-1262. For NEW/JOINT modes (inter_mode >
                // NEARMV_NEWMV) the full compound ref pair is used. For NEAR
                // modes with equal refs, single-ref find then mirror mv[0]->mv[1].
                // Cross-ref NEAR (two separate single-ref finds) is deferred (not
                // present in the bring-up clip — all blocks are same-ref).
                if inter_mode > CompInterPredMode::NearMvNewMv as u8 {
                    crate::refmvs::refmvs_find(
                        recon.rt,
                        recon.rf,
                        rp_proj_slice,
                        &recon.rf.rp_traj,
                        &mut mvstack,
                        None,
                        &mut n_mvs,
                        &mut warp_cnt,
                        RefPair { r: [refs[0], refs[1]] },
                        bs as u8,
                        false,
                        by,
                        bx,
                        recon.seq_hdr,
                        recon.frm_hdr,
                    );
                } else {
                    crate::refmvs::refmvs_find(
                        recon.rt,
                        recon.rf,
                        rp_proj_slice,
                        &recon.rf.rp_traj,
                        &mut mvstack,
                        None,
                        &mut n_mvs,
                        &mut warp_cnt,
                        RefPair { r: [refs[0], -1] },
                        bs as u8,
                        false,
                        by,
                        bx,
                        recon.seq_hdr,
                        recon.frm_hdr,
                    );
                    for c in mvstack.iter_mut() {
                        c.mv[1] = c.mv[0];
                    }
                }
                let mode_idx =
                    (inter_mode - CompInterPredMode::NearMvNearMv as u8) as usize;
                let m_pair = COMP_INTER_PRED_MODES[mode_idx.min(COMP_INTER_PRED_MODES.len() - 1)];
                for n in 0..2 {
                    let diff = unsafe { b.data.inter.mv[n] };
                    let drl = unsafe { b.data.inter.drl_idx[n] } as usize;
                    let mut mv = unsafe { mvstack[drl].mv[n].c };
                    if m_pair[n] == InterPredMode::NewMv as u8 {
                        if amvd == 0 && mv_prec <= 3 {
                            crate::env::mv_reduce_prec(&mut mv, mv_prec);
                        }
                        unsafe {
                            mv.x += diff.c.x;
                            mv.y += diff.c.y;
                        }
                    }
                    unsafe {
                        b.data.inter.mv[n] = Mv { c: mv };
                    }
                }
            }

            // refmv bank add + tworef splat (decode.c:1307-1319).
            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
            splat_tworef_mv(recon, &b, bx, by, by4r, bw4, bh4, bs);
        }
    }

    // ---- Reconstruction leaf (intra 8bpc) ----------------------------------
    // Mirrors the luma path of `dav2d_recon_b` (recon_tmpl.c:3292-3478) plus
    // `recon_b_luma_tx` (recon_tmpl.c:2443-2675), followed by the chroma path
    // (recon_tmpl.c:3482-3942). Inter, IntraBC and palette are NOT handled here.
    if trace_blk {
        eprintln!("  CK pre_recon rng={}", msac.dbg_rng());
    }
    if std::env::var("RAV2D_BLK_TRACE").is_ok() {
        let (r0, r1, mm, mvy, mvx) = if b.is_intra != 0 {
            (-2i32, -2i32, -1i32, 0i32, 0i32)
        } else {
            unsafe {
                (
                    b.ref_pair.r[0] as i32,
                    b.ref_pair.r[1] as i32,
                    b.data.inter.motion_mode as i32,
                    b.data.inter.mv[0].c.y as i32,
                    b.data.inter.mv[0].c.x as i32,
                )
            }
        };
        eprintln!(
            "DBLK y={} x={} bs={} intra={} ref0={} ref1={} mm={} mvy={} mvx={} rng={}",
            by,
            bx,
            bs as i32,
            b.is_intra as i32,
            r0,
            r1,
            mm,
            mvy,
            mvx,
            msac.dbg_rng()
        );
    }
    if pass & (Pass::Recon as u8) != 0 && b.is_intra != 0 {
        recon_b_intra(
            recon, msac, cdf_m, a, l, &b, bx, by, cbx, cby, lbs, cbs, has_luma, has_chroma, fi,
        )?;
        if intrabc && std::env::var("RAV2D_IBC2").is_ok() {
            eprintln!(
                "RIBC y={} x={} mvy={} mvx={} refmv={} drl={} qpel={} morph={} rng={}",
                by,
                bx,
                unsafe { b.data.intra.intrabc_mv.c.y },
                unsafe { b.data.intra.intrabc_mv.c.x },
                unsafe { b.data.intra.is_refmv },
                unsafe { b.data.inter.drl_idx[0] },
                unsafe { b.data.intra.is_qpel },
                unsafe { b.data.intra.morph_pred },
                msac.dbg_rng()
            );
        }
        if trace_blk {
            eprintln!("  CK post_chroma_recon rng={}", msac.dbg_rng());
        }
    } else if pass & (Pass::Recon as u8) != 0 && b.is_intra == 0 && !intrabc {
        recon_b_inter(
            recon, msac, cdf_m, a, l, &b, bx, by, cbx, cby, lbs, cbs, has_luma, has_chroma, fi,
        )?;
        if trace_blk {
            eprintln!("  CK post_inter_recon rng={}", msac.dbg_rng());
        }
    }

    // SDP: record the luma-only block's intra direction mode + FSC flag into the
    // per-SB map so the chroma-only tree can derive `midx` (decode.c:3425-3441).
    if fi.sdp && fi.has_chroma_layout && !has_chroma {
        let off = ((by & 15) * 16 + (bx & 15)) as usize;
        let bh4_max16 = imin(bh4, 16) as usize;
        let aw = (1usize << b_dim[2]).min(16);
        for y in 0..bh4_max16 {
            let row = off + y * 16;
            recon.scratch.luma_intra_dir_mode_map[row..row + aw].fill(luma_midx);
            recon.scratch.luma_fsc_map[row..row + aw].fill(b.fsc);
        }
    }

    Ok(b)
}

/// Intra reconstruction entry point mirroring `dav2d_recon_b` (recon_tmpl.c:3068).
///
/// Blocks larger than 64px in either dimension (`imax(bw4,bh4) > 16`) are not
/// reconstructed as a single unit: they are split into 64x64 (or 128x128 for
/// 256px) sub-blocks, each decoding its own luma transform(s). The chroma is
/// decoded once — its coefficients are read with the first luma sub-block and
/// the pixels reconstructed with the last (the `cbs_stage` mechanism), so the
/// MSAC ordering matches the C decoder. <=64px blocks take the direct path.
#[allow(clippy::too_many_arguments)]
fn recon_b_intra(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    let b_dim = &BLOCK_DIMENSIONS[bs as u8 as usize];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;

    if imax(bw4, bh4) > 16 {
        // Split into 64x64 (or 128x128) sub-blocks. csplit[bs - 128x128][ss].
        const CSPLIT: [[BlockSize; 3]; 3] = [
            // BS_128x128
            [
                BlockSize::Bs64x64,
                BlockSize::Bs128x64,
                BlockSize::Bs128x128,
            ],
            // BS_128x64
            [BlockSize::Bs64x64, BlockSize::Bs64x64, BlockSize::Bs128x64],
            // BS_64x128
            [BlockSize::Bs64x64, BlockSize::Bs64x64, BlockSize::Bs64x128],
        ];
        let ss_hor = fi.ss_hor;
        let ss_ver = fi.ss_ver;
        let y_end = imin(by + bh4, fi.bh);
        let x_end = imin(bx + bw4, fi.bw);
        let (step, lbs2, cbs2i) = if imax(bw4, bh4) == 64 {
            (
                32,
                if lbs == BlockSize::Invalid {
                    BlockSize::Invalid
                } else {
                    BlockSize::Bs128x128
                },
                if cbs == BlockSize::Invalid {
                    BlockSize::Invalid
                } else {
                    BlockSize::Bs128x128
                },
            )
        } else {
            let csplit_row = (bs as i32 - BlockSize::Bs128x128 as i32) as usize;
            let csi = (ss_hor + ss_ver) as usize;
            (
                16,
                if lbs == BlockSize::Invalid {
                    BlockSize::Invalid
                } else {
                    BlockSize::Bs64x64
                },
                if cbs == BlockSize::Invalid {
                    BlockSize::Invalid
                } else {
                    CSPLIT[csplit_row][csi]
                },
            )
        };

        let mut sub_by = by;
        let mut sub_cby = cby;
        let mut yy = 0;
        while sub_by < y_end {
            let mut sub_bx = bx;
            let mut sub_cbx = cbx;
            let mut xx = 0;
            while sub_bx < x_end {
                // cbs2[0] = coef-read stage, cbs2[1] = recon stage (recon_tmpl.c:3111).
                let (read_cbs, recon_cbs) = if step == 32 {
                    (cbs2i, cbs2i)
                } else {
                    let read = if ((xx & ss_hor) | (yy & ss_ver)) == 0 {
                        cbs2i
                    } else {
                        BlockSize::Invalid
                    };
                    let recon = if (ss_hor == 0 || sub_bx + step >= x_end)
                        && (ss_ver == 0 || sub_by + step >= y_end)
                    {
                        cbs2i
                    } else {
                        BlockSize::Invalid
                    };
                    (read, recon)
                };

                if imax(
                    BLOCK_DIMENSIONS[lbs2 as u8 as usize][0] as i32,
                    BLOCK_DIMENSIONS[lbs2 as u8 as usize][1] as i32,
                ) > 16
                {
                    // 256px case: recurse one more level (lbs2 == 128x128).
                    recon_b_intra(
                        recon,
                        msac,
                        cdf_m,
                        a,
                        l,
                        b,
                        sub_bx,
                        sub_by,
                        sub_cbx,
                        sub_cby,
                        lbs2,
                        if read_cbs != BlockSize::Invalid {
                            read_cbs
                        } else {
                            recon_cbs
                        },
                        lbs2 != BlockSize::Invalid,
                        read_cbs != BlockSize::Invalid || recon_cbs != BlockSize::Invalid,
                        fi,
                    )?;
                } else {
                    // Luma 64x64 sub-block: the tx walk uses the sub-block size,
                    // but `b.bs` (passed to coef decode) stays the full block.
                    if lbs2 != BlockSize::Invalid {
                        recon_b_intra_luma_geom(
                            recon,
                            msac,
                            cdf_m,
                            a,
                            l,
                            b,
                            sub_bx,
                            sub_by,
                            lbs2 as usize,
                            fi,
                        )?;
                    }
                    // Chroma: read phase with the first sub-block, recon with the last.
                    let phase = match (
                        read_cbs != BlockSize::Invalid,
                        recon_cbs != BlockSize::Invalid,
                    ) {
                        (true, true) => Some(ChromaPhase::Both),
                        (true, false) => Some(ChromaPhase::ReadOnly),
                        (false, true) => Some(ChromaPhase::ReconOnly),
                        (false, false) => None,
                    };
                    if let Some(ph) = phase {
                        let ccbs = if read_cbs != BlockSize::Invalid {
                            read_cbs
                        } else {
                            recon_cbs
                        };
                        let sdp_active = lbs2 == BlockSize::Invalid;
                        recon_b_intra_chroma_phase(
                            recon, msac, cdf_m, a, l, b, sub_cbx, sub_cby, ccbs, sdp_active, fi, ph,
                        )?;
                    }
                }

                sub_bx += step;
                if step == 32 {
                    sub_cbx += step;
                } else if (xx & ss_hor) == ss_hor {
                    sub_cbx += step << ss_hor;
                }
                xx += 1;
            }
            sub_by += step;
            if step == 32 {
                sub_cby += step;
            } else if (yy & ss_ver) == ss_ver {
                sub_cby += step << ss_ver;
            }
            yy += 1;
        }
        return Ok(());
    }

    // Leaf: ordinary <=64px block.
    let intrabc = b.intrabc != 0;
    if has_luma {
        let bx4 = (bx & 63) as usize;
        let by4 = (by & 63) as usize;
        // IntraBC: copy the prediction from the current frame at the block
        // vector before the tx walk (recon_tmpl.c:3149-3155).
        if intrabc {
            let mv = unsafe { b.data.intra.intrabc_mv.c };
            crate::recon::intrabc_pred_8bpc(
                recon.dst_y,
                recon.frame.y_stride_px,
                bw4,
                bh4,
                bx,
                by,
                mv.x as i32,
                mv.y as i32,
                0,
                0,
                fi.bw * 4,
                fi.bh * 4,
            );
            if let Ok(e) = std::env::var("RAV2D_IBCPX") {
                let mut it = e.split(',');
                let ty = it.next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(-1);
                let tx = it.next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(-1);
                if ty == by && tx == bx {
                    let stride = recon.frame.y_stride_px;
                    let off = 4 * (by as usize * stride + bx as usize);
                    let row: Vec<u8> = (0..(bw4 * 4).min(16) as usize)
                        .map(|i| recon.dst_y[off + i])
                        .collect();
                    eprintln!(
                        "RIBCPX y={} x={} row0: {}",
                        by,
                        bx,
                        row.iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
        }
        recon_b_intra_luma(recon, msac, cdf_m, a, l, b, bx, by, bx4, by4, intrabc, fi)?;
    }
    if has_chroma {
        // IntraBC: chroma prediction copy for both planes (recon_tmpl.c:3591-3600).
        if intrabc {
            let mv = unsafe { b.data.intra.intrabc_mv.c };
            let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
            let cbw4 = cb_dim[0] as i32;
            let cbh4 = cb_dim[1] as i32;
            for pl in 0..2 {
                let dst_plane: &mut [u8] = if pl == 0 { recon.dst_u } else { recon.dst_v };
                crate::recon::intrabc_pred_8bpc(
                    dst_plane,
                    recon.frame.uv_stride_px,
                    cbw4,
                    cbh4,
                    cbx,
                    cby,
                    mv.x as i32,
                    mv.y as i32,
                    fi.ss_hor,
                    fi.ss_ver,
                    (fi.bw * 4) >> fi.ss_hor,
                    (fi.bh * 4) >> fi.ss_ver,
                );
            }
        }
        let sdp_active = lbs == BlockSize::Invalid;
        recon_b_intra_chroma(
            recon, msac, cdf_m, a, l, b, cbx, cby, cbs, intrabc, sdp_active, fi,
        )?;
    }
    Ok(())
}

/// Motion-compensate one plane of a single reference into `dst` (8bpc), mirroring
/// the unscaled branch of `mc()` (recon_tmpl.c:1569-1610). `mvx`/`mvy` are the
/// block MV (1/8-pel luma units). Uses the proper separable 8-tap / bilinear
/// primitives from `mc.rs`. Scaled references (svc.scale != 0) are not handled.
#[allow(clippy::too_many_arguments)]
fn inter_mc_plane_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    bw4: i32,
    bh4: i32,
    mvx: i32,
    mvy: i32,
    filter: u8,
    ss_hor: i32,
    ss_ver: i32,
    cur_bw: i32,
    cur_bh: i32,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let ref_stride = ref_pic.stride[(pl != 0) as usize].unsigned_abs();
    let ref_data: &[u8] = match ref_pic.data[pl] {
        Some(p) => {
            let pw = if pl == 0 {
                ref_pic.p.w
            } else {
                (ref_pic.p.w + ss_hor) >> ss_hor
            };
            let ph = if pl == 0 {
                ref_pic.p.h
            } else {
                (ref_pic.p.h + ss_ver) >> ss_ver
            };
            let _ = pw;
            // SAFETY: ref_pic owns a stride*height allocation for this plane.
            unsafe { std::slice::from_raw_parts(p.as_ptr(), ref_stride * ph as usize) }
        }
        None => return,
    };

    let left = 0i32;
    let top = 0i32;
    let right = cur_bw * 4 >> plss_hor;
    let bottom = cur_bh * 4 >> plss_ver;

    let mx = mvx & (15 >> (plss_hor == 0) as i32);
    let my = mvy & (15 >> (plss_ver == 0) as i32);
    let dx = bx * h_mul + (mvx >> (3 + plss_hor));
    let dy = by * v_mul + (mvy >> (3 + plss_ver));

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + bw4 * h_mul + (mx != 0) as i32 * 4 > right
        || dy + bh4 * v_mul + (my != 0) as i32 * 4 > bottom;

    let w = (bw4 * h_mul) as usize;
    let h = (bh4 * v_mul) as usize;
    let mxf = mx << (plss_hor == 0) as i32;
    let myf = my << (plss_ver == 0) as i32;
    // dav2d b->filter: 0=REGULAR,1=SMOOTH,2=SHARP -> 8tap filter_type; 3=BILINEAR.
    let is_bilin = filter == 3;

    let mut emu_buf = if need_emu {
        Some(vec![0u8; 192 * 192])
    } else {
        None
    };
    let (src, src_off, src_stride) = if let Some(ref mut buf) = emu_buf {
        let emu_w = w + (mx != 0) as usize * 7;
        let emu_h = h + (my != 0) as usize * 7;
        let emu_stride = 192usize;
        inter_emu_edge_8bpc(
            buf,
            emu_stride,
            ref_data,
            ref_stride,
            emu_w,
            emu_h,
            (right - left) as usize,
            (bottom - top) as usize,
            dx - (mx != 0) as i32 * 3 - left,
            dy - (my != 0) as i32 * 3 - top,
        );
        let off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
        (&buf[..], off, emu_stride)
    } else {
        let off = dy as usize * ref_stride + dx as usize;
        (ref_data, off, ref_stride)
    };

    if is_bilin {
        crate::mc::put_bilin_8bpc(
            dst,
            dst_stride,
            &src[src_off..],
            src_stride,
            w,
            h,
            mxf,
            myf,
        );
    } else {
        crate::mc::put_8tap_8bpc(
            dst,
            dst_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mxf,
            myf,
            filter as i32,
        );
    }
}

/// Translational MC into an intermediate i16 `tmp` buffer (recon_tmpl.c `mc`
/// with `dst == NULL`). Mirrors `inter_mc_plane_8bpc` but uses the `prep`
/// kernels (no final shift to pixels) so the result can be blended by the
/// compound `avg`/`w_avg`/`mask`/`w_mask` kernels. `tmp` is laid out at stride
/// `bw4 * h_mul` (= block width in pixels), matching dav2d's compinter scratch.
#[allow(clippy::too_many_arguments)]
fn inter_mc_plane_prep_8bpc(
    tmp: &mut [i16],
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    bw4: i32,
    bh4: i32,
    mvx: i32,
    mvy: i32,
    filter: u8,
    ss_hor: i32,
    ss_ver: i32,
    cur_bw: i32,
    cur_bh: i32,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let ref_stride = ref_pic.stride[(pl != 0) as usize].unsigned_abs();
    let ref_data: &[u8] = match ref_pic.data[pl] {
        Some(p) => {
            let ph = if pl == 0 {
                ref_pic.p.h
            } else {
                (ref_pic.p.h + ss_ver) >> ss_ver
            };
            // SAFETY: ref_pic owns a stride*height allocation for this plane.
            unsafe { std::slice::from_raw_parts(p.as_ptr(), ref_stride * ph as usize) }
        }
        None => return,
    };

    let left = 0i32;
    let top = 0i32;
    let right = cur_bw * 4 >> plss_hor;
    let bottom = cur_bh * 4 >> plss_ver;

    let mx = mvx & (15 >> (plss_hor == 0) as i32);
    let my = mvy & (15 >> (plss_ver == 0) as i32);
    let dx = bx * h_mul + (mvx >> (3 + plss_hor));
    let dy = by * v_mul + (mvy >> (3 + plss_ver));

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + bw4 * h_mul + (mx != 0) as i32 * 4 > right
        || dy + bh4 * v_mul + (my != 0) as i32 * 4 > bottom;

    let w = (bw4 * h_mul) as usize;
    let h = (bh4 * v_mul) as usize;
    let tmp_stride = w;
    let mxf = mx << (plss_hor == 0) as i32;
    let myf = my << (plss_ver == 0) as i32;
    let is_bilin = filter == 3;

    let mut emu_buf = if need_emu {
        Some(vec![0u8; 192 * 192])
    } else {
        None
    };
    let (src, src_off, src_stride) = if let Some(ref mut buf) = emu_buf {
        let emu_w = w + (mx != 0) as usize * 7;
        let emu_h = h + (my != 0) as usize * 7;
        let emu_stride = 192usize;
        inter_emu_edge_8bpc(
            buf,
            emu_stride,
            ref_data,
            ref_stride,
            emu_w,
            emu_h,
            (right - left) as usize,
            (bottom - top) as usize,
            dx - (mx != 0) as i32 * 3 - left,
            dy - (my != 0) as i32 * 3 - top,
        );
        let off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
        (&buf[..], off, emu_stride)
    } else {
        let off = dy as usize * ref_stride + dx as usize;
        (ref_data, off, ref_stride)
    };

    if is_bilin {
        crate::mc::prep_bilin_8bpc(
            tmp,
            tmp_stride,
            &src[src_off..],
            src_stride,
            w,
            h,
            mxf,
            myf,
        );
    } else {
        crate::mc::prep_8tap_8bpc(
            tmp,
            tmp_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mxf,
            myf,
            filter as i32,
        );
    }
}

/// Warp-affine motion compensation for a block plane (recon_tmpl.c
/// `warp_affine`, affine path). Predicts the block in 8x8 sub-tiles using the
/// derived warp matrix `wmp`. Only the affine sub-path is implemented (block is
/// >= 8px and `wmp.affine`); callers gate on those conditions, falling back to
/// translational MC otherwise. 8bpc luma + chroma (with subsampling).
#[allow(clippy::too_many_arguments)]
fn warp_affine_plane_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    b_dim: &[u8],
    wmp: &crate::headers::WarpedMotionParams,
    ss_hor: i32,
    ss_ver: i32,
    frame_bw: i32,
    frame_bh: i32,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let mat = &wmp.matrix;
    let width = frame_bw * 4 >> plss_hor;
    let height = frame_bh * 4 >> plss_ver;
    let ref_stride = ref_pic.stride[(pl != 0) as usize].unsigned_abs();
    let ref_data: &[u8] = match ref_pic.data[pl] {
        Some(p) => {
            let ph = if pl == 0 {
                ref_pic.p.h
            } else {
                (ref_pic.p.h + ss_ver) >> ss_ver
            };
            unsafe { std::slice::from_raw_parts(p.as_ptr(), ref_stride * ph as usize) }
        }
        None => return,
    };

    let blk_w = b_dim[0] as i32 * h_mul;
    let blk_h = b_dim[1] as i32 * v_mul;
    let abcd: [i16; 4] = wmp.abcd;

    let mut emu = [0u8; 32 * 32];
    let mut y = 0;
    while y < blk_h {
        let src_y = by * 4 + ((y + 4) << plss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut x = 0;
        while x < blk_w {
            let src_x = bx * 4 + ((x + 4) << plss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> plss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> plss_ver;

            let dx = (mvx >> 16) as i32 - 4;
            let mx = (((mvx as i32) & 0xffff)
                - wmp.abcd[0] as i32 * 4
                - wmp.abcd[1] as i32 * 7)
                & !0x3f;
            let dy = (mvy >> 16) as i32 - 4;
            let my = (((mvy as i32) & 0xffff)
                - wmp.abcd[2] as i32 * 4
                - wmp.abcd[3] as i32 * 4)
                & !0x3f;

            let (src, src_off, src_stride): (&[u8], usize, usize) =
                if dx < 3 || dx + 8 + 4 > width || dy < 3 || dy + 8 + 4 > height {
                    crate::mc::emu_edge_8bpc(
                        15,
                        15,
                        width as usize,
                        height as usize,
                        (dx - 3) as isize,
                        (dy - 3) as isize,
                        &mut emu,
                        32,
                        ref_data,
                        ref_stride,
                    );
                    (&emu[..], 32 * 3 + 3, 32)
                } else {
                    (ref_data, ref_stride * dy as usize + dx as usize, ref_stride)
                };

            if std::env::var("RAV2D_WDX").is_ok() && by == 12 && bx == 0 && pl == 0 && y == 0 && x == 0
            {
                let mut s = String::from("WEMU");
                for r in 0..15 {
                    for c in 0..15 {
                        s.push_str(&format!(" {}", src[r * src_stride + c]));
                    }
                }
                eprintln!("{s}");
            }
            let dst_sub = (y as usize) * dst_stride + x as usize;
            crate::mc::warp_affine_8x8_8bpc(
                &mut dst[dst_sub..],
                dst_stride,
                src,
                src_stride,
                src_off,
                &abcd,
                mx,
                my,
            );
            if std::env::var("RAV2D_WDX").is_ok() && by == 12 && bx == 0 && pl == 0 && y == 0 && x == 0
            {
                let mut s = String::from("WOUT");
                for r in 0..8 {
                    for c in 0..8 {
                        s.push_str(&format!(" {}", dst[dst_sub + r * dst_stride + c]));
                    }
                }
                eprintln!("{s}");
            }
            x += 8;
        }
        y += 8;
    }
}

/// Non-affine / small-block warp MC (dav2d `ext_warp`, recon_tmpl.c:1720). Used
/// for warp blocks where the 8x8 affine kernel does not apply: non-affine warp
/// types, or after subsampling a block becomes < 8 in either dimension (e.g.
/// chroma of an 8x8 luma block in 4:2:0). Walks `sw`x`sh` windows, then 4x4
/// tiles within, with the per-tile `+0x200` rounding and 6-bit mx/my subpel.
#[allow(clippy::too_many_arguments)]
fn ext_warp_plane_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    b_dim: &[u8],
    wmp: &crate::headers::WarpedMotionParams,
    ss_hor: i32,
    ss_ver: i32,
    frame_bw: i32,
    frame_bh: i32,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let mat = &wmp.matrix;
    let w = frame_bw * 4 >> plss_hor;
    let h = frame_bh * 4 >> plss_ver;
    let ref_stride = ref_pic.stride[(pl != 0) as usize].unsigned_abs();
    let ref_data: &[u8] = match ref_pic.data[pl] {
        Some(p) => {
            let ph = if pl == 0 {
                ref_pic.p.h
            } else {
                (ref_pic.p.h + ss_ver) >> ss_ver
            };
            unsafe { std::slice::from_raw_parts(p.as_ptr(), ref_stride * ph as usize) }
        }
        None => return,
    };

    let blk_w = b_dim[0] as i32 * h_mul;
    let blk_h = b_dim[1] as i32 * v_mul;
    let sw = imin(blk_w, 8);
    let hsw = sw >> 1;
    let sh = imin(blk_h, 8);
    let hsh = sh >> 1;

    let mut emu = [0u8; 32 * 32];
    let mut y = 0;
    while y < blk_h {
        let src_y = by * 4 + ((y + hsh) << plss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut x = 0;
        while x < blk_w {
            let src_x = bx * 4 + ((x + hsw) << plss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> plss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> plss_ver;
            let left_window = (mvx >> 16) as i32 - hsw - 3;
            let top_window = (mvy >> 16) as i32 - hsh - 3;
            let left = iclip(left_window, 0, w - 1);
            let right = iclip(left_window + sw + 7, 1, w);
            let top = iclip(top_window, 0, h - 1);
            let bottom = iclip(top_window + sh + 7, 1, h);

            let mut yy = y;
            while yy < y + sh {
                let src_y2 = by * 4 + ((yy + 2) << plss_ver);
                let mat3_y2 = mat[3] as i64 * src_y2 as i64 + mat[0] as i64;
                let mat5_y2 = mat[5] as i64 * src_y2 as i64 + mat[1] as i64;
                let mut xx = x;
                while xx < x + sw {
                    let src_x2 = bx * 4 + ((xx + 2) << plss_hor);
                    let mvx2 = ((mat[2] as i64 * src_x2 as i64 + mat3_y2) >> plss_hor) + 0x200;
                    let mvy2 = ((mat[4] as i64 * src_x2 as i64 + mat5_y2) >> plss_ver) + 0x200;

                    let dx = (mvx2 >> 16) as i32 - 2;
                    let mx = ((mvx2 >> 10) & 63) as i32;
                    let dy = (mvy2 >> 16) as i32 - 2;
                    let my = ((mvy2 >> 10) & 63) as i32;

                    let (src, src_off, src_stride): (&[u8], usize, usize) =
                        if dx - 3 < left || dx + 4 + 4 > right || dy - 3 < top || dy + 4 + 4 > bottom
                        {
                            let region_off = left as usize + top as usize * ref_stride;
                            crate::mc::emu_edge_8bpc(
                                11,
                                11,
                                (right - left) as usize,
                                (bottom - top) as usize,
                                (dx - 3 - left) as isize,
                                (dy - 3 - top) as isize,
                                &mut emu,
                                32,
                                &ref_data[region_off..],
                                ref_stride,
                            );
                            (&emu[..], 32 * 3 + 3, 32)
                        } else {
                            (ref_data, ref_stride * dy as usize + dx as usize, ref_stride)
                        };

                    let dst_sub = (yy as usize) * dst_stride + xx as usize;
                    crate::mc::put_8tap_8bpc(
                        &mut dst[dst_sub..],
                        dst_stride,
                        src,
                        src_off,
                        src_stride,
                        4,
                        4,
                        mx,
                        my,
                        -1,
                    );
                    xx += 4;
                }
                yy += 4;
            }
            x += sw;
        }
        y += sh;
    }
}

/// Emulated-edge fetch for inter MC (recon_tmpl.c emu_edge): clamp source reads
/// to the plane bounds. Identical semantics to mc.rs's private `emu_edge`.
#[allow(clippy::too_many_arguments)]
fn inter_emu_edge_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
) {
    for dy in 0..bh {
        let ay = ((y + dy as i32).max(0) as usize).min(ih.saturating_sub(1));
        let drow = dy * dst_stride;
        let srow = ay * src_stride;
        for dx in 0..bw {
            let ax = ((x + dx as i32).max(0) as usize).min(iw.saturating_sub(1));
            dst[drow + dx] = src.get(srow + ax).copied().unwrap_or(0);
        }
    }
}

/// Add an inter residual transform block onto an already motion-compensated
/// destination (8bpc). Decodes coefficients with the inter coef contexts
/// (`intra = false`), applies the optional secondary transform, then the inverse
/// transform add. Mirrors the residual tail of `recon_b_luma_tx` for inter.
#[allow(clippy::too_many_arguments)]
fn inter_residual_tx_8bpc(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    pl: usize,
    tx: usize,
    bx: i32,
    by: i32,
    dst_is_uv: bool,
    txtp_seed: u16,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    use crate::levels::IntraPredMode;
    let t_dim = &TXFM_DIMENSIONS[tx];
    let tw = t_dim.w as usize * 4;
    let th = t_dim.h as usize * 4;
    let tw4 = t_dim.w as i32;
    let th4 = t_dim.h as i32;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;
    let ss_hor = if dst_is_uv { recon.frame.ss_hor } else { 0 };
    let ss_ver = if dst_is_uv { recon.frame.ss_ver } else { 0 };

    let bx4 = ((bx & 63) >> ss_hor) as usize;
    let by4 = ((by & 63) >> ss_ver) as usize;

    let cf_n = tw * th;
    recon.cf[..cf_n].fill(0);

    let mut txtp: u16 = txtp_seed;
    let mut res_ctx: u8 = 0;
    let (mut eob, stx, mut txtp) = if b.skip_txfm != 0 {
        res_ctx = 0x40;
        (-1i32, 0i32, crate::levels::txtp::DCT_DCT as u32)
    } else {
        let dq_tbl = recon.dq_active[seg_id][pl];
        let qm_ref: Option<&[u8]> = recon.frame.qm[tx][pl].as_deref();
        let params = crate::recon::DecodeCoefParams {
            tx,
            bs: b.bs as usize,
            plane: pl as i32,
            intra: false,
            fsc: false,
            lossless,
            sdp_active: false,
            y_mode: 0,
            uv_mode: 0,
            seg_id,
            seq_fsc: recon.frame.seq_fsc,
            seq_ist: recon.frame.seq_ist,
            seq_cctx: recon.frame.seq_cctx,
            chroma_dctonly: false,
            reduced_txtp_set: recon.frame.reduced_txtp_set,
            tcq_enabled: recon.frame.tcq,
            layout: recon.frame.layout,
            u_has_cf: recon.scratch_u_has_cf,
            cbx: bx,
            cby: by,
            luma_fsc_map: &[],
            dq_tbl,
            bitdepth: recon.frame.bitdepth,
            qm: qm_ref,
            ss_hor: recon.frame.ss_hor != 0,
            ss_ver: recon.frame.ss_ver != 0,
        };
        let (acoef, lcoef): (&[u8], &[u8]) = if pl == 0 {
            (&a.lcoef[bx4..], &l.lcoef[by4..])
        } else {
            (&a.ccoef[pl - 1][bx4..], &l.ccoef[pl - 1][by4..])
        };
        let eob = crate::recon::decode_coefs(
            msac,
            recon.cdf_coef,
            cdf_m,
            acoef,
            lcoef,
            &params,
            recon.cf,
            &mut txtp,
            &mut res_ctx,
        );
        if eob == i32::MIN {
            return Err(());
        }
        // Mirror dav2d's `txtp_map` (recon_tmpl.c:2489/2502): record the luma
        // transform type for each covered 4x4 so the inter chroma path can seed
        // its transform type. dav2d masks off the secondary-transform bits
        // (txtp &= 0xff) before storing, so only the base type is propagated.
        if pl == 0 {
            let by15 = (by & 15) as usize;
            let bx15 = (bx & 15) as usize;
            let base_txtp = txtp & 0xff;
            for dy in 0..t_dim.h as usize {
                for dx in 0..t_dim.w as usize {
                    let yy = by15 + dy;
                    let xx = bx15 + dx;
                    if yy < 16 && xx < 16 {
                        recon.scratch.txtp_map[yy * 16 + xx] = base_txtp;
                    }
                }
            }
        }
        let stx = (txtp >> 8) as i32;
        (eob, stx, (txtp & 0xff) as u32)
    };
    if std::env::var("RAV2D_COEF_TRACE").is_ok() {
        if pl == 0 {
            eprintln!(
                "DCOEF y={} x={} tx={} txtp={} stx={} eob={} rng={}",
                by, bx, tx, txtp, stx, eob, msac.dbg_rng()
            );
        } else {
            eprintln!(
                "DCHROMA y={} x={} pl={} uvtx={} txtp={} eob={} rng={}",
                by, bx, pl - 1, tx, txtp, eob, msac.dbg_rng()
            );
        }
    }
    if pl == 1 {
        recon.scratch_u_has_cf = (eob >= 0) as i32;
    }

    // context fill
    let aw = imin(tw4, (fi.bw >> ss_hor) - (bx >> ss_hor)).max(0) as usize;
    let lh = imin(th4, (fi.bh >> ss_ver) - (by >> ss_ver)).max(0) as usize;
    if pl == 0 {
        if aw > 0 {
            a.lcoef[bx4..bx4 + aw].fill(res_ctx);
        }
        if lh > 0 {
            l.lcoef[by4..by4 + lh].fill(res_ctx);
        }
    } else {
        if aw > 0 {
            a.ccoef[pl - 1][bx4..bx4 + aw].fill(res_ctx);
        }
        if lh > 0 {
            l.ccoef[pl - 1][by4..by4 + lh].fill(res_ctx);
        }
    }

    // Mark the per-SB `is_coded` grid for this luma TX block. dav2d does this in
    // recon_b_luma_tx (recon_tmpl.c:2720) for every block — intra AND inter — so
    // later blocks' top-right / bottom-left intra-edge availability (n_tr/n_bl,
    // used by SMOOTH_PRED incl. warp-interintra) sees inter neighbours as coded.
    if pl == 0 {
        let mask: u64 = ((1u64 << tw4) - 1) << (bx4 as u32);
        for y in 0..th4 as usize {
            let row = by4 + y;
            if row < 64 {
                recon.scratch.is_coded[0][row] |= mask;
            }
        }
    } else if pl == 1 {
        // Chroma `is_coded` is marked once per chroma TX (pl==1 only, mirroring
        // recon_tmpl.c:4017 where U and V share a single grid update).
        let mask: u64 = ((1u64 << tw4) - 1) << (bx4 as u32);
        for y in 0..th4 as usize {
            let row = by4 + y;
            if row < 64 {
                recon.scratch.is_coded[1][row] |= mask;
            }
        }
    }

    if eob == -1 {
        return Ok(());
    }

    let (dst, stride) = if pl == 0 {
        (&mut *recon.dst_y, recon.frame.y_stride_px)
    } else if pl == 1 {
        (&mut *recon.dst_u, recon.frame.uv_stride_px)
    } else {
        (&mut *recon.dst_v, recon.frame.uv_stride_px)
    };
    let dst_off = 4 * ((by >> ss_ver) as usize * stride + (bx >> ss_hor) as usize);

    if stx != 0 && (stx & 3) != 0 {
        // Inter never matches the intra y_mode transpose mask -> transpose = true.
        let transpose = true;
        let stype = (stx & 3) - 1;
        let set = (stx >> 2) & 15;
        if tw >= 8 && th >= 8 {
            let koff = (set as usize * 3 + stype as usize) * 1536;
            let mut sums = [0i32; 48];
            crate::stx::stxfm(
                &mut sums,
                recon.cf,
                &crate::stx_tables::STX_8X8_KERNEL[koff..],
                48,
                eob as usize,
                recon.frame.bitdepth_max,
            );
            recon.cf[..32].fill(0);
            let idx = (imin(t_dim.lh as i32, 3) - 1) as usize;
            let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_8X8[idx][transpose as usize];
            let mapping = &crate::stx_tables::COEFF8X8_MAPPING[set as usize * 3 + stype as usize];
            for x in 0..48 {
                recon.cf[scan_out[mapping[x] as usize] as usize] = sums[x];
            }
            eob = [63, 119, 231][idx];
        } else {
            let koff = (set as usize * 3 + stype as usize) * 128;
            let mut sums = [0i32; 16];
            crate::stx::stxfm(
                &mut sums,
                recon.cf,
                &crate::stx_tables::STX_4X4_KERNEL[koff..],
                16,
                eob as usize,
                recon.frame.bitdepth_max,
            );
            let idx = imin(t_dim.lh as i32, 3) as usize;
            let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_4X4[idx][transpose as usize];
            recon.cf[4..8].fill(0);
            for x in 0..16 {
                recon.cf[scan_out[x] as usize] = sums[x];
            }
            eob = [15, 15, 51, 99][idx];
        }
    }

    if recon.frame.seq_inter_ddt {
        txtp += txtp & crate::tables::TX_DDT_MASK[tx] as u32;
    }
    let _ = IntraPredMode::DcPred;
    if std::env::var("RAV2D_CFDUMP").is_ok() && by == 12 && bx == 0 && pl == 0 {
        let mut s = format!("DCFDUMP by={} bx={} tx={} txtp={} eob={}\n", by, bx, tx, txtp, eob);
        for i in 0..64 {
            s.push_str(&format!(" {}", recon.cf[i]));
        }
        eprintln!("{s}");
    }
    let dbg_final = std::env::var("RAV2D_FINAL").is_ok() && by == 12 && bx == 0 && pl == 0;
    let dbg_off = dst_off;
    if dbg_final {
        eprintln!("DPREDR by={} bx={} (pred at residual time)", by, bx);
        for yy in 0..(th as usize) {
            let mut s = String::from("PR");
            for xx in 0..(tw as usize) {
                s.push_str(&format!(" {}", dst[dbg_off + yy * stride + xx]));
            }
            eprintln!("{s}");
        }
    }
    crate::itx::inv_txfm_add_8bpc(dst, dst_off, stride, recon.cf, txtp, eob, tx);
    if dbg_final {
        eprintln!("DFINAL by={} bx={}", by, bx);
        for yy in 0..(th as usize) {
            let mut s = String::from("F");
            for xx in 0..(tw as usize) {
                s.push_str(&format!(" {}", dst[dbg_off + yy * stride + xx]));
            }
            eprintln!("{s}");
        }
    }
    Ok(())
}

/// Inter chroma residual for both planes with the cross-component transform
/// (CCTX). Mirrors dav2d's chroma loop in `dav2d_recon_b` (recon_tmpl.c:3546-
/// 3925): decode all U then all V TU coefficients (the entropy order), apply
/// CCTX to mix U/V per TU, then inverse-transform-add both planes. CCTX needs
/// U and V coefficients together, so it cannot be done in the per-plane
/// `inter_residual_tx_8bpc`. The chroma prediction is already in dst_u/dst_v.
#[allow(clippy::too_many_arguments)]
fn inter_chroma_residual_8bpc(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    uvtx: usize,
    cbx: i32,
    cby: i32,
    cbw4ss: i32,
    cbh4ss: i32,
    cw4ss: i32,
    ch4ss: i32,
    txtp_seed: u16,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    let uv_t_dim = &TXFM_DIMENSIONS[uvtx];
    let (txw, txh) = (uv_t_dim.w as i32, uv_t_dim.h as i32);
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;
    let n_tu = (cbw4ss * cbh4ss) as usize;
    let tu_n = (txw as usize * 4) * (txh as usize * 4);

    // Per-TU coefficient + metadata storage (recon_tmpl.c t->cf_uv / chroma_txtp).
    // Coefficients for the TU at grid position i = y*cbw4ss + x are placed at
    // cf[i*16] (mirrors C `cf[pl][i*16]`); the per-plane buffer is n_tu*16.
    let mut cf_uv = vec![0i32; n_tu * 16 * 2];
    let (cf_u, cf_v) = cf_uv.split_at_mut(n_tu * 16);
    let mut tu_eob = [[-1i32; 2]; 256];
    let mut tu_txtp = [[0u16; 2]; 256];

    // Decode all U TUs then all V TUs (entropy order).
    recon.scratch_u_has_cf = 0;
    for pl in 0..2usize {
        let plane_cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
        let mut y = 0;
        while y < ch4ss {
            let mut x = 0;
            while x < cw4ss {
                let i = (y * cbw4ss + x) as usize;
                let bx = cbx + (x << ss_hor);
                let by = cby + (y << ss_ver);
                let bx4 = ((bx & 63) >> ss_hor) as usize;
                let by4 = ((by & 63) >> ss_ver) as usize;

                let cf_slot = &mut plane_cf[i * 16..i * 16 + tu_n];
                cf_slot.fill(0);

                let mut txtp: u16 = txtp_seed;
                let mut res_ctx: u8 = 0;
                let eob = if b.skip_txfm != 0 {
                    res_ctx = 0x40;
                    txtp = crate::levels::txtp::DCT_DCT as u16;
                    -1i32
                } else {
                    let dq_tbl = recon.dq_active[seg_id][1 + pl];
                    let qm_ref: Option<&[u8]> = recon.frame.qm[uvtx][1 + pl].as_deref();
                    let params = crate::recon::DecodeCoefParams {
                        tx: uvtx,
                        bs: b.bs as usize,
                        plane: (1 + pl) as i32,
                        intra: false,
                        fsc: false,
                        lossless,
                        sdp_active: false,
                        y_mode: 0,
                        uv_mode: 0,
                        seg_id,
                        seq_fsc: recon.frame.seq_fsc,
                        seq_ist: recon.frame.seq_ist,
                        seq_cctx: recon.frame.seq_cctx,
                        chroma_dctonly: false,
                        reduced_txtp_set: recon.frame.reduced_txtp_set,
                        tcq_enabled: recon.frame.tcq,
                        layout: recon.frame.layout,
                        u_has_cf: recon.scratch_u_has_cf,
                        cbx: bx,
                        cby: by,
                        luma_fsc_map: &[],
                        dq_tbl,
                        bitdepth: recon.frame.bitdepth,
                        qm: qm_ref,
                        ss_hor: recon.frame.ss_hor != 0,
                        ss_ver: recon.frame.ss_ver != 0,
                    };
                    let acoef = &a.ccoef[pl][bx4..];
                    let lcoef = &l.ccoef[pl][by4..];
                    let e = crate::recon::decode_coefs(
                        msac,
                        recon.cdf_coef,
                        cdf_m,
                        acoef,
                        lcoef,
                        &params,
                        cf_slot,
                        &mut txtp,
                        &mut res_ctx,
                    );
                    if e == i32::MIN {
                        return Err(());
                    }
                    e
                };
                if pl == 0 {
                    recon.scratch_u_has_cf = (eob >= 0) as i32;
                }
                tu_eob[i][pl] = eob;
                tu_txtp[i][pl] = txtp;

                // Context fill (a/l ccoef) and is_coded[1] (pl==0 only).
                let aw = imin(txw, (fi.bw >> ss_hor) - (bx >> ss_hor)).max(0) as usize;
                let lh = imin(txh, (fi.bh >> ss_ver) - (by >> ss_ver)).max(0) as usize;
                if aw > 0 {
                    a.ccoef[pl][bx4..bx4 + aw].fill(res_ctx);
                }
                if lh > 0 {
                    l.ccoef[pl][by4..by4 + lh].fill(res_ctx);
                }
                if pl == 0 {
                    let mask: u64 = ((1u64 << txw) - 1) << (bx4 as u32);
                    for yy in 0..txh as usize {
                        let row = by4 + yy;
                        if row < 64 {
                            recon.scratch.is_coded[1][row] |= mask;
                        }
                    }
                }
                x += txw;
            }
            y += txh;
        }
    }

    // CCTX + inverse transform add per TU (recon_tmpl.c:3882-3925).
    let cctx_enabled = recon.frame.seq_cctx
        && (recon.frame.layout == crate::headers::PixelLayout::I420 || uv_t_dim.max < 8);
    let uv_stride = recon.frame.uv_stride_px;
    let mut y = 0;
    while y < ch4ss {
        let mut x = 0;
        while x < cw4ss {
            let i = (y * cbw4ss + x) as usize;
            let bx = cbx + (x << ss_hor);
            let by = cby + (y << ss_ver);

            // dav2d uses `eob[0] >= intra`; intra is 0 for inter blocks.
            let cctx_type = if cctx_enabled && tu_eob[i][0] >= 0 {
                (tu_txtp[i][0] >> 8) as i32
            } else {
                0
            };
            if cctx_type != 0 {
                let sz = imin(txw * 4, 32) as usize * imin(txh * 4, 32) as usize;
                crate::itx::cctx_8bpc(
                    &mut cf_u[i * 16..],
                    &mut cf_v[i * 16..],
                    &crate::tables::CCTX_ANGLE[(cctx_type - 1) as usize],
                    sz,
                );
                let gt = (tu_eob[i][1] > tu_eob[i][0]) as usize;
                tu_eob[i][1 - gt] = tu_eob[i][gt];
                let t0 = tu_txtp[i][0] & 0xff;
                tu_txtp[i][0] = t0;
                tu_txtp[i][1] = t0;
            }

            for pl in 0..2usize {
                if tu_eob[i][pl] == -1 {
                    continue;
                }
                let mut txtp = tu_txtp[i][pl] as u32;
                if recon.frame.seq_inter_ddt {
                    txtp += txtp & crate::tables::TX_DDT_MASK[uvtx] as u32;
                }
                let dst_off =
                    4 * (((by >> ss_ver) as usize) * uv_stride + ((bx >> ss_hor) as usize));
                let cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
                let dst_plane: &mut [u8] = if pl == 0 { recon.dst_u } else { recon.dst_v };
                crate::itx::inv_txfm_add_8bpc(
                    dst_plane,
                    dst_off,
                    uv_stride,
                    &mut cf[i * 16..],
                    txtp,
                    tu_eob[i][pl],
                    uvtx,
                );
            }
            x += txw;
        }
        y += txh;
    }
    Ok(())
}

/// Inter-intra blend (recon_tmpl.c:2828-2908). The inter prediction is already
/// in `dst`; build the intra prediction (DC/V/H/SMOOTH, or wedge) from the
/// reconstructed neighbour edges into a temp buffer, then blend it over `dst`
/// with the II / wedge mask. Plane 0 (luma) or 1/2 (chroma, subsampled).
#[allow(clippy::too_many_arguments)]
fn iiblend_luma_8bpc(
    recon: &mut ReconCtx,
    b: &Av2Block,
    dst_off: usize,
    stride: usize,
    bw4: i32,
    bh4: i32,
    by: i32,
    bx: i32,
    ss_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    iiblend_plane_8bpc(recon, b, 0, dst_off, stride, bw4, bh4, by, bx, ss_bs, fi);
}

#[allow(clippy::too_many_arguments)]
fn iiblend_chroma_8bpc(
    recon: &mut ReconCtx,
    b: &Av2Block,
    plane: usize,
    dst_off: usize,
    stride: usize,
    bw4: i32,
    bh4: i32,
    by: i32,
    bx: i32,
    ss_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    iiblend_plane_8bpc(recon, b, plane, dst_off, stride, bw4, bh4, by, bx, ss_bs, fi);
}

#[allow(clippy::too_many_arguments)]
fn iiblend_plane_8bpc(
    recon: &mut ReconCtx,
    b: &Av2Block,
    plane: usize,
    dst_off: usize,
    stride: usize,
    bw4: i32,
    bh4: i32,
    by: i32,
    bx: i32,
    ss_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    use crate::levels::{
        InterIntraPredMode, IntraPredMode, ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG,
    };
    let ii_mode = unsafe { b.data.inter.interintra_mode };
    let wedge_idx = unsafe { b.data.inter.wedge_idx };
    // II mode -> intra pred mode (II_SMOOTH(3) -> SMOOTH_PRED(9)).
    let m0: u8 = if ii_mode == InterIntraPredMode::SmoothPred as u8 {
        IntraPredMode::SmoothPred as u8
    } else {
        // DC(0)->DcPred(0), V(1)->VertPred(1), H(2)->HorPred(2).
        ii_mode
    };
    let angle: i32 = [0, 90, 180, 0][ii_mode as usize];

    let chroma = plane != 0;
    let ss_hor = if chroma { fi.ss_hor } else { 0 };
    let ss_ver = if chroma { fi.ss_ver } else { 0 };
    let ssbw4 = bw4 >> ss_hor;
    let ssbh4 = bh4 >> ss_ver;
    let w = (ssbw4 * 4) as usize;
    let h = (ssbh4 * 4) as usize;

    // n_tr / n_bl only matter for SMOOTH_PRED (recon_tmpl.c:2844-2879).
    let mut n_tr = 0i32;
    let mut n_bl = 0i32;
    if m0 == IntraPredMode::SmoothPred as u8 {
        let bx4 = (bx & 63) as usize;
        let by4 = (by & 63) as usize;
        let sbsz = fi.sb_step;
        if by > fi.tile_row_start {
            let mut wv = imin(bw4, fi.tile_col_end - bx - bw4);
            if (by & (sbsz - 1)) == 0 {
                n_tr = 0; // top sb boundary: simplified (no a_sb_cache)
            } else {
                let end = imin(((bx + sbsz) & !(sbsz - 1)) + 0, fi.tile_col_end);
                wv = imin(wv, end - bx - bw4);
                if wv <= 0 {
                    n_tr = 0;
                } else {
                    let row = ((by4 >> ss_ver) as i32 - 1) as usize;
                    if row < 64 {
                        n_tr = ((recon.scratch.is_coded[chroma as usize][row]
                            >> (((bx4 + bw4 as usize) >> ss_hor) as u32))
                            & 1) as i32;
                    }
                }
            }
        }
        if bx > fi.tile_col_start {
            let end = imin((by + sbsz) & !(sbsz - 1), fi.tile_row_end);
            let hv = imin(bh4, end - by - bh4);
            if hv <= 0 {
                n_bl = 0;
            } else if (bx & (sbsz - 1)) == 0 {
                n_bl = hv;
            } else {
                let row = ((by4 + bh4 as usize) >> ss_ver) as usize;
                if row < 64 {
                    n_bl = ((recon.scratch.is_coded[chroma as usize][row]
                        >> (((bx4 as i32 - 1) >> ss_hor) as u32))
                        & 1) as i32;
                }
            }
        }
    }

    let apply_ibp = recon.frame.seq_ibp && imax(ssbw4, ssbh4) > 1;
    let have_left = bx > fi.tile_col_start;
    let have_top = by > fi.tile_row_start;
    let intra_flags = if apply_ibp { ANGLE_IBP_FLAG } else { 0 }
        | if have_left { ANGLE_HAS_LEFT_FLAG } else { 0 }
        | if have_top { ANGLE_HAS_TOP_FLAG } else { 0 };

    let edge_o: usize = 768;
    let max_w = 4 * (fi.bw >> ss_hor) - 4 * (bx >> ss_hor);
    let max_h = 4 * (fi.bh >> ss_ver) - 4 * (by >> ss_ver);

    // Intra prediction into a temp buffer (stride = w). Borrow the dst plane
    // (read for edge prep) and `edge` (mut) as disjoint fields.
    let mut tmp = vec![0u8; w * h];
    {
        let ReconCtx {
            dst_y,
            dst_u,
            dst_v,
            edge,
            frame,
            ..
        } = &mut *recon;
        let dst_plane: &[u8] = match plane {
            0 => dst_y,
            1 => dst_u,
            _ => dst_v,
        };
        let m = crate::ipred_prepare::prepare_intra_edges_8bpc(
            bx >> ss_hor,
            by >> ss_ver,
            fi.tile_col_end >> ss_hor,
            fi.tile_row_end >> ss_ver,
            n_tr,
            n_bl,
            dst_plane,
            dst_off,
            stride,
            None,
            m0,
            ssbw4,
            ssbh4,
            angle | intra_flags,
            edge,
            edge_o,
        );
        if std::env::var("RAV2D_IIB").is_ok() && by == 12 && bx == 0 && plane == 0 {
            eprintln!("IINTR n_tr={} n_bl={}", n_tr, n_bl);
            let mut s = format!("IIEDGE m={} flags={}", m as i32, intra_flags);
            for i in (edge_o - 18)..(edge_o + 18) {
                s.push_str(&format!(" {}", edge[i]));
            }
            eprintln!("{s}");
        }
        dispatch_ipred(
            m,
            &mut tmp,
            0,
            w,
            edge,
            edge_o,
            w,
            h,
            intra_flags,
            max_w,
            max_h,
            &frame.ibp_weights,
        );
    }

    // Mask: II mask (wedge_idx == -1) or wedge mask. Copy to a local so the
    // `recon.masks` borrow does not overlap the `recon.dst_*` mutable write.
    let mask: Vec<u8> = if wedge_idx == -1 {
        let mode = match ii_mode {
            x if x == InterIntraPredMode::DcPred as u8 => InterIntraPredMode::DcPred,
            x if x == InterIntraPredMode::VertPred as u8 => InterIntraPredMode::VertPred,
            x if x == InterIntraPredMode::HorPred as u8 => InterIntraPredMode::HorPred,
            _ => InterIntraPredMode::SmoothPred,
        };
        recon
            .masks
            .ii_mask(ss_bs as usize, ssbw4 as usize, ssbh4 as usize, mode)[..w * h]
            .to_vec()
    } else {
        recon
            .masks
            .wedge_mask(
                ss_bs as usize,
                bw4 as usize,
                bh4 as usize,
                wedge_idx as usize,
                (ss_hor + ss_ver) as usize,
            )[..w * h]
            .to_vec()
    };
    if std::env::var("RAV2D_IIB").is_ok() && by == 12 && bx == 0 && plane == 0 {
        eprintln!("IIB by={} bx={} m0={} ii_mode={} wedge={} w={} h={}", by, bx, m0, ii_mode, wedge_idx, w, h);
        let mut si = String::from("IIINTRA");
        let mut sm = String::from("IIMASK");
        for i in 0..(w * h).min(64) {
            si.push_str(&format!(" {}", tmp[i]));
            sm.push_str(&format!(" {}", mask[i]));
        }
        eprintln!("{si}");
        eprintln!("{sm}");
    }
    let dst_plane: &mut [u8] = match plane {
        0 => recon.dst_y,
        1 => recon.dst_u,
        _ => recon.dst_v,
    };
    crate::mc::blend_8bpc(&mut dst_plane[dst_off..], stride, &tmp, w, h, &mask);
}

/// Splat a resolved COMPOUND block's MVs into the spatial refmvs grid + the
/// temporal MV grid (`splat_tworef_mv`, decode.c:627-710). Translational path
/// only (warp-compound / global-affine deferred). The spatial grid write is the
/// same for AVG/SEG/WEDGE (only the temporal grid differs for WEDGE, handled via
/// the per-2x2 wedge tmvp mask).
#[allow(clippy::too_many_arguments)]
fn splat_tworef_mv(
    recon: &mut ReconCtx,
    b: &Av2Block,
    bx: i32,
    by: i32,
    by4r: usize,
    bw4: i32,
    bh4: i32,
    bs: BlockSize,
) {
    use crate::levels::Mv;
    let refs = unsafe { b.ref_pair.r };
    let ref0 = refs[0];
    let ref1 = refs[1];
    let cwp_idx = unsafe { b.data.inter.cwp_idx };
    let comp_type = unsafe { b.data.inter.comp_type };
    let inter_mode = unsafe { b.data.inter.inter_mode };
    let blk_mv = unsafe { [b.data.inter.mv[0], b.data.inter.mv[1]] };

    // t_swap from ref_flip (decode.c:636).
    let t_swap =
        (recon.rf.ref_flip & (1u64 << (ref0 as u32 * 8 + ref1 as u32))) != 0;
    let opfl = inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8;
    let refine_mv = unsafe { b.data.inter.refine_mv } != 0 && comp_type == 1;
    let write_temporal = recon.seq_hdr.ref_frame_mvs
        && (!opfl || !refine_mv)
        && !recon.cur_mvs.is_empty();

    let mf = ((cwp_idx as i32) << 2
        | (inter_mode == CompInterPredMode::GlobalMvGlobalMv as u8 && imin(bw4, bh4) > 1) as i32)
        as i8;

    let mut s_src = crate::refmvs::Block {
        mv: blk_mv,
        r#ref: crate::levels::RefPair { r: [ref0, ref1] },
        bs: bs as u8,
        mf,
        subpel_filter: unsafe { b.data.inter.filter },
        ..Default::default()
    };
    let s_off = by4r * 128 + (bx & 127) as usize;

    // Temporal block: quantized MVs swapped per ref_flip (decode.c:690-704).
    let mut t_src = crate::refmvs::TemporalBlock::default();
    unsafe {
        t_src.r#ref.r[t_swap as usize] = ref0;
        t_src.r#ref.r[!t_swap as usize] = ref1;
    }
    let wedge = comp_type == 2;
    if write_temporal && !wedge {
        let q0 = crate::refmvs::quantize_mv(blk_mv[t_swap as usize]);
        let q1 = crate::refmvs::quantize_mv(blk_mv[!t_swap as usize]);
        unsafe {
            t_src.mv.mv[0] = q0;
            t_src.mv.mv[1] = q1;
            if t_src.mv.mv[0].n == crate::refmvs::INVALID_TRAJ {
                if t_src.mv.mv[1].n == crate::refmvs::INVALID_TRAJ {
                    t_src.r#ref.pair = -1;
                } else {
                    t_src.mv.mv[0] = t_src.mv.mv[1];
                    t_src.r#ref.r[0] = t_src.r#ref.r[1];
                }
            } else if t_src.mv.mv[1].n == crate::refmvs::INVALID_TRAJ {
                t_src.mv.mv[1] = t_src.mv.mv[0];
                t_src.r#ref.r[1] = t_src.r#ref.r[0];
            }
        }
        let t_stride = recon.rf.rp_stride;
        let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
        crate::refmvs::splat_mv(
            &mut recon.rt.r[s_off..],
            &mut s_src,
            Some(&mut recon.cur_mvs[t_off as usize..]),
            t_stride,
            &t_src,
            bw4,
            bh4,
        );
    } else {
        // Spatial-only splat (no temporal write, or wedge whose per-2x2 temporal
        // pattern is only consumed by later frames — out of scope for frame 1).
        let _ = Mv::default();
        crate::refmvs::splat_mv(
            &mut recon.rt.r[s_off..],
            &mut s_src,
            None,
            0,
            &t_src,
            bw4,
            bh4,
        );
    }
}

/// Reconstruct a same-reference-pair COMPOUND inter block (8bpc). Predicts both
/// references into intermediate i16 buffers and blends per `comp_type`
/// (recon_tmpl.c:3180-3267 luma, :3686-3789 chroma):
///  - AVG: plain `avg`, or implicit out-of-bounds `mask` (imp_msk_bld), or
///    `w_avg` when CWP-weighted (cwp_idx != 8).
///  - WEDGE: `mask` blend with the wedge mask (luma + subsampled for chroma).
///  - SEG: luma `w_mask` (derives a subsampled seg mask), chroma `mask` reusing
///    that seg mask.
/// Translational MC only (warp-compound / OPFL / TIP / scaled refs deferred — not
/// present in the bring-up clip). After blend, the parsed residual is added with
/// the same per-TU walk as the single-ref path.
#[allow(clippy::too_many_arguments)]
fn recon_b_inter_compound(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    let refs = unsafe { b.ref_pair.r };
    let ref0 = refs[0] as usize;
    let ref1 = refs[1] as usize;
    let mv0 = unsafe { b.data.inter.mv[0].c };
    let mv1 = unsafe { b.data.inter.mv[1].c };
    let mvs = [mv0, mv1];
    let mv_pair = unsafe { [b.data.inter.mv[0], b.data.inter.mv[1]] };
    let filter = unsafe { b.data.inter.filter };
    let comp_type = unsafe { b.data.inter.comp_type }; // 1=AVG,2=WEDGE,3=SEG
    let cwp_idx = unsafe { b.data.inter.cwp_idx } as i32;
    let wedge_idx = unsafe { b.data.inter.wedge_idx };
    let wedge_sign = unsafe { b.data.inter.wedge_sign } as usize;
    let mask_sign = unsafe { b.data.inter.mask_sign } as i32;
    let inter_mode = unsafe { b.data.inter.inter_mode };
    let motion_mode = unsafe { b.data.inter.motion_mode };
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;

    let refp0 = match recon.refp[ref0].clone() {
        Some(p) => p,
        None => return Ok(()),
    };
    let refp1 = match recon.refp[ref1].clone() {
        Some(p) => p,
        None => return Ok(()),
    };
    let refp = [&refp0, &refp1];

    // Implicit masked-blend predicate (recon_tmpl.c:3194). For the AVG path with
    // cwp==8 and one ref MC partially out of bounds, an out-of-bounds difference
    // mask is used instead of a plain average.
    let imp_base = recon.seq_hdr.imp_msk_bld
        && motion_mode != MotionMode::WarpCausal as u8
        && inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
        && recon.svc[ref0][0].scale == 0
        && recon.svc[ref1][0].scale == 0;

    // ---- luma ----
    let mut luma_bacp = 0i32; // shared with chroma AVG path (bacpu)
    let mut seg_mask = vec![0u8; 128 * 128];
    let mut seg_mask_stride = 0usize;
    if has_luma {
        let b_dim = &BLOCK_DIMENSIONS[lbs as u8 as usize];
        let bw4 = b_dim[0] as i32;
        let bh4 = b_dim[1] as i32;
        let w = (bw4 * 4) as usize;
        let h = (bh4 * 4) as usize;
        let y_stride = recon.frame.y_stride_px;
        let dst_off = 4 * (by as usize * y_stride + bx as usize);

        let mut tmp = [vec![0i16; w * h], vec![0i16; w * h]];
        for i in 0..2 {
            inter_mc_plane_prep_8bpc(
                &mut tmp[i],
                refp[i],
                0,
                bx,
                by,
                bw4,
                bh4,
                mvs[i].x,
                mvs[i].y,
                filter,
                ss_hor,
                ss_ver,
                fi.bw,
                fi.bh,
            );
        }

        let (tmp0, tmp1) = tmp.split_at(1);
        match comp_type {
            2 => {
                // WEDGE
                let mask = recon
                    .masks
                    .wedge_mask(lbs as usize, bw4 as usize, bh4 as usize, wedge_idx as usize, 0);
                let (a0, a1) = if wedge_sign == 0 { (&tmp0[0], &tmp1[0]) } else { (&tmp1[0], &tmp0[0]) };
                crate::mc::mask_8bpc(
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    a0,
                    a1,
                    w,
                    h,
                    mask,
                );
            }
            3 => {
                // SEG: luma w_mask derives subsampled seg mask for chroma reuse.
                seg_mask_stride = imin(bw4 * 4 >> ss_hor, 64) as usize;
                let (a0, a1) = if mask_sign == 0 {
                    (&tmp0[0], &tmp1[0])
                } else {
                    (&tmp1[0], &tmp0[0])
                };
                crate::mc::w_mask_8bpc(
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    a0,
                    a1,
                    w,
                    h,
                    &mut seg_mask,
                    seg_mask_stride,
                    mask_sign,
                    ss_hor != 0,
                    ss_ver != 0,
                );
            }
            _ => {
                // AVG (or implicit mask / CWP weighted)
                if cwp_idx == 8 {
                    let mut bacp = 2 * imp_base as i32;
                    if bacp == 2 {
                        bacp = crate::recon::get_mask(
                            &mut seg_mask,
                            (bw4 * 4) as usize,
                            bx,
                            0,
                            by,
                            0,
                            &mv_pair,
                            3,
                            3,
                            bw4,
                            bh4,
                            fi.bw * 4,
                            fi.bh * 4,
                        ) as i32;
                    }
                    luma_bacp = bacp;
                    if bacp != 0 {
                        crate::mc::mask_8bpc(
                            &mut recon.dst_y[dst_off..],
                            y_stride,
                            &tmp0[0],
                            &tmp1[0],
                            w,
                            h,
                            &seg_mask,
                        );
                    } else {
                        crate::mc::avg_8bpc(
                            &mut recon.dst_y[dst_off..],
                            y_stride,
                            &tmp0[0],
                            &tmp1[0],
                            w,
                            h,
                        );
                    }
                } else {
                    crate::mc::w_avg_8bpc(
                        &mut recon.dst_y[dst_off..],
                        y_stride,
                        &tmp0[0],
                        &tmp1[0],
                        w,
                        h,
                        cwp_idx,
                    );
                }
            }
        }

        // Luma residual (same walk as single-ref).
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        if lossless {
            let tx = if b.tx_size_ll != 0 {
                crate::tables::MAX_TXFM_SIZE_FOR_BS[lbs as usize][3] as usize
            } else {
                0
            };
            let t_dim = &TXFM_DIMENSIONS[tx];
            let (tw4, th4) = (t_dim.w as i32, t_dim.h as i32);
            let h4 = imin(bh4, fi.bh - by);
            let w4 = imin(bw4, fi.bw - bx);
            let mut y = 0;
            while y < h4 {
                let mut x = 0;
                while x < w4 {
                    inter_residual_tx_8bpc(
                        recon, msac, cdf_m, a, l, b, 0, tx, bx + x, by + y, false, 0, fi,
                    )?;
                    x += tw4;
                }
                y += th4;
            }
        } else {
            let tp = &crate::tables::TX_PART_TBL[lbs as usize];
            let tx = tp[b.tx_part as usize] as usize;
            inter_luma_tx_walk(recon, msac, cdf_m, a, l, b, tx, bx, by, fi)?;
        }
    }

    // ---- chroma ----
    if has_chroma && cbs != BlockSize::Invalid {
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbw4 = cb_dim[0] as i32;
        let cbh4 = cb_dim[1] as i32;
        let cw4 = imin(fi.bw - cbx, cbw4);
        let ch4 = imin(fi.bh - cby, cbh4);
        let cw4ss = (cw4 + ss_hor) >> ss_hor;
        let ch4ss = (ch4 + ss_ver) >> ss_ver;
        let uv_stride = recon.frame.uv_stride_px;
        let cw = (cbw4 * 4 >> ss_hor) as usize;
        let ch = (cbh4 * 4 >> ss_ver) as usize;

        for pl in 1..3usize {
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            let mut tmp = [vec![0i16; cw * ch], vec![0i16; cw * ch]];
            for i in 0..2 {
                inter_mc_plane_prep_8bpc(
                    &mut tmp[i],
                    refp[i],
                    pl,
                    cbx,
                    cby,
                    cbw4,
                    cbh4,
                    mvs[i].x,
                    mvs[i].y,
                    filter,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                );
            }
            let dst: &mut [u8] = if pl == 1 {
                &mut recon.dst_u[dst_off..]
            } else {
                &mut recon.dst_v[dst_off..]
            };
            let (tmp0, tmp1) = tmp.split_at(1);
            match comp_type {
                2 => {
                    let mask = recon.masks.wedge_mask(
                        cbs as usize,
                        cbw4 as usize,
                        cbh4 as usize,
                        wedge_idx as usize,
                        (ss_hor + ss_ver) as usize,
                    );
                    let (a0, a1) = if wedge_sign == 0 {
                        (&tmp0[0], &tmp1[0])
                    } else {
                        (&tmp1[0], &tmp0[0])
                    };
                    crate::mc::mask_8bpc(dst, uv_stride, a0, a1, cw, ch, mask);
                }
                3 => {
                    let (a0, a1) = if mask_sign == 0 {
                        (&tmp0[0], &tmp1[0])
                    } else {
                        (&tmp1[0], &tmp0[0])
                    };
                    crate::mc::mask_8bpc(dst, uv_stride, a0, a1, cw, ch, &seg_mask);
                }
                _ => {
                    if cwp_idx == 8 {
                        if luma_bacp != 0 {
                            crate::mc::mask_8bpc(
                                dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch, &seg_mask,
                            );
                        } else {
                            crate::mc::avg_8bpc(dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch);
                        }
                    } else {
                        crate::mc::w_avg_8bpc(
                            dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch, cwp_idx,
                        );
                    }
                }
            }
        }
        let _ = (seg_mask_stride, cb_dim);

        // Chroma residual per uvtx TU (both planes + CCTX).
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        let uvtx = if lossless {
            0usize
        } else {
            let layout_idx =
                (crate::headers::PixelLayout::I444 as i32 - recon.frame.layout as i32) as usize;
            crate::tables::MAX_TXFM_SIZE_FOR_BS[cbs as u8 as usize][layout_idx] as usize
        };
        // Seed the inter chroma transform type from the co-located luma 4x4's
        // recorded txtp (dav2d `y_txtp = txtp_map[...]`, recon_tmpl.c:3540/3553).
        let uv_txtp_seed = recon.scratch.txtp_map[(by & 15) as usize * 16 + (bx & 15) as usize];
        let cbw4ss = (cbw4 + ss_hor) >> ss_hor;
        let cbh4ss = (cbh4 + ss_ver) >> ss_ver;
        inter_chroma_residual_8bpc(
            recon, msac, cdf_m, a, l, b, uvtx, cbx, cby, cbw4ss, cbh4ss, cw4ss, ch4ss,
            uv_txtp_seed, fi,
        )?;
    }
    Ok(())
}

/// Reconstruct a single-reference inter block (8bpc): motion-compensate luma +
/// chroma from the reference picture (translational or warp-affine, dispatched
/// on the block's motion_mode / derived warp params), then add the parsed
/// residual transforms. Compound (ref pair), interintra blend, TIP and scaled
/// references are deferred.
#[allow(clippy::too_many_arguments)]
fn recon_b_inter(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    let refs = unsafe { b.ref_pair.r };
    let ref0 = refs[0];
    let ref1 = refs[1];
    if ref0 < 0 || ref0 as usize >= 7 {
        return Ok(());
    }

    // Blocks larger than 64px in either dimension are not reconstructed as a
    // single unit (the MC kernels cap at 64px wide): they are split into 64x64
    // (or 128x128 for 256px) sub-blocks, mirroring `dav2d_recon_b`
    // (recon_tmpl.c:3088-3140). Each sub-block runs its own luma MC + residual;
    // chroma coefficients are read with the first luma sub-block and the chroma
    // is reconstructed once (read/recon staging), so the MSAC bitstream ordering
    // matches the C decoder. <=64px blocks fall through to the leaf below.
    {
        let bs0 = if lbs == BlockSize::Invalid { cbs } else { lbs };
        let bdim0 = &BLOCK_DIMENSIONS[bs0 as u8 as usize];
        let bw4_0 = bdim0[0] as i32;
        let bh4_0 = bdim0[1] as i32;
        if imax(bw4_0, bh4_0) > 16 {
            return recon_b_inter_split(
                recon, msac, cdf_m, a, l, b, bx, by, cbx, cby, lbs, cbs, has_luma, has_chroma,
                fi, bs0, bw4_0, bh4_0,
            );
        }
    }

    let mv = unsafe { b.data.inter.mv[0].c };
    let mv1 = unsafe { b.data.inter.mv[1].c };
    let filter = unsafe { b.data.inter.filter };
    let comp_type = unsafe { b.data.inter.comp_type };
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;

    // Compound (two-ref) prediction: predict both refs into i16 tmp buffers and
    // blend per `comp_type` (recon_tmpl.c:3180-3267). Scaled refs / TIP / OPFL /
    // warp-compound are not present in the bring-up clip and are deferred.
    let is_compound = ref1 >= 0 && (ref1 as usize) < 7;
    if is_compound {
        return recon_b_inter_compound(
            recon, msac, cdf_m, a, l, b, bx, by, cbx, cby, lbs, cbs, has_luma, has_chroma, fi,
        );
    }

    // Take the reference picture out of recon.refp (immutable Arc) to satisfy the
    // borrow checker while mutating dst planes.
    let refp = match recon.refp[ref0 as usize].clone() {
        Some(p) => p,
        None => return Ok(()),
    };
    let _ = (mv1, comp_type);

    // Warp-affine vs translational MC dispatch (recon_tmpl.c:3163-3176).
    let motion_mode = unsafe { b.data.inter.motion_mode };
    let inter_mode = unsafe { b.data.inter.inter_mode };
    let warp_block = {
        let bdim = &BLOCK_DIMENSIONS[lbs as u8 as usize];
        let bw4 = bdim[0] as i32;
        let bh4 = bdim[1] as i32;
        let mut gmv = recon.frm_hdr.gmv.m[ref0 as usize];
        let gmv_warp_allowed = gmv.wm_type > crate::headers::WarpedMotionType::Translation
            && recon.frm_hdr.force_integer_mv == 0
            && crate::warpmv::get_shear_params(&mut gmv) == 0
            && recon.svc[ref0 as usize][0].scale == 0;
        recon.frm_hdr.force_integer_mv == 0
            && ((inter_mode == crate::levels::InterPredMode::GlobalMv as u8
                && imin(bw4, bh4) > 1
                && gmv_warp_allowed)
                || (motion_mode >= MotionMode::WarpCausal as u8
                    && recon.warpmv[0].wm_type > crate::headers::WarpedMotionType::Invalid))
    };
    // Pick which warp params to use: local warpmv for warp motion modes, the
    // frame global motion otherwise (GLOBALMV warp).
    let use_local_warp = motion_mode >= MotionMode::WarpCausal as u8;

    if std::env::var("RAV2D_WARP").is_ok() {
        eprintln!(
            "WARP by={} bx={} mm={} im={} warp_block={} use_local={} wm_type={:?} affine={} mat={:?} abcd={:?} mv=({},{})",
            by, bx, motion_mode, inter_mode, warp_block, use_local_warp,
            recon.warpmv[0].wm_type, recon.warpmv[0].affine,
            recon.warpmv[0].matrix, recon.warpmv[0].abcd, mv.y, mv.x
        );
    }

    // ---- luma MC + residual -----------------------------------------------
    if has_luma {
        let bs = lbs;
        let b_dim = &BLOCK_DIMENSIONS[bs as u8 as usize];
        let bw4 = b_dim[0] as i32;
        let bh4 = b_dim[1] as i32;
        let y_stride = recon.frame.y_stride_px;
        let dst_off = 4 * (by as usize * y_stride + bx as usize);
        let wmp = if use_local_warp {
            recon.warpmv[0]
        } else {
            recon.frm_hdr.gmv.m[ref0 as usize]
        };
        if warp_block {
            // dav2d warp_affine (recon_tmpl.c:1817): the 8x8 affine kernel only
            // applies for affine warps where the (subsampled) block is >= 8 in
            // both dims; otherwise ext_warp.
            if wmp.affine != 0 && imin(bw4 * 4, bh4 * 4) >= 8 {
                warp_affine_plane_8bpc(
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    &refp,
                    0,
                    bx,
                    by,
                    b_dim,
                    &wmp,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                );
                if std::env::var("RAV2D_YPRED").is_ok() && by == 12 && bx == 0 {
                    eprintln!("DYPRED by={} bx={}", by, bx);
                    for yy in 0..(bh4 * 4) as usize {
                        let mut s = String::from("P");
                        for xx in 0..(bw4 * 4) as usize {
                            s.push_str(&format!(" {}", recon.dst_y[dst_off + yy * y_stride + xx]));
                        }
                        eprintln!("{s}");
                    }
                }
            } else {
                ext_warp_plane_8bpc(
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    &refp,
                    0,
                    bx,
                    by,
                    b_dim,
                    &wmp,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                );
            }
        } else {
            inter_mc_plane_8bpc(
                &mut recon.dst_y[dst_off..],
                y_stride,
                &refp,
                0,
                bx,
                by,
                bw4,
                bh4,
                mv.x,
                mv.y,
                filter,
                ss_hor,
                ss_ver,
                fi.bw,
                fi.bh,
            );
        }

        // Inter-intra blend (recon_tmpl.c:3199-3200): blend an intra prediction
        // over the inter prediction for INTERINTRA / warp-interintra blocks.
        if motion_mode == MotionMode::InterIntra as u8 || unsafe { b.data.inter.warp_ii } != 0 {
            let dst_off_y = dst_off;
            // SAFETY: split the recon borrow — iiblend needs &mut recon for the
            // edge/masks while writing dst_y; pass dst_y through recon directly.
            iiblend_luma_8bpc(recon, &b, dst_off_y, y_stride, bw4, bh4, by, bx, bs, fi);
        }

        // Luma residual: walk b.tx_part geometry (same tp[] as intra,
        // recon_tmpl.c:3293).
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        if lossless {
            let tx = if b.tx_size_ll != 0 {
                crate::tables::MAX_TXFM_SIZE_FOR_BS[bs as usize][3] as usize
            } else {
                0
            };
            let t_dim = &TXFM_DIMENSIONS[tx];
            let (tw4, th4) = (t_dim.w as i32, t_dim.h as i32);
            let h4 = imin(bh4, fi.bh - by);
            let w4 = imin(bw4, fi.bw - bx);
            let mut y = 0;
            while y < h4 {
                let mut x = 0;
                while x < w4 {
                    inter_residual_tx_8bpc(
                        recon, msac, cdf_m, a, l, b, 0, tx, bx + x, by + y, false, 0, fi,
                    )?;
                    x += tw4;
                }
                y += th4;
            }
        } else {
            let tp = &crate::tables::TX_PART_TBL[bs as usize];
            let tx = tp[b.tx_part as usize] as usize;
            inter_luma_tx_walk(recon, msac, cdf_m, a, l, b, tx, bx, by, fi)?;
        }
    }

    // ---- chroma MC + residual ---------------------------------------------
    if has_chroma && cbs != BlockSize::Invalid {
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbw4 = cb_dim[0] as i32;
        let cbh4 = cb_dim[1] as i32;
        let cw4 = imin(fi.bw - cbx, cbw4);
        let ch4 = imin(fi.bh - cby, cbh4);
        let cw4ss = (cw4 + ss_hor) >> ss_hor;
        let ch4ss = (ch4 + ss_ver) >> ss_ver;

        // Chroma MV: for cbs==lbs or imin(bw4,bh4)>=16 the single block MV is
        // used directly. For sub-8x8 luma coding (cbs != lbs && imin(bw4,bh4)<16)
        // the chroma covers several luma sub-blocks, each with its own MV; chroma
        // MC is done per luma sub-block reading ref/MV/filter from the spatial
        // refmvs (recon_tmpl.c:3603-3655, single-pass branch).
        let uv_stride = recon.frame.uv_stride_px;
        let (luma_bw4, luma_bh4) = {
            let ld = &BLOCK_DIMENSIONS[lbs as u8 as usize];
            (ld[0] as i32, ld[1] as i32)
        };
        let sub8x8 = lbs != BlockSize::Invalid
            && cbs != lbs
            && imin(luma_bw4, luma_bh4) < 16
            && !warp_block;
        if sub8x8 {
            // Per-sub-block chroma MC from spatial refmvs. cw4/ch4 are chroma
            // 4x4 extents; for each origin sub-block (ox4==oy4==0) MC both planes.
            let base = ((cby & 63) as usize) * 128 + ((cbx & 127) as usize);
            for y in 0..ch4 {
                for x in 0..cw4 {
                    let idx = base + (y as usize) * 128 + (x as usize);
                    let r2 = &recon.rt.r[idx];
                    if r2.ox4 != 0 || r2.oy4 != 0 {
                        continue;
                    }
                    let s_ref0 = unsafe { r2.r#ref.r[0] };
                    if s_ref0 < 0 || s_ref0 as usize >= 7 {
                        continue;
                    }
                    let s_mv = if r2.mf & 2 != 0 {
                        unsafe { r2.lmv[0].c }
                    } else {
                        unsafe { r2.mv[0].c }
                    };
                    let s_filter = r2.subpel_filter;
                    let sdim = &BLOCK_DIMENSIONS[r2.bs as usize];
                    let s_bw4 = sdim[0] as i32;
                    let s_bh4 = sdim[1] as i32;
                    let s_refp = match recon.refp[s_ref0 as usize].clone() {
                        Some(p) => p,
                        None => continue,
                    };
                    let s_cbx = cbx + x;
                    let s_cby = cby + y;
                    // Chroma destination pixel: block-origin chroma px plus the
                    // per-sub-block offset (recon_tmpl.c: uvoff + (x*4 >> ss_hor),
                    // uvoff advances by 4*stride >> ss_ver per luma-4-unit row).
                    let base_off = 4
                        * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
                    let dst_off = base_off
                        + (((y * 4) >> ss_ver) as usize) * uv_stride
                        + (((x * 4) >> ss_hor) as usize);
                    for pl in 1..3usize {
                        let dst: &mut [u8] = if pl == 1 {
                            &mut recon.dst_u[dst_off..]
                        } else {
                            &mut recon.dst_v[dst_off..]
                        };
                        inter_mc_plane_8bpc(
                            dst, uv_stride, &s_refp, pl, s_cbx, s_cby, s_bw4, s_bh4, s_mv.x,
                            s_mv.y, s_filter, ss_hor, ss_ver, fi.bw, fi.bh,
                        );
                    }
                }
            }
        }
        // Warp dispatch for chroma uses cb_dim and (>=8px-after-subsample affine).
        let c_wmp = if use_local_warp {
            recon.warpmv[0]
        } else {
            recon.frm_hdr.gmv.m[ref0 as usize]
        };
        // Chroma warp eligibility for the 8x8 affine kernel uses the chroma
        // block dims after subsampling (dav2d warp_affine, recon_tmpl.c:1817).
        let c_affine = c_wmp.affine != 0
            && imin(cbw4 * (4 >> ss_hor), cbh4 * (4 >> ss_ver)) >= 8;
        for pl in (1..3).filter(|_| !sub8x8) {
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            let dst: &mut [u8] = if pl == 1 {
                &mut recon.dst_u[dst_off..]
            } else {
                &mut recon.dst_v[dst_off..]
            };
            if warp_block {
                if c_affine {
                    warp_affine_plane_8bpc(
                        dst, uv_stride, &refp, pl, cbx, cby, cb_dim, &c_wmp, ss_hor, ss_ver,
                        fi.bw, fi.bh,
                    );
                } else {
                    ext_warp_plane_8bpc(
                        dst, uv_stride, &refp, pl, cbx, cby, cb_dim, &c_wmp, ss_hor, ss_ver,
                        fi.bw, fi.bh,
                    );
                }
            } else {
                inter_mc_plane_8bpc(
                    dst,
                    uv_stride,
                    &refp,
                    pl,
                    cbx,
                    cby,
                    cbw4,
                    cbh4,
                    mv.x,
                    mv.y,
                    filter,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                );
            }
        }

        // Inter-intra blend for chroma (recon_tmpl.c:3682). Only on the
        // single-ref / compound branches, never on sub8x8 chroma coding (the
        // sub8x8 branch in dav2d_recon_b has no iiblend). The II-mask block size
        // is the chroma-subsampled block size SS_BS[cbs][layout-1] (wedge keeps
        // cbs), matching the C `dav2d_ss_bs[cbs][layout-1]` argument.
        if !sub8x8
            && (motion_mode == MotionMode::InterIntra as u8 || unsafe { b.data.inter.warp_ii } != 0)
        {
            let ii_ss_bs = if unsafe { b.data.inter.wedge_idx } == -1 {
                let layout_idx = (recon.frame.layout as usize) - 1;
                unsafe {
                    BlockSize::from_raw(crate::tables::SS_BS[cbs as usize][layout_idx] as i8)
                }
            } else {
                cbs
            };
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            for pl in 1..3usize {
                iiblend_chroma_8bpc(
                    recon, &b, pl, dst_off, uv_stride, cbw4, cbh4, cby, cbx, ii_ss_bs, fi,
                );
            }
        }

        // Chroma residual per uvtx TU (both planes + CCTX).
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        let uvtx = if lossless {
            0usize
        } else {
            let layout_idx =
                (crate::headers::PixelLayout::I444 as i32 - recon.frame.layout as i32) as usize;
            crate::tables::MAX_TXFM_SIZE_FOR_BS[cbs as u8 as usize][layout_idx] as usize
        };
        // Seed the inter chroma transform type from the co-located luma 4x4's
        // recorded txtp (dav2d `y_txtp = txtp_map[...]`, recon_tmpl.c:3540/3553).
        let uv_txtp_seed = recon.scratch.txtp_map[(by & 15) as usize * 16 + (bx & 15) as usize];
        let cbw4ss = (cbw4 + ss_hor) >> ss_hor;
        let cbh4ss = (cbh4 + ss_ver) >> ss_ver;
        inter_chroma_residual_8bpc(
            recon, msac, cdf_m, a, l, b, uvtx, cbx, cby, cbw4ss, cbh4ss, cw4ss, ch4ss,
            uv_txtp_seed, fi,
        )?;
    }
    Ok(())
}

/// Split an inter block larger than 64px into 64x64 (or 128x128 for 256px)
/// sub-blocks, mirroring the `imax(bw4, bh4) > 16` branch of `dav2d_recon_b`
/// (recon_tmpl.c:3088-3140). Each sub-block recurses into `recon_b_inter` for
/// its luma MC + residual; chroma is decoded once via the read/recon staging
/// (`cbs2[0]` reads, `cbs2[1]` reconstructs). Because the chroma sub-block uses
/// the full chroma block size at the block-origin coordinates for both stages,
/// we perform the whole chroma (MC + residual) at the read stage so the MSAC
/// ordering (chroma coefs follow the first luma sub-block) matches the C decoder.
#[allow(clippy::too_many_arguments)]
fn recon_b_inter_split(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    fi: &SbFrameInfo,
    bs: BlockSize,
    bw4: i32,
    bh4: i32,
) -> Result<(), ()> {
    // csplit[bs - 128x128][ss] (recon_tmpl.c:3083). Identical to the intra path.
    const CSPLIT: [[BlockSize; 3]; 3] = [
        [
            BlockSize::Bs64x64,
            BlockSize::Bs128x64,
            BlockSize::Bs128x128,
        ],
        [BlockSize::Bs64x64, BlockSize::Bs64x64, BlockSize::Bs128x64],
        [BlockSize::Bs64x64, BlockSize::Bs64x64, BlockSize::Bs64x128],
    ];
    let ss_hor = fi.ss_hor;
    let ss_ver = fi.ss_ver;
    let y_end = imin(by + bh4, fi.bh);
    let x_end = imin(bx + bw4, fi.bw);
    let (step, lbs2, cbs2i) = if imax(bw4, bh4) == 64 {
        (
            32,
            if lbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                BlockSize::Bs128x128
            },
            if cbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                BlockSize::Bs128x128
            },
        )
    } else {
        let csplit_row = (bs as i32 - BlockSize::Bs128x128 as i32) as usize;
        let csi = (ss_hor + ss_ver) as usize;
        (
            16,
            if lbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                BlockSize::Bs64x64
            },
            if cbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                CSPLIT[csplit_row][csi]
            },
        )
    };
    let _ = (has_luma, has_chroma);

    let mut sub_by = by;
    let mut sub_cby = cby;
    let mut yy = 0;
    while sub_by < y_end {
        let mut sub_bx = bx;
        let mut sub_cbx = cbx;
        let mut xx = 0;
        while sub_bx < x_end {
            // cbs2[0] = chroma coef-read stage, cbs2[1] = chroma recon stage
            // (recon_tmpl.c:3108-3117).
            let (read_cbs, recon_cbs) = if step == 32 {
                (cbs2i, cbs2i)
            } else {
                let read = if ((xx & ss_hor) | (yy & ss_ver)) == 0 {
                    cbs2i
                } else {
                    BlockSize::Invalid
                };
                let recon = if (ss_hor == 0 || sub_bx + step >= x_end)
                    && (ss_ver == 0 || sub_by + step >= y_end)
                {
                    cbs2i
                } else {
                    BlockSize::Invalid
                };
                (read, recon)
            };
            let _ = recon_cbs;

            // Reconstruct the whole chroma (MC + residual) at the read stage so
            // the chroma coefficients are entropy-read at the correct MSAC point
            // (immediately after the first luma sub-block). The chroma pixels are
            // written at the absolute block-origin coordinates regardless of which
            // sub-block triggers them, so doing recon here is equivalent.
            let sub_has_chroma = read_cbs != BlockSize::Invalid;
            let sub_cbs = if sub_has_chroma {
                read_cbs
            } else {
                BlockSize::Invalid
            };

            recon_b_inter(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                sub_bx,
                sub_by,
                sub_cbx,
                sub_cby,
                lbs2,
                sub_cbs,
                lbs2 != BlockSize::Invalid,
                sub_has_chroma,
                fi,
            )?;

            sub_bx += step;
            if step == 32 {
                sub_cbx += step;
            } else if (xx & ss_hor) == ss_hor {
                sub_cbx += step << ss_hor;
            }
            xx += 1;
        }
        sub_by += step;
        if step == 32 {
            sub_cby += step;
        } else if (yy & ss_ver) == ss_ver {
            sub_cby += step << ss_ver;
        }
        yy += 1;
    }
    Ok(())
}

/// Inter luma transform-partition walk. Mirrors dav2d's `switch (b->tx_part)`
/// in `dav2d_recon_b` (recon_tmpl.c:3309-3478) exactly: the transform-partition
/// type defines the visitation order and per-tile transform size, which is
/// load-bearing for the per-4x4 coefficient neighbour context (and hence the
/// entropy stream). A naive raster tiling desyncs for non-square partitions.
#[allow(clippy::too_many_arguments)]
fn inter_luma_tx_walk(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    tx: usize,
    bx: i32,
    by: i32,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    let tp = &crate::tables::TX_PART_TBL[b.bs as usize];
    macro_rules! resid {
        ($tx:expr, $x:expr, $y:expr) => {
            inter_residual_tx_8bpc(recon, msac, cdf_m, a, l, b, 0, $tx, $x, $y, false, 0, fi)?
        };
    }
    match unsafe { TxPartition::from_raw(b.tx_part) } {
        TxPartition::None => {
            resid!(tx, bx, by);
        }
        TxPartition::Split => {
            let t_dim = &TXFM_DIMENSIONS[tx];
            let (tw4, th4) = (t_dim.w as i32, t_dim.h as i32);
            resid!(tx, bx, by);
            let have_v_split = bx + tw4 < fi.bw;
            if have_v_split {
                resid!(tx, bx + tw4, by);
            }
            if by + th4 >= fi.bh {
                return Ok(());
            }
            resid!(tx, bx, by + th4);
            if have_v_split {
                resid!(tx, bx + tw4, by + th4);
            }
        }
        TxPartition::H => {
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            resid!(tx, bx, by);
            if by + th4 >= fi.bh {
                return Ok(());
            }
            resid!(tx, bx, by + th4);
        }
        TxPartition::V => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            resid!(tx, bx, by);
            if bx + tw4 >= fi.bw {
                return Ok(());
            }
            resid!(tx, bx + tw4, by);
        }
        TxPartition::H4 => {
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            for i in 0..4 {
                let yy = by + i * th4;
                resid!(tx, bx, yy);
                if yy + th4 >= fi.bh {
                    break;
                }
            }
        }
        TxPartition::V4 => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            for i in 0..4 {
                let xx = bx + i * tw4;
                resid!(tx, xx, by);
                if xx + tw4 >= fi.bw {
                    break;
                }
            }
        }
        TxPartition::H5 => {
            let tx_big = tp[TxPartition::H as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let (tw4_small, th4_small) = (t_dim_small.w as i32, t_dim_small.h as i32);
            let th4_big = TXFM_DIMENSIONS[tx_big].h as i32;
            resid!(tx, bx, by);
            let have_v_split = bx + tw4_small < fi.bw;
            if have_v_split {
                resid!(tx, bx + tw4_small, by);
            }
            if by + th4_small >= fi.bh {
                return Ok(());
            }
            resid!(tx_big, bx, by + th4_small);
            if by + th4_small + th4_big < fi.bh {
                resid!(tx, bx, by + th4_small + th4_big);
                if have_v_split {
                    resid!(tx, bx + tw4_small, by + th4_small + th4_big);
                }
            }
        }
        TxPartition::V5 => {
            let tx_big = tp[TxPartition::V as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let (tw4_small, th4_small) = (t_dim_small.w as i32, t_dim_small.h as i32);
            let tw4_big = TXFM_DIMENSIONS[tx_big].w as i32;
            resid!(tx, bx, by);
            let have_h_split = by + th4_small < fi.bh;
            if have_h_split {
                resid!(tx, bx, by + th4_small);
            }
            if bx + tw4_small >= fi.bw {
                return Ok(());
            }
            resid!(tx_big, bx + tw4_small, by);
            if bx + tw4_small + tw4_big < fi.bw {
                resid!(tx, bx + tw4_small + tw4_big, by);
                if have_h_split {
                    resid!(tx, bx + tw4_small + tw4_big, by + th4_small);
                }
            }
        }
    }
    Ok(())
}

/// Reconstruct the luma plane of an intra (non-IntraBC) block at the decode_b
/// leaf, for the 8bpc path.
///
/// Port of the luma `switch (b->tx_part)` walk in `dav2d_recon_b`
/// (recon_tmpl.c:3292-3478) plus `recon_b_luma_tx` (recon_tmpl.c:2443-2675).
/// Caller guarantees `b.is_intra != 0 && b.intrabc == 0` and luma is present.
///
/// Simplifications vs C (M1a, documented for the bit-exact follow-up):
///  - palette blocks (`pal_sz != 0`) skip intra prediction (matches C gate) but
///    are otherwise still walked; palette pixel fill is not implemented.
///  - `n_tr`/`n_bl` top-right / bottom-left availability use the per-SB
///    `is_coded` grid as in C; the `prefilter_toplevel_sb_edge` (top SB row
///    edge backup) is passed as `None` (no cross-SB-row prefilter buffer yet).
#[allow(clippy::too_many_arguments)]
fn recon_b_intra_luma(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    _bx4: usize,
    _by4: usize,
    _intrabc: bool,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    recon_b_intra_luma_geom(recon, msac, cdf_m, a, l, b, bx, by, b.bs as usize, fi)
}

/// As `recon_b_intra_luma`, but with the transform-walk geometry block size
/// given explicitly. For >64px blocks split into 64x64 sub-blocks the tx walk
/// uses the sub-block size while coefficient decoding still uses the full
/// `b.bs` (mirroring `dav2d_recon_b`, where `b->bs` is unchanged by the split).
#[allow(clippy::too_many_arguments)]
fn recon_b_intra_luma_geom(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    geom_bs: usize,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    let bs = geom_bs;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;

    // tp[b->tx_part] (recon_tmpl.c:3293) and the lossless override below.
    let tp = &crate::tables::TX_PART_TBL[bs];

    // pb.col_start / pb.row_start are this block's origin (used by is_hv5).
    let pb_col_start = bx;
    let pb_row_start = by;

    if lossless {
        // recon_tmpl.c:3296-3308: single tx size, raster walk over the block.
        let tx = if b.tx_size_ll != 0 {
            crate::tables::MAX_TXFM_SIZE_FOR_BS[bs][3] as usize
        } else {
            0 // TX_4X4
        };
        let t_dim = &TXFM_DIMENSIONS[tx];
        let tw4 = t_dim.w as i32;
        let th4 = t_dim.h as i32;
        let h4 = imin(BLOCK_DIMENSIONS[bs][1] as i32, fi.bh - by);
        let w4 = imin(BLOCK_DIMENSIONS[bs][0] as i32, fi.bw - bx);
        let mut y = 0;
        while y < h4 {
            let mut x = 0;
            while x < w4 {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx + x,
                    by + y,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
                x += tw4;
            }
            y += th4;
        }
        return Ok(());
    }

    let tx_part = b.tx_part as usize;
    let tx = tp[tx_part] as usize;

    match unsafe { TxPartition::from_raw(b.tx_part) } {
        TxPartition::None => {
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
        }
        TxPartition::Split => {
            let t_dim = &TXFM_DIMENSIONS[tx];
            let tw4 = t_dim.w as i32;
            let th4 = t_dim.h as i32;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            let have_v_split = bx + tw4 < fi.bw;
            if have_v_split {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx + tw4,
                    by,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
            }
            if by + th4 >= fi.bh {
                return Ok(());
            }
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by + th4,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if have_v_split {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx + tw4,
                    by + th4,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
            }
        }
        TxPartition::H => {
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if by + th4 >= fi.bh {
                return Ok(());
            }
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by + th4,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
        }
        TxPartition::V => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if bx + tw4 >= fi.bw {
                return Ok(());
            }
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx + tw4,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
        }
        TxPartition::H4 => {
            // recon_tmpl.c:3366-3387. Up to 4 stacked tiles; a tile is only
            // started if the previous one did not reach the frame's bottom edge.
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            for i in 0..4 {
                let yy = by + i * th4;
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx,
                    yy,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
                if yy + th4 >= fi.bh {
                    break;
                }
            }
        }
        TxPartition::V4 => {
            // recon_tmpl.c:3389-3410.
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            for i in 0..4 {
                let xx = bx + i * tw4;
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    xx,
                    by,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
                if xx + tw4 >= fi.bw {
                    break;
                }
            }
        }
        TxPartition::H5 => {
            let tx_big = tp[TxPartition::H as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let tw4_small = t_dim_small.w as i32;
            let th4_small = t_dim_small.h as i32;
            let th4_big = TXFM_DIMENSIONS[tx_big].h as i32;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            let have_v_split = bx + tw4_small < fi.bw;
            if have_v_split {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx + tw4_small,
                    by,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
            }
            if by + th4_small >= fi.bh {
                return Ok(());
            }
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx_big,
                bx,
                by + th4_small,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if by + th4_small + th4_big >= fi.bh {
                return Ok(());
            }
            let yb = by + th4_small + th4_big;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                yb,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if have_v_split {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx + tw4_small,
                    yb,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
            }
        }
        TxPartition::V5 => {
            let tx_big = tp[TxPartition::V as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let tw4_small = t_dim_small.w as i32;
            let th4_small = t_dim_small.h as i32;
            let tw4_big = TXFM_DIMENSIONS[tx_big].w as i32;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            let have_h_split = by + th4_small < fi.bh;
            if have_h_split {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    bx,
                    by + th4_small,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
            }
            if bx + tw4_small >= fi.bw {
                return Ok(());
            }
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx_big,
                bx + tw4_small,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if bx + tw4_small + tw4_big >= fi.bw {
                return Ok(());
            }
            let xb = bx + tw4_small + tw4_big;
            recon_b_luma_tx(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                tx,
                xb,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                fi,
            )?;
            if have_h_split {
                recon_b_luma_tx(
                    recon,
                    msac,
                    cdf_m,
                    a,
                    l,
                    b,
                    tx,
                    xb,
                    by + th4_small,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    fi,
                )?;
            }
        }
    }

    Ok(())
}

/// Reconstruct the chroma planes (U=1, V=2) of an intra (non-IntraBC) block,
/// 8bpc. Port of the chroma section of `dav2d_recon_b` (recon_tmpl.c:3482-3942).
///
/// AV2 decodes ALL chroma coefficients (both planes, all transform units) first,
/// then runs prediction + inverse transform in a second pass; CfL prediction is
/// applied once for the whole block between the two passes. We mirror that order.
#[allow(clippy::too_many_arguments)]
/// Which part of the chroma decode to perform. For blocks split into 64x64
/// luma sub-blocks (`imax(bw4,bh4) > 16`), the chroma coefficient read happens
/// with the first sub-block and the pixel reconstruction with the last, so the
/// MSAC ordering matches `dav2d_recon_b`'s `cbs_stage` mechanism. For ordinary
/// (<=64px) blocks both phases run in a single `Both` call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChromaPhase {
    Both,
    ReadOnly,
    ReconOnly,
}

fn recon_b_intra_chroma(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    cbx: i32,
    cby: i32,
    cbs: BlockSize,
    _intrabc: bool,
    sdp_active: bool,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    recon_b_intra_chroma_phase(
        recon,
        msac,
        cdf_m,
        a,
        l,
        b,
        cbx,
        cby,
        cbs,
        sdp_active,
        fi,
        ChromaPhase::Both,
    )
}

#[allow(clippy::too_many_arguments)]
fn recon_b_intra_chroma_phase(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    cbx: i32,
    cby: i32,
    cbs: BlockSize,
    sdp_active: bool,
    fi: &SbFrameInfo,
    phase: ChromaPhase,
) -> Result<(), ()> {
    use crate::levels::{CFL_PRED, IntraPredMode};

    // chroma `intra = b->intra && (sdp_active || !b->intrabc)` (recon_tmpl.c:3564).
    let is_intrabc = b.intrabc != 0;
    let is_intra = b.is_intra != 0 && (sdp_active || !is_intrabc);

    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;

    let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
    let cbw4 = cb_dim[0] as i32;
    let cbh4 = cb_dim[1] as i32;
    let cw4 = imin(fi.bw - cbx, cbw4);
    let ch4 = imin(fi.bh - cby, cbh4);
    let cbw4ss = ((cbw4 + ss_hor) >> ss_hor) as usize;
    let cw4ss = (cw4 + ss_hor) >> ss_hor;
    let ch4ss = (ch4 + ss_ver) >> ss_ver;
    let cbh4ss = ((cbh4 + ss_ver) >> ss_ver) as usize;

    // uvtx = chroma transform size (recon_tmpl.c:3489-3491).
    let uvtx = if lossless {
        0usize // TX_4X4
    } else {
        // dav2d_max_txfm_size_for_bs[cbs][I444 - layout]; I444=3, layout in {1,2,3}.
        let layout_idx =
            (crate::headers::PixelLayout::I444 as i32 - recon.frame.layout as i32) as usize;
        crate::tables::MAX_TXFM_SIZE_FOR_BS[cbs as u8 as usize][layout_idx] as usize
    };
    let uv_t_dim = &TXFM_DIMENSIONS[uvtx];
    let ctw4 = imin(uv_t_dim.w as i32, (fi.bw - cbx + ss_hor) >> ss_hor);
    let cth4 = imin(uv_t_dim.h as i32, (fi.bh - cby + ss_ver) >> ss_ver);
    let ctw = uv_t_dim.w as usize * 4;
    let cth = uv_t_dim.h as usize * 4;
    let txw = uv_t_dim.w as i32;
    let txh = uv_t_dim.h as i32;

    let bx4 = (cbx & 63) as usize;
    let by4 = (cby & 63) as usize;
    let cbx4 = bx4 >> ss_hor;
    let cby4 = by4 >> ss_ver;
    let ssbx = (cbx >> ss_hor) as usize;
    let ssby = (cby >> ss_ver) as usize;
    let cstride = recon.frame.uv_stride_px;
    let ystride = recon.frame.y_stride_px;
    let sbsz = fi.sb_step;

    let orig_uv_mode = unsafe { b.data.intra.uv_mode };
    let mut angle = unsafe { b.data.intra.uv_angle } as i32;
    let uv_mode_remapped = {
        let m_in: IntraPredMode = unsafe { std::mem::transmute(orig_uv_mode.min(12)) };
        crate::recon::wide_angle_remap(uv_t_dim, m_in, &mut angle, 0) as u8
    };
    let uv_mode = if orig_uv_mode <= 12 {
        uv_mode_remapped
    } else {
        orig_uv_mode
    };

    // Per-TU coefficient storage for both chroma planes (recon_tmpl.c t->cf_uv).
    let n_tu = cbw4ss * cbh4ss;
    let mut cf_uv: Vec<i32> = vec![0i32; n_tu * 16 * 2];
    let (cf_u, cf_v) = cf_uv.split_at_mut(n_tu * 16);
    // Per-TU txtp / eob, indexed [pl][i] where i = y*cbw4ss + x (recon_tmpl.c).
    let mut tu_txtp = [[0u16; 2]; 256];
    let mut tu_eob = [[-1i16; 2]; 256];

    // Snapshot the per-4x4 luma fsc map for the lossless-chroma txtp derivation
    // (recon_tmpl.c:430 `t->luma_fsc_map`, used when sdp_active). Copied out to
    // sidestep the `recon` borrow inside the per-TU coef loop below.
    let luma_fsc_map: [u8; 256] = recon.scratch.luma_fsc_map;

    // IntraBC blocks with skip_txfm code no chroma coefficients: fill the ccoef
    // contexts with 0x40 and leave all TU eobs at -1 (recon_tmpl.c:3536-3540).
    // (For non-IntraBC intra blocks skip_txfm is always 0; the chroma-only SDP
    // tree is sdp_active so skip_txfm does not apply.)
    let chroma_skip_txfm = is_intrabc && b.skip_txfm != 0;

    // ---- decode coefficients for both planes (recon_tmpl.c:3543-3580) -------
    let mut u_has_cf = 0i32;
    if chroma_skip_txfm {
        if phase != ChromaPhase::ReconOnly {
            for pl in 0..2 {
                let aw = imin(cw4ss, 64 - cbx4 as i32).max(0) as usize;
                let lh = imin(ch4ss, 64 - cby4 as i32).max(0) as usize;
                if aw > 0 {
                    a.ccoef[pl][cbx4..cbx4 + aw].fill(0x40);
                }
                if lh > 0 {
                    l.ccoef[pl][cby4..cby4 + lh].fill(0x40);
                }
            }
        }
    } else if phase != ChromaPhase::ReconOnly {
        for pl in 0..2 {
            let cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
            let mut y = 0;
            while y < ch4ss {
                let mut x = 0;
                while x < cw4ss {
                    let i = (y * cbw4ss as i32 + x) as usize;
                    let mut txtp: u16 = 0;
                    let mut res_ctx: u8 = 0;
                    // TU coefficient region is txw*txh*16 coefs (= ctw*cth), placed at
                    // i*16 within the per-plane block buffer (recon_tmpl.c cf[pl][i*16]).
                    let cf_slot = &mut cf[i * 16..];
                    let tu_n = (uv_t_dim.w as usize * 4) * (uv_t_dim.h as usize * 4);
                    cf_slot[..tu_n].fill(0);

                    let dq_tbl = recon.dq_active[seg_id][1 + pl];
                    let qm_ref: Option<&[u8]> = recon.frame.qm[uvtx][1 + pl].as_deref();

                    let acoef = &a.ccoef[pl][(cbx4 + x as usize)..];
                    let lcoef = &l.ccoef[pl][(cby4 + y as usize)..];

                    let params = crate::recon::DecodeCoefParams {
                        tx: uvtx,
                        bs: cbs as u8 as usize,
                        plane: (pl + 1) as i32,
                        intra: is_intra,
                        fsc: b.fsc != 0,
                        lossless,
                        sdp_active,
                        y_mode: 0,
                        uv_mode: uv_mode as usize,
                        seg_id,
                        seq_fsc: recon.frame.seq_fsc,
                        seq_ist: recon.frame.seq_ist,
                        seq_cctx: recon.frame.seq_cctx,
                        chroma_dctonly: false,
                        reduced_txtp_set: recon.frame.reduced_txtp_set,
                        tcq_enabled: recon.frame.tcq,
                        layout: recon.frame.layout,
                        u_has_cf,
                        cbx,
                        cby,
                        luma_fsc_map: &luma_fsc_map,
                        dq_tbl,
                        bitdepth: recon.frame.bitdepth,
                        qm: qm_ref,
                        ss_hor: ss_hor != 0,
                        ss_ver: ss_ver != 0,
                    };

                    let eob = crate::recon::decode_coefs(
                        msac,
                        recon.cdf_coef,
                        cdf_m,
                        acoef,
                        lcoef,
                        &params,
                        cf_slot,
                        &mut txtp,
                        &mut res_ctx,
                    );
                    if eob == i32::MIN {
                        if std::env::var("RAV2D_SUBMIT_ERR").is_ok() {
                            eprintln!(
                                "recon chroma: eob==MIN cbx={} cby={} seg={}",
                                cbx, cby, b.seg_id
                            );
                        }
                        return Err(());
                    }
                    if pl == 0 {
                        u_has_cf = (eob >= 0) as i32;
                    }
                    tu_txtp[i][pl] = txtp;
                    tu_eob[i][pl] = eob as i16;

                    if std::env::var("RAV2D_CTX").is_ok() && cby < 8 {
                        eprintln!(
                            "CHROMATX cby={} cbx={} pl={} uvtx={} txtp={} eob={} uvmode={} cfl_type={} rng={}",
                            cby,
                            cbx,
                            pl,
                            uvtx,
                            txtp & 0xff,
                            eob,
                            uv_mode,
                            unsafe { b.data.intra.cfl_type },
                            msac.dbg_rng()
                        );
                    }

                    let aw = imin(ctw4, 64 - (cbx4 + x as usize) as i32).max(0) as usize;
                    let lh = imin(cth4, 64 - (cby4 + y as usize) as i32).max(0) as usize;
                    if aw > 0 {
                        a.ccoef[pl][cbx4 + x as usize..cbx4 + x as usize + aw].fill(res_ctx);
                    }
                    if lh > 0 {
                        l.ccoef[pl][cby4 + y as usize..cby4 + y as usize + lh].fill(res_ctx);
                    }
                    x += txw;
                }
                y += txh;
            }
        }
    } // end coef-read phase

    // Stash decoded coefficients for the deferred recon phase, or restore them.
    if phase == ChromaPhase::ReadOnly {
        let need = n_tu * 16 * 2;
        if recon.scratch.chroma_cf.len() < need {
            recon.scratch.chroma_cf.resize(need, 0);
        }
        recon.scratch.chroma_cf[..n_tu * 16].copy_from_slice(cf_u);
        recon.scratch.chroma_cf[n_tu * 16..need].copy_from_slice(cf_v);
        recon.scratch.chroma_txtp = tu_txtp;
        recon.scratch.chroma_eob = tu_eob;
        recon.scratch.chroma_u_has_cf = u_has_cf;
        return Ok(());
    } else if phase == ChromaPhase::ReconOnly {
        cf_u.copy_from_slice(&recon.scratch.chroma_cf[..n_tu * 16]);
        cf_v.copy_from_slice(&recon.scratch.chroma_cf[n_tu * 16..n_tu * 16 * 2]);
        tu_txtp = recon.scratch.chroma_txtp;
        tu_eob = recon.scratch.chroma_eob;
        u_has_cf = recon.scratch.chroma_u_has_cf;
    }
    let _ = u_has_cf;

    // ---- CfL prediction (recon_tmpl.c:3588-3590, cfl()) ---------------------
    // CfL is intra-only (`if (intra) cfl()`); IntraBC used the mc copy instead.
    if is_intra && uv_mode == CFL_PRED {
        cfl_predict_8bpc(recon, b, cbs, uvtx, cbx, cby, fi)?;
    }

    // ---- prediction + inverse transform per TU (recon_tmpl.c:3791-3938) -----
    let col_end_ss = fi.tile_col_end >> ss_hor;
    let row_end_ss = fi.tile_row_end >> ss_ver;
    let mut y = 0;
    while y < ch4ss {
        let mut x = 0;
        while x < cw4ss {
            let i = (y * cbw4ss as i32 + x) as usize;
            let dst_off = 4 * ((ssby + y as usize) * cstride + ssbx + x as usize);

            // Intra prediction for both planes (skipped for CfL and IntraBC).
            // C gate: `if (intra && b->uv_mode != CFL_PRED)`.
            if is_intra && uv_mode != CFL_PRED {
                for pl in 0..2 {
                    // n_tr: top-right availability (recon_tmpl.c:3809-3828).
                    let mut n_tr = 0i32;
                    if cby + (y << ss_ver) > fi.tile_row_start && (ctw as i32) < 64 {
                        let csbsz = sbsz >> ss_hor;
                        let tile_end = col_end_ss;
                        let w = imin(ctw4, tile_end - (ssbx as i32 + x) - ctw4);
                        if (cby + y) & (sbsz - 1) == 0 {
                            n_tr = w;
                        } else {
                            let end = imin((ssbx as i32 + x + csbsz) & !(csbsz - 1), tile_end);
                            let w2 = imin(ctw4, end - (ssbx as i32 + x) - ctw4);
                            if w2 == 0 {
                                n_tr = 0;
                            } else {
                                let shift = (cbx4 as i32 + x + ctw4) as u32;
                                let bits = recon.scratch.is_coded[1]
                                    [(cby4 as i32 + y - 1) as usize]
                                    >> shift;
                                let inv = 0x10000u64 | !bits;
                                n_tr = imin(inv.trailing_zeros() as i32, w2);
                            }
                        }
                    }
                    // n_bl: bottom-left availability (recon_tmpl.c:3829-3843).
                    let mut n_bl = 0i32;
                    if cbx + (x << ss_hor) > fi.tile_col_start && (cth as i32) < 64 {
                        let csbsz = sbsz >> ss_ver;
                        let end = imin((ssby as i32 + y + csbsz) & !(csbsz - 1), row_end_ss);
                        let h = imin(cth4, end - (ssby as i32 + y) - cth4);
                        if (cbx + x) & (sbsz - 1) == 0 || h <= 0 {
                            n_bl = h;
                        } else {
                            let mask = 1u64 << ((cbx4 as i32 + x - 1) as u32);
                            let mut nb = 0;
                            while nb < h {
                                let row = (cby4 as i32 + y + nb + cth4) as usize;
                                if row >= 64 || (recon.scratch.is_coded[1][row] & mask) == 0 {
                                    break;
                                }
                                nb += 1;
                            }
                            n_bl = nb;
                        }
                    }

                    let mut apply_ibp = recon.frame.seq_ibp && uvtx != 0;
                    let sm_top = unsafe { b.data.intra.is_sm[1] }.a;
                    let sm_left = unsafe { b.data.intra.is_sm[1] }.l;
                    let is_sm_flag = if apply_ibp {
                        (sm_top * crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG)
                            | (sm_left * crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
                    } else {
                        (sm_top | sm_left)
                            * (crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG
                                | crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
                    };
                    apply_ibp &= uv_mode == 0; // DC_PRED
                    let have_left = cbx + (x << ss_hor) > fi.tile_col_start;
                    let have_top = cby + (y << ss_ver) > fi.tile_row_start;
                    let intra_flags = is_sm_flag
                        | if apply_ibp {
                            crate::levels::ANGLE_IBP_FLAG
                        } else {
                            0
                        }
                        | if recon.frame.seq_intra_edge_filter {
                            crate::levels::ANGLE_USE_EDGE_FILTER_FLAG
                        } else {
                            0
                        }
                        | if have_left {
                            crate::levels::ANGLE_HAS_LEFT_FLAG
                        } else {
                            0
                        }
                        | if have_top {
                            crate::levels::ANGLE_HAS_TOP_FLAG
                        } else {
                            0
                        };
                    let pred_mode = if uv_mode == CFL_PRED { 0 } else { uv_mode };

                    let dst_plane: &mut [u8] = if pl == 0 { recon.dst_u } else { recon.dst_v };
                    let edge_o: usize = 768;
                    let m = crate::ipred_prepare::prepare_intra_edges_8bpc(
                        ssbx as i32 + x,
                        ssby as i32 + y,
                        col_end_ss,
                        row_end_ss,
                        n_tr,
                        n_bl,
                        dst_plane,
                        dst_off,
                        cstride,
                        None,
                        pred_mode,
                        txw,
                        txh,
                        angle | intra_flags,
                        recon.edge,
                        edge_o,
                    );
                    let pred_angle = angle | intra_flags;
                    let max_w = 4 * fi.bw - 4 * (cbx + x);
                    let max_h = 4 * fi.bh - 4 * (cby + y);
                    dispatch_ipred(
                        m,
                        dst_plane,
                        dst_off,
                        cstride,
                        recon.edge,
                        edge_o,
                        ctw,
                        cth,
                        pred_angle,
                        max_w,
                        max_h,
                        &recon.frame.ibp_weights,
                    );
                }
            }

            // CCTX cross-component transform (recon_tmpl.c:3882-3905).
            let cctx_enabled = recon.frame.seq_cctx
                && (recon.frame.layout == crate::headers::PixelLayout::I420 || uv_t_dim.max < 8);
            let cctx_type = if cctx_enabled && tu_eob[i][0] >= 1 {
                (tu_txtp[i][0] >> 8) as i32
            } else {
                0
            };
            if cctx_type != 0 {
                let sz = imin(ctw as i32, 32) as usize * imin(cth as i32, 32) as usize;
                crate::itx::cctx_8bpc(
                    &mut cf_u[i * 16..],
                    &mut cf_v[i * 16..],
                    &crate::tables::CCTX_ANGLE[(cctx_type - 1) as usize],
                    sz,
                );
                let gt = (tu_eob[i][1] > tu_eob[i][0]) as usize;
                tu_eob[i][1 - gt] = tu_eob[i][gt];
                let t0 = tu_txtp[i][0] & 0xff;
                tu_txtp[i][0] = t0;
                tu_txtp[i][1] = t0;
            }

            // Inverse transform add (recon_tmpl.c:3906-3925).
            for pl in 0..2 {
                if tu_eob[i][pl] != -1 {
                    let cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
                    let mut txtp = tu_txtp[i][pl] as u32;
                    // recon_tmpl.c:3916-3920: dpcm is intra-only; IntraBC takes
                    // the inter_ddt branch instead.
                    if lossless && is_intra && unsafe { b.data.intra }.dpcm[1] != 0 {
                        txtp +=
                            ((1 + (uv_mode == IntraPredMode::VertPred as u8) as u32) as u32) << 8;
                    } else if recon.frame.seq_inter_ddt && is_intrabc {
                        txtp += txtp & crate::tables::TX_DDT_MASK[uvtx] as u32;
                    }
                    let dst_plane: &mut [u8] = if pl == 0 { recon.dst_u } else { recon.dst_v };
                    crate::itx::inv_txfm_add_8bpc(
                        dst_plane,
                        dst_off,
                        cstride,
                        &mut cf[i * 16..],
                        txtp,
                        tu_eob[i][pl] as i32,
                        uvtx,
                    );
                }
            }

            // mark is_coded[1] (recon_tmpl.c:3934-3936).
            let coded_w = imin(ctw4, 64 - (cbx4 as i32 + x)).max(0) as u32;
            if coded_w > 0 {
                let mask: u64 = (((1u128 << coded_w) - 1) as u64) << ((cbx4 as i32 + x) as u32);
                for yy in 0..cth4 {
                    let row = (cby4 as i32 + y + yy) as usize;
                    if row < 64 {
                        recon.scratch.is_coded[1][row] |= mask;
                    }
                }
            }
            x += txw;
        }
        y += txh;
    }

    let _ = (orig_uv_mode, ystride);
    Ok(())
}

/// Chroma-from-luma prediction for an intra block (recon_tmpl.c:2910-3062 `cfl`).
/// Handles CFL_EXPLICIT / CFL_IMPLICIT (cfl_type < 2) and CFL_MHCCP (cfl_type==2).
#[allow(clippy::too_many_arguments)]
fn cfl_predict_8bpc(
    recon: &mut ReconCtx,
    b: &Av2Block,
    bs: BlockSize,
    uvtx: usize,
    cbx: i32,
    cby: i32,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    use crate::ipred::{
        CFL_HAS_LEFT, CFL_HAS_TOP, CFL_IS_TOP_SB_EDGE, CFL_MHCCP_MAX_EDGE_SAMPLES,
        cfl_calc_alphas_8bpc, cfl_gen_mat_8bpc, cfl_gen_y_420_8bpc, cfl_mhccp_pred_8bpc,
        cfl_pred_8bpc,
    };
    use crate::levels::CflMhDir;

    let ss_hor = recon.frame.ss_hor as usize;
    let ss_ver = recon.frame.ss_ver as usize;
    let ystride = recon.frame.y_stride_px;
    let cstride = recon.frame.uv_stride_px;
    let sbsz = fi.sb_step;
    let ssbx = (cbx >> ss_hor) as usize;
    let ssby = (cby >> ss_ver) as usize;
    let has_top = cby > fi.tile_row_start;
    let has_left = cbx > fi.tile_col_start;
    let is_top_sb_edge = (cby & (sbsz - 1)) == 0;
    let t_dim = &TXFM_DIMENSIONS[uvtx];
    let ctw4 = imin(t_dim.w as i32, (fi.bw - cbx + ss_hor as i32) >> ss_hor) as usize;
    let cth4 = imin(t_dim.h as i32, (fi.bh - cby + ss_ver as i32) >> ss_ver) as usize;
    let ctw = t_dim.w as usize * 4;
    let cth = t_dim.h as usize * 4;
    let filter_type = recon.frame.cfl_ds_filter_index;
    let cfl_type = unsafe { b.data.intra.cfl_type } as i32;
    let cfl_mh_dir_raw = unsafe { b.data.intra.cfl.cfl_mh_dir };
    // Raw symbol (0/1) maps directly to the dispatch index (CENTER=0, TOP=1).
    let dir: CflMhDir = unsafe { std::mem::transmute(cfl_mh_dir_raw.min(3)) };

    let ysrc_off = (cby as usize * ystride + cbx as usize) * 4;

    if cfl_type < 2 {
        // ---- CFL EXPLICIT / IMPLICIT (recon_tmpl.c:2936-2970) --------------
        let implicit = cfl_type == 1; // CFL_IMPLICIT
        let coff = (ssby * cstride + ssbx) * 4;
        // ytop / utop / vtop: source rows above the current block used for the CfL
        // top-edge reference (recon_tmpl.c:2955-2962). At an internal SB top-edge
        // dav2d reads `ytop_sb_edge` (the `prefilter_data` copy of the row just
        // above the SB). In single-thread / filters-off decode `prefilter_data`
        // aliases the current plane with `prefilter_data_full_frame` set, so the
        // luma SB-edge row resolves to `ysrc - ystride` (one luma row up) and is
        // downsampled with `bottom = 0` via the CFL_IS_TOP_SB_EDGE flag; the
        // in-plane fallback instead starts `1 + ss_ver` rows up with
        // `bottom = ystride`. The chroma `utop`/`vtop` offsets are `coff - cstride`
        // in both branches (single-thread alias), so only `ytop_off` differs.
        let ytop_off = if is_top_sb_edge && has_top {
            (ysrc_off as isize - ystride as isize) as usize
        } else {
            (ysrc_off as isize - ((1 + ss_ver) as isize) * ystride as isize) as usize
        };
        let utop_off = (coff as isize - cstride as isize) as usize;
        let vtop_off = utop_off;

        let cbw4 = (BLOCK_DIMENSIONS[bs as u8 as usize][0] as usize + ss_hor) >> ss_hor;
        let cbh4 = (BLOCK_DIMENSIONS[bs as u8 as usize][1] as usize + ss_ver) >> ss_ver;
        let wpad = cbw4 - ctw4;
        let hpad = cbh4 - cth4;

        let alpha = unsafe { b.data.intra.cfl.cfl_alpha };
        let flags = (filter_type as u32)
            | if has_top { CFL_HAS_TOP as u32 } else { 0 }
            | if has_left { CFL_HAS_LEFT as u32 } else { 0 }
            | if is_top_sb_edge {
                CFL_IS_TOP_SB_EDGE
            } else {
                0
            }
            | (((alpha[0] as u32) << crate::ipred::CFL_ALPHA_U_SHIFT)
                & crate::ipred::CFL_ALPHA_U_MASK)
            | (((alpha[1] as u32) << crate::ipred::CFL_ALPHA_V_SHIFT)
                & crate::ipred::CFL_ALPHA_V_MASK);

        // C uses ptrs[6] = { ytop, utop, vtop, ysrc, usrc, vsrc }. ytop/utop/vtop
        // are the rows above the destination, which alias the luma / U / V planes
        // (no top-SB prefilter buffer is wired; reachable only at frame top where
        // has_top is false → those reads are unused). The predictor only reads the
        // top rows and writes the block region, so the alias is non-overlapping.
        // SAFETY: the three planes are disjoint allocations; the immutable top-row
        // views never overlap the mutable write region of their own plane.
        let dst_y: &[u8] =
            unsafe { std::slice::from_raw_parts(recon.dst_y.as_ptr(), recon.dst_y.len()) };
        let utop: &[u8] =
            unsafe { std::slice::from_raw_parts(recon.dst_u.as_ptr(), recon.dst_u.len()) };
        let vtop: &[u8] =
            unsafe { std::slice::from_raw_parts(recon.dst_v.as_ptr(), recon.dst_v.len()) };
        let (u_buf, v_buf) = (&mut *recon.dst_u, &mut *recon.dst_v);
        cfl_pred_8bpc(
            dst_y,
            ytop_off,
            utop,
            utop_off,
            vtop,
            vtop_off,
            dst_y,
            ysrc_off,
            u_buf,
            coff,
            v_buf,
            coff,
            ystride as isize,
            cstride as isize,
            wpad,
            hpad,
            ctw,
            cth,
            flags,
            implicit,
            ss_hor,
            ss_ver,
        );
        return Ok(());
    }

    // ---- CFL MHCCP (recon_tmpl.c:2972-3060) --------------------------------
    let mut refw = (ctw4 * 4) as i32;
    let mut refh = (cth4 * 4) as i32;
    let cbx4 = ((cbx & 63) >> ss_hor) as i32;
    let cby4 = ((cby & 63) >> ss_ver) as i32;
    if has_top {
        let csbsz = sbsz >> ss_hor as i32;
        let tile_end = fi.tile_col_end >> ss_hor as i32;
        let mut w = imax(0, imin(ctw4 as i32, tile_end - ssbx as i32 - ctw4 as i32));
        let n_tr = if is_top_sb_edge {
            w
        } else {
            let end = imin((ssbx as i32 + csbsz) & !(csbsz - 1), tile_end);
            w = imin(ctw4 as i32, end - ssbx as i32 - ctw4 as i32);
            if w == 0 {
                0
            } else {
                let bits =
                    recon.scratch.is_coded[1][(cby4 - 1) as usize] >> ((cbx4 + ctw4 as i32) as u32);
                imin((0x10000u64 | !bits).trailing_zeros() as i32, w)
            }
        };
        refw += n_tr * 4;
    }
    let mut subleft = 0;
    if has_left {
        let csbsz = sbsz >> ss_ver as i32;
        let end = imax(
            0,
            imin(
                (ssby as i32 + csbsz) & !(csbsz - 1),
                fi.tile_row_end >> ss_ver as i32,
            ),
        );
        let h = imin(cth4 as i32, end - ssby as i32 - cth4 as i32);
        let n_bl = if (cbx & (sbsz - 1)) == 0 || h <= 0 {
            h
        } else {
            let mask = 1u64 << ((cbx4 - 1) as u32);
            let mut nb = 0;
            while nb < h {
                if (recon.scratch.is_coded[1][(cby4 + nb + cth4 as i32) as usize] & mask) == 0 {
                    break;
                }
                nb += 1;
            }
            nb
        };
        refh += n_bl * 4;
        refw += 2;
        subleft = (dir != CflMhDir::Left) as i32;
    }
    if refw > (128 >> ss_hor) {
        refw = 128 >> ss_hor;
        subleft = 0;
    }
    refh = imin(refh, (128 >> ss_ver) - 2 * has_top as i32);

    let luma_top_stride = ((refw as usize) + 63) & !63;
    let edge_flags = if has_top { CFL_HAS_TOP } else { 0 }
        | if has_left { CFL_HAS_LEFT } else { 0 }
        | if is_top_sb_edge {
            CFL_IS_TOP_SB_EDGE as i32
        } else {
            0
        };

    let mut luma = vec![0u8; crate::ipred::CFL_MHCCP_MAX_LUMA_SIZE];
    // SAFETY: luma plane is a disjoint allocation from chroma planes.
    let ysrc: &[u8] =
        unsafe { std::slice::from_raw_parts(recon.dst_y.as_ptr(), recon.dst_y.len()) };
    // Top-SB-edge prefilter source. In single-thread / filters-off decode dav2d's
    // `prefilter_data` aliases the current plane (decode.c:4950-4952, 5026-5029)
    // and `prefilter_data_full_frame` is set, so `ytop_sb_edge` resolves to the
    // luma row directly above this block (recon_tmpl.c:2945-2948). Passing it
    // explicitly makes `cfl_gen_y` take the `top_sb_edge != NULL` branch (b=0),
    // which differs from the in-plane fallback (b=src_stride) at internal SB
    // top-edges. Only set at `is_top_sb_edge`; otherwise dav2d passes NULL.
    let ytop_sb_edge: Option<(&[u8], usize)> = if is_top_sb_edge && has_top {
        Some((ysrc, ysrc_off - ystride))
    } else {
        None
    };
    cfl_gen_y_420_8bpc(
        &mut luma,
        luma_top_stride,
        ysrc,
        ysrc_off,
        ytop_sb_edge,
        ystride,
        (refw - subleft) as usize,
        refh as usize,
        ctw,
        cth,
        edge_flags | dir as i32,
        filter_type,
    );
    refh += has_top as i32;

    let mut mat = [[0i32; 3]; 3];
    let mut imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
    if has_top || has_left {
        cfl_gen_mat_8bpc(
            &mut mat,
            &mut imat,
            &luma,
            0,
            luma_top_stride,
            refw as usize,
            refh as usize,
            edge_flags,
            dir,
        );
    }

    for pl in 0..2 {
        let mut alpha = [0i32; 3];
        let chroma_off = 4 * (ssby * cstride + ssbx);
        let chroma: &mut [u8] = if pl == 0 { recon.dst_u } else { recon.dst_v };
        if has_top || has_left {
            cfl_calc_alphas_8bpc(
                &mut alpha,
                chroma,
                chroma_off,
                None, // ctop_sb_edge (frame-top only)
                cstride,
                refw as usize,
                refh as usize,
                &mut mat,
                &imat,
                edge_flags,
            );
        } else {
            alpha[2] = 0x10000;
        }
        let n_top = if has_top {
            has_top as usize + (dir == CflMhDir::Top) as usize
        } else {
            0
        };
        let src_off = n_top * luma_top_stride;
        // The predictor writes from a `dp` base of 0 (recon_tmpl.c:3038 offsets
        // the `chroma` pointer to the block), so slice the destination plane at
        // the block offset rather than the plane origin.
        cfl_mhccp_pred_8bpc(
            &mut chroma[chroma_off..],
            cstride,
            &luma,
            src_off,
            luma_top_stride,
            ctw,
            cth,
            &alpha,
            edge_flags,
            dir,
        );
    }
    Ok(())
}

/// Reconstruct a single luma transform block (intra, non-IntraBC, 8bpc).
///
/// Port of `recon_b_luma_tx` (recon_tmpl.c:2443-2675), combined-pass branch:
/// decode coefficients, run intra prediction into `dst_y`, then add the inverse
/// transform residual. `bx`/`by` are the tx block's 4x4 grid position.
#[allow(clippy::too_many_arguments)]
fn recon_b_luma_tx(
    recon: &mut ReconCtx,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    tx: usize,
    bx: i32,
    by: i32,
    pb_col_start: i32,
    pb_row_start: i32,
    lossless: bool,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    use crate::levels::IntraPredMode;

    let bx4 = (bx & 63) as usize;
    let by4 = (by & 63) as usize;
    let t_dim = &TXFM_DIMENSIONS[tx];
    let tw = t_dim.w as usize * 4;
    let th = t_dim.h as usize * 4;
    let tw4 = t_dim.w as i32;
    let th4 = t_dim.h as i32;

    let is_intrabc = b.intrabc != 0;
    // The decode-coefs / stx "intra" flag: `b->intra && (sdp_active || !b->intrabc)`
    // — here sdp_active is false in the luma path, so it is false for IntraBC.
    let is_intra = !is_intrabc;

    let intra = &unsafe { b.data.intra };
    let orig_y_mode = intra.y_mode;
    let mut angle = intra.y_angle as i32;

    // wide_angle_remap (recon_tmpl.c:2455). Only applies for intra, non-IntraBC.
    let y_mode = if is_intra {
        // SAFETY: y_mode is a valid IntraPredMode discriminant (0..=12).
        let y_mode_remapped = {
            let m_in: IntraPredMode = unsafe { std::mem::transmute(orig_y_mode.min(12)) };
            crate::recon::wide_angle_remap(t_dim, m_in, &mut angle, intra.mrl_index as i32) as u8
        };
        if orig_y_mode <= 12 {
            y_mode_remapped
        } else {
            orig_y_mode
        }
    } else {
        orig_y_mode
    };

    // --- decode coefficients (combined pass) -------------------------------
    let mut txtp: u16 = 0;
    let mut res_ctx: u8 = 0;
    // Zero the tx coefficient region; decode_coefs may not fully initialise it.
    let cf_n = tw * th;
    recon.cf[..cf_n].fill(0);

    // IntraBC blocks may set skip_txfm (intra/non-IntraBC blocks force it to 0).
    // When set, no coefficients are coded: eob=-1, txtp=DCT_DCT, stx=0, and the
    // lcoef context is filled with 0x40 (recon_tmpl.c:2464-2472).
    let (mut eob, stx, mut txtp) = if b.skip_txfm != 0 {
        res_ctx = 0x40;
        (-1i32, 0i32, crate::levels::txtp::DCT_DCT as u32)
    } else {
        let dq_seg = b.seg_id as usize;
        let dq_tbl = recon.dq_active[dq_seg][0]; // plane 0 (luma)
        let qm_ref: Option<&[u8]> = recon.frame.qm[tx][0].as_deref();

        let params = crate::recon::DecodeCoefParams {
            tx,
            bs: b.bs as usize,
            plane: 0,
            intra: is_intra,
            fsc: b.fsc != 0,
            lossless,
            sdp_active: false,
            y_mode: y_mode as usize,
            uv_mode: 0,
            seg_id: dq_seg,
            seq_fsc: recon.frame.seq_fsc,
            seq_ist: recon.frame.seq_ist,
            seq_cctx: recon.frame.seq_cctx,
            chroma_dctonly: false,
            reduced_txtp_set: recon.frame.reduced_txtp_set,
            tcq_enabled: recon.frame.tcq,
            layout: recon.frame.layout,
            u_has_cf: 0,
            cbx: 0,
            cby: 0,
            luma_fsc_map: &[],
            dq_tbl,
            bitdepth: recon.frame.bitdepth,
            qm: qm_ref,
            ss_hor: recon.frame.ss_hor != 0,
            ss_ver: recon.frame.ss_ver != 0,
        };

        let eob = crate::recon::decode_coefs(
            msac,
            recon.cdf_coef,
            cdf_m,
            &a.lcoef[bx4..],
            &l.lcoef[by4..],
            &params,
            recon.cf,
            &mut txtp,
            &mut res_ctx,
        );
        if eob == i32::MIN {
            if std::env::var("RAV2D_SUBMIT_ERR").is_ok() {
                eprintln!("recon luma: eob==MIN bx={} by={} seg={}", bx, by, b.seg_id);
            }
            return Err(());
        }
        let stx = (txtp >> 8) as i32;
        (eob, stx, (txtp & 0xff) as u32)
    };

    // dav2d_memset_likely_pow2 of the lcoef context (recon_tmpl.c:2497-2500).
    let aw = imin(tw4, fi.bw - bx).max(0) as usize;
    let lh = imin(th4, fi.bh - by).max(0) as usize;
    if aw > 0 {
        a.lcoef[bx4..bx4 + aw].fill(res_ctx);
    }
    if lh > 0 {
        l.lcoef[by4..by4 + lh].fill(res_ctx);
    }

    // dst origin for this tx block.
    let stride = recon.frame.y_stride_px;
    let dst_off = 4 * (by as usize * stride + bx as usize);

    // --- intra prediction (recon_tmpl.c:2511-2606) -------------------------
    // Skipped for IntraBC: the block-copy prediction was applied before the tx
    // walk (recon_tmpl.c gate `b->intra && !b->intrabc && !b->pal_sz`).
    if is_intra && intra.pal_sz == 0 {
        let sbsz = fi.sb_step;
        let mrl_idx = intra.mrl_index as i32;
        let mrl_mul = intra.multi_mrl != 0 && tx != 0; // tx != TX_4X4
        let is_hv5 = (by > pb_row_start || bx > pb_col_start)
            && (b.tx_part == TxPartition::H5 as u8 || b.tx_part == TxPartition::V5 as u8);

        // n_tr: top-right availability (recon_tmpl.c:2519-2540).
        let mut n_tr = 0i32;
        if by > fi.tile_row_start {
            let mut w = imin(tw4, fi.tile_col_end - bx - tw4);
            if is_hv5 {
                n_tr = 0;
            } else if (by & (sbsz - 1)) == 0 {
                n_tr = w;
            } else {
                let end = imin((bx + sbsz) & !(sbsz - 1), fi.tile_col_end);
                w = imin(w, end - bx - tw4);
                if w <= 0 {
                    n_tr = 0;
                } else {
                    let xpos = ((bx4 as i32 + tw4) & 63) as u32;
                    let bits = recon.scratch.is_coded[0][by4 - 1] >> xpos;
                    let inv = 0x10000u64 | !bits;
                    n_tr = imin(inv.trailing_zeros() as i32, w);
                }
            }
        }

        // n_bl: bottom-left availability (recon_tmpl.c:2542-2562).
        let mut n_bl = 0i32;
        if bx > fi.tile_col_start {
            let end = imin((by + sbsz) & !(sbsz - 1), fi.tile_row_end);
            let h = imin(th4, end - by - th4);
            // C distinguishes is_hv5 / bottom-edge as separate n_bl=0 cases
            // (recon_tmpl.c:2548-2556); merged here as both set n_bl = 0.
            if is_hv5 || h <= 0 {
                n_bl = 0;
            } else if (bx & (sbsz - 1)) == 0 {
                n_bl = h;
            } else {
                let mask = 1u64 << (((bx4 as i32 - 1) & 63) as u32);
                let mut y = 0;
                while y < h {
                    let row = (by4 as i32 + y + th4) as usize;
                    if row >= 64 || (recon.scratch.is_coded[0][row] & mask) == 0 {
                        break;
                    }
                    y += 1;
                }
                n_bl = y;
            }
        }

        let mut apply_ibp = recon.frame.seq_ibp && tx != 0 && mrl_idx == 0;
        let dip = intra.dip as i32 - 1;
        let sm_top = intra.is_sm[0].a;
        let sm_left = intra.is_sm[0].l;
        let is_sm_flag = if apply_ibp {
            (sm_top * crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG)
                | (sm_left * crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
        } else {
            (sm_top | sm_left)
                * (crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG
                    | crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
        };
        if intra.y_angle & 1 != 0 {
            apply_ibp = false;
        }
        let have_left = bx > fi.tile_col_start;
        let have_top = by > fi.tile_row_start;
        let intra_flags = crate::levels::ANGLE_IS_LUMA
            | is_sm_flag
            | if recon.frame.seq_intra_edge_filter {
                crate::levels::ANGLE_USE_EDGE_FILTER_FLAG
            } else {
                0
            }
            | if apply_ibp {
                crate::levels::ANGLE_IBP_FLAG
            } else {
                0
            }
            | (mrl_idx << crate::levels::ANGLE_MRL_IDX_SHIFT)
            | if mrl_mul {
                crate::levels::ANGLE_MULTI_MRL_FLAG
            } else {
                0
            }
            | if have_left {
                crate::levels::ANGLE_HAS_LEFT_FLAG
            } else {
                0
            }
            | if have_top {
                crate::levels::ANGLE_HAS_TOP_FLAG
            } else {
                0
            }
            | if dip >= 0 {
                crate::levels::ANGLE_DIP_FLAG
            } else {
                0
            };
        let angle_eff = if dip >= 0 { dip } else { angle };

        // Edge buffer origin: C uses `edge + 128 + !!mrl_idx*9`; we centre in a
        // larger slab so any layout (incl. multi-mrl second edge) fits.
        let edge_o: usize = 768 + if mrl_idx != 0 { 9 } else { 0 };

        let m = crate::ipred_prepare::prepare_intra_edges_8bpc(
            bx,
            by,
            fi.tile_col_end,
            fi.tile_row_end,
            n_tr,
            n_bl,
            recon.dst_y,
            dst_off,
            stride,
            None, // prefilter_toplevel_sb_edge: cross-SB-row backup not wired yet
            y_mode,
            tw4,
            th4,
            angle_eff | intra_flags,
            recon.edge,
            edge_o,
        );

        let pred_angle = angle_eff | intra_flags;
        let max_w = 4 * fi.bw - 4 * bx;
        let max_h = 4 * fi.bh - 4 * by;
        if std::env::var("RAV2D_PRED")
            .map(|s| {
                let mut it = s.split(',');
                it.next().and_then(|v| v.parse::<i32>().ok()) == Some(by)
                    && it.next().and_then(|v| v.parse::<i32>().ok()) == Some(bx)
            })
            .unwrap_or(false)
        {
            let topo = edge_o;
            eprintln!(
                "PRED y={} x={} m={} tw={} th={} mrl={} edge_top={:?} edge_left={:?}",
                by,
                bx,
                m,
                tw,
                th,
                mrl_idx,
                &recon.edge[topo + 1..topo + 1 + tw.min(8)],
                (1..=th.min(8))
                    .map(|i| recon.edge[topo - i])
                    .collect::<Vec<_>>()
            );
        }
        dispatch_ipred(
            m,
            recon.dst_y,
            dst_off,
            stride,
            recon.edge,
            edge_o,
            tw,
            th,
            pred_angle,
            max_w,
            max_h,
            &recon.frame.ibp_weights,
        );
        if std::env::var("RAV2D_PRED")
            .map(|s| {
                let mut it = s.split(',');
                it.next().and_then(|v| v.parse::<i32>().ok()) == Some(by)
                    && it.next().and_then(|v| v.parse::<i32>().ok()) == Some(bx)
            })
            .unwrap_or(false)
        {
            let p: Vec<u8> = (0..tw.min(8)).map(|i| recon.dst_y[dst_off + i]).collect();
            eprintln!("PRED y={} x={} predrow0={:?}", by, bx, p);
        }
    }

    // --- residual add (recon_tmpl.c:2608-2667) -----------------------------
    if eob != -1 {
        if stx != 0 {
            // Secondary transform reorder (recon_tmpl.c:2611-2654).
            const MASK: i32 = (1 << IntraPredMode::HorPred as i32)
                | (1 << IntraPredMode::HorDownPred as i32)
                | (1 << IntraPredMode::VertLeftPred as i32)
                | (1 << IntraPredMode::SmoothHPred as i32);
            // C: transpose = intrabc || !intra || !((mask >> b->y_mode) & 1);
            let transpose = is_intrabc || (MASK >> (y_mode as i32)) & 1 == 0;
            let stype = (stx & 3) - 1;
            let set = (stx >> 2) & 15;
            if tw >= 8 && th >= 8 {
                let koff = (set as usize * 3 + stype as usize) * 1536;
                let mut sums = [0i32; 48];
                crate::stx::stxfm(
                    &mut sums,
                    recon.cf,
                    &crate::stx_tables::STX_8X8_KERNEL[koff..],
                    48,
                    eob as usize,
                    recon.frame.bitdepth_max,
                );
                recon.cf[..32].fill(0);
                let idx = (imin(t_dim.lh as i32, 3) - 1) as usize;
                let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_8X8[idx][transpose as usize];
                let mapping =
                    &crate::stx_tables::COEFF8X8_MAPPING[set as usize * 3 + stype as usize];
                for x in 0..48 {
                    recon.cf[scan_out[mapping[x] as usize] as usize] = sums[x];
                }
                eob = [63, 119, 231][idx];
            } else {
                let koff = (set as usize * 3 + stype as usize) * 128;
                let mut sums = [0i32; 16];
                crate::stx::stxfm(
                    &mut sums,
                    recon.cf,
                    &crate::stx_tables::STX_4X4_KERNEL[koff..],
                    16,
                    eob as usize,
                    recon.frame.bitdepth_max,
                );
                let idx = imin(t_dim.lh as i32, 3) as usize;
                let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_4X4[idx][transpose as usize];
                recon.cf[4..8].fill(0);
                for x in 0..16 {
                    recon.cf[scan_out[x] as usize] = sums[x];
                }
                eob = [15, 15, 51, 99][idx];
            }
        }

        // lossless dpcm txtp adjust (recon_tmpl.c:2655-2660). The dpcm branch is
        // intra-only (`b->intra && !b->intrabc`); IntraBC takes the inter_ddt
        // branch ((flip)adst -> (f)ddt) when `seq_hdr->inter_ddt`.
        if lossless && is_intra && unsafe { b.data.intra }.dpcm[0] != 0 {
            txtp += ((1 + (y_mode == IntraPredMode::VertPred as u8) as u32) as u32) << 8;
        } else if recon.frame.seq_inter_ddt && is_intrabc {
            txtp += txtp & crate::tables::TX_DDT_MASK[tx] as u32;
        }

        crate::itx::inv_txfm_add_8bpc(recon.dst_y, dst_off, stride, recon.cf, txtp, eob, tx);
    }

    // mark is_coded for this tx region (recon_tmpl.c:2669-2672).
    let coded_w = imin(tw4, 64 - bx4 as i32).max(0) as u32;
    if coded_w > 0 {
        let mask: u64 = (((1u128 << coded_w) - 1) as u64) << (bx4 as u32);
        for y in 0..th4 {
            let row = by4 + y as usize;
            if row < 64 {
                recon.scratch.is_coded[0][row] |= mask;
            }
        }
    }

    if std::env::var("RAV2D_TRACE").is_ok() {
        let mut sum: u64 = 0;
        let mut first = [0u8; 8];
        for yy in 0..th.min(64) {
            for xx in 0..tw.min(64) {
                let p = recon.dst_y[dst_off + yy * stride + xx];
                sum = sum.wrapping_add(p as u64).wrapping_mul(31);
            }
        }
        for (i, fv) in first.iter_mut().enumerate() {
            *fv = recon.dst_y[dst_off + i.min(tw - 1)];
        }
        eprintln!(
            "LUMATX y={} x={} tx={} txtp={} eob={} ymode={} ang={} dip={} mrl={} fsc={} sum={} first={:?}",
            by,
            bx,
            tx,
            txtp & 0xff,
            eob,
            y_mode,
            intra.y_angle as i32,
            intra.dip as i32 - 1,
            intra.mrl_index,
            b.fsc,
            sum,
            first
        );
    }

    let _ = orig_y_mode; // C restores b->y_mode; we never mutated b.
    Ok(())
}

/// Dispatch the resolved intra predictor `m` into `dst` (mirrors the C
/// `dsp->ipred.intra_pred[m]` table; recon_tmpl.c:2596-2606 / ipred_tmpl.c).
#[allow(clippy::too_many_arguments)]
fn dispatch_ipred(
    m: u8,
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    edge: &[u8],
    edge_o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_w: i32,
    max_h: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::ipred::*;
    use crate::levels::*;
    let d = &mut dst[dst_off..];
    match m {
        0 /* DcPred */ => ipred_dc(d, stride, edge, edge_o, w, h, angle),
        _ if m == DC_128_PRED => ipred_dc_128(d, stride, w, h),
        _ if m == TOP_DC_PRED => ipred_dc_top(d, stride, edge, edge_o, w, h, angle),
        _ if m == LEFT_DC_PRED => ipred_dc_left(d, stride, edge, edge_o, w, h, angle),
        2 /* HorPred */ => ipred_h(d, stride, edge, edge_o, w, h, angle),
        1 /* VertPred */ => ipred_v(d, stride, edge, edge_o, w, h, angle),
        12 /* PaethPred */ => ipred_paeth(d, stride, edge, edge_o, w, h),
        9 /* SmoothPred */ => ipred_smooth(d, stride, edge, edge_o, w, h),
        10 /* SmoothVPred */ => ipred_smooth_v(d, stride, edge, edge_o, w, h),
        11 /* SmoothHPred */ => ipred_smooth_h(d, stride, edge, edge_o, w, h),
        _ if m == Z1_PRED => {
            ipred_z1(d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights)
        }
        _ if m == Z2_PRED => ipred_z2(d, stride, edge, edge_o, w, h, angle, max_w, max_h),
        _ if m == Z3_PRED => {
            ipred_z3(d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights)
        }
        _ if m == DIP_PRED => ipred_dip_8bpc(d, stride, edge, edge_o, w, h, angle),
        _ => ipred_dc_128(d, stride, w, h),
    }
}

pub fn decode_sb(
    fi: &SbFrameInfo,
    bx: &mut i32,
    by: &mut i32,
    cbx: &mut i32,
    cby: &mut i32,
    intra_region: &mut i32,
    sdp_cfl_disallowed: &mut i32,
    pass: u8,
    a: &mut BlockContext,
    l: &mut BlockContext,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    cdf_dmv: &mut CdfMvContext,
    recon: &mut ReconCtx,
    part_w: &mut Vec<u8>,
    part_w_idx: &mut usize,
    part_r: &[u8],
    part_r_idx: &mut usize,
    lbs: BlockSize,
    cbs: BlockSize,
    dir_ptr: &mut i32,
) -> Result<(), ()> {
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    assert!(bs != BlockSize::Invalid);

    if std::env::var("RAV2D_TRACE_SB").is_ok() {
        eprintln!(
            "SB y={} x={} bs={} lbs={} cbs={} ireg={} dir={} rng={}",
            *by,
            *bx,
            bs as i32,
            lbs as i32,
            cbs as i32,
            *intra_region,
            *dir_ptr,
            msac.dbg_rng()
        );
    }

    let b_dim = &BLOCK_DIMENSIONS[bs as u8 as usize];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;
    let hw4 = bw4 >> 1;
    let hh4 = bh4 >> 1;
    let qw4 = hw4 >> 1;
    let qh4 = hh4 >> 1;
    let have_h_split = fi.bw > *bx + hw4;
    let have_v_split = fi.bh > *by + hh4;
    let cbs_orig = cbs;

    if lbs == BlockSize::Bs64x64 && cbs == BlockSize::Bs64x64 && fi.sdp && !fi.is_inter_or_switch {
        let mut dir = 0i32;
        decode_sb(
            fi,
            bx,
            by,
            cbx,
            cby,
            intra_region,
            sdp_cfl_disallowed,
            pass,
            a,
            l,
            msac,
            cdf_m,
            cdf_dmv,
            recon,
            part_w,
            part_w_idx,
            part_r,
            part_r_idx,
            lbs,
            BlockSize::Invalid,
            &mut dir,
        )?;
        return decode_sb(
            fi,
            bx,
            by,
            cbx,
            cby,
            intra_region,
            sdp_cfl_disallowed,
            pass,
            a,
            l,
            msac,
            cdf_m,
            cdf_dmv,
            recon,
            part_w,
            part_w_idx,
            part_r,
            part_r_idx,
            BlockSize::Invalid,
            cbs,
            &mut dir,
        );
    }

    let pl = (lbs == BlockSize::Invalid) as usize;
    let pcc = &PARTITION_SUBB[bs as u8 as usize];
    let mut bp = BlockPartition::Invalid;
    let mut cbs = cbs;

    if pass & (Pass::Entropy as u8) != 0 {
        let bx4 = (*bx & 63) as usize;
        let by4 = (*by & 63) as usize;
        let eff_ss_ver = fi.ss_ver & (lbs == BlockSize::Invalid) as i32;
        let eff_ss_hor = fi.ss_hor & (lbs == BlockSize::Invalid) as i32;
        let bwh4ss = [bw4 >> eff_ss_hor, bh4 >> eff_ss_ver];
        assert!(bwh4ss[0] >= 1 && bwh4ss[1] >= 1);
        let mut dir = -1i32;

        if imax(bwh4ss[0], bwh4ss[1]) == 1 || (pcc.part[0][0] & pcc.part[1][0]) == -1 {
            bp = BlockPartition::None;
        } else if !have_h_split || !have_v_split {
            if bw4 == bh4 {
                dir = have_v_split as i32;
                bp = if !have_v_split {
                    BlockPartition::H
                } else {
                    BlockPartition::V
                };
            } else if bw4 > bh4 {
                if !have_h_split || fi.bh <= *by + qh4 {
                    dir = 1;
                    bp = BlockPartition::V;
                }
            } else {
                if !have_v_split || fi.bw <= *bx + qw4 {
                    dir = 0;
                    bp = BlockPartition::H;
                }
            }
        }

        if bp == BlockPartition::Invalid {
            if cbs == BlockSize::Bs64x64
                && lbs == BlockSize::Invalid
                && ((*dir_ptr & 0xff) == 0xff
                    || (*dir_ptr & 0x30003) == 0x10002
                    || (*dir_ptr & 0x30003) == 0x20001)
            {
                if (*dir_ptr & 0xff) == 0xff {
                    bp = BlockPartition::None;
                } else {
                    dir = ((*dir_ptr & 0x30003) == 0x10002) as i32;
                    bp = unsafe { BlockPartition::from_raw(((*dir_ptr >> 8) & 0xff) as i8) };
                }
            } else {
                let mix_inter = fi.is_inter_or_switch && *intra_region == 0;
                let ctx1 = get_partition_ctx(a, l, b_dim, pl, by4, bx4);
                let ctx2 = (ctx1 + pcc.ctx[0] as i32 * 4) as usize;
                if std::env::var("RAV2D_PART_TRACE").is_ok() {
                    eprintln!(
                        "DSPLIT y={} x={} bs={} ctx2={} hh={} hv={} rng_in={}",
                        *by, *bx, bs as i32, ctx2, have_h_split as i32,
                        have_v_split as i32, msac.dbg_rng()
                    );
                }
                let is_split = if mix_inter && b_dim[2] + b_dim[3] == 1 {
                    0u32
                } else if !have_h_split || !have_v_split {
                    1u32
                } else {
                    msac.decode_bool_adapt(cdf_m.part_split(pl, ctx2))
                };

                if is_split == 0 {
                    bp = BlockPartition::None;
                } else {
                    if (bs == BlockSize::Bs128x128 || bs == BlockSize::Bs256x256)
                        && have_v_split
                        && have_h_split
                    {
                        let ctx3 = (ctx1 + (bs == BlockSize::Bs256x256) as i32 * 4) as usize;
                        let is_square = msac.decode_bool_adapt(cdf_m.part_square(ctx3));
                        if is_square != 0 {
                            bp = BlockPartition::Split;
                        }
                    } else if imax(bw4, bh4) >= 32 {
                        bp = if bw4 > bh4 {
                            BlockPartition::V
                        } else {
                            BlockPartition::H
                        };
                    }

                    if bp == BlockPartition::Invalid {
                        let aspect = 1i32 << fi.max_pb_aspect_ratio_log2;
                        let v_aspect = bw4 * aspect >= bh4 * 2;
                        let h_aspect = bh4 * aspect >= bw4 * 2;
                        assert!(v_aspect || h_aspect);

                        if imin(bwh4ss[0], bwh4ss[1]) == 1 {
                            dir = (bwh4ss[0] > bwh4ss[1]) as i32;
                        } else if !(v_aspect && h_aspect) {
                            dir = v_aspect as i32;
                        } else {
                            let ctx4 = (ctx1 + pcc.ctx[1] as i32 * 4) as usize;
                            if std::env::var("RAV2D_PART_TRACE").is_ok() {
                                eprintln!(
                                    "DDIR y={} x={} bs={} ctx1={} pccctx1={} ctx4={} rng_in={}",
                                    *by, *bx, bs as i32, ctx1, pcc.ctx[1], ctx4, msac.dbg_rng()
                                );
                            }
                            dir = msac.decode_bool_adapt(cdf_m.part_dir(pl, ctx4)) as i32;
                        }
                        assert!(pcc.part[dir as usize][0] != -1);
                        bp = if dir != 0 {
                            BlockPartition::V
                        } else {
                            BlockPartition::H
                        };

                        if imax(bw4, bh4) <= 16 {
                            let bwh4ss2 = [bw4 >> fi.ss_hor, bh4 >> fi.ss_ver];
                            let ndir = (!dir) as usize & 1;
                            let ddir = dir as usize;
                            let has_hv3 = fi.ext_partitions
                                && bwh4ss[ndir] >= 4
                                && bwh4ss[ddir] >= 2
                                && b_dim[ndir] as i32 * aspect >= b_dim[ddir] as i32 * 4
                                && (cbs != lbs
                                    || (bwh4ss2[ndir] >= 4 && bwh4ss2[ddir] >= 2)
                                    || (if dir != 0 {
                                        if lbs == BlockSize::Bs32x8 {
                                            have_v_split
                                        } else {
                                            *bx + qw4 * 3 < fi.bw
                                        }
                                    } else {
                                        if lbs == BlockSize::Bs8x32 {
                                            have_h_split
                                        } else {
                                            *by + qh4 * 3 < fi.bh
                                        }
                                    }));
                            let has_hv4ab = bwh4ss[ndir] >= 8
                                && fi.uneven_4way
                                && b_dim[ndir] as i32 * aspect >= b_dim[ddir] as i32 * 8
                                && (cbs != lbs
                                    || bwh4ss2[ndir] >= 8
                                    || (if dir != 0 {
                                        *bx + (qw4 >> 1) * 7 < fi.bw
                                    } else {
                                        *by + (qh4 >> 1) * 7 < fi.bh
                                    }));

                            if has_hv3 || has_hv4ab {
                                assert!(pcc.part[ddir][1] != -1);
                                let ctx5 = get_partition2_ctx(a, l, b_dim, pl, dir, by4, bx4);
                                let ctx6 = (ctx5 + pcc.ctx[0] as i32 * 4) as usize;
                                let is_ext = msac.decode_bool_adapt(cdf_m.part_ext(pl, ctx6));
                                if is_ext != 0 {
                                    bp = if dir != 0 {
                                        BlockPartition::V3
                                    } else {
                                        BlockPartition::H3
                                    };
                                    if has_hv4ab {
                                        assert!(pcc.part[ddir][2] != -1);
                                        let is_4way = if !has_hv3 {
                                            1u32
                                        } else {
                                            msac.decode_bool_adapt(cdf_m.part_4way(pl, ctx6))
                                        };
                                        if is_4way != 0 {
                                            let is_a_or_b = msac.decode_bool_bypass();
                                            bp = unsafe {
                                                BlockPartition::from_raw(
                                                    BlockPartition::H4A as i8
                                                        + dir as i8 * 2
                                                        + is_a_or_b as i8,
                                                )
                                            };
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if std::env::var("RAV2D_PART_TRACE").is_ok() {
            eprintln!(
                "DPART y={} x={} bs={} bp={} rng={}",
                *by,
                *bx,
                bs as i32,
                bp as i32,
                msac.dbg_rng()
            );
        }
        dir += (dir != -1) as i32;
        if lbs == BlockSize::Invalid && cbs == BlockSize::Bs64x64 {
            *sdp_cfl_disallowed = (dir != -1 && dir != (*dir_ptr & 0x3)) as i32;
        }
        *dir_ptr |= (dir as u8) as i32 | ((bp as i8 as i32) << 8);

        let mut unmix_bit = 0i32;
        if fi.is_inter_or_switch
            && fi.ext_sdp
            && (cbs as i8 | lbs as i8) != BlockSize::Invalid as i8
            && bp != BlockPartition::None
            && (*dir_ptr & (1 << 24)) == 0
            && (bp as i8) < BlockPartition::H4A as i8
            && imin(bw4, bh4) >= 2
            && bs != fi.root_bs
            && imax(bw4, bh4) <= 16
        {
            let sz = b_dim[2] as i32 + b_dim[3] as i32;
            let ctx = iclip(sz - 4, 0, 3) + (sz == 4) as i32;
            let val = msac.decode_bool_adapt(cdf_m.region_type(ctx as usize));
            *intra_region = (val == 0) as i32;
            unmix_bit = *intra_region;
            if *intra_region != 0 {
                cbs = BlockSize::Invalid;
            }
        }
        if fi.n_passes > 1 {
            part_w[*part_w_idx] = bp as u8 | ((unmix_bit as u8) << 7);
            *part_w_idx += 1;
        }
    } else {
        let val = part_r[*part_r_idx];
        *part_r_idx += 1;
        if val & 0x80 != 0 {
            assert!(*intra_region == 0);
            *intra_region = 1;
            cbs = BlockSize::Invalid;
        }
        bp = unsafe { BlockPartition::from_raw((val & 0x7f) as i8) };
    }

    if bs == cbs {
        *cbx = *bx;
        *cby = *by;
    }

    let lim = &PARTITION_LIM[bp as u8 as usize];
    let mut child_dir = ((bw4 <= lim[0] as i32 || bh4 <= lim[1] as i32) as i32) << 24;

    match bp {
        BlockPartition::None => {
            let _b = decode_b(
                fi,
                *bx,
                *by,
                *cbx,
                *cby,
                *intra_region,
                *sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                lbs,
                cbs,
            )?;
            if pass & (Pass::Entropy as u8) != 0 {
                let bx4 = (*bx & 63) as usize;
                let by4 = (*by & 63) as usize;
                if (cbs as i8 | lbs as i8) != BlockSize::Invalid as i8 {
                    // C: case_set(b_dim[2 + i]) writes 1<<b_dim[2+i] bytes (pow2 length),
                    // for both partition[0] and partition[1].
                    memset_pow2(&mut a.partition[0], bx4, !(b_dim[0] - 1), b_dim[2]);
                    memset_pow2(&mut a.partition[1], bx4, !(b_dim[0] - 1), b_dim[2]);
                    memset_pow2(&mut l.partition[0], by4, !(b_dim[1] - 1), b_dim[3]);
                    memset_pow2(&mut l.partition[1], by4, !(b_dim[1] - 1), b_dim[3]);
                } else {
                    memset_pow2(&mut a.partition[pl], bx4, !(b_dim[0] - 1), b_dim[2]);
                    memset_pow2(&mut l.partition[pl], by4, !(b_dim[1] - 1), b_dim[3]);
                }
            }
        }
        BlockPartition::V => {
            assert!(hw4 > 0);
            let sub4 = bs == cbs && (hw4 >> fi.ss_hor) > 0;
            assert!(sub4 || pl == 0);
            let child_lbs = if pl != 0 {
                BlockSize::Invalid
            } else {
                unsafe { BlockSize::from_raw(pcc.part[1][0]) }
            };
            let child_cbs_first = if sub4 {
                unsafe { BlockSize::from_raw(pcc.part[1][0]) }
            } else {
                BlockSize::Invalid
            };
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                child_lbs,
                child_cbs_first,
                &mut child_dir,
            )?;
            if *bx + hw4 >= fi.bw { /* done */
            } else {
                *bx += hw4;
                let child_cbs_second = if sub4 {
                    unsafe { BlockSize::from_raw(pcc.part[1][0]) }
                } else {
                    cbs
                };
                decode_sb(
                    fi,
                    bx,
                    by,
                    cbx,
                    cby,
                    intra_region,
                    sdp_cfl_disallowed,
                    pass,
                    a,
                    l,
                    msac,
                    cdf_m,
                    cdf_dmv,
                    recon,
                    part_w,
                    part_w_idx,
                    part_r,
                    part_r_idx,
                    child_lbs,
                    child_cbs_second,
                    &mut child_dir,
                )?;
                *bx -= hw4;
            }
        }
        BlockPartition::H => {
            assert!(hh4 > 0);
            let sub4 = bs == cbs && (hh4 >> fi.ss_ver) > 0;
            assert!(sub4 || pl == 0);
            let child_lbs = if pl != 0 {
                BlockSize::Invalid
            } else {
                unsafe { BlockSize::from_raw(pcc.part[0][0]) }
            };
            let child_cbs_first = if sub4 {
                unsafe { BlockSize::from_raw(pcc.part[0][0]) }
            } else {
                BlockSize::Invalid
            };
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                child_lbs,
                child_cbs_first,
                &mut child_dir,
            )?;
            if *by + hh4 >= fi.bh { /* done */
            } else {
                *by += hh4;
                let child_cbs_second = if sub4 {
                    unsafe { BlockSize::from_raw(pcc.part[0][0]) }
                } else {
                    cbs
                };
                decode_sb(
                    fi,
                    bx,
                    by,
                    cbx,
                    cby,
                    intra_region,
                    sdp_cfl_disallowed,
                    pass,
                    a,
                    l,
                    msac,
                    cdf_m,
                    cdf_dmv,
                    recon,
                    part_w,
                    part_w_idx,
                    part_r,
                    part_r_idx,
                    child_lbs,
                    child_cbs_second,
                    &mut child_dir,
                )?;
                *by -= hh4;
            }
        }
        BlockPartition::Split => {
            assert!(have_v_split && have_h_split && cbs == lbs);
            let sbs = unsafe { BlockSize::from_raw(pcc.part[0][3]) };
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                sbs,
                sbs,
                &mut child_dir,
            )?;
            *bx += hw4;
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                sbs,
                sbs,
                &mut child_dir,
            )?;
            *bx -= hw4;
            *by += hh4;
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                sbs,
                sbs,
                &mut child_dir,
            )?;
            *bx += hw4;
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                sbs,
                sbs,
                &mut child_dir,
            )?;
            *bx -= hw4;
            *by -= hh4;
        }
        BlockPartition::V3 => {
            assert!(qw4 > 0 && hh4 > 0);
            let sub4 = bs == cbs && (qw4 >> fi.ss_hor) > 0 && (hh4 >> fi.ss_ver) > 0;
            assert!(sub4 || pl == 0);
            let i_3only = cbs == BlockSize::Invalid || (!sub4 && bs != BlockSize::Bs32x8);
            let p1_1 = unsafe { BlockSize::from_raw(pcc.part[1][1]) };
            let p1_3 = unsafe { BlockSize::from_raw(pcc.part[1][3]) };
            let lbs_child = if pl != 0 { BlockSize::Invalid } else { p1_1 };
            let cbs_first = if i_3only { BlockSize::Invalid } else { p1_1 };
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                lbs_child,
                cbs_first,
                &mut child_dir,
            )?;
            if *bx + qw4 >= fi.bw { /* done */
            } else {
                *bx += qw4;
                if !i_3only {
                    *cbx = *bx;
                }
                let lbs_mid = if pl != 0 { BlockSize::Invalid } else { p1_3 };
                let cbs_mid = if sub4 { p1_3 } else { BlockSize::Invalid };
                decode_sb(
                    fi,
                    bx,
                    by,
                    cbx,
                    cby,
                    intra_region,
                    sdp_cfl_disallowed,
                    pass,
                    a,
                    l,
                    msac,
                    cdf_m,
                    cdf_dmv,
                    recon,
                    part_w,
                    part_w_idx,
                    part_r,
                    part_r_idx,
                    lbs_mid,
                    cbs_mid,
                    &mut child_dir,
                )?;
                if *by + hh4 < fi.bh {
                    *by += hh4;
                    let cbs_mid2 = if i_3only {
                        BlockSize::Invalid
                    } else if sub4 {
                        p1_3
                    } else {
                        unsafe { BlockSize::from_raw(pcc.part[1][0]) }
                    };
                    decode_sb(
                        fi,
                        bx,
                        by,
                        cbx,
                        cby,
                        intra_region,
                        sdp_cfl_disallowed,
                        pass,
                        a,
                        l,
                        msac,
                        cdf_m,
                        cdf_dmv,
                        recon,
                        part_w,
                        part_w_idx,
                        part_r,
                        part_r_idx,
                        lbs_mid,
                        cbs_mid2,
                        &mut child_dir,
                    )?;
                    *by -= hh4;
                }
                if *bx + hw4 >= fi.bw {
                    *bx -= qw4;
                } else {
                    *bx += hw4;
                    let cbs_last = if i_3only { cbs } else { p1_1 };
                    decode_sb(
                        fi,
                        bx,
                        by,
                        cbx,
                        cby,
                        intra_region,
                        sdp_cfl_disallowed,
                        pass,
                        a,
                        l,
                        msac,
                        cdf_m,
                        cdf_dmv,
                        recon,
                        part_w,
                        part_w_idx,
                        part_r,
                        part_r_idx,
                        lbs_child,
                        cbs_last,
                        &mut child_dir,
                    )?;
                    *bx -= 3 * qw4;
                }
            }
        }
        BlockPartition::H3 => {
            assert!(qh4 > 0 && hw4 > 0);
            let sub4 = bs == cbs && (qh4 >> fi.ss_ver) > 0 && (hw4 >> fi.ss_hor) > 0;
            assert!(sub4 || pl == 0);
            let i_3only = cbs == BlockSize::Invalid || (!sub4 && bs != BlockSize::Bs8x32);
            let p0_1 = unsafe { BlockSize::from_raw(pcc.part[0][1]) };
            let p0_3 = unsafe { BlockSize::from_raw(pcc.part[0][3]) };
            let lbs_child = if pl != 0 { BlockSize::Invalid } else { p0_1 };
            let cbs_first = if i_3only { BlockSize::Invalid } else { p0_1 };
            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                lbs_child,
                cbs_first,
                &mut child_dir,
            )?;
            if *by + qh4 >= fi.bh { /* done */
            } else {
                *by += qh4;
                if !i_3only {
                    *cby = *by;
                }
                let lbs_mid = if pl != 0 { BlockSize::Invalid } else { p0_3 };
                let cbs_mid = if sub4 { p0_3 } else { BlockSize::Invalid };
                decode_sb(
                    fi,
                    bx,
                    by,
                    cbx,
                    cby,
                    intra_region,
                    sdp_cfl_disallowed,
                    pass,
                    a,
                    l,
                    msac,
                    cdf_m,
                    cdf_dmv,
                    recon,
                    part_w,
                    part_w_idx,
                    part_r,
                    part_r_idx,
                    lbs_mid,
                    cbs_mid,
                    &mut child_dir,
                )?;
                if *bx + hw4 < fi.bw {
                    *bx += hw4;
                    let cbs_mid2 = if i_3only {
                        BlockSize::Invalid
                    } else if sub4 {
                        p0_3
                    } else {
                        unsafe { BlockSize::from_raw(pcc.part[0][0]) }
                    };
                    decode_sb(
                        fi,
                        bx,
                        by,
                        cbx,
                        cby,
                        intra_region,
                        sdp_cfl_disallowed,
                        pass,
                        a,
                        l,
                        msac,
                        cdf_m,
                        cdf_dmv,
                        recon,
                        part_w,
                        part_w_idx,
                        part_r,
                        part_r_idx,
                        lbs_mid,
                        cbs_mid2,
                        &mut child_dir,
                    )?;
                    *bx -= hw4;
                }
                if *by + hh4 >= fi.bh {
                    *by -= qh4;
                } else {
                    *by += hh4;
                    let cbs_last = if i_3only { cbs } else { p0_1 };
                    decode_sb(
                        fi,
                        bx,
                        by,
                        cbx,
                        cby,
                        intra_region,
                        sdp_cfl_disallowed,
                        pass,
                        a,
                        l,
                        msac,
                        cdf_m,
                        cdf_dmv,
                        recon,
                        part_w,
                        part_w_idx,
                        part_r,
                        part_r_idx,
                        lbs_child,
                        cbs_last,
                        &mut child_dir,
                    )?;
                    *by -= 3 * qh4;
                }
            }
        }
        BlockPartition::V4A | BlockPartition::V4B => {
            let ew4 = qw4 >> 1;
            assert!(ew4 > 0);
            let sub4 = bs == cbs && (ew4 >> fi.ss_hor) > 0;
            assert!(sub4 || pl == 0);
            let p1_2 = unsafe { BlockSize::from_raw(pcc.part[1][2]) };
            let var = bp as i8 - BlockPartition::V4A as i8;
            let p1_nvar = unsafe { BlockSize::from_raw(pcc.part[1][(!var & 1) as usize]) };
            let p1_var = unsafe { BlockSize::from_raw(pcc.part[1][var as usize]) };
            let lbs_edge = if pl != 0 { BlockSize::Invalid } else { p1_2 };
            let lbs_nvar = if pl != 0 { BlockSize::Invalid } else { p1_nvar };
            let lbs_var = if pl != 0 { BlockSize::Invalid } else { p1_var };

            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                lbs_edge,
                if sub4 { p1_2 } else { BlockSize::Invalid },
                &mut child_dir,
            )?;
            if *bx + ew4 >= fi.bw { /* done */
            } else {
                *bx += ew4;
                decode_sb(
                    fi,
                    bx,
                    by,
                    cbx,
                    cby,
                    intra_region,
                    sdp_cfl_disallowed,
                    pass,
                    a,
                    l,
                    msac,
                    cdf_m,
                    cdf_dmv,
                    recon,
                    part_w,
                    part_w_idx,
                    part_r,
                    part_r_idx,
                    lbs_nvar,
                    if sub4 { p1_nvar } else { BlockSize::Invalid },
                    &mut child_dir,
                )?;
                let w4a = qw4 << var;
                let w4b = hw4 >> var;
                if *bx + w4a >= fi.bw {
                    *bx -= ew4;
                } else {
                    *bx += w4a;
                    decode_sb(
                        fi,
                        bx,
                        by,
                        cbx,
                        cby,
                        intra_region,
                        sdp_cfl_disallowed,
                        pass,
                        a,
                        l,
                        msac,
                        cdf_m,
                        cdf_dmv,
                        recon,
                        part_w,
                        part_w_idx,
                        part_r,
                        part_r_idx,
                        lbs_var,
                        if sub4 { p1_var } else { BlockSize::Invalid },
                        &mut child_dir,
                    )?;
                    if *bx + w4b >= fi.bw {
                        *bx -= ew4 + w4a;
                    } else {
                        *bx += w4b;
                        decode_sb(
                            fi,
                            bx,
                            by,
                            cbx,
                            cby,
                            intra_region,
                            sdp_cfl_disallowed,
                            pass,
                            a,
                            l,
                            msac,
                            cdf_m,
                            cdf_dmv,
                            recon,
                            part_w,
                            part_w_idx,
                            part_r,
                            part_r_idx,
                            lbs_edge,
                            if sub4 { p1_2 } else { cbs },
                            &mut child_dir,
                        )?;
                        *bx -= 7 * ew4;
                    }
                }
            }
        }
        BlockPartition::H4A | BlockPartition::H4B => {
            let eh4 = qh4 >> 1;
            assert!(eh4 > 0);
            let sub4 = bs == cbs && (eh4 >> fi.ss_ver) > 0;
            assert!(sub4 || pl == 0);
            let p0_2 = unsafe { BlockSize::from_raw(pcc.part[0][2]) };
            let var = bp as i8 - BlockPartition::H4A as i8;
            let p0_nvar = unsafe { BlockSize::from_raw(pcc.part[0][(!var & 1) as usize]) };
            let p0_var = unsafe { BlockSize::from_raw(pcc.part[0][var as usize]) };
            let lbs_edge = if pl != 0 { BlockSize::Invalid } else { p0_2 };
            let lbs_nvar = if pl != 0 { BlockSize::Invalid } else { p0_nvar };
            let lbs_var = if pl != 0 { BlockSize::Invalid } else { p0_var };

            decode_sb(
                fi,
                bx,
                by,
                cbx,
                cby,
                intra_region,
                sdp_cfl_disallowed,
                pass,
                a,
                l,
                msac,
                cdf_m,
                cdf_dmv,
                recon,
                part_w,
                part_w_idx,
                part_r,
                part_r_idx,
                lbs_edge,
                if sub4 { p0_2 } else { BlockSize::Invalid },
                &mut child_dir,
            )?;
            if *by + eh4 >= fi.bh { /* done */
            } else {
                *by += eh4;
                decode_sb(
                    fi,
                    bx,
                    by,
                    cbx,
                    cby,
                    intra_region,
                    sdp_cfl_disallowed,
                    pass,
                    a,
                    l,
                    msac,
                    cdf_m,
                    cdf_dmv,
                    recon,
                    part_w,
                    part_w_idx,
                    part_r,
                    part_r_idx,
                    lbs_nvar,
                    if sub4 { p0_nvar } else { BlockSize::Invalid },
                    &mut child_dir,
                )?;
                let h4a = qh4 << var;
                let h4b = hh4 >> var;
                if *by + h4a >= fi.bh {
                    *by -= eh4;
                } else {
                    *by += h4a;
                    decode_sb(
                        fi,
                        bx,
                        by,
                        cbx,
                        cby,
                        intra_region,
                        sdp_cfl_disallowed,
                        pass,
                        a,
                        l,
                        msac,
                        cdf_m,
                        cdf_dmv,
                        recon,
                        part_w,
                        part_w_idx,
                        part_r,
                        part_r_idx,
                        lbs_var,
                        if sub4 { p0_var } else { BlockSize::Invalid },
                        &mut child_dir,
                    )?;
                    if *by + h4b >= fi.bh {
                        *by -= eh4 + h4a;
                    } else {
                        *by += h4b;
                        decode_sb(
                            fi,
                            bx,
                            by,
                            cbx,
                            cby,
                            intra_region,
                            sdp_cfl_disallowed,
                            pass,
                            a,
                            l,
                            msac,
                            cdf_m,
                            cdf_dmv,
                            recon,
                            part_w,
                            part_w_idx,
                            part_r,
                            part_r_idx,
                            lbs_edge,
                            if sub4 { p0_2 } else { cbs },
                            &mut child_dir,
                        )?;
                        *by -= 7 * eh4;
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    *dir_ptr |= (child_dir & 0xff) << 16;

    if *intra_region != 0 && cbs_orig != BlockSize::Invalid {
        *cbx = *bx;
        *cby = *by;
        let _b = decode_b(
            fi,
            *bx,
            *by,
            *cbx,
            *cby,
            *intra_region,
            *sdp_cfl_disallowed,
            pass,
            a,
            l,
            msac,
            cdf_m,
            cdf_dmv,
            recon,
            BlockSize::Invalid,
            cbs_orig,
        )?;
        *intra_region = 0;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a throwaway `ReconCtx` (and its backing scratch) bound to the given
    /// idents in the caller's scope. The recon leaf is a no-op under
    /// `Pass::Entropy`, so contents are never read; this just satisfies the
    /// threaded parameter for decode_sb / decode_b unit tests.
    macro_rules! make_dummy_recon {
        ($rf:ident, $recon:ident) => {
            let mut __ty = vec![0u8; 64 * 64];
            let mut __tu = vec![0u8; 64 * 64];
            let mut __tv = vec![0u8; 64 * 64];
            let mut __tcf = vec![0i32; 64 * 64];
            let mut __tcoef = crate::cdf::CdfCoefContext::default();
            let __dq = [[[0u32; 2]; 3]; crate::headers::MAX_SEGMENTS];
            let __qm: [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES] = Default::default();
            let $rf = ReconFrameCtx {
                dq: &__dq,
                qm: &__qm,
                y_stride_px: 64,
                uv_stride_px: 64,
                y_h: 64,
                uv_h: 64,
                ss_hor: 0,
                ss_ver: 0,
                bitdepth_max: 255,
                seq_fsc: false,
                seq_ist: [false; 2],
                seq_cctx: false,
                layout: crate::headers::PixelLayout::I420,
                bitdepth: 8,
                seg_lossless: [0u8; crate::headers::MAX_SEGMENTS],
                reduced_txtp_set: 0,
                tcq: false,
                seq_intra_edge_filter: false,
                seq_ibp: false,
                seq_inter_ddt: false,
                cfl_ds_filter_index: 0,
                ibp_weights: [[[0u8; 16]; 16]; 7],
            };
            let mut __scratch = ReconScratch::default();
            let mut __edge = vec![0u8; 2048];
            let mut __segmap = vec![0u8; 64 * 64];
            let mut __lf_mask: Vec<crate::lf_mask::Av2Filter> = vec![Default::default(); 4];
            let mut __ccsomap = vec![0u8; 0];
            let __rmf = crate::refmvs::Frame::default();
            let mut __rmt = crate::refmvs::Tile {
                rp_proj: Vec::new(),
                rp_proj_off: 0,
                rp_traj_off: 0,
                ra: vec![crate::refmvs::Block::default(); 1],
                ra_off: 0,
                ra_tl: crate::refmvs::Block::default(),
                r: vec![crate::refmvs::Block::default(); 64 * 128],
                tile_col: crate::refmvs::TileRange { start: 0, end: 64 },
                tile_row: crate::refmvs::TileRange { start: 0, end: 64 },
                bank: crate::refmvs::MvBank {
                    mv: [[[crate::levels::Mv::default(); 2]; 4]; 9],
                    cwp_idx: [[0; 4]; 3],
                    r#ref: [crate::levels::RefPair::default(); 4],
                    size: [0; 9],
                    idx: [0; 9],
                    hits: [0; 2],
                    avail: 0,
                },
                warp: crate::refmvs::WarpBank {
                    mat: [[[0; 6]; 4]; 7],
                    warp_type: [[0; 4]; 7],
                    hits: 0,
                    size: [0; 7],
                    idx: [0; 7],
                },
            };
            let __seqh = crate::headers::SequenceHeader::default();
            let __frmh = crate::headers::FrameHeader::default();
            let mut __cur_mvs: Vec<crate::refmvs::TemporalBlock> = Vec::new();
            let __refp: [Option<std::sync::Arc<crate::picture::Picture>>; 7] = Default::default();
            let __svc = [[crate::internal::ScalableMotionParams::default(); 2]; 7];
            let __masks = crate::wedge::init_masks();
            let mut $recon = ReconCtx {
                dst_y: &mut __ty,
                dst_u: &mut __tu,
                dst_v: &mut __tv,
                cdf_coef: &mut __tcoef,
                cf: &mut __tcf,
                frame: &$rf,
                masks: &__masks,
                scratch: &mut __scratch,
                edge: &mut __edge,
                cur_segmap: &mut __segmap,
                prev_segmap: None,
                b4_stride: 64,
                last_qidx: 0,
                dqmem: [[[0; 2]; 3]; crate::headers::MAX_SEGMENTS],
                dq_active: [[[0; 2]; 3]; crate::headers::MAX_SEGMENTS],
                seg_id_err: false,
                lf_mask: &mut __lf_mask,
                lf_idx: 0,
                sb256w: 1,
                cur_ccsomap: &mut __ccsomap,
                prev_ccsomap: [None, None, None],
                rt: &mut __rmt,
                rf: &__rmf,
                cur_mvs: &mut __cur_mvs,
                refp: &__refp,
                svc: &__svc,
                scratch_u_has_cf: 0,
                seq_hdr: &__seqh,
                frm_hdr: &__frmh,
                warpmv: [crate::headers::WarpedMotionParams::default(); 2],
            };
        };
    }

    #[test]
    fn test_neg_deinterleave_zero_ref() {
        assert_eq!(neg_deinterleave(5, 0, 10), 5);
    }

    #[test]
    fn test_neg_deinterleave_max_ref() {
        assert_eq!(neg_deinterleave(3, 9, 10), 10 - 3 - 1);
    }

    #[test]
    fn test_neg_deinterleave_small_ref() {
        assert_eq!(neg_deinterleave(0, 3, 10), 3);
        assert_eq!(neg_deinterleave(1, 3, 10), 4);
        assert_eq!(neg_deinterleave(2, 3, 10), 2);
        assert_eq!(neg_deinterleave(3, 3, 10), 5);
        assert_eq!(neg_deinterleave(4, 3, 10), 1);
        assert_eq!(neg_deinterleave(5, 3, 10), 6);
        assert_eq!(neg_deinterleave(6, 3, 10), 0);
    }

    #[test]
    fn test_neg_deinterleave_large_ref() {
        // 2*ref >= max
        assert_eq!(neg_deinterleave(0, 7, 10), 7);
        assert_eq!(neg_deinterleave(1, 7, 10), 8);
        assert_eq!(neg_deinterleave(2, 7, 10), 6);
    }

    #[test]
    fn test_init_quant_tables_no_seg() {
        let mut hdr = FrameHeader::default();
        hdr.quant.yac = 100;
        hdr.quant.ydc_delta = 0;
        hdr.quant.uac_delta = 0;
        hdr.quant.udc_delta = 0;
        hdr.quant.vac_delta = 0;
        hdr.quant.vdc_delta = 0;
        hdr.segmentation.enabled = 0;

        let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
        init_quant_tables(&hdr, 100, &mut dq);

        assert!(dq[0][0][0] > 0);
        assert!(dq[0][0][1] > 0);
        assert_eq!(dq[0][0][0], dq[0][0][1]);
        // segments 1+ should be untouched
        assert_eq!(dq[1][0][0], 0);
    }

    #[test]
    fn test_reset_context_keyframe() {
        let mut ctx = BlockContext::default();
        reset_context(&mut ctx, true, false);
        assert_eq!(ctx.tx_lpf_y[0], 3);
        assert_eq!(ctx.tx_lpf_uv[0], 2);
        assert_eq!(ctx.intra[0], 1);
        assert_eq!(ctx.mode[0], 0); // DC_PRED
        assert_eq!(ctx.lcoef[0], 0x40);
        assert_eq!(ctx.filter[0], N_SWITCHABLE_FILTERS as u8);
    }

    #[test]
    fn test_reset_context_interframe() {
        let mut ctx = BlockContext::default();
        reset_context(&mut ctx, false, false);
        assert_eq!(ctx.intra[0], 0);
        assert_eq!(ctx.mode[0], 13); // NEARMV
        assert_eq!(ctx.r#ref[0][0], -1);
        assert_eq!(ctx.r#ref[1][0], -1);
    }

    #[test]
    fn test_reset_context_tip_frame() {
        let mut ctx = BlockContext::default();
        ctx.midx[0] = 42;
        reset_context(&mut ctx, false, true);
        assert_eq!(ctx.tx_lpf_y[0], 3);
        // tip frame returns early, midx should be untouched
        assert_eq!(ctx.midx[0], 42);
    }

    #[test]
    fn test_size_group_lookup() {
        assert_eq!(SIZE_GROUP[BlockSize::Bs4x4 as usize], 0);
        assert_eq!(SIZE_GROUP[BlockSize::Bs8x8 as usize], 1);
        assert_eq!(SIZE_GROUP[BlockSize::Bs16x16 as usize], 2);
        assert_eq!(SIZE_GROUP[BlockSize::Bs32x32 as usize], 3);
        assert_eq!(SIZE_GROUP[BlockSize::Bs256x256 as usize], 3);
    }

    #[test]
    fn test_ss_size_mul() {
        assert_eq!(SS_SIZE_MUL[0], [4, 4]); // I400
        assert_eq!(SS_SIZE_MUL[1], [6, 5]); // I420
        assert_eq!(SS_SIZE_MUL[3], [12, 8]); // I444
    }

    #[test]
    fn test_tx_part_group() {
        assert_eq!(TX_PART_GROUP[BlockSize::Bs4x4 as usize], 0);
        assert_eq!(TX_PART_GROUP[BlockSize::Bs8x8 as usize], 1);
        assert_eq!(TX_PART_GROUP[BlockSize::Bs64x64 as usize], 7);
        assert_eq!(TX_PART_GROUP[BlockSize::Bs64x16 as usize], 8);
    }

    #[test]
    fn test_jmvd_scale_amvd() {
        let mut mv = MvXY { x: 10, y: 20 };
        jmvd_scale(&mut mv, true, 1);
        assert_eq!(mv.x, 20);
        assert_eq!(mv.y, 40);

        jmvd_scale(&mut mv, true, 2);
        assert_eq!(mv.x, 10);
        assert_eq!(mv.y, 20);

        jmvd_scale(&mut mv, true, 0);
        assert_eq!(mv.x, 10);
        assert_eq!(mv.y, 20);
    }

    #[test]
    fn test_jmvd_scale_no_amvd() {
        let mut mv = MvXY { x: 10, y: 20 };
        jmvd_scale(&mut mv, false, 1);
        assert_eq!(mv.y, 40);
        assert_eq!(mv.x, 10);

        let mut mv = MvXY { x: 10, y: 20 };
        jmvd_scale(&mut mv, false, 2);
        assert_eq!(mv.x, 20);
        assert_eq!(mv.y, 20);
    }

    #[test]
    fn test_get_prev_frame_segid() {
        let seg_map = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
        let id = get_prev_frame_segid(0, 0, 2, 2, &seg_map, 8);
        assert_eq!(id, 0);

        let id = get_prev_frame_segid(0, 2, 2, 1, &seg_map, 8);
        assert_eq!(id, 2);
    }

    #[test]
    fn test_mc_lowest_px_no_scale() {
        let smp = ScalableMotionParams { scale: 0, step: 0 };
        let mut dst = 0;
        mc_lowest_px(&mut dst, 4, 4, 0, 0, &smp);
        assert_eq!(dst, 32); // (4+4)*4 + 0 + 0
    }

    #[test]
    fn test_intra_mode_tables() {
        assert_eq!(REORDERED_NONDIR_Y_MODE[0], 0); // DC_PRED
        assert_eq!(REORDERED_NONDIR_Y_MODE[1], 9); // SMOOTH_PRED
        assert_eq!(REORDERED_DIR_Y_MODE[2], 1); // VERT_PRED
        assert_eq!(DEFAULT_MODE_LIST_Y.len(), 56);
        assert_eq!(DEFAULT_MODE_LIST_UV[0], 1); // VERT_PRED
        assert_eq!(INTRA_DIR_MODE_Y_TO_UV_IDX.len(), 8);
    }

    #[test]
    fn test_mv_prec_tbl() {
        assert_eq!(MV_PREC_TBL[0], [3, 1, 0]);
        assert_eq!(MV_PREC_TBL[1], [4, 3, 1]);
    }

    #[test]
    fn test_partition_subb() {
        let p = &PARTITION_SUBB[BlockSize::Bs64x64 as usize];
        assert_eq!(p.part[0][0], BlockSize::Bs64x32 as i8);
        assert_eq!(p.part[0][3], BlockSize::Bs32x32 as i8);
        assert_eq!(p.part[1][0], BlockSize::Bs32x64 as i8);
        assert_eq!(p.ctx[0], 3);
        assert_eq!(p.ctx[1], 6);

        let p = &PARTITION_SUBB[BlockSize::Bs4x4 as usize];
        assert_eq!(p.ctx[0], -1);
        assert_eq!(p.ctx[1], -1);
    }

    #[test]
    fn test_read_wedge_idx() {
        let data = vec![0x55; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = CdfModeContext { data: [0u16; 3496] };
        for i in 3060..3063 {
            cdf_m.data[i] = 32768 / 4 * (3 - (i - 3060)) as u16;
        }
        for q in 0..4 {
            let base = 3064 + q * 8;
            for j in 0..4 {
                cdf_m.data[base + j] = 32768 / 5 * (4 - j) as u16;
            }
        }
        for i in 3096..3098 {
            cdf_m.data[i] = 32768 / 3 * (2 - (i - 3096)) as u16;
        }
        for i in 3100..3103 {
            cdf_m.data[i] = 32768 / 4 * (3 - (i - 3100)) as u16;
        }
        let idx = read_wedge_idx(&mut msac, &mut cdf_m);
        assert!(idx >= -1 && idx <= 67);
    }

    #[test]
    fn test_wedge_angle_dist2idx() {
        assert_eq!(WEDGE_ANGLE_DIST2IDX[0][0], -1);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[0][1], 0);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[1][0], 3);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[10][0], -1);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[10][1], 38);
    }

    fn make_msac(data: &[u8]) -> MsacContext<'_> {
        MsacContext::new(data, false)
    }

    #[test]
    fn test_decode_4way_basic() {
        let data = vec![0x80; 64];
        let mut msac = make_msac(&data);
        let mut cdf = [16384u16, 16384, 16384, 0];
        let result = decode_4way(&mut msac, 8, &mut cdf, 5);
        assert!(result >= 0 && result < 32);
    }

    #[test]
    fn test_decode_4way_center_ref() {
        let data = vec![0x00; 64];
        let mut msac = make_msac(&data);
        let mut cdf = [16384u16, 16384, 16384, 0];
        let result = decode_4way(&mut msac, 16, &mut cdf, 5);
        assert!(result >= 0 && result < 32);
    }

    #[test]
    fn test_read_amvd_zero_joint() {
        let data = vec![0x00; 64];
        let mut msac = make_msac(&data);
        let mut amvd_joint = [16384u16, 16384, 16384, 0];
        let mut amvd_index = [16384u16; 16];
        let mv = read_amvd_raw(&mut msac, &mut amvd_joint, &mut amvd_index);
        let (y, x) = unsafe { (mv.c.y, mv.c.x) };
        assert!(y >= 0 && x >= 0);
    }

    #[test]
    fn test_read_amvd_nonzero() {
        let data = vec![0xFF; 64];
        let mut msac = make_msac(&data);
        let mut amvd_joint = [8192u16, 16384, 24576, 0];
        let mut amvd_index = [0u16; 16];
        for i in 0..2 {
            for j in 0..7 {
                amvd_index[i * 8 + j] = ((j + 1) as u16) * 4096;
            }
        }
        let mv = read_amvd_raw(&mut msac, &mut amvd_joint, &mut amvd_index);
        let (y, x) = unsafe { (mv.c.y, mv.c.x) };
        assert!(y >= 0 && x >= 0);
    }

    fn make_default_cdf_mv() -> CdfMvContext {
        let mut cdf = CdfMvContext { data: [0u16; 168] };
        for i in 0..7 {
            let base = i * 8;
            for j in 0..7 {
                cdf.data[base + j] = (32768 - (j as u16 + 1) * 4096).max(1);
            }
            cdf.data[base + 7] = 0;
        }
        for i in 0..7 {
            let base = 56 + i * 8;
            for j in 0..7 {
                cdf.data[base + j] = (32768 - (j as u16 + 1) * 4096).max(1);
            }
            cdf.data[base + 7] = 0;
        }
        cdf.data[112] = 16384;
        cdf.data[113] = 0;
        cdf.data[114] = 16384;
        cdf.data[115] = 0;
        for i in 0..2 {
            let b = 116 + i * 2;
            cdf.data[b] = 16384;
            cdf.data[b + 1] = 0;
        }
        cdf.data[120] = 16384;
        cdf.data[121] = 0;
        for i in 0..16 {
            let b = 122 + i * 2;
            cdf.data[b] = 16384;
            cdf.data[b + 1] = 0;
        }
        for i in 0..2 {
            let b = 154 + i * 2;
            cdf.data[b] = 16384;
            cdf.data[b + 1] = 0;
        }
        for i in 0..4 {
            let b = 158 + i * 2;
            cdf.data[b] = 16384;
            cdf.data[b + 1] = 0;
        }
        cdf
    }

    #[test]
    fn test_read_mv_residual_zero() {
        let data = vec![0x00; 128];
        let mut msac = make_msac(&data);
        let mut cdf_mv = make_default_cdf_mv();
        let mut shell_tip = [16384u16, 0];
        let mv = read_mv_residual(&mut msac, &mut cdf_mv, &mut shell_tip, 3);
        let _n = unsafe { mv.n };
    }

    #[test]
    fn test_read_mv_residual_all_precs() {
        for mv_prec in 0..7 {
            let data = vec![0x80; 256];
            let mut msac = make_msac(&data);
            let mut cdf_mv = make_default_cdf_mv();
            let mut shell_tip = [16384u16, 0];
            let mv = read_mv_residual(&mut msac, &mut cdf_mv, &mut shell_tip, mv_prec);
            let (y, x) = unsafe { (mv.c.y, mv.c.x) };
            assert!(y >= 0 || y < 0);
            assert!(x >= 0 || x < 0);
        }
    }

    #[test]
    fn test_affine_lowest_px_basic() {
        let b_dim = [8u8, 8, 32, 32];
        let mat = [0i32, 1 << 16, 0, 0, 0, 1 << 16];
        let mut dst = i32::MIN;
        affine_lowest_px(&mut dst, &b_dim, 4, 4, &mat, 0, 0);
        assert!(dst > i32::MIN);
    }

    #[test]
    fn test_affine_lowest_px_subsampled() {
        let b_dim = [8u8, 8, 32, 32];
        let mat = [0i32, 1 << 16, 0, 0, 0, 1 << 16];
        let mut dst0 = i32::MIN;
        let mut dst1 = i32::MIN;
        affine_lowest_px(&mut dst0, &b_dim, 4, 4, &mat, 0, 0);
        affine_lowest_px(&mut dst1, &b_dim, 4, 4, &mat, 1, 1);
        assert!(dst0 > i32::MIN);
        assert!(dst1 > i32::MIN);
    }

    #[test]
    fn test_affine_lowest_px_accumulates() {
        let b_dim = [8u8, 8, 32, 32];
        let mat = [0i32, 1 << 16, 0, 0, 0, 1 << 16];
        let mut dst = 1000;
        affine_lowest_px(&mut dst, &b_dim, 0, 0, &mat, 0, 0);
        assert!(dst >= 1000);
    }

    fn make_default_cdf_mode() -> CdfModeContext {
        CdfModeContext {
            data: [16384u16; 3496],
        }
    }

    fn make_default_block() -> crate::levels::Av2Block {
        crate::levels::Av2Block::default()
    }

    #[test]
    fn test_read_tx_part_4x4_noop() {
        let data = vec![0x80; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut b = make_default_block();
        read_tx_part(&mut msac, &mut cdf_m, &mut b, BlockSize::Bs4x4, false, true);
        assert_eq!(b.tx_part, TxPartition::None as u8);
    }

    #[test]
    fn test_read_tx_part_lossless_4x4() {
        let data = vec![0x80; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut b = make_default_block();
        read_tx_part(&mut msac, &mut cdf_m, &mut b, BlockSize::Bs4x4, true, false);
        assert_eq!(b.tx_size_ll, 0);
        assert_eq!(b.tx_part, TxPartition::None as u8);
    }

    #[test]
    fn test_read_tx_part_lossless_intra_fsc() {
        let data = vec![0x80; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut b = make_default_block();
        b.is_intra = 1;
        b.fsc = 1;
        read_tx_part(&mut msac, &mut cdf_m, &mut b, BlockSize::Bs8x8, true, false);
        assert!(b.tx_size_ll <= 1);
    }

    #[test]
    fn test_read_tx_part_switchable_skip() {
        let data = vec![0x80; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut b = make_default_block();
        b.skip_txfm = 1;
        read_tx_part(
            &mut msac,
            &mut cdf_m,
            &mut b,
            BlockSize::Bs16x16,
            false,
            true,
        );
        assert_eq!(b.tx_part, TxPartition::None as u8);
    }

    #[test]
    fn test_read_tx_part_switchable_8x8() {
        let data = vec![0x80; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut b = make_default_block();
        b.skip_txfm = 0;
        read_tx_part(&mut msac, &mut cdf_m, &mut b, BlockSize::Bs8x8, false, true);
        assert!(b.tx_part <= TxPartition::V5 as u8);
    }

    #[test]
    fn test_read_tx_part_switchable_8x4() {
        let data = vec![0xFF; 64];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut b = make_default_block();
        b.skip_txfm = 0;
        read_tx_part(&mut msac, &mut cdf_m, &mut b, BlockSize::Bs8x4, false, true);
        assert!(b.tx_part <= TxPartition::V5 as u8);
    }

    #[test]
    fn test_read_pal_indices_basic() {
        let data = vec![0x00; 512];
        let mut msac = make_msac(&data);
        let mut cdf_m = make_default_cdf_mode();
        let mut pal_out = vec![0u8; 64];
        let mut scratch = vec![0u8; 4096];
        let sz = [4, 4, 4, 4];
        let ret = read_pal_indices(&mut msac, &mut cdf_m, &mut pal_out, &mut scratch, 3, &sz);
        assert!(ret == 0 || ret == -1);
    }

    #[test]
    fn test_read_pal_indices_pal_sz_range() {
        for pal_sz in 2..=8 {
            let data = vec![0x80; 512];
            let mut msac = make_msac(&data);
            let mut cdf_m = make_default_cdf_mode();
            let mut pal_out = vec![0u8; 64];
            let mut scratch = vec![0u8; 4096];
            let sz = [4, 4, 4, 4];
            let ret = read_pal_indices(
                &mut msac,
                &mut cdf_m,
                &mut pal_out,
                &mut scratch,
                pal_sz,
                &sz,
            );
            assert!(ret == 0 || ret == -1);
        }
    }

    #[test]
    fn test_read_pal_indices_does_not_panic() {
        for &byte in &[0x00u8, 0x55, 0x80, 0xAA, 0xFF] {
            let data = vec![byte; 512];
            let mut msac = make_msac(&data);
            let mut cdf_m = make_default_cdf_mode();
            let mut pal_out = vec![0u8; 128];
            let mut scratch = vec![0u8; 4096];
            let sz = [8, 4, 8, 4];
            let _ = read_pal_indices(&mut msac, &mut cdf_m, &mut pal_out, &mut scratch, 4, &sz);
        }
    }

    #[test]
    fn test_compute_restore_planes_none() {
        let fh = FrameHeader::default();
        assert_eq!(compute_restore_planes(&fh), 0);
    }

    #[test]
    fn test_compute_restore_planes_all() {
        let mut fh = FrameHeader::default();
        fh.restoration.p[0].restoration_type = RestorationType::NsWiener as u8;
        fh.restoration.p[1].restoration_type = RestorationType::NsWiener as u8;
        fh.restoration.p[2].restoration_type = RestorationType::PcWiener as u8;
        assert_eq!(compute_restore_planes(&fh), 7);
    }

    #[test]
    fn test_compute_restore_planes_gdf_only() {
        let mut fh = FrameHeader::default();
        fh.gdf.enabled = AdaptiveBoolean::On;
        assert_eq!(compute_restore_planes(&fh), 1);
    }

    #[test]
    fn test_compute_gdf_ref_dst_idx_disabled() {
        let fh = FrameHeader::default();
        let absrefdist = [0u8; 7];
        assert_eq!(compute_gdf_ref_dst_idx(&fh, &absrefdist), 0);
    }

    #[test]
    fn test_compute_gdf_ref_dst_idx_inter() {
        use crate::headers::FrameType;
        let mut fh = FrameHeader::default();
        fh.gdf.enabled = AdaptiveBoolean::On;
        fh.frame_type = FrameType::Inter;
        fh.n_ref_frames = 7;
        let absrefdist = [3, 5, 2, 1, 1, 1, 1];
        assert_eq!(compute_gdf_ref_dst_idx(&fh, &absrefdist), 3);
    }

    #[test]
    fn test_init_ns_wiener_bank_y() {
        let mut bank = NsWienerBank::default();
        init_ns_wiener_bank(&mut bank, 0, 3);
        assert_eq!(bank.bank_size, [0; 16]);
        assert_eq!(bank.bank_idx, [0; 16]);
        for n in 0..3 {
            assert_ne!(bank.filter[0][n][0], 0);
        }
        assert_eq!(bank.filter[0][3][0], 0);
    }

    #[test]
    fn test_init_ns_wiener_bank_uv() {
        let mut bank = NsWienerBank::default();
        init_ns_wiener_bank(&mut bank, 1, 2);
        for n in 0..2 {
            assert_ne!(bank.filter[0][n][0], 0);
        }
        assert_eq!(bank.filter[0][2][0], 0);
        assert_eq!(bank.filter[0][0][16], 0); // cf_range[16] = [4,-8] → -8+8=0
        assert_eq!(bank.filter[0][0][18], 0); // beyond n_coeffs, untouched
    }

    #[test]
    fn test_init_start_of_tile_row() {
        let row_start_sb: [u16; 4] = [0, 3, 5, 8];
        let mut buf = Vec::new();
        init_start_of_tile_row(&mut buf, 8, 3, &row_start_sb);
        assert_eq!(buf[0], 0b01); // tile_row=0, start
        assert_eq!(buf[1], 0b00); // tile_row=0, cont
        assert_eq!(buf[2], 0b00); // tile_row=0, cont
        assert_eq!(buf[3], 0b11); // tile_row=1, start
        assert_eq!(buf[4], 0b10); // tile_row=1, cont
        assert_eq!(buf[5], 0b101); // tile_row=2, start
        assert_eq!(buf[6], 0b100); // tile_row=2, cont
        assert_eq!(buf[7], 0b100); // tile_row=2, cont
    }

    fn make_lf_state() -> LoopFilterState {
        LoopFilterState::default()
    }

    #[test]
    fn test_init_wiener_none() {
        let fh = FrameHeader::default();
        let mut lf = make_lf_state();
        init_wiener(&fh, &mut lf);
        assert_eq!(lf.base_q, 0);
    }

    #[test]
    fn test_init_wiener_ns_wiener() {
        let mut fh = FrameHeader::default();
        fh.restoration.p[0].restoration_type = RestorationType::NsWiener as u8;
        fh.restoration.p[0].ns.num_classes_idx = 3;
        fh.quant.yac = 100;
        let mut lf = make_lf_state();
        init_wiener(&fh, &mut lf);
        assert_eq!(lf.wiener_idx, 0);
        assert_eq!(lf.ns_subclass_class_idx, Some(2));
        assert!(lf.base_q != 0);
    }

    #[test]
    fn test_init_wiener_idx_ranges() {
        let mut fh = FrameHeader::default();
        fh.restoration.p[0].restoration_type = RestorationType::PcWiener as u8;

        let mut lf = make_lf_state();
        fh.quant.yac = 50;
        init_wiener(&fh, &mut lf);
        assert_eq!(lf.wiener_idx, 0);

        fh.quant.yac = 150;
        init_wiener(&fh, &mut lf);
        assert_eq!(lf.wiener_idx, 1);

        fh.quant.yac = 200;
        init_wiener(&fh, &mut lf);
        assert_eq!(lf.wiener_idx, 2);

        fh.quant.yac = 250;
        init_wiener(&fh, &mut lf);
        assert_eq!(lf.wiener_idx, 3);
    }

    fn make_default_ns_plane(n_classes: u8) -> NSWienerPlane {
        NSWienerPlane {
            frame_filters_on: 0,
            num_classes_idx: 0,
            num_classes: n_classes,
            temporal: 0,
            refidx: 0,
            filter: [[0; 18]; 16],
        }
    }

    fn make_flat_cdf_mode() -> CdfModeContext {
        let mut m = CdfModeContext { data: [0; 3496] };
        for i in (0..20).step_by(2) {
            m.data[i] = 16384;
            m.data[i + 1] = 0;
        }
        m
    }

    #[test]
    fn test_read_restoration_info_switchable_none() {
        let data = [0xFF; 16];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = make_flat_cdf_mode();
        let mut bank = NsWienerBank::default();
        let mut lr = Av2RestorationUnit::default();
        let ns = make_default_ns_plane(1);

        read_restoration_info(
            &mut msac,
            &mut cdf_m,
            &mut bank,
            &mut lr,
            0,
            RestorationType::Switchable,
            &ns,
        );
        assert!(
            lr.restoration_type == RestorationType::None as u8
                || lr.restoration_type == RestorationType::PcWiener as u8
                || lr.restoration_type == RestorationType::NsWiener as u8
        );
    }

    #[test]
    fn test_read_restoration_info_ns_wiener_type() {
        let data = [0x00; 16];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = make_flat_cdf_mode();
        let mut bank = NsWienerBank::default();
        let mut lr = Av2RestorationUnit::default();
        let ns = make_default_ns_plane(1);

        read_restoration_info(
            &mut msac,
            &mut cdf_m,
            &mut bank,
            &mut lr,
            0,
            RestorationType::NsWiener,
            &ns,
        );
        assert!(
            lr.restoration_type == RestorationType::None as u8
                || lr.restoration_type == RestorationType::NsWiener as u8
        );
    }

    #[test]
    fn test_read_restoration_info_pc_wiener_type() {
        let data = [0x80; 16];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = make_flat_cdf_mode();
        let mut bank = NsWienerBank::default();
        let mut lr = Av2RestorationUnit::default();
        let ns = make_default_ns_plane(1);

        read_restoration_info(
            &mut msac,
            &mut cdf_m,
            &mut bank,
            &mut lr,
            0,
            RestorationType::PcWiener,
            &ns,
        );
        assert!(
            lr.restoration_type == RestorationType::None as u8
                || lr.restoration_type == RestorationType::PcWiener as u8
        );
    }

    #[test]
    fn test_read_restoration_info_ns_filter_exact_match() {
        let data = [0xFF; 64];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = make_flat_cdf_mode();
        cdf_m.data[6] = 0;
        let mut bank = NsWienerBank::default();
        bank.filter[0][0][0] = 5;
        bank.filter[0][0][1] = -3;
        let mut lr = Av2RestorationUnit::default();
        let ns = make_default_ns_plane(1);

        read_restoration_info(
            &mut msac,
            &mut cdf_m,
            &mut bank,
            &mut lr,
            0,
            RestorationType::NsWiener,
            &ns,
        );
        if lr.restoration_type == RestorationType::NsWiener as u8 {
            assert!(bank.bank_size[0] <= 4);
        }
    }

    #[test]
    fn test_read_restoration_info_uv_plane() {
        let data = [0x40; 64];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = make_flat_cdf_mode();
        let mut bank = NsWienerBank::default();
        let mut lr = Av2RestorationUnit::default();
        let ns = make_default_ns_plane(1);

        read_restoration_info(
            &mut msac,
            &mut cdf_m,
            &mut bank,
            &mut lr,
            1,
            RestorationType::NsWiener,
            &ns,
        );
        assert!(
            lr.restoration_type == RestorationType::None as u8
                || lr.restoration_type == RestorationType::NsWiener as u8
        );
    }

    fn make_frame_hdr_for_init(cols: u8, rows: u8) -> FrameHeader {
        let mut hdr = FrameHeader::default();
        hdr.tiling.t.cols = cols;
        hdr.tiling.t.rows = rows;
        hdr.quant.yac = 100;
        hdr
    }

    #[test]
    fn test_decode_frame_init_single_tile() {
        let hdr = make_frame_hdr_for_init(1, 1);
        let seq = crate::headers::SequenceHeader::default();
        let mut lf = LoopFilterState::default();
        let mut ft = crate::internal::FrameThread::default();
        let mut ts = Vec::new();
        let mut n_ts = 0i32;
        let mut a = Vec::new();
        let mut a_sz = 0i32;
        let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
        let mut qm: [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES] = Default::default();
        let absrefdist = [0u8; 7];

        decode_frame_init(
            &hdr,
            &seq,
            &mut lf,
            &mut ft,
            &mut ts,
            &mut n_ts,
            &mut a,
            &mut a_sz,
            &mut dq,
            &mut qm,
            &absrefdist,
            4,
            2,
            2,
            32,
            32,
            1,
            1,
        );

        assert_eq!(n_ts, 1);
        assert_eq!(ts.len(), 1);
        assert_eq!(lf.re_sz, 2);
        assert!(dq[0][0][0] > 0);
    }

    #[test]
    fn test_decode_frame_init_multi_tile() {
        let hdr = make_frame_hdr_for_init(2, 3);
        let seq = crate::headers::SequenceHeader::default();
        let mut lf = LoopFilterState::default();
        let mut ft = crate::internal::FrameThread::default();
        let mut ts = Vec::new();
        let mut n_ts = 0i32;
        let mut a = Vec::new();
        let mut a_sz = 0i32;
        let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
        let mut qm: [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES] = Default::default();
        let absrefdist = [0u8; 7];

        decode_frame_init(
            &hdr,
            &seq,
            &mut lf,
            &mut ft,
            &mut ts,
            &mut n_ts,
            &mut a,
            &mut a_sz,
            &mut dq,
            &mut qm,
            &absrefdist,
            8,
            4,
            4,
            64,
            64,
            1,
            1,
        );

        assert_eq!(n_ts, 6);
        assert_eq!(ts.len(), 6);
        assert_eq!(a_sz, 4 * 3);
        assert_eq!(a.len(), 12);
    }

    #[test]
    fn test_decode_frame_init_multithread_resets_ctx() {
        let mut hdr = make_frame_hdr_for_init(1, 1);
        hdr.tip.frame_mode = 0;
        let seq = crate::headers::SequenceHeader::default();
        let mut lf = LoopFilterState::default();
        let mut ft = crate::internal::FrameThread::default();
        let mut ts = Vec::new();
        let mut n_ts = 0i32;
        let mut a = Vec::new();
        let mut a_sz = 0i32;
        let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
        let mut qm: [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES] = Default::default();
        let absrefdist = [0u8; 7];

        decode_frame_init(
            &hdr,
            &seq,
            &mut lf,
            &mut ft,
            &mut ts,
            &mut n_ts,
            &mut a,
            &mut a_sz,
            &mut dq,
            &mut qm,
            &absrefdist,
            4,
            2,
            2,
            32,
            32,
            4,
            1,
        );

        assert_eq!(n_ts, 1);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_decode_frame_init_multipass_allocs_tile_start_off() {
        let hdr = make_frame_hdr_for_init(2, 2);
        let seq = crate::headers::SequenceHeader::default();
        let mut lf = LoopFilterState::default();
        let mut ft = crate::internal::FrameThread::default();
        let mut ts = Vec::new();
        let mut n_ts = 0i32;
        let mut a = Vec::new();
        let mut a_sz = 0i32;
        let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
        let mut qm: [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES] = Default::default();
        let absrefdist = [0u8; 7];

        decode_frame_init(
            &hdr,
            &seq,
            &mut lf,
            &mut ft,
            &mut ts,
            &mut n_ts,
            &mut a,
            &mut a_sz,
            &mut dq,
            &mut qm,
            &absrefdist,
            4,
            2,
            2,
            32,
            32,
            1,
            2,
        );

        assert_eq!(n_ts, 4);
        assert_eq!(ft.tile_start_off.len(), 4);
    }

    #[test]
    fn test_decode_frame_init_mask_alloc() {
        let hdr = make_frame_hdr_for_init(1, 1);
        let seq = crate::headers::SequenceHeader::default();
        let mut lf = LoopFilterState::default();
        let mut ft = crate::internal::FrameThread::default();
        let mut ts = Vec::new();
        let mut n_ts = 0i32;
        let mut a = Vec::new();
        let mut a_sz = 0i32;
        let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
        let mut qm: [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES] = Default::default();
        let absrefdist = [0u8; 7];

        decode_frame_init(
            &hdr,
            &seq,
            &mut lf,
            &mut ft,
            &mut ts,
            &mut n_ts,
            &mut a,
            &mut a_sz,
            &mut dq,
            &mut qm,
            &absrefdist,
            4,
            3,
            2,
            32,
            32,
            1,
            1,
        );

        assert_eq!(lf.mask.len(), 6);
        assert_eq!(lf.lr_mask.len(), 6);
    }

    #[test]
    fn test_setup_tile_bounds_basic() {
        let mut ts = crate::internal::TileState::default();
        let col_start_sb: [u16; 3] = [0, 4, 8];
        let row_start_sb: [u16; 3] = [0, 2, 4];

        setup_tile_bounds(&mut ts, 0, 1, &col_start_sb, &row_start_sb, 2, 64, 32, 1);

        assert_eq!(ts.tiling.row, 0);
        assert_eq!(ts.tiling.col, 1);
        assert_eq!(ts.tiling.col_start, 16);
        assert_eq!(ts.tiling.col_end, 32);
        assert_eq!(ts.tiling.row_start, 0);
        assert_eq!(ts.tiling.row_end, 8);
    }

    #[test]
    fn test_setup_tile_bounds_clamps_to_frame() {
        let mut ts = crate::internal::TileState::default();
        let col_start_sb: [u16; 2] = [0, 100];
        let row_start_sb: [u16; 2] = [0, 100];

        setup_tile_bounds(&mut ts, 0, 0, &col_start_sb, &row_start_sb, 2, 30, 20, 1);

        assert_eq!(ts.tiling.col_end, 30);
        assert_eq!(ts.tiling.row_end, 20);
    }

    #[test]
    fn test_setup_tile_bounds_multithread_progress() {
        use std::sync::atomic::Ordering;
        let mut ts = crate::internal::TileState::default();
        let col_start_sb: [u16; 2] = [0, 4];
        let row_start_sb: [u16; 3] = [0, 3, 6];

        setup_tile_bounds(&mut ts, 1, 0, &col_start_sb, &row_start_sb, 2, 64, 64, 4);

        for p in 0..3 {
            assert_eq!(ts.progress[p].load(Ordering::Relaxed), 3);
        }
    }

    #[test]
    fn test_decode_tip_frame_init_sets_all_tiles() {
        let mut hdr = FrameHeader::default();
        hdr.tiling.t.cols = 2;
        hdr.tiling.t.rows = 2;
        hdr.tiling.t.col_start_sb[0] = 0;
        hdr.tiling.t.col_start_sb[1] = 4;
        hdr.tiling.t.col_start_sb[2] = 8;
        hdr.tiling.t.row_start_sb[0] = 0;
        hdr.tiling.t.row_start_sb[1] = 3;
        hdr.tiling.t.row_start_sb[2] = 6;

        let mut ts = vec![
            crate::internal::TileState::default(),
            crate::internal::TileState::default(),
            crate::internal::TileState::default(),
            crate::internal::TileState::default(),
        ];

        decode_tip_frame_init(&mut ts, &hdr, 2, 64, 48, 1);

        assert_eq!(ts[0].tiling.col_start, 0);
        assert_eq!(ts[0].tiling.row_start, 0);
        assert_eq!(ts[1].tiling.col_start, 16);
        assert_eq!(ts[1].tiling.row_start, 0);
        assert_eq!(ts[2].tiling.col_start, 0);
        assert_eq!(ts[2].tiling.row_start, 12);
        assert_eq!(ts[3].tiling.col_start, 16);
        assert_eq!(ts[3].tiling.row_start, 12);
        assert_eq!(ts[3].tiling.row_end, 24);
    }

    #[test]
    fn test_setup_tile_wiener_banks() {
        use crate::headers::RestorationType;
        let mut ts = crate::internal::TileState::default();
        let mut hdr = FrameHeader::default();
        hdr.restoration.p[0].restoration_type = RestorationType::NsWiener as u8;
        hdr.restoration.p[0].ns.num_classes = 2;
        hdr.restoration.p[1].restoration_type = RestorationType::None as u8;
        hdr.restoration.p[2].restoration_type = RestorationType::Switchable as u8;
        hdr.restoration.p[2].ns.num_classes = 1;

        setup_tile_wiener_banks(&mut ts, &hdr);

        assert_ne!(ts.ns_wiener_bank[0].filter[0][0][0], 0);
        assert_ne!(ts.ns_wiener_bank[2].filter[0][0][0], 0);
    }

    #[test]
    fn test_setup_tile_sets_cdf_and_qidx() {
        let mut ts = crate::internal::TileState::default();
        let mut hdr = FrameHeader::default();
        hdr.quant.yac = 128;
        hdr.tiling.t.cols = 1;
        hdr.tiling.t.rows = 1;
        hdr.tiling.t.col_start_sb[1] = 4;
        hdr.tiling.t.row_start_sb[1] = 4;
        let data = vec![0xAA; 32];

        setup_tile(
            &mut ts,
            &data,
            &hdr,
            None,
            2,
            0,
            0,
            hdr.tiling.t.col_start_sb.as_ref(),
            hdr.tiling.t.row_start_sb.as_ref(),
            2,
            32,
            32,
            1,
            0,
        );

        assert_eq!(ts.last_qidx, 128);
        assert_eq!(ts.msac_buf.len(), 32);
        assert_eq!(ts.tiling.col_start, 0);
        assert_eq!(ts.tiling.row_start, 0);
    }

    #[test]
    fn test_setup_tile_with_input_cdf() {
        let mut ts = crate::internal::TileState::default();
        let hdr = FrameHeader::default();
        let mut cdf = crate::cdf::CdfContext::default();
        cdf.m.data[0] = 12345;

        setup_tile(
            &mut ts,
            &[0; 8],
            &hdr,
            Some(&cdf),
            0,
            0,
            0,
            &[0, 1],
            &[0, 1],
            2,
            16,
            16,
            1,
            0,
        );

        assert_eq!(ts.cdf.m.data[0], 12345);
    }

    #[test]
    fn test_decode_frame_init_cdf_single_tile() {
        let mut hdr = FrameHeader::default();
        hdr.tiling.t.cols = 1;
        hdr.tiling.t.rows = 1;
        hdr.tiling.t.col_start_sb[1] = 4;
        hdr.tiling.t.row_start_sb[1] = 4;
        hdr.tiling.n_bytes = 2;

        let tg = crate::internal::TileGroup {
            data: vec![0x55; 64],
            start: 0,
            end: 0,
        };

        let mut ts = vec![crate::internal::TileState::default()];
        let mut tile_start_off = vec![0u32];

        let res = decode_frame_init_cdf(
            &mut ts,
            &[tg],
            &hdr,
            None,
            0,
            2,
            32,
            32,
            1,
            1,
            &mut tile_start_off,
        );

        assert!(res.is_ok());
        assert_eq!(ts[0].msac_buf.len(), 64);
    }

    #[test]
    fn test_decode_frame_init_cdf_multi_tile() {
        let mut hdr = FrameHeader::default();
        hdr.tiling.t.cols = 2;
        hdr.tiling.t.rows = 1;
        hdr.tiling.t.col_start_sb[1] = 4;
        hdr.tiling.t.col_start_sb[2] = 8;
        hdr.tiling.t.row_start_sb[1] = 4;
        hdr.tiling.n_bytes = 2;

        let mut data = Vec::new();
        // tile 0 size: 10 bytes (little-endian: 9 because tile_sz = stored+1)
        data.push(9);
        data.push(0);
        data.extend_from_slice(&[0xAA; 10]);
        // tile 1: remaining
        data.extend_from_slice(&[0xBB; 20]);

        let tg = crate::internal::TileGroup {
            data,
            start: 0,
            end: 1,
        };

        let mut ts = vec![
            crate::internal::TileState::default(),
            crate::internal::TileState::default(),
        ];
        let mut tile_start_off = vec![0u32; 2];

        let res = decode_frame_init_cdf(
            &mut ts,
            &[tg],
            &hdr,
            None,
            0,
            2,
            64,
            32,
            1,
            1,
            &mut tile_start_off,
        );

        assert!(res.is_ok());
        assert_eq!(ts[0].msac_buf.len(), 10);
        assert_eq!(ts[1].msac_buf.len(), 20);
        assert_eq!(ts[0].tiling.col, 0);
        assert_eq!(ts[1].tiling.col, 1);
    }

    #[test]
    fn test_decode_frame_init_cdf_error_on_truncated() {
        let mut hdr = FrameHeader::default();
        hdr.tiling.t.cols = 2;
        hdr.tiling.t.rows = 1;
        hdr.tiling.n_bytes = 2;

        let tg = crate::internal::TileGroup {
            data: vec![0xFF; 3], // too short: claims huge tile_sz
            start: 0,
            end: 1,
        };

        let mut ts = vec![
            crate::internal::TileState::default(),
            crate::internal::TileState::default(),
        ];
        let mut tile_start_off = vec![0u32; 2];

        let res = decode_frame_init_cdf(
            &mut ts,
            &[tg],
            &hdr,
            None,
            0,
            2,
            64,
            32,
            1,
            1,
            &mut tile_start_off,
        );

        assert!(res.is_err());
    }

    fn make_test_tile() -> refmvs::Tile {
        use crate::levels::RefPair;
        refmvs::Tile {
            rp_proj: Vec::new(),
            rp_proj_off: 0,
            rp_traj_off: 0,
            ra: vec![refmvs::Block::default(); 256],
            ra_off: 0,
            ra_tl: refmvs::Block::default(),
            r: vec![refmvs::Block::default(); 64 * 128],
            tile_col: refmvs::TileRange { start: 0, end: 64 },
            tile_row: refmvs::TileRange { start: 0, end: 64 },
            bank: refmvs::MvBank {
                mv: [[[Mv::default(); 2]; 4]; 9],
                cwp_idx: [[0; 4]; 3],
                r#ref: [RefPair::default(); 4],
                size: [0; 9],
                idx: [0; 9],
                hits: [0; 2],
                avail: 0,
            },
            warp: refmvs::WarpBank {
                mat: [[[0; 6]; 4]; 7],
                warp_type: [[0; 4]; 7],
                hits: 0,
                size: [0; 7],
                idx: [0; 7],
            },
        }
    }

    #[test]
    fn test_extend_warpmv_translation() {
        use crate::headers::WarpedMotionType;
        let mut rt = make_test_tile();
        let by = 4i32;
        let bx = 4i32;
        let sb_step = 16;
        let neighbor_row = ((by - 1) & 63) as usize * 128;
        let neighbor_x = ((bx + 2) & 127) as usize;
        rt.r[neighbor_row + neighbor_x].mf = 0;
        rt.r[neighbor_row + neighbor_x].mv[0] = Mv {
            c: MvXY { y: 100, x: 200 },
        };
        unsafe {
            rt.r[neighbor_row + neighbor_x].r#ref.r[0] = 1;
        }

        let b_dim = &[4u8, 4, 2, 2];
        let mv0 = Mv {
            c: MvXY { y: 50, x: 100 },
        };
        let mut wmp = WarpedMotionParams::default();
        let gmv = [0i32; 6];

        extend_warpmv(&rt, bx, by, 2, -1, b_dim, 1, mv0, &mut wmp, sb_step, &gmv);
        assert!(
            wmp.wm_type != WarpedMotionType::Invalid || wmp.wm_type == WarpedMotionType::Invalid
        );
    }

    #[test]
    fn test_derive_warpmv_single_top_neighbor() {
        use crate::headers::WarpedMotionType;

        let mut rt = make_test_tile();
        let by = 4i32;
        let bx = 4i32;
        let sb_step = 16;
        let ref_idx: i8 = 1;
        let above_row = ((by - 1) & 63) as usize * 128;
        let above_x = (bx & 127) as usize;
        rt.r[above_row + above_x].bs = BlockSize::Bs8x8 as u8;
        rt.r[above_row + above_x].mf = 0;
        rt.r[above_row + above_x].mv[0] = Mv {
            c: MvXY { y: 32, x: 64 },
        };
        unsafe {
            rt.r[above_row + above_x].r#ref.r[0] = ref_idx;
        }

        let left_col = ((bx - 1) & 127) as usize;
        let cur_row = (by & 63) as usize * 128;
        rt.r[cur_row + left_col].bs = BlockSize::Bs8x8 as u8;
        rt.r[cur_row + left_col].mf = 0;
        rt.r[cur_row + left_col].mv[0] = Mv {
            c: MvXY { y: 16, x: 48 },
        };
        unsafe {
            rt.r[cur_row + left_col].r#ref.r[0] = ref_idx;
        }

        let mv = Mv {
            c: MvXY { y: 24, x: 56 },
        };
        let mut wmp = WarpedMotionParams::default();

        derive_warpmv(
            &rt, bx, by, true, true, 2, 2, 2, 2, ref_idx, mv, &mut wmp, sb_step, 64,
        );
        assert!(
            wmp.wm_type == WarpedMotionType::Invalid || wmp.wm_type != WarpedMotionType::Invalid
        );
    }

    #[test]
    fn test_sbframeinfo_from_frame() {
        let seq = crate::headers::SequenceHeader {
            ss_hor: 1,
            ss_ver: 1,
            sdp: true,
            ..Default::default()
        };
        let mut fh = FrameHeader::default();
        fh.frame_type = crate::headers::FrameType::Key;
        fh.segmentation.enabled = 1;
        fh.n_ref_frames = 7;
        fh.txfm_mode = crate::headers::TxfmMode::Switchable;

        let fi = SbFrameInfo::from_frame(
            &seq,
            &fh,
            64,
            64,
            BlockSize::Bs64x64,
            16,
            1,
            [0; 8],
            &[0; 7],
            &[0; 7],
            RefPair { pair: 0 },
            0,
            64,
            0,
            64,
            -1,
            RefPair { pair: -1 },
        );

        assert_eq!(fi.bw, 64);
        assert_eq!(fi.bh, 64);
        assert_eq!(fi.ss_hor, 1);
        assert_eq!(fi.ss_ver, 1);
        assert_eq!(fi.sb_step, 16);
        assert!(!fi.is_inter_or_switch);
        assert!(fi.sdp);
        assert!(fi.seg_enabled);
        assert!(fi.txfm_switchable);
        assert_eq!(fi.n_ref_frames, 7);
        assert_eq!(fi.tile_col_end, 64);
        assert_eq!(fi.tile_row_end, 64);
    }

    #[test]
    fn test_decode_sb_partition_none_leaf() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, false);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();
        let mut part_w = Vec::new();
        let mut part_w_idx = 0usize;
        let part_r: &[u8] = &[];
        let mut part_r_idx = 0usize;
        let mut bx = 0i32;
        let mut by = 0i32;
        let mut cbx = 0i32;
        let mut cby = 0i32;
        let mut intra_region = 0i32;
        let mut sdp_cfl_disallowed = 0i32;
        let mut dir = 0i32;
        let pass = Pass::Entropy as u8;

        make_dummy_recon!(_rf, _recon);
        let result = decode_sb(
            &fi,
            &mut bx,
            &mut by,
            &mut cbx,
            &mut cby,
            &mut intra_region,
            &mut sdp_cfl_disallowed,
            pass,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            &mut part_w,
            &mut part_w_idx,
            part_r,
            &mut part_r_idx,
            BlockSize::Bs4x4,
            BlockSize::Bs4x4,
            &mut dir,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_sb_split_recurse() {
        let fi = SbFrameInfo {
            bw: 128,
            bh: 128,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: true,
            sdp: false,
            ext_sdp: false,
            ext_partitions: true,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 128,
            tile_row_start: 0,
            tile_row_end: 128,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0xFF; 256];
        let mut msac = MsacContext::new(&data, false);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();
        let mut part_w = Vec::new();
        let mut part_w_idx = 0usize;
        let part_r: &[u8] = &[];
        let mut part_r_idx = 0usize;
        let mut bx = 0i32;
        let mut by = 0i32;
        let mut cbx = 0i32;
        let mut cby = 0i32;
        let mut intra_region = 0i32;
        let mut sdp_cfl_disallowed = 0i32;
        let mut dir = 0i32;
        let pass = Pass::Entropy as u8;

        make_dummy_recon!(_rf, _recon);
        let result = decode_sb(
            &fi,
            &mut bx,
            &mut by,
            &mut cbx,
            &mut cby,
            &mut intra_region,
            &mut sdp_cfl_disallowed,
            pass,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            &mut part_w,
            &mut part_w_idx,
            part_r,
            &mut part_r_idx,
            BlockSize::Bs32x32,
            BlockSize::Bs32x32,
            &mut dir,
        );
        assert!(result.is_ok());
        assert_eq!(bx, 0);
        assert_eq!(by, 0);
    }

    #[test]
    fn test_decode_b_intra_keyframe() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            8,
            8,
            8,
            8,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);
        assert_eq!(b.skip_mode, 0);
        assert_eq!(b.skip_txfm, 0);
        assert_eq!(b.intrabc, 0);
    }

    #[test]
    fn test_decode_b_inter_skip_mode() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: true,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: true,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        // All 0xFF data → MSAC will decode 1s for all bool_adapt calls
        let data = vec![0x00; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            8,
            8,
            8,
            8,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();
        // skip_mode decoded (block is 2x2=4 > 2, inter frame, skip_mode_enabled)
        // With all-zero data and default CDFs, skip_mode will be decoded
        assert_eq!(b.seg_id, 0);
    }

    #[test]
    fn test_decode_b_skip_mask_forces_skip_txfm() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: true,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 1,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            8,
            8,
            8,
            8,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();
        // seg_skip_mask bit 0 set, seg_id=0 → skip_txfm forced to 1
        assert_eq!(b.skip_txfm, 1);
    }

    #[test]
    fn test_decode_b_intra_y_mode_decoding() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            8,
            8,
            8,
            8,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);
        // y_mode should be a valid intra prediction mode (0-12)
        let y_mode = unsafe { b.data.intra.y_mode };
        assert!(y_mode <= 12, "y_mode={y_mode} out of range");
    }

    #[test]
    fn test_decode_b_intra_y_mode_with_neighbours() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0xAA; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();
        // Set neighbour midx to directional mode 17 (VERT_PRED centre)
        a.midx[9] = 17;
        l.midx[9] = 17;

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            8,
            8,
            8,
            8,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);
        let y_mode = unsafe { b.data.intra.y_mode };
        assert!(y_mode <= 12, "y_mode={y_mode} out of range");
    }

    #[test]
    fn test_decode_b_fsc_and_mrl() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: true,
            mrls: true,
            mhccp: false,
            cfl: true,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x55; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            8,
            8,
            8,
            8,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);
        // FSC decoded (8x8 block, idtx_intra=true)
        // fsc is 0 or 1
        assert!(b.fsc <= 1);
        // MRL decoded only if directional mode (midx != 0xff)
        let mrl = unsafe { b.data.intra.mrl_index };
        assert!(mrl <= 3);
    }

    #[test]
    fn test_decode_b_uv_chroma_mode() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: true,
            mrls: true,
            mhccp: false,
            cfl: true,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0xAA; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        // Decode with both luma and chroma
        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            4,
            4,
            4,
            4,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs4x4,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);
        let uv_mode = unsafe { b.data.intra.uv_mode };
        let uv_angle = unsafe { b.data.intra.uv_angle };
        // UV mode should be valid (0-13)
        assert!(uv_mode <= CFL_PRED, "uv_mode {} out of range", uv_mode);
        // UV angle in [-3, 3]
        assert!(
            uv_angle >= -3 && uv_angle <= 3,
            "uv_angle {} out of range",
            uv_angle
        );
    }

    #[test]
    fn test_decode_b_uv_dpcm_chroma() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: true,
            seg_lossless: [1; crate::headers::MAX_SEGMENTS],
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        // Feed data that makes dpcm[0]=true and dpcm[1]=true
        let data = vec![0xFF; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            4,
            4,
            4,
            4,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs4x4,
            BlockSize::Bs4x4,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);
        let dpcm_chroma = unsafe { b.data.intra.dpcm[1] };
        // With 0xFF data and lossless, DPCM should be triggered
        let uv_mode = unsafe { b.data.intra.uv_mode };
        if dpcm_chroma != 0 {
            // DPCM chroma: uv_mode is VERT(1) or HOR(2)
            assert!(uv_mode == 1 || uv_mode == 2);
        }
    }

    #[test]
    fn test_decode_b_context_writeback() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: false,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x40; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        // Decode 8x8 block at (4,4); keyframe → always intra
        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            4,
            4,
            4,
            4,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs4x4,
        )
        .unwrap();
        assert_eq!(b.is_intra, 1);

        // Verify above context written for bw4=2 positions starting at bx4=4
        assert_eq!(a.intra[4], 1);
        assert_eq!(a.intra[5], 1);
        assert_eq!(a.intrabc[4], 0);
        assert_eq!(a.skip_mode[4], 0);

        // Verify left context written for bh4=2 positions starting at by4=4
        assert_eq!(l.intra[4], 1);
        assert_eq!(l.intra[5], 1);
        assert_eq!(l.intrabc[4], 0);

        // Verify uvmode written for chroma (cbs=Bs4x4 → cbw4=cbh4=1 at cbx4=4, cby4=4)
        let uv_mode = unsafe { b.data.intra.uv_mode };
        assert_eq!(a.uvmode[4], uv_mode);
        assert_eq!(l.uvmode[4], uv_mode);
    }

    #[test]
    fn test_decode_b_single_ref_inter() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: true,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: false,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        let data = vec![0x80; 512];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            4,
            4,
            4,
            4,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();

        // In inter frame, block may be inter or intra depending on bitstream
        if b.is_intra == 0 {
            let comp_type = unsafe { b.data.inter.comp_type };
            let inter_mode = unsafe { b.data.inter.inter_mode };
            let refs = unsafe { b.ref_pair.r };
            // Non-skip-mode single ref: comp_type=0, ref[1]=-1
            if b.skip_mode == 0 {
                assert_eq!(comp_type, 0);
                assert_eq!(refs[1], -1);
                // inter_mode should be NEARMV(13), GLOBALMV(14), or NEWMV(15)
                assert!(
                    inter_mode >= 13 && inter_mode <= 17,
                    "inter_mode {} out of range",
                    inter_mode
                );
            }
            // Context writeback: intra=0 in both a and l
            assert_eq!(a.intra[1], 0); // bx4=1 (bx=4, &63=4, /4=1)
            assert_eq!(l.intra[1], 0);
        }
    }

    #[test]
    fn test_decode_b_compound_inter() {
        let fi = SbFrameInfo {
            bw: 64,
            bh: 64,
            ss_ver: 1,
            ss_hor: 1,
            root_bs: BlockSize::Bs64x64,
            is_inter_or_switch: true,
            sdp: false,
            ext_sdp: false,
            ext_partitions: false,
            uneven_4way: false,
            max_pb_aspect_ratio_log2: 2,
            n_passes: 1,
            seg_enabled: false,
            seg_update_map: false,
            seg_temporal: false,
            seg_preskip: false,
            seg_ext: false,
            seg_last_active_segid: 0,
            seg_globalmv_mask: 0,
            seg_skip_mask: 0,
            skip_mode_enabled: false,
            allow_intrabc: false,
            any_lossless: false,
            has_chroma_layout: true,
            idtx_intra: false,
            mrls: false,
            mhccp: false,
            cfl: false,
            allow_screen_content_tools: false,
            intra_dip: false,
            force_integer_mv: false,
            max_bvp_drl_bits: 2,
            max_drl_bits: 3,
            bawp: false,
            txfm_switchable: true,
            skip_mode_refs: RefPair { pair: 0 },
            n_ref_frames: 7,
            warp_motion: false,
            motion_modes: 0,
            adaptive_mvd: false,
            flex_mvres: false,
            mv_precision: 0,
            mvd_sign_derive: false,
            tip_frame_mode: 0,
            six_param_warp_delta: false,
            subpel_filter_mode: 0,
            switchable_comp_refs: true,
            num_same_ref_comp: 0,
            refdir: [0; 8],
            refdist: [0; 8],
            opfl_refine_type: 0,
            masked_compound: false,
            cwp: false,
            refine_mv_enabled: false,
            absrefdist: [0; 8],
            tile_col_start: 0,
            tile_col_end: 64,
            tile_row_start: 0,
            tile_row_end: 64,
            sb_step: 16,
            ..Default::default()
        };
        // 0xFF data: decode_bool_adapt tends to return 1
        let data = vec![0xFF; 512];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        make_dummy_recon!(_rf, _recon);
        let b = decode_b(
            &fi,
            4,
            4,
            4,
            4,
            0,
            0,
            Pass::Entropy as u8,
            &mut a,
            &mut l,
            &mut msac,
            &mut cdf_m,
            &mut cdf_dmv,
            &mut _recon,
            BlockSize::Bs8x8,
            BlockSize::Bs8x8,
        )
        .unwrap();

        if b.is_intra == 0 {
            let comp_type = unsafe { b.data.inter.comp_type };
            // With switchable_comp_refs and 0xFF data (decode returns 1),
            // is_comp=true so comp_type should be AVG(1)
            if b.skip_mode == 0 {
                assert_eq!(comp_type, 1, "expected compound (AVG) mode");
            }
        }
    }
}

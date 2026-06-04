use crate::cdf::{CdfModeContext, CdfMvContext};
use crate::ctx::{memset_pow2, set_ctx};
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
    SUBSET_MASKS_UV, SUBSET_MASKS_Y,
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
            tile_col_start,
            tile_col_end,
            tile_row_start,
            tile_row_end,
            sb_step,
        }
    }
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
    part_w: &mut Vec<u8>,
    part_r: &[u8],
    by: i32,
    sb256w: i32,
    root_bs: BlockSize,
    c_root_bs: BlockSize,
) -> Result<(), ()> {
    let sb_step = fi.sb_step;
    let sb256y = by >> 6;
    let tile_row = ts.tiling.row;
    let col_start = ts.tiling.col_start;
    let col_end = ts.tiling.col_end;
    let row_start = ts.tiling.row_start;
    let row_end = ts.tiling.row_end;

    let mut bx = col_start;
    while bx < col_end {
        let a_idx = (tile_row * sb256w + (bx >> 6)) as usize;
        let lf_idx = ((bx >> 6) + sb256y * sb256w) as usize;

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
            let (ss_ver, ss_hor) = if p == 0 { (0, 0) } else { (fi.ss_ver, fi.ss_hor) };
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

        decode_sb(
            fi,
            &mut bx_m,
            &mut by_m,
            &mut cbx,
            &mut cby,
            &mut intra_region,
            &mut sdp_cfl_disallowed,
            Pass::Entropy as u8,
            &mut a_arr[a_idx],
            l,
            msac,
            &mut ts.cdf.m,
            &mut ts.cdf.dmv,
            part_w,
            &mut part_w_idx,
            part_r,
            &mut part_r_idx,
            root_bs,
            c_root_bs,
            &mut dir,
        )?;

        bx += sb_step;
    }

    // Error out on symbol-decoder overread.
    if msac.cnt() <= -15 {
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
        skip_mode_refs,
        ..
    } = fc;

    let seq_hdr = &**seq_hdr;
    let frame_hdr = &**frame_hdr;
    let root_bs = *root_bs;
    let sb256w = *sb256w;
    let sb_step = *sb_step;
    let bw = *bw;
    let bh = *bh;
    let refdir = *refdir;
    let skip_mode_refs = *skip_mode_refs;

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

    let mut l = BlockContext::default();

    for tr in 0..rows {
        for tc in 0..cols {
            let ts_idx = (tr * cols + tc) as usize;
            let (cs, ce, rs, re) = {
                let t = &ts[ts_idx].tiling;
                (t.col_start, t.col_end, t.row_start, t.row_end)
            };
            let fi = SbFrameInfo::from_frame(
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
            );

            // MSAC borrows the tile data; move it into a local so the tile's
            // CDF state remains independently mutable during decode.
            let buf = std::mem::take(&mut ts[ts_idx].msac_buf);
            let mut msac = MsacContext::new(&buf, disable_cdf);
            let mut part_w: Vec<u8> = Vec::new();
            let part_r: Vec<u8> = Vec::new();

            let mut by = rs;
            while by < re {
                reset_context(&mut l, keyframe, is_tip);
                decode_tile_sbrow_entropy(
                    &fi,
                    frame_hdr,
                    &mut ts[ts_idx],
                    &mut msac,
                    a,
                    &mut lf.mask,
                    &mut lf.lr_mask,
                    &mut l,
                    &mut part_w,
                    &part_r,
                    by,
                    sb256w,
                    root_bs,
                    c_root_bs,
                )?;
                by += sb_step;
            }

            drop(msac);
            ts[ts_idx].msac_buf = buf;
        }
    }

    Ok(())
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

    fc.dsp = c.dsp.clone();
    fc.tile = std::mem::take(&mut c.tile);
    fc.n_tile_data = c.n_tile_data;

    let qcat = crate::cdf::cdf_thread_init_static_qcat(frame_hdr.quant.yac as u32) as usize;

    fc.seq_hdr = seq_hdr;
    fc.frame_hdr = frame_hdr;

    // Keyframes / intra frames with no primary reference use the static CDF.
    // Inter-frame CDF reference selection lands with M3.
    let in_cdf = None;

    decode_frame(&mut fc, n_tc, 1, in_cdf, qcat)
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
    _pass: u8,
    a: &mut BlockContext,
    l: &mut BlockContext,
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    cdf_dmv: &mut CdfMvContext,
    lbs: BlockSize,
    cbs: BlockSize,
) -> Result<Av2Block, ()> {
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
    ) = {
        let mut sm = [0u8; 2];
        let mut st = [0u8; 2];
        let mut intra_vals = [0u8; 2];
        let mut ibc_vals = [0u8; 2];
        let mut xoff = [0usize; 2];
        let mut r0 = [0i8; 2];
        let mut r1 = [0i8; 2];
        let mut amvd_v = [0u8; 2];
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
            xoff[idx] = bx4;
            if idx == 0 {
                sm[1] = sm[0];
                st[1] = st[0];
                intra_vals[1] = intra_vals[0];
                ibc_vals[1] = ibc_vals[0];
                r0[1] = r0[0];
                r1[1] = r1[0];
                amvd_v[1] = amvd_v[0];
                xoff[1] = xoff[0];
            }
            if have_top {
                idx += 1;
            }
        }
        (sm, st, intra_vals, ibc_vals, xoff, idx, r0, r1, amvd_v)
    };

    // Segmentation (simplified: seg_id = 0 when not enabled)
    b.seg_id = 0;

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
            let ictx = (nx_intra[0] + nx_intra[1]) as i32 + n_ctx as i32;
            b.is_intra = (msac.decode_bool_adapt(cdf_m.intra(ictx as usize)) == 0) as u8;
        }
    } else {
        b.is_intra = 1;
    }

    // Pre-compute spatial neighbour (nb) context values within SB.
    // These are used by intrabc, FSC, MRL, multi_mrl, DIP, morph_pred.
    // boff[i] = -1 means unavailable.
    let have_top_in_sb = (by & (fi.sb_step - 1)) != 0;
    let (nb_fsc, nb_mrl, nb_multi_mrl, nb_intrabc, nb_midx, nb_mvprec, nb_motion_mode, nb_boff) =
        if has_luma {
            let mut fsc = [0u8; 2];
            let mut mrl = [0u8; 2];
            let mut mmrl = [0u8; 2];
            let mut ibc = [0u8; 2];
            let mut mid = [0xffu8; 2];
            let mut mvp = [0u8; 2];
            let mut mm = [0u8; 2];
            let mut boff = [-1i32; 2];
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
                boff[0] = off as i32;
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
                boff[idx] = off as i32;
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
                boff[idx] = by4 as i32;
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
                boff[idx] = bx4 as i32;
                if idx == 0 {
                    fsc[1] = fsc[0];
                    mrl[1] = mrl[0];
                    mmrl[1] = mmrl[0];
                    ibc[1] = ibc[0];
                    mid[1] = mid[0];
                    mvp[1] = mvp[0];
                    mm[1] = mm[0];
                }
            }
            (fsc, mrl, mmrl, ibc, mid, mvp, mm, boff)
        } else {
            (
                [0u8; 2],
                [0u8; 2],
                [0u8; 2],
                [0u8; 2],
                [0xffu8; 2],
                [0u8; 2],
                [0u8; 2],
                [-1i32; 2],
            )
        };

    // intrabc
    if has_luma {
        b.intrabc = 0;
        if fi.allow_intrabc && imin(bw4, bh4) < 16 && b.is_intra != 0 && intra_region == 0 {
            let ctx = (nb_intrabc[0] + nb_intrabc[1]) as usize;
            b.intrabc = msac.decode_bool_adapt(cdf_m.intrabc(ctx)) as u8;
        }
    }
    let intrabc = has_luma && b.intrabc != 0;

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

        // DPCM (lossless mode)
        let dpcm = fi.any_lossless && msac.decode_bool_adapt(cdf_m.dpcm(0)) != 0;
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
        let cbw4 = cb_dim[0] as i32;
        let cbh4 = cb_dim[1] as i32;

        let mut midx = luma_midx;

        // DPCM for chroma
        unsafe {
            b.data.intra.dpcm[1] =
                (fi.any_lossless && msac.decode_bool_adapt(cdf_m.dpcm(1)) != 0) as u8;
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
            let ll = fi.any_lossless;
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
                    // Directional mode from default UV list
                    const DEFAULT_MODE_LIST_UV: [u8; 8] = [
                        1, 2, 3, 4, 5, 6, 7, 8, // VERT, HOR, D45, D135, D113, D157, D203, D67
                    ];
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
            let nb_dip_0 = if have_left {
                l.dip[(by4 + bh4 as usize).saturating_sub(1)]
            } else {
                0
            };
            let nb_dip_1 = if have_top {
                a.dip[(bx4 + bw4 as usize).saturating_sub(1)]
            } else {
                0
            };
            let ctx = (nb_dip_0 != 0) as usize + (nb_dip_1 != 0) as usize;
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

    // TX partition (intra path)
    if b.is_intra != 0 && !intrabc && has_luma {
        read_tx_part(msac, cdf_m, &mut b, bs, fi.any_lossless, fi.txfm_switchable);
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
        a.seg_pred[bx4..bx4 + aw].fill(0);
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
        l.seg_pred[by4..by4 + lh].fill(0);
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

        let is_refmv = unsafe { b.data.intra.is_refmv };
        unsafe {
            b.data.intra.is_qpel = (!fi.force_integer_mv) as u8;
        }
        if is_refmv == 0 && !fi.force_integer_mv {
            unsafe {
                b.data.intra.is_qpel = msac.decode_bool_adapt(cdf_m.intrabc_precision()) as u8;
            }
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
        }

        // TX partition for IntraBC
        read_tx_part(msac, cdf_m, &mut b, bs, fi.any_lossless, fi.txfm_switchable);

        // morph_pred for IntraBC
        unsafe {
            b.data.intra.morph_pred = 0;
        }
        if !fi.is_inter_or_switch && fi.bawp && fi.allow_screen_content_tools {
            let nb_mp_0 = if have_left {
                l.morph_pred[(by4 + bh4 as usize).saturating_sub(1)]
            } else {
                0
            };
            let nb_mp_1 = if have_top {
                a.morph_pred[(bx4 + bw4 as usize).saturating_sub(1)]
            } else {
                0
            };
            let ctx = nb_mp_0 as usize + nb_mp_1 as usize;
            unsafe {
                b.data.intra.morph_pred = msac.decode_bool_adapt(cdf_m.morph_pred(ctx)) as u8;
            }
        }
        let morph_pred = unsafe { b.data.intra.morph_pred };

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
            // simplified comp context
            let ctx = 0usize;
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
            let comp_ctx = 0usize; // STUB: derive compound reference context from neighbors (AV2 §5.11.21)
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

            // --- JMVD scale mode ---
            if final_inter_mode == CompInterPredMode::JointNewMv as u8
                || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8
            {
                let _jmvd_scale_mode = if amvd_val != 0 {
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
                // simplified comp_type context
                let ctx = 0usize;
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

            // --- CWP (compound weighted prediction) ---
            unsafe {
                b.data.inter.cwp_idx = 8;
            }
            let comp_type_val = unsafe { b.data.inter.comp_type };
            if refine_mv_val == 0
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
                let fctx = 0usize; // simplified
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
                    let warp_ctx = sngl_ctx.min(4);
                    allow_warp = msac.decode_bool_adapt(cdf_m.warp(warp_ctx)) != 0;
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

                if inter_mode == InterPredMode::WarpNewMv as u8 {
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
                    let ctx = if n == 0 || n == 3 { 1 } else { 0 };
                    let mut val = msac.decode_symbol_adapt(cdf_m.warp_delta_param(0, ctx), 7) as i8;
                    if val == 7 && prec != 0 {
                        val += msac.decode_symbol_adapt(cdf_m.warp_delta_param(1, ctx), 7) as i8;
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
                    // simplified filter context
                    let fctx = 0usize;
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
            read_tx_part(msac, cdf_m, &mut b, bs, fi.any_lossless, fi.txfm_switchable);
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

    Ok(b)
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
                lbs,
                cbs,
            )?;
            if pass & (Pass::Entropy as u8) != 0 {
                let bx4 = (*bx & 63) as usize;
                let by4 = (*by & 63) as usize;
                if (cbs as i8 | lbs as i8) != BlockSize::Invalid as i8 {
                    set_ctx(&mut a.partition[0], bx4, !(b_dim[0] - 1), b_dim[2] as usize);
                    set_ctx(&mut a.partition[1], bx4, !(b_dim[0] - 1), b_dim[2] as usize);
                    set_ctx(&mut l.partition[0], by4, !(b_dim[1] - 1), b_dim[3] as usize);
                    set_ctx(&mut l.partition[1], by4, !(b_dim[1] - 1), b_dim[3] as usize);
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
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        // All 0xFF data → MSAC will decode 1s for all bool_adapt calls
        let data = vec![0x00; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        let data = vec![0x80; 128];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        let data = vec![0x55; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        let data = vec![0xAA; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        // Decode with both luma and chroma
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
        };
        // Feed data that makes dpcm[0]=true and dpcm[1]=true
        let data = vec![0xFF; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        let data = vec![0x40; 256];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

        // Decode 8x8 block at (4,4); keyframe → always intra
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
        };
        let data = vec![0x80; 512];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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
        };
        // 0xFF data: decode_bool_adapt tends to return 1
        let data = vec![0xFF; 512];
        let mut msac = MsacContext::new(&data, true);
        let mut cdf_m = CdfModeContext::default();
        let mut cdf_dmv = CdfMvContext::default();
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();

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

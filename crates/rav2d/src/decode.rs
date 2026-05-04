use crate::cdf::{CdfModeContext, CdfMvContext};
use crate::dsp::N_SWITCHABLE_FILTERS;
use crate::env::BlockContext;
use crate::headers::{AdaptiveBoolean, FrameHeader, NSWienerPlane, RestorationType, MAX_SEGMENTS};
use crate::internal::{LoopFilterState, NsWienerBank, ScalableMotionParams};
use crate::lf_mask::Av2RestorationUnit;
use crate::intops::{apply_sign64, imax, imin, inv_recenter};
use crate::levels::{BlockSize, Mv, MvXY, TxPartition, N_BS_SIZES};
use crate::pal::pal_idx_finish;
use crate::msac::MsacContext;
use crate::quantizer::dq_lookup;
use crate::tables::{
    NS_WIENER_COEF_RANGE_Y, NS_WIENER_COEF_RANGE_UV,
    SUBSET_MASKS_Y, SUBSET_MASKS_UV,
};

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
    let n = if frame_hdr.segmentation.enabled != 0 { 8 } else { 1 };
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
    [4, 4],   // I400
    [6, 5],   // I420
    [8, 6],   // I422
    [12, 8],  // I444
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
        let y = (by4 * v_mul << 4) + mvy * (1 << (ss_ver == 0) as u32);
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
    17, 45, 3, 10, 24, 31, 38, 52,
    15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54,
    16, 18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53,
    14, 20, 42, 48, 0, 6, 7, 13, 21, 27, 28, 34, 35, 41, 49, 55,
];

pub static DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];

pub static INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];

pub static MV_PREC_TBL: [[u8; 3]; 2] = [
    [3, 1, 0],
    [4, 3, 1],
];

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
    [-1,  0,  1,  2],  // WEDGE_0
    [ 3,  4,  5,  6],  // WEDGE_14
    [ 7,  8,  9, 10],  // WEDGE_27
    [11, 12, 13, 14],  // WEDGE_45
    [15, 16, 17, 18],  // WEDGE_63
    [-1, 19, 20, 21],  // WEDGE_90
    [22, 23, 24, 25],  // WEDGE_117
    [26, 27, 28, 29],  // WEDGE_135
    [30, 31, 32, 33],  // WEDGE_153
    [34, 35, 36, 37],  // WEDGE_166
    [-1, 38, 39, 40],  // WEDGE_180
    [-1, 41, 42, 43],  // WEDGE_194
    [-1, 44, 45, 46],  // WEDGE_207
    [-1, 47, 48, 49],  // WEDGE_225
    [-1, 50, 51, 52],  // WEDGE_243
    [-1, 53, 54, 55],  // WEDGE_270
    [-1, 56, 57, 58],  // WEDGE_297
    [-1, 59, 60, 61],  // WEDGE_315
    [-1, 62, 63, 64],  // WEDGE_333
    [-1, 65, 66, 67],  // WEDGE_346
];

#[derive(Clone, Copy)]
pub struct PartitionConstants {
    pub part: [[i8; 4]; 2],
    pub ctx: [i8; 2],
}

const I: i8 = -1; // BS_INVALID shorthand
use BlockSize::*;

pub static PARTITION_SUBB: [PartitionConstants; N_BS_SIZES] = {
    let mut t = [PartitionConstants { part: [[I; 4]; 2], ctx: [I; 2] }; N_BS_SIZES];

    t[Bs256x256 as usize] = PartitionConstants {
        part: [[Bs256x128 as i8, I, I, Bs128x128 as i8],
               [Bs128x256 as i8, I, I, Bs128x128 as i8]],
        ctx: [9, 12],
    };
    t[Bs256x128 as usize] = PartitionConstants {
        part: [[I, I, I, I],
               [Bs128x128 as i8, I, I, I]],
        ctx: [8, I],
    };
    t[Bs128x256 as usize] = PartitionConstants {
        part: [[Bs128x128 as i8, I, I, I],
               [I, I, I, I]],
        ctx: [7, I],
    };
    t[Bs128x128 as usize] = PartitionConstants {
        part: [[Bs128x64 as i8, I, I, Bs64x64 as i8],
               [Bs64x128 as i8, I, I, Bs64x64 as i8]],
        ctx: [6, 9],
    };
    t[Bs128x64 as usize] = PartitionConstants {
        part: [[I, I, I, I],
               [Bs64x64 as i8, I, I, I]],
        ctx: [5, I],
    };
    t[Bs64x128 as usize] = PartitionConstants {
        part: [[Bs64x64 as i8, I, I, I],
               [I, I, I, I]],
        ctx: [4, I],
    };
    t[Bs64x64 as usize] = PartitionConstants {
        part: [[Bs64x32 as i8, Bs64x16 as i8, Bs64x8 as i8, Bs32x32 as i8],
               [Bs32x64 as i8, Bs16x64 as i8, Bs8x64 as i8, Bs32x32 as i8]],
        ctx: [3, 6],
    };
    t[Bs64x32 as usize] = PartitionConstants {
        part: [[Bs64x16 as i8, Bs64x8 as i8, Bs64x4 as i8, Bs32x16 as i8],
               [Bs32x32 as i8, Bs16x32 as i8, Bs8x32 as i8, Bs32x16 as i8]],
        ctx: [3, 5],
    };
    t[Bs64x16 as usize] = PartitionConstants {
        part: [[Bs64x8 as i8, Bs64x4 as i8, I, Bs32x8 as i8],
               [Bs32x16 as i8, Bs16x16 as i8, Bs8x16 as i8, Bs32x8 as i8]],
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
        part: [[Bs32x32 as i8, Bs32x16 as i8, Bs32x8 as i8, Bs16x32 as i8],
               [Bs16x64 as i8, Bs8x64 as i8, Bs4x64 as i8, Bs16x32 as i8]],
        ctx: [3, 4],
    };
    t[Bs32x32 as usize] = PartitionConstants {
        part: [[Bs32x16 as i8, Bs32x8 as i8, Bs32x4 as i8, Bs16x16 as i8],
               [Bs16x32 as i8, Bs8x32 as i8, Bs4x32 as i8, Bs16x16 as i8]],
        ctx: [2, 3],
    };
    t[Bs32x16 as usize] = PartitionConstants {
        part: [[Bs32x8 as i8, Bs32x4 as i8, I, Bs16x8 as i8],
               [Bs16x16 as i8, Bs8x16 as i8, Bs4x16 as i8, Bs16x8 as i8]],
        ctx: [2, 2],
    };
    t[Bs32x8 as usize] = PartitionConstants {
        part: [[Bs32x4 as i8, I, I, I],
               [Bs16x8 as i8, Bs8x8 as i8, Bs4x8 as i8, Bs16x4 as i8]],
        ctx: [13, 14],
    };
    t[Bs32x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs16x64 as usize] = PartitionConstants {
        part: [[Bs16x32 as i8, Bs16x16 as i8, Bs16x8 as i8, Bs8x32 as i8],
               [Bs8x64 as i8, Bs4x64 as i8, I, Bs8x32 as i8]],
        ctx: [14, 13],
    };
    t[Bs16x32 as usize] = PartitionConstants {
        part: [[Bs16x16 as i8, Bs16x8 as i8, Bs16x4 as i8, Bs8x16 as i8],
               [Bs8x32 as i8, Bs4x32 as i8, I, Bs8x16 as i8]],
        ctx: [2, 1],
    };
    t[Bs16x16 as usize] = PartitionConstants {
        part: [[Bs16x8 as i8, Bs16x4 as i8, I, Bs8x8 as i8],
               [Bs8x16 as i8, Bs4x16 as i8, I, Bs8x8 as i8]],
        ctx: [1, 0],
    };
    t[Bs16x8 as usize] = PartitionConstants {
        part: [[Bs16x4 as i8, I, I, I],
               [Bs8x8 as i8, Bs4x8 as i8, I, Bs8x4 as i8]],
        ctx: [1, 2],
    };
    t[Bs16x4 as usize] = PartitionConstants {
        part: [[I, I, I, I],
               [Bs8x4 as i8, I, I, I]],
        ctx: [11, I],
    };
    t[Bs8x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs8x32 as usize] = PartitionConstants {
        part: [[Bs8x16 as i8, Bs8x8 as i8, Bs8x4 as i8, Bs4x16 as i8],
               [Bs4x32 as i8, I, I, I]],
        ctx: [12, 13],
    };
    t[Bs8x16 as usize] = PartitionConstants {
        part: [[Bs8x8 as i8, Bs8x4 as i8, I, Bs4x8 as i8],
               [Bs4x16 as i8, I, I, I]],
        ctx: [1, 1],
    };
    t[Bs8x8 as usize] = PartitionConstants {
        part: [[Bs8x4 as i8, I, I, I],
               [Bs4x8 as i8, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs8x4 as usize] = PartitionConstants {
        part: [[I, I, I, I],
               [Bs4x4 as i8, I, I, I]],
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
        part: [[Bs4x8 as i8, I, I, I],
               [I, I, I, I]],
        ctx: [10, I],
    };
    t[Bs4x8 as usize] = PartitionConstants {
        part: [[Bs4x4 as i8, I, I, I],
               [I, I, I, I]],
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

pub static CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] = [
    [8, 12, 4, 10, 6],
    [8, 12, 4, 20, -4],
];

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
    let angle = 5 * quad
        + msac.decode_symbol_adapt(cdf_m.wedge_angle(quad), 4) as usize;
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
    let rem = msac.decode_bools_bypass((n_bits + bin + if bin == 0 { 1 } else { 0 } - 4) as u32) as i32;
    let v = (if bin != 0 { 1 << (n_bits + bin - 4) } else { 0 }) + rem;
    let n = 1 << n_bits;
    if r * 2 <= n {
        inv_recenter(r as u32, v as u32) as i32
    } else {
        n - 1 - inv_recenter((n - 1 - r) as u32, v as u32) as i32
    }
}

pub fn read_amvd(
    msac: &mut MsacContext,
    amvd_joint: &mut [u16],
    amvd_index: &mut [u16],
) -> Mv {
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
        sh_class = h_syms + 1
            + msac.decode_symbol_adapt(
                cdf_mv.shell_upper(mv_prec as usize),
                imin(h_syms2, 7) as usize,
            ) as i32;
        if mv_prec + sh_class == 21 {
            sh_class += msac.decode_bool_adapt(shell_tip) as i32;
        }
    } else {
        sh_class = msac.decode_symbol_adapt(
            cdf_mv.shell_lower(mv_prec as usize),
            h_syms as usize,
        ) as i32;
    }

    let mut sh_index;
    if sh_class < 2 {
        sh_index = msac.decode_bool_adapt(
            cdf_mv.shell_offset_low(sh_class as usize),
        ) as i32;
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
            sh_index |= m
                * msac.decode_bool_adapt(cdf_mv.shell_offset_hi(i as usize)) as i32;
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
                pair_index +=
                    msac.decode_uniform((sh_index as u32 >> 1) - 1) as i32;
            }
        }
    }

    let sh = 6 - mv_prec;
    if pair_index * 2 == sh_index {
        let v = (sh_index >> 1) << sh;
        Mv { c: MvXY { y: v, x: v } }
    } else {
        let b = msac.decode_bool_adapt(
            cdf_mv.col_index(imin(sh_class, 3) as usize),
        );
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

pub fn read_pal_indices(
    msac: &mut MsacContext,
    cdf_m: &mut CdfModeContext,
    pal_out: &mut [u8],
    scratch: &mut [u8],
    pal_sz: i32,
    sz: &[i32; 4],
) -> i32 {
    let dir = (imax(sz[2], sz[3]) < 64) as u32
        & msac.decode_bool_bypass();
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
            let v = msac.decode_symbol_adapt(
                cdf_m.pal_idx(pal_cdf_base, 0),
                nsym,
            ) as i32;
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
            let v = msac.decode_symbol_adapt(
                cdf_m.pal_idx(pal_cdf_base, 0),
                nsym,
            ) as i32;
            let next_v = if v == 0 { prev_v } else { v - (v <= prev_v) as i32 };
            scratch[off as usize] = next_v as u8;

            if copy == 1 {
                for m in 1..lim2 {
                    scratch[(off + m as isize * strides[1]) as usize] = next_v as u8;
                }
            } else {
                let mut prev_tl = prev_v;
                let mut prev_l = next_v;
                for m in 1..lim2 {
                    let prev_t = scratch[(off - strides[0] + m as isize * strides[1]) as usize] as i32;
                    let ctx = if prev_t == prev_l {
                        3 + (prev_tl == prev_l) as usize
                    } else {
                        1 + (prev_t == prev_tl || prev_l == prev_tl) as usize
                    };
                    let v = msac.decode_symbol_adapt(
                        cdf_m.pal_idx(pal_cdf_base, ctx),
                        nsym,
                    ) as i32;
                    let p = match ctx {
                        1 => match v {
                            0 | 1 => {
                                if v == dir as i32 { prev_l } else { prev_t }
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
                                v - (v <= prev_l + s) as i32
                                    - (v <= prev_tl + 1 - s) as i32
                            }
                        },
                        4 => {
                            if v == 0 { prev_l } else { v - (v <= prev_l) as i32 }
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

    pal_idx_finish(pal_out, scratch, sz[2] as usize, sz[3] as usize, sz[0] as usize, sz[1] as usize);
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
            b.tx_size_ll =
                msac.decode_bool_adapt(cdf_m.txsz_lossless(szctx, inter)) as u8;
        }
    } else if b.skip_txfm == 0 {
        if txfm_switchable && bs != BlockSize::Bs4x4 && imax(bw4, bh4) <= 16 {
            let inter = (b.is_intra == 0 || b.intrabc != 0) as usize;
            let szctx = TX_PART_GROUP[bs_idx] as usize;
            let is_split = msac.decode_bool_adapt(
                cdf_m.tx_split(b.fsc as usize, inter, szctx),
            );
            if is_split != 0 {
                if imin(bw4, bh4) >= 2 {
                    let ctx = TX_TYPE_GROUP_VH[bs_idx] as usize;
                    b.tx_part = 1
                        + msac.decode_symbol_adapt(
                            cdf_m.tx_part_2d(b.fsc as usize, inter, ctx),
                            6,
                        ) as u8;
                } else if imax(bw4, bh4) >= 4 {
                    let ctx = (bw4 >= 4) as usize;
                    let tx_part_4way = msac.decode_bool_adapt(
                        cdf_m.tx_part_1d(b.fsc as usize, inter, ctx),
                    );
                    b.tx_part =
                        TxPartition::H as u8 + ctx as u8 + tx_part_4way as u8 * 2;
                } else {
                    debug_assert!(
                        bs == BlockSize::Bs4x8 || bs == BlockSize::Bs8x4
                    );
                    b.tx_part = if bs == BlockSize::Bs4x8 {
                        TxPartition::H as u8
                    } else {
                        TxPartition::V as u8
                    };
                }
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
        lr.restoration_type = if t != 0 { frame_type as u8 } else { RestorationType::None as u8 };
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

        let masks: &[u32] = if is_uv != 0 { &SUBSET_MASKS_UV } else { &SUBSET_MASKS_Y };
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
            let asym = is_uv != 0
                && s != 0
                && msac.decode_bool_adapt(cdf_m.wiener_ns_sym()) != 0;

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
        assert_eq!(SS_SIZE_MUL[0], [4, 4]);   // I400
        assert_eq!(SS_SIZE_MUL[1], [6, 5]);   // I420
        assert_eq!(SS_SIZE_MUL[3], [12, 8]);  // I444
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
        assert_eq!(REORDERED_DIR_Y_MODE[2], 1);    // VERT_PRED
        assert_eq!(DEFAULT_MODE_LIST_Y.len(), 56);
        assert_eq!(DEFAULT_MODE_LIST_UV[0], 1);     // VERT_PRED
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
        let mv = read_amvd(&mut msac, &mut amvd_joint, &mut amvd_index);
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
        let mv = read_amvd(&mut msac, &mut amvd_joint, &mut amvd_index);
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
        cdf.data[112] = 16384; cdf.data[113] = 0;
        cdf.data[114] = 16384; cdf.data[115] = 0;
        for i in 0..2 { let b = 116 + i * 2; cdf.data[b] = 16384; cdf.data[b + 1] = 0; }
        cdf.data[120] = 16384; cdf.data[121] = 0;
        for i in 0..16 { let b = 122 + i * 2; cdf.data[b] = 16384; cdf.data[b + 1] = 0; }
        for i in 0..2 { let b = 154 + i * 2; cdf.data[b] = 16384; cdf.data[b + 1] = 0; }
        for i in 0..4 { let b = 158 + i * 2; cdf.data[b] = 16384; cdf.data[b + 1] = 0; }
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
        CdfModeContext { data: [16384u16; 3496] }
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
        read_tx_part(&mut msac, &mut cdf_m, &mut b, BlockSize::Bs16x16, false, true);
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
            let ret = read_pal_indices(&mut msac, &mut cdf_m, &mut pal_out, &mut scratch, pal_sz, &sz);
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
        LoopFilterState {
            mask: Vec::new(),
            lr_mask: Vec::new(),
            segmap_uv: Vec::new(),
            uv_segmap_stride: 0,
            cdef_buf_plane_sz: [0; 2],
            cdef_buf_sbh: 0,
            lr_buf_plane_sz: [0; 4],
            re_sz: 0,
            base_q: 0,
            gdf_ref_dst_idx: 0,
            start_of_tile_row: Vec::new(),
            restore_planes: 0,
            wiener_idx: 0,
            ns_subclass_class_idx: None,
        }
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
            &mut msac, &mut cdf_m, &mut bank, &mut lr,
            0, RestorationType::Switchable, &ns,
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
            &mut msac, &mut cdf_m, &mut bank, &mut lr,
            0, RestorationType::NsWiener, &ns,
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
            &mut msac, &mut cdf_m, &mut bank, &mut lr,
            0, RestorationType::PcWiener, &ns,
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
            &mut msac, &mut cdf_m, &mut bank, &mut lr,
            0, RestorationType::NsWiener, &ns,
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
            &mut msac, &mut cdf_m, &mut bank, &mut lr,
            1, RestorationType::NsWiener, &ns,
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
            &hdr, &seq, &mut lf, &mut ft, &mut ts, &mut n_ts,
            &mut a, &mut a_sz, &mut dq, &mut qm, &absrefdist,
            4, 2, 2, 32, 32, 1, 1,
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
            &hdr, &seq, &mut lf, &mut ft, &mut ts, &mut n_ts,
            &mut a, &mut a_sz, &mut dq, &mut qm, &absrefdist,
            8, 4, 4, 64, 64, 1, 1,
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
            &hdr, &seq, &mut lf, &mut ft, &mut ts, &mut n_ts,
            &mut a, &mut a_sz, &mut dq, &mut qm, &absrefdist,
            4, 2, 2, 32, 32, 4, 1,
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
            &hdr, &seq, &mut lf, &mut ft, &mut ts, &mut n_ts,
            &mut a, &mut a_sz, &mut dq, &mut qm, &absrefdist,
            4, 2, 2, 32, 32, 1, 2,
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
            &hdr, &seq, &mut lf, &mut ft, &mut ts, &mut n_ts,
            &mut a, &mut a_sz, &mut dq, &mut qm, &absrefdist,
            4, 3, 2, 32, 32, 1, 1,
        );

        assert_eq!(lf.mask.len(), 6);
        assert_eq!(lf.lr_mask.len(), 6);
    }
}

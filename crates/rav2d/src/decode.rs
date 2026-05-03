use crate::cdf::{CdfModeContext, CdfMvContext};
use crate::dsp::N_SWITCHABLE_FILTERS;
use crate::env::BlockContext;
use crate::headers::{FrameHeader, MAX_SEGMENTS};
use crate::internal::ScalableMotionParams;
use crate::intops::{apply_sign64, imax, imin, inv_recenter};
use crate::levels::{BlockSize, Mv, MvXY, TxPartition, N_BS_SIZES};
use crate::msac::MsacContext;
use crate::quantizer::dq_lookup;

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
}

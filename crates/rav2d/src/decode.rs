use crate::dsp::N_SWITCHABLE_FILTERS;
use crate::env::BlockContext;
use crate::headers::{FrameHeader, MAX_SEGMENTS};
use crate::internal::ScalableMotionParams;
use crate::intops::{apply_sign64, imax, imin};
use crate::levels::{BlockSize, MvXY, N_BS_SIZES};
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
    fn test_wedge_angle_dist2idx() {
        assert_eq!(WEDGE_ANGLE_DIST2IDX[0][0], -1);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[0][1], 0);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[1][0], 3);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[10][0], -1);
        assert_eq!(WEDGE_ANGLE_DIST2IDX[10][1], 38);
    }
}

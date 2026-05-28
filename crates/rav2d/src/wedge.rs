// Copyright (c) 2018-2026, VideoLAN and dav2d authors
// Copyright (c) 2018-2026, Two Orioles, LLC
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::intops::{iclip, imin};
use crate::levels::{BlockSize, InterIntraPredMode, N_BS_SIZES};
use crate::tables::BLOCK_DIMENSIONS;

const BS_64X64: usize = BlockSize::Bs64x64 as u8 as usize;

/// Number of wedge offset entries: block sizes from Bs64x64 onward,
/// excluding the 6 largest (256x256..64x128).
const WEDGE_OFFSETS_LEN: usize = N_BS_SIZES - BS_64X64 - 6; // 19

/// Number of inter-intra offset entries: all block sizes from Bs64x64 onward.
const II_NONDC_OFFSETS_LEN: usize = N_BS_SIZES - BS_64X64; // 25

const WEDGE_444_LEN: usize = 68 * (64 + 32 + 16 + 8) * (64 + 32 + 16 + 8);
const WEDGE_422_LEN: usize = 68 * (32 + 16 + 8 + 4) * (64 + 32 + 16 + 8);
const WEDGE_420_LEN: usize = 68 * (32 + 16 + 8 + 4) * (32 + 16 + 8 + 4);
const WEDGE_TMVP_LEN: usize = 68 * (8 + 4 + 2 + 1) * (8 + 4 + 2 + 1);
const II_DC_LEN: usize = 64 * 64;
const II_NONDC_LEN: usize = (64 + 32 + 16 + 8 + 4) * (64 + 32 + 16 + 8 + 4) * 3;

/// Pre-computed wedge and inter-intra masks.
///
/// This struct is ~1.7 MB and should be heap-allocated via [`init_masks`].
pub struct Masks {
    wedge_offsets: [u8; WEDGE_OFFSETS_LEN],
    ii_nondc_offsets: [u16; II_NONDC_OFFSETS_LEN],
    wedge_444: Vec<u8>,
    wedge_422: Vec<u8>,
    wedge_420: Vec<u8>,
    wedge_tmvp: Vec<u8>,
    ii_dc: Vec<u8>,
    ii_nondc: Vec<u8>,
}

impl Masks {
    /// Returns the inter-intra mask for the given block size and prediction mode.
    ///
    /// `bs` is the `BlockSize` discriminant as `usize`, `bw4` and `bh4` are
    /// the block width/height in 4-sample units, and `ii_mode` is the
    /// `InterIntraPredMode`.
    pub fn ii_mask(&self, bs: usize, bw4: usize, bh4: usize, ii_mode: InterIntraPredMode) -> &[u8] {
        if ii_mode == InterIntraPredMode::DcPred {
            &self.ii_dc
        } else {
            let off = self.ii_nondc_offsets[bs - BS_64X64] as usize * 0x60
                + 16 * bw4 * bh4 * (ii_mode as usize - 1);
            &self.ii_nondc[off..]
        }
    }

    /// Returns the wedge mask for the given block size, wedge index, and
    /// chroma subsampling index (0 = 444, 1 = 422, 2 = 420).
    pub fn wedge_mask(
        &self,
        bs: usize,
        bw4: usize,
        bh4: usize,
        widx: usize,
        ss_idx: usize,
    ) -> &[u8] {
        let off =
            (self.wedge_offsets[bs - BS_64X64] as usize * 0x1100 + 16 * bw4 * bh4 * widx) >> ss_idx;
        self.wedge_buf(ss_idx, off)
    }

    /// Returns the TMVP wedge mask for the given block size and wedge index.
    pub fn wedge_tmvp(&self, bs: usize, bw4: usize, bh4: usize, widx: usize) -> &[u8] {
        let off = self.wedge_offsets[bs - BS_64X64] as usize * 68 + (bw4 * bh4 / 4) * widx;
        &self.wedge_tmvp[off..]
    }

    fn wedge_buf(&self, ss_idx: usize, off: usize) -> &[u8] {
        match ss_idx {
            0 => &self.wedge_444[off..],
            1 => &self.wedge_422[off..],
            2 => &self.wedge_420[off..],
            _ => unreachable!(),
        }
    }
}

// -- Wedge direction types --

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum WedgeDirection {
    Wedge0 = 0,
    Wedge14 = 1,
    Wedge27 = 2,
    Wedge45 = 3,
    Wedge63 = 4,
    Wedge90 = 5,
    Wedge117 = 6,
    Wedge135 = 7,
    Wedge153 = 8,
    Wedge166 = 9,
    Wedge180 = 10,
    Wedge194 = 11,
    Wedge207 = 12,
    Wedge225 = 13,
    Wedge243 = 14,
    Wedge270 = 15,
    Wedge297 = 16,
    Wedge315 = 17,
    Wedge333 = 18,
    Wedge346 = 19,
}

const N_WEDGE_DIRECTIONS: usize = 20;

struct WedgeCodeType {
    direction: WedgeDirection,
    x_offset: u8,
    y_offset: u8,
}

use WedgeDirection::*;

static WEDGE_CODEBOOK_16: [WedgeCodeType; 68] = [
    WedgeCodeType {
        direction: Wedge0,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge0,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge0,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge14,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge14,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge14,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge14,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge27,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge27,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge27,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge27,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge45,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge45,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge45,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge45,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge63,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge63,
        x_offset: 4,
        y_offset: 3,
    },
    WedgeCodeType {
        direction: Wedge63,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCodeType {
        direction: Wedge63,
        x_offset: 4,
        y_offset: 1,
    },
    WedgeCodeType {
        direction: Wedge90,
        x_offset: 4,
        y_offset: 3,
    },
    WedgeCodeType {
        direction: Wedge90,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCodeType {
        direction: Wedge90,
        x_offset: 4,
        y_offset: 1,
    },
    WedgeCodeType {
        direction: Wedge117,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge117,
        x_offset: 4,
        y_offset: 3,
    },
    WedgeCodeType {
        direction: Wedge117,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCodeType {
        direction: Wedge117,
        x_offset: 4,
        y_offset: 1,
    },
    WedgeCodeType {
        direction: Wedge135,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge135,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge135,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge135,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge153,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge153,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge153,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge153,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge166,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge166,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge166,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge166,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge180,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge180,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge180,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge194,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge194,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge194,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge207,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge207,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge207,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge225,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge225,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge225,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge243,
        x_offset: 4,
        y_offset: 5,
    },
    WedgeCodeType {
        direction: Wedge243,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCodeType {
        direction: Wedge243,
        x_offset: 4,
        y_offset: 7,
    },
    WedgeCodeType {
        direction: Wedge270,
        x_offset: 4,
        y_offset: 5,
    },
    WedgeCodeType {
        direction: Wedge270,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCodeType {
        direction: Wedge270,
        x_offset: 4,
        y_offset: 7,
    },
    WedgeCodeType {
        direction: Wedge297,
        x_offset: 4,
        y_offset: 5,
    },
    WedgeCodeType {
        direction: Wedge297,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCodeType {
        direction: Wedge297,
        x_offset: 4,
        y_offset: 7,
    },
    WedgeCodeType {
        direction: Wedge315,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge315,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge315,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge333,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge333,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge333,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge346,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge346,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCodeType {
        direction: Wedge346,
        x_offset: 7,
        y_offset: 4,
    },
];

// -- Helper functions --

fn copy2d(dst: &mut [u8], src: &[u8], w8: usize, h8: usize, x_off: usize, y_off: usize) {
    let src_start = (64 - y_off * h8) * 128 + (64 - x_off * w8);
    let row_len = w8 * 8;
    for y in 0..h8 * 8 {
        let s = src_start + y * 128;
        let d = y * row_len;
        dst[d..d + row_len].copy_from_slice(&src[s..s + row_len]);
    }
}

fn subsample_420(dst: &mut [u8], src: &[u8], w8: usize, h8: usize) {
    let stride = w8 * 8;
    for y in 0..h8 * 4 {
        for x in 0..w8 * 4 {
            dst[y * w8 * 4 + x] = ((src[y * 2 * stride + x * 2] as u16
                + src[y * 2 * stride + x * 2 + 1] as u16
                + src[(y * 2 + 1) * stride + x * 2] as u16
                + src[(y * 2 + 1) * stride + x * 2 + 1] as u16
                + 2)
                >> 2) as u8;
        }
    }
}

fn subsample_422(dst: &mut [u8], src: &[u8], w8: usize, h8: usize) {
    let stride = w8 * 8;
    for y in 0..h8 * 8 {
        for x in 0..w8 * 4 {
            dst[y * w8 * 4 + x] =
                ((src[y * stride + x * 2] as u16 + src[y * stride + x * 2 + 1] as u16 + 1) >> 1)
                    as u8;
        }
    }
}

fn fill_tmvp(dst: &mut [u8], src: &[u8], w8: usize, h8: usize) {
    let stride = w8 * 8;
    for y in 0..h8 {
        for x in 0..w8 {
            let mut score = [0i32; 2];
            for yy in y * 8..y * 8 + 8 {
                for xx in x * 8..x * 8 + 8 {
                    score[0] += (src[yy * stride + xx] < 4) as i32;
                    score[1] += (src[yy * stride + xx] > 60) as i32;
                }
            }
            dst[y * w8 + x] = if score[0] >= 60 {
                0
            } else if score[1] >= 60 {
                1
            } else {
                2
            };
        }
    }
}

fn gen_master(master: &mut [u8; 128 * 128], mul: i32, wd: WedgeDirection) {
    static COS_LUT: [i8; N_WEDGE_DIRECTIONS] = [
        4, 4, 4, 2, 2, 0, -2, -2, -4, -4, -4, -4, -4, -2, -2, 0, 2, 2, 4, 4,
    ];
    static SIN_LUT: [i8; N_WEDGE_DIRECTIONS] = [
        0, -1, -2, -2, -4, -4, -4, -2, -2, -1, 0, 1, 2, 2, 4, 4, 4, 2, 2, 1,
    ];
    static WEIGHT: [i8; 29] = [
        8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    ];

    let idx = wd as usize;
    let s = SIN_LUT[idx] as i32 * mul;
    let c = COS_LUT[idx] as i32 * mul;
    for y in 0..128 {
        let dy = (2 * y as i32 - 127) * s;
        for x in 0..128 {
            let d = iclip((2 * x as i32 - 127) * c + dy, -28, 28);
            master[y * 128 + x] = (4 * if d >= 0 {
                16 - WEIGHT[d as usize] as i32
            } else {
                WEIGHT[(-d) as usize] as i32
            }) as u8;
        }
    }
}

fn build_nondc_ii_masks(mask_v: &mut [u8], w: usize, h: usize, step: usize) {
    static II_WEIGHTS_1D: [u8; 64] = [
        60, 56, 52, 48, 45, 42, 39, 37, 34, 32, 30, 28, 26, 24, 22, 21, 19, 18, 17, 16, 15, 14, 13,
        12, 11, 10, 10, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
        2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];

    let area = w * h;
    // mask_v[0..area]     = vertical mask
    // mask_v[area..2*area] = horizontal mask (mask_h)
    // mask_v[2*area..3*area] = smooth mask (mask_sm)
    for y in 0..h {
        let off = y * w;
        let wt = II_WEIGHTS_1D[y * step];
        for x in 0..w {
            mask_v[off + x] = wt;
            mask_v[area + off + x] = II_WEIGHTS_1D[x * step];
            mask_v[2 * area + off + x] = II_WEIGHTS_1D[imin(x as i32, y as i32) as usize * step];
        }
    }
}

// -- Block size helpers for the fill macro equivalent --

/// Returns the `BlockSize` discriminant as `usize` for a given variant.
const fn bs(b: BlockSize) -> usize {
    b as i8 as u8 as usize
}

fn init_wedge_masks(masks: &mut Masks) {
    let mut o: usize = 0;
    for bsi in BS_64X64..N_BS_SIZES {
        let b_dim = &BLOCK_DIMENSIONS[bsi];
        if b_dim[0] == 1 || b_dim[1] == 1 {
            continue;
        }
        masks.wedge_offsets[bsi - BS_64X64] = o as u8;
        o += (b_dim[0] as usize * b_dim[1] as usize) >> 2;
    }
    debug_assert_eq!(o * 0x1100, WEDGE_444_LEN);
    debug_assert!(o < 256);

    let mut master = [0u8; 128 * 128];
    let mut wd = N_WEDGE_DIRECTIONS; // sentinel: no direction generated yet

    // Helper closure: fill wedge masks for one block size.
    // We need a macro because we borrow masks mutably multiple times.
    macro_rules! fill {
        ($masks:expr, $master:expr, $cb:expr, $w8:expr, $h8:expr, $bs_variant:expr, $widx:expr) => {{
            let bsi = bs($bs_variant);
            let bw4 = $w8 * 2;
            let bh4 = $h8 * 2;
            let w8: usize = $w8;
            let h8: usize = $h8;
            let widx: usize = $widx;

            // Copy 444 mask from master
            {
                let off_444 = ($masks.wedge_offsets[bsi - BS_64X64] as usize * 0x1100
                    + 16 * bw4 * bh4 * widx)
                    >> 0;
                let dst = &mut $masks.wedge_444[off_444..];
                copy2d(
                    dst,
                    $master,
                    w8,
                    h8,
                    $cb.x_offset as usize,
                    $cb.y_offset as usize,
                );
            }

            // Build a temporary copy of the 444 mask for subsampling and tmvp
            let wm_len = w8 * 8 * h8 * 8;
            let off_444 = ($masks.wedge_offsets[bsi - BS_64X64] as usize * 0x1100
                + 16 * bw4 * bh4 * widx)
                >> 0;
            let mut wm_tmp = vec![0u8; wm_len];
            wm_tmp.copy_from_slice(&$masks.wedge_444[off_444..off_444 + wm_len]);

            // Subsample 422
            {
                let off_422 = ($masks.wedge_offsets[bsi - BS_64X64] as usize * 0x1100
                    + 16 * bw4 * bh4 * widx)
                    >> 1;
                let dst = &mut $masks.wedge_422[off_422..];
                subsample_422(dst, &wm_tmp, w8, h8);
            }

            // Subsample 420
            {
                let off_420 = ($masks.wedge_offsets[bsi - BS_64X64] as usize * 0x1100
                    + 16 * bw4 * bh4 * widx)
                    >> 2;
                let dst = &mut $masks.wedge_420[off_420..];
                subsample_420(dst, &wm_tmp, w8, h8);
            }

            // Fill TMVP
            {
                let off_tmvp =
                    $masks.wedge_offsets[bsi - BS_64X64] as usize * 68 + (bw4 * bh4 / 4) * widx;
                let dst = &mut $masks.wedge_tmvp[off_tmvp..];
                fill_tmvp(dst, &wm_tmp, w8, h8);
            }
        }};
    }

    // First pass: sharp edge (mul=2) for small block sizes
    for widx in 0..68 {
        let cb = &WEDGE_CODEBOOK_16[widx];
        if cb.direction as usize != wd {
            gen_master(&mut master, 2, cb.direction);
            wd = cb.direction as usize;
        }
        fill!(masks, &master, cb, 1, 1, BlockSize::Bs8x8, widx);
        fill!(masks, &master, cb, 1, 2, BlockSize::Bs8x16, widx);
        fill!(masks, &master, cb, 2, 1, BlockSize::Bs16x8, widx);
        fill!(masks, &master, cb, 2, 2, BlockSize::Bs16x16, widx);
    }

    // Second pass: soft edge (mul=1) for larger block sizes
    for widx in 0..68 {
        let cb = &WEDGE_CODEBOOK_16[widx];
        if cb.direction as usize != wd {
            gen_master(&mut master, 1, cb.direction);
            wd = cb.direction as usize;
        }
        fill!(masks, &master, cb, 1, 4, BlockSize::Bs8x32, widx);
        fill!(masks, &master, cb, 1, 8, BlockSize::Bs8x64, widx);
        fill!(masks, &master, cb, 2, 4, BlockSize::Bs16x32, widx);
        fill!(masks, &master, cb, 2, 8, BlockSize::Bs16x64, widx);
        fill!(masks, &master, cb, 4, 1, BlockSize::Bs32x8, widx);
        fill!(masks, &master, cb, 4, 2, BlockSize::Bs32x16, widx);
        fill!(masks, &master, cb, 4, 4, BlockSize::Bs32x32, widx);
        fill!(masks, &master, cb, 4, 8, BlockSize::Bs32x64, widx);
        fill!(masks, &master, cb, 8, 1, BlockSize::Bs64x8, widx);
        fill!(masks, &master, cb, 8, 2, BlockSize::Bs64x16, widx);
        fill!(masks, &master, cb, 8, 4, BlockSize::Bs64x32, widx);
        fill!(masks, &master, cb, 8, 8, BlockSize::Bs64x64, widx);
    }
}

fn init_ii_masks(masks: &mut Masks) {
    // DC mask: all 32
    for v in masks.ii_dc.iter_mut() {
        *v = 32;
    }

    let mut o: usize = 0;
    for bsi in BS_64X64..N_BS_SIZES {
        let b_dim = &BLOCK_DIMENSIONS[bsi];
        masks.ii_nondc_offsets[bsi - BS_64X64] = o as u16;
        o += (b_dim[0] as usize * b_dim[1] as usize) >> 1;
    }
    debug_assert_eq!(o * 0x60 + 0x30, II_NONDC_LEN);
    debug_assert!(o < u16::MAX as usize);

    macro_rules! fill_ii {
        ($masks:expr, $w:expr, $h:expr, $step:expr) => {{
            // The C code uses: fill(W, H, S) => build_nondc_ii_masks(II_MASK(BS_WxH, 0, 0, 1), W, H, S)
            // II_MASK(bs, 0, 0, 1) = &ii_nondc[offsets.ii_nondc[bs - BS_64x64] * 0x60 + 0]
            // since bw4=0,bh4=0 => 16*0*0*(1-1)=0
            //
            // We need to find the correct BlockSize for WxH.
            let bsi = bs(block_size_for_dims($w, $h));
            let base = $masks.ii_nondc_offsets[bsi - BS_64X64] as usize * 0x60;
            let dst = &mut $masks.ii_nondc[base..];
            build_nondc_ii_masks(dst, $w, $h, $step);
        }};
    }

    fill_ii!(masks, 4, 4, 16);
    fill_ii!(masks, 4, 8, 8);
    fill_ii!(masks, 4, 16, 4);
    fill_ii!(masks, 4, 32, 2);
    fill_ii!(masks, 4, 64, 1);
    fill_ii!(masks, 8, 4, 8);
    fill_ii!(masks, 8, 8, 8);
    fill_ii!(masks, 8, 16, 4);
    fill_ii!(masks, 8, 32, 2);
    fill_ii!(masks, 8, 64, 1);
    fill_ii!(masks, 16, 4, 4);
    fill_ii!(masks, 16, 8, 4);
    fill_ii!(masks, 16, 16, 4);
    fill_ii!(masks, 16, 32, 2);
    fill_ii!(masks, 16, 64, 1);
    fill_ii!(masks, 32, 4, 2);
    fill_ii!(masks, 32, 8, 2);
    fill_ii!(masks, 32, 16, 2);
    fill_ii!(masks, 32, 32, 2);
    fill_ii!(masks, 32, 64, 1);
    fill_ii!(masks, 64, 4, 1);
    fill_ii!(masks, 64, 8, 1);
    fill_ii!(masks, 64, 16, 1);
    fill_ii!(masks, 64, 32, 1);
    fill_ii!(masks, 64, 64, 1);
}

/// Map pixel dimensions (w, h) to the corresponding `BlockSize`.
///
/// The C code uses token-pasting (`BS_##w##x##h`); we do a match.
const fn block_size_for_dims(w: usize, h: usize) -> BlockSize {
    match (w, h) {
        (4, 4) => BlockSize::Bs4x4,
        (4, 8) => BlockSize::Bs4x8,
        (4, 16) => BlockSize::Bs4x16,
        (4, 32) => BlockSize::Bs4x32,
        (4, 64) => BlockSize::Bs4x64,
        (8, 4) => BlockSize::Bs8x4,
        (8, 8) => BlockSize::Bs8x8,
        (8, 16) => BlockSize::Bs8x16,
        (8, 32) => BlockSize::Bs8x32,
        (8, 64) => BlockSize::Bs8x64,
        (16, 4) => BlockSize::Bs16x4,
        (16, 8) => BlockSize::Bs16x8,
        (16, 16) => BlockSize::Bs16x16,
        (16, 32) => BlockSize::Bs16x32,
        (16, 64) => BlockSize::Bs16x64,
        (32, 4) => BlockSize::Bs32x4,
        (32, 8) => BlockSize::Bs32x8,
        (32, 16) => BlockSize::Bs32x16,
        (32, 32) => BlockSize::Bs32x32,
        (32, 64) => BlockSize::Bs32x64,
        (64, 4) => BlockSize::Bs64x4,
        (64, 8) => BlockSize::Bs64x8,
        (64, 16) => BlockSize::Bs64x16,
        (64, 32) => BlockSize::Bs64x32,
        (64, 64) => BlockSize::Bs64x64,
        _ => panic!(),
    }
}

/// Initialize all wedge and inter-intra masks.
///
/// Returns a heap-allocated `Masks` struct (~1.7 MB).
pub fn init_masks() -> Box<Masks> {
    let mut masks = Box::new(Masks {
        wedge_offsets: [0u8; WEDGE_OFFSETS_LEN],
        ii_nondc_offsets: [0u16; II_NONDC_OFFSETS_LEN],
        wedge_444: vec![0u8; WEDGE_444_LEN],
        wedge_422: vec![0u8; WEDGE_422_LEN],
        wedge_420: vec![0u8; WEDGE_420_LEN],
        wedge_tmvp: vec![0u8; WEDGE_TMVP_LEN],
        ii_dc: vec![0u8; II_DC_LEN],
        ii_nondc: vec![0u8; II_NONDC_LEN],
    });
    init_wedge_masks(&mut masks);
    init_ii_masks(&mut masks);
    masks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_size_assertions() {
        // Reproduce the C assertions that validate offset accumulation.
        let mut wedge_o: usize = 0;
        for bsi in BS_64X64..N_BS_SIZES {
            let b_dim = &BLOCK_DIMENSIONS[bsi];
            if b_dim[0] == 1 || b_dim[1] == 1 {
                continue;
            }
            wedge_o += (b_dim[0] as usize * b_dim[1] as usize) >> 2;
        }
        assert_eq!(wedge_o * 0x1100, WEDGE_444_LEN);
        assert!(wedge_o < 256);

        let mut ii_o: usize = 0;
        for bsi in BS_64X64..N_BS_SIZES {
            let b_dim = &BLOCK_DIMENSIONS[bsi];
            ii_o += (b_dim[0] as usize * b_dim[1] as usize) >> 1;
        }
        assert_eq!(ii_o * 0x60 + 0x30, II_NONDC_LEN);
        assert!(ii_o < u16::MAX as usize);
    }

    #[test]
    fn test_init_masks_does_not_panic() {
        let _masks = init_masks();
    }

    #[test]
    fn test_gen_master_center() {
        // Wedge90 with mul=2 should produce a vertical split around x=64.
        let mut master = [0u8; 128 * 128];
        gen_master(&mut master, 2, WedgeDirection::Wedge90);
        // At y=64 (center row), the mask should transition around x=64.
        // x=0 => d = (2*0 - 127)*0 + 0 = 0 (cos=0 for Wedge90, sin=-4)
        // Actually cos_lut[Wedge90]=0, sin_lut[Wedge90]=-4
        // So c=0, s=-4*2=-8
        // d = (2x-127)*0 + (2y-127)*(-8) = (2y-127)*(-8)
        // At y=64: d = (128-127)*(-8) = -8, clamped to -8
        // weight[-(-8)] = weight[8] = 4, so val = 4*4 = 16
        let val = master[64 * 128 + 0];
        assert_eq!(val, 16);
        // At y=0: d = (0-127)*(-8) = 1016, clamped to 28
        // weight[28] = 0, val = 4*(16-0) = 64
        let val = master[0 * 128 + 0];
        assert_eq!(val, 64);
    }

    #[test]
    fn test_ii_dc_mask_all_32() {
        let masks = init_masks();
        assert!(masks.ii_dc.iter().all(|&v| v == 32));
    }

    #[test]
    fn test_wedge_offsets_monotonic() {
        let masks = init_masks();
        // Offsets should be non-decreasing (they accumulate).
        let mut prev = 0u8;
        for &off in &masks.wedge_offsets {
            // Entries for skipped block sizes stay 0, but valid ones increase.
            if off > 0 {
                assert!(off >= prev, "wedge offsets should be non-decreasing");
                prev = off;
            }
        }
    }

    #[test]
    fn test_wedge_mask_nonzero() {
        let masks = init_masks();
        // The 8x8 wedge mask for index 0 should have some non-zero values.
        let bsi = bs(BlockSize::Bs8x8);
        let wm = masks.wedge_mask(bsi, 2, 2, 0, 0);
        let sum: u32 = wm[..64].iter().map(|&v| v as u32).sum();
        assert!(sum > 0, "8x8 wedge mask should contain non-zero values");
    }

    #[test]
    fn test_ii_nondc_mask_range() {
        let masks = init_masks();
        // All non-DC inter-intra mask values should be in [0, 64].
        for &v in &masks.ii_nondc {
            assert!(v <= 64, "ii_nondc values should be <= 64, got {}", v);
        }
    }

    #[test]
    fn test_wedge_444_value_range() {
        let masks = init_masks();
        for &v in &masks.wedge_444 {
            assert!(v <= 64, "wedge_444 values should be <= 64, got {}", v);
        }
    }

    #[test]
    fn test_wedge_tmvp_values() {
        let masks = init_masks();
        // TMVP values should be 0, 1, or 2.
        for &v in &masks.wedge_tmvp {
            assert!(v <= 2, "wedge_tmvp values should be 0, 1, or 2, got {}", v);
        }
    }

    #[test]
    fn test_subsample_422_basic() {
        // 1x1 block (w8=1, h8=1) => 8x8 src, 4x8 dst
        let mut src = [0u8; 64];
        for i in 0..64 {
            src[i] = (i * 2) as u8;
        }
        let mut dst = [0u8; 32];
        subsample_422(&mut dst, &src, 1, 1);
        // dst[0] = (src[0] + src[1] + 1) >> 1 = (0 + 2 + 1) >> 1 = 1
        assert_eq!(dst[0], 1);
    }

    #[test]
    fn test_subsample_420_basic() {
        let src = [32u8; 64];
        let mut dst = [0u8; 16];
        subsample_420(&mut dst, &src, 1, 1);
        // All source values are 32, so all dst values should be 32.
        assert!(dst.iter().all(|&v| v == 32));
    }

    #[test]
    fn test_block_size_for_dims() {
        assert_eq!(block_size_for_dims(8, 8), BlockSize::Bs8x8);
        assert_eq!(block_size_for_dims(64, 64), BlockSize::Bs64x64);
        assert_eq!(block_size_for_dims(16, 32), BlockSize::Bs16x32);
        assert_eq!(block_size_for_dims(4, 4), BlockSize::Bs4x4);
    }
}

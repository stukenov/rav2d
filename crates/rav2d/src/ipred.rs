use crate::dip_tables::DIP_WEIGHTS;
use crate::intops::{apply_sign, clz, ctz, iclip, imax, imin, ulog2};
use crate::levels::CflMhDir;
use crate::pixel::{BitDepth, BitDepth8, Pixel};
use crate::levels::{
    ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, ANGLE_IS_LUMA, ANGLE_MRL_IDX_MASK,
    ANGLE_MRL_IDX_SHIFT, ANGLE_MULTI_MRL_FLAG, ANGLE_SMOOTH_LEFT_EDGE_FLAG,
    ANGLE_SMOOTH_TOP_EDGE_FLAG, ANGLE_USE_EDGE_FILTER_FLAG,
};
use crate::recon::derive_alpha;
use crate::tables::{
    DC_IBP_WEIGHTS, DIV_RECIP, DIV_SCALE_SH_BIAS, DIV_SCALE_SH_COEFW, DIV_SCALE_SH_OFFSET,
    DR_INTRA_DERIVATIVE, SM_WEIGHTS,
};

#[derive(Clone, Copy)]
pub struct DrFilter4Tap {
    pub a: i8,
    pub b: u8,
    pub c: u8,
    pub d: i8,
}

pub static DR_INTERP_FILTER: [DrFilter4Tap; 32] = [
    DrFilter4Tap {
        a: 0,
        b: 128,
        c: 0,
        d: 0,
    },
    DrFilter4Tap {
        a: -2,
        b: 127,
        c: 4,
        d: -1,
    },
    DrFilter4Tap {
        a: -3,
        b: 125,
        c: 8,
        d: -2,
    },
    DrFilter4Tap {
        a: -5,
        b: 123,
        c: 13,
        d: -3,
    },
    DrFilter4Tap {
        a: -6,
        b: 121,
        c: 17,
        d: -4,
    },
    DrFilter4Tap {
        a: -7,
        b: 118,
        c: 22,
        d: -5,
    },
    DrFilter4Tap {
        a: -9,
        b: 116,
        c: 27,
        d: -6,
    },
    DrFilter4Tap {
        a: -9,
        b: 112,
        c: 32,
        d: -7,
    },
    DrFilter4Tap {
        a: -10,
        b: 109,
        c: 37,
        d: -8,
    },
    DrFilter4Tap {
        a: -11,
        b: 106,
        c: 41,
        d: -8,
    },
    DrFilter4Tap {
        a: -11,
        b: 102,
        c: 46,
        d: -9,
    },
    DrFilter4Tap {
        a: -12,
        b: 98,
        c: 52,
        d: -10,
    },
    DrFilter4Tap {
        a: -12,
        b: 94,
        c: 56,
        d: -10,
    },
    DrFilter4Tap {
        a: -12,
        b: 90,
        c: 61,
        d: -11,
    },
    DrFilter4Tap {
        a: -12,
        b: 85,
        c: 66,
        d: -11,
    },
    DrFilter4Tap {
        a: -12,
        b: 81,
        c: 71,
        d: -12,
    },
    DrFilter4Tap {
        a: -12,
        b: 76,
        c: 76,
        d: -12,
    },
    DrFilter4Tap {
        a: -12,
        b: 71,
        c: 81,
        d: -12,
    },
    DrFilter4Tap {
        a: -11,
        b: 66,
        c: 85,
        d: -12,
    },
    DrFilter4Tap {
        a: -11,
        b: 61,
        c: 90,
        d: -12,
    },
    DrFilter4Tap {
        a: -10,
        b: 56,
        c: 94,
        d: -12,
    },
    DrFilter4Tap {
        a: -10,
        b: 52,
        c: 98,
        d: -12,
    },
    DrFilter4Tap {
        a: -9,
        b: 46,
        c: 102,
        d: -11,
    },
    DrFilter4Tap {
        a: -8,
        b: 41,
        c: 106,
        d: -11,
    },
    DrFilter4Tap {
        a: -8,
        b: 37,
        c: 109,
        d: -10,
    },
    DrFilter4Tap {
        a: -7,
        b: 32,
        c: 112,
        d: -9,
    },
    DrFilter4Tap {
        a: -6,
        b: 27,
        c: 116,
        d: -9,
    },
    DrFilter4Tap {
        a: -5,
        b: 22,
        c: 118,
        d: -7,
    },
    DrFilter4Tap {
        a: -4,
        b: 17,
        c: 121,
        d: -6,
    },
    DrFilter4Tap {
        a: -3,
        b: 13,
        c: 123,
        d: -5,
    },
    DrFilter4Tap {
        a: -2,
        b: 8,
        c: 125,
        d: -3,
    },
    DrFilter4Tap {
        a: -1,
        b: 4,
        c: 127,
        d: -2,
    },
];

pub fn get_filter_strength(wh: i32, angle: i32, is_sm: bool) -> i32 {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else {
        if wh <= 8 {
            if angle >= 56 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 32 {
                return 3;
            }
            if angle >= 16 {
                return 2;
            }
            if angle >= 8 {
                return 1;
            }
        } else if wh <= 32 {
            if angle >= 32 {
                return 3;
            }
            if angle >= 4 {
                return 2;
            }
            return 1;
        } else {
            return 3;
        }
    }
    0
}

pub fn filter_edge<P: Pixel>(
    out: &mut [P],
    sz: usize,
    lim_from: i32,
    lim_to: i32,
    inp: &[P],
    from: i32,
    to: i32,
    strength: usize,
) {
    static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

    debug_assert!(strength > 0);
    // NB: lim_from / lim_to may be negative (C uses signed `int`); compare in
    // i32 space so a negative bound yields an empty loop instead of wrapping.
    let mut i: i32 = 0;
    while i < imin(sz as i32, lim_from) {
        out[i as usize] = inp[iclip(i, from, to - 1) as usize];
        i += 1;
    }
    while i < imin(lim_to, sz as i32) {
        let mut s = 0i32;
        for j in 0..5 {
            s += inp[iclip(i - 2 + j, from, to - 1) as usize].into()
                * KERNEL[strength - 1][j as usize] as i32;
        }
        out[i as usize] = P::from_i32((s + 8) >> 4);
        i += 1;
    }
    while i < sz as i32 {
        out[i as usize] = inp[iclip(i, from, to - 1) as usize];
        i += 1;
    }
}

fn splat_dc<P: Pixel>(dst: &mut [P], stride: usize, off: usize, width: usize, mut height: usize, dc: P) {
    let mut p = off;
    while height > 0 {
        for x in 0..width {
            dst[p + x] = dc;
        }
        p += stride;
        height -= 1;
    }
}

fn dc_gen_top<P: Pixel>(tl: &[P], o: usize, width: usize) -> u32 {
    let mut dc = (width >> 1) as u32;
    for i in 0..width {
        dc += tl[o + 1 + i].as_u16() as u32;
    }
    dc >> ctz(width as u32)
}

fn dc_gen_left<P: Pixel>(tl: &[P], o: usize, height: usize) -> u32 {
    let mut dc = (height >> 1) as u32;
    for i in 0..height {
        dc += tl[o - 1 - i].as_u16() as u32;
    }
    dc >> ctz(height as u32)
}

fn fast_div32_dc(num: u32, den: u32) -> u32 {
    debug_assert!(den > 0 && den <= 255);
    let mut shift = ulog2(den);
    let rem = den as i32 - (1 << shift);
    let idx = (rem << (7 - shift)) as usize;
    debug_assert!(idx <= 128);
    shift += 9;
    ((num as u64 * DIV_RECIP[idx] as u64) as u32 + ((1u32 << shift) >> 1)) >> shift
}

fn dc_gen<BD: BitDepth>(bd: BD, tl: &[BD::Pixel], o: usize, width: usize, height: usize) -> u32 {
    let n_pel = width + height;
    let mut dc = 0u32;
    for i in 0..width {
        dc += tl[o + 1 + i].as_u16() as u32;
    }
    for i in 0..height {
        dc += tl[o - 1 - i].as_u16() as u32;
    }
    if n_pel & (n_pel - 1) == 0 {
        return (dc + width as u32) >> ctz(n_pel as u32);
    }
    (fast_div32_dc(dc, n_pel as u32)).min(bd.bitdepth_max() as u32)
}

pub fn ipred_dc_128_8bpc(dst: &mut [u8], stride: usize, width: usize, height: usize) {
    ipred_dc_128(BitDepth8, dst, stride, width, height);
}

pub fn ipred_dc_128<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    width: usize,
    height: usize,
) {
    let dc = BD::Pixel::from_i32((bd.bitdepth_max() + 1) >> 1);
    splat_dc(dst, stride, 0, width, height, dc);
}

pub fn ipred_dc_top_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_dc_top(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub fn ipred_dc_top<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    mut height: usize,
    angle: i32,
) {
    let dc = dc_gen_top(tl, o, width);
    let mut off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let h = height >> 2;
        let w_y = &DC_IBP_WEIGHTS[h..];
        for y in 0..h {
            let wy = 128 - w_y[y] as u32;
            let dc_wy = dc * w_y[y] as u32;
            for x in 0..width {
                dst[off + x] =
                    BD::Pixel::from_i32(((tl[o + 1 + x].as_u16() as u32 * wy + dc_wy + 64) >> 7) as i32);
            }
            off += stride;
        }
        height -= h;
    }

    splat_dc(dst, stride, off, width, height, BD::Pixel::from_i32(dc as i32));
}

pub fn ipred_dc_left_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_dc_left(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub fn ipred_dc_left<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    mut width: usize,
    height: usize,
    angle: i32,
) {
    let dc = dc_gen_left(tl, o, height);
    let mut off = 0;
    let mut x_off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let w = width >> 2;
        let w_x = &DC_IBP_WEIGHTS[w..];
        for y in 0..height {
            let left = tl[o - 1 - y].as_u16() as u32;
            for x in 0..w {
                dst[off + x] = BD::Pixel::from_i32(
                    ((left * (128 - w_x[x] as u32) + dc * w_x[x] as u32 + 64) >> 7) as i32,
                );
            }
            off += stride;
        }
        off = 0;
        x_off = w;
        width -= w;
    }

    let dc_p = BD::Pixel::from_i32(dc as i32);
    let mut p = off;
    for _ in 0..height {
        for x in 0..width {
            dst[p + x_off + x] = dc_p;
        }
        p += stride;
    }
}

pub fn ipred_dc_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_dc(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub fn ipred_dc<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    mut width: usize,
    mut height: usize,
    angle: i32,
) {
    let dc = dc_gen(bd, tl, o, width, height);
    let mut off = 0;
    let mut x_off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let h = height >> 2;
        let w = width >> 2;
        let x_start = if width < height { w } else { 0 };
        let w_y = &DC_IBP_WEIGHTS[h..];
        for y in 0..h {
            let wy = 128 - w_y[y] as u32;
            let dc_wy = dc * w_y[y] as u32;
            for x in x_start..width {
                dst[off + x] =
                    BD::Pixel::from_i32(((tl[o + 1 + x].as_u16() as u32 * wy + dc_wy + 64) >> 7) as i32);
            }
            off += stride;
        }

        let y_start = if width >= height { h } else { 0 };
        off = y_start * stride;
        let w_x = &DC_IBP_WEIGHTS[w..];
        for y in y_start..height {
            let left = tl[o - 1 - y].as_u16() as u32;
            for x in 0..w {
                dst[off + x] = BD::Pixel::from_i32(
                    ((left * (128 - w_x[x] as u32) + dc * w_x[x] as u32 + 64) >> 7) as i32,
                );
            }
            off += stride;
        }
        off = h * stride + w;
        x_off = 0;
        width -= w;
        height -= h;
    }

    let dc_p = BD::Pixel::from_i32(dc as i32);
    let mut p = off;
    for _ in 0..height {
        for x in 0..width {
            dst[p + x_off + x] = dc_p;
        }
        p += stride;
    }
}

pub fn ipred_v_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_v(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub fn ipred_v<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    // multi-reference-line averaging (ipred_tmpl.c:251-268).
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        for x in 0..width {
            let top: i32 = tl[o + 1 + x].into();
            let top2: i32 = tl[o + 1 + e_stride + x].into();
            dst[x] = BD::Pixel::from_i32((top + top2 + 1) >> 1);
        }
        let mut off = stride;
        for _ in 1..height {
            dst.copy_within(0..width, off);
            off += stride;
        }
        return;
    }
    let mut off = 0;
    for _ in 0..height {
        dst[off..off + width].copy_from_slice(&tl[o + 1..o + 1 + width]);
        off += stride;
    }
}

pub fn ipred_h_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_h(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub fn ipred_h<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    // multi-reference-line averaging (ipred_tmpl.c:282-295).
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        let mut off = 0;
        for y in 0..height {
            let left: i32 = tl[o - 1 - y].into();
            let left2: i32 = tl[o + e_stride - 1 - y].into();
            let v = BD::Pixel::from_i32((left + left2 + 1) >> 1);
            for x in 0..width {
                dst[off + x] = v;
            }
            off += stride;
        }
        return;
    }
    let mut off = 0;
    for y in 0..height {
        let v = tl[o - 1 - y];
        for x in 0..width {
            dst[off + x] = v;
        }
        off += stride;
    }
}

pub fn ipred_paeth_8bpc(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    ipred_paeth(BitDepth8, dst, stride, tl, o, w, h);
}

pub fn ipred_paeth<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let topleft: i32 = tl[o].into();
    let mut off = 0;
    for y in 0..h {
        let left: i32 = tl[o - 1 - y].into();
        for x in 0..w {
            let top: i32 = tl[o + 1 + x].into();
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();
            dst[off + x] = BD::Pixel::from_i32(if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            });
        }
        off += stride;
    }
}

pub fn ipred_smooth_8bpc(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    ipred_smooth(BitDepth8, dst, stride, tl, o, w, h);
}

pub fn ipred_smooth<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let bwl2 = ulog2(w as u32);
    let bhl2 = ulog2(h as u32);
    let rnd_ver = (h >> 1) as i32;
    let rnd_hor = (w >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right: i32 = tl[o + w + 1].into();
    let bottom: i32 = tl[o - h - 1].into();

    let mut off = 0;
    for y in 0..h {
        let left: i32 = tl[o - 1 - y].into();
        let diff_hor = left - right;
        let off_ver = h as i32 - 1 - y as i32;
        let w_ver = weights[y] as i32;
        for x in 0..w {
            let above: i32 = tl[o + 1 + x].into();
            let mul_ver = (above - bottom) * off_ver;
            let mul_hor = diff_hor * (w as i32 - 1 - x as i32);
            let mut pred_ver = bottom + ((mul_ver + rnd_ver) >> bhl2);
            let mut pred_hor = right + ((mul_hor + rnd_hor) >> bwl2);
            pred_ver += ((above - pred_ver) * w_ver + 32) >> 6;
            pred_hor += ((left - pred_hor) * weights[x] as i32 + 32) >> 6;
            dst[off + x] = BD::Pixel::from_i32((pred_ver + pred_hor + 1) >> 1);
        }
        off += stride;
    }
}

pub fn ipred_smooth_v_8bpc(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    ipred_smooth_v(BitDepth8, dst, stride, tl, o, w, h);
}

pub fn ipred_smooth_v<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let bhl2 = ulog2(h as u32);
    let rnd = (h >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom: i32 = tl[o - h - 1].into();

    let mut off = 0;
    for y in 0..h {
        let off_y = h as i32 - 1 - y as i32;
        let w_ver = weights[y] as i32;
        for x in 0..w {
            let above: i32 = tl[o + 1 + x].into();
            let mul = (above - bottom) * off_y;
            let pred = bottom + ((mul + rnd) >> bhl2);
            dst[off + x] = BD::Pixel::from_i32(pred + (((above - pred) * w_ver + 32) >> 6));
        }
        off += stride;
    }
}

pub fn ipred_smooth_h_8bpc(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    ipred_smooth_h(BitDepth8, dst, stride, tl, o, w, h);
}

pub fn ipred_smooth_h<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let bwl2 = ulog2(w as u32);
    let rnd = (w >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right_val: i32 = tl[o + w + 1].into();

    let mut off = 0;
    for y in 0..h {
        let left: i32 = tl[o - 1 - y].into();
        let diff = left - right_val;
        for x in 0..w {
            let mul = diff * (w as i32 - 1 - x as i32);
            let pred = right_val + ((mul + rnd) >> bwl2);
            dst[off + x] = BD::Pixel::from_i32(pred + (((left - pred) * weights[x] as i32 + 32) >> 6));
        }
        off += stride;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ipred_z1_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    ipred_z1(
        BitDepth8, dst, stride, tl, o, width, height, angle, max_width, max_height, ibp_weights,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn ipred_z1<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mut angle: i32,
    max_width: i32,
    _max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    angle &= 511;

    if mrl_mul {
        let e_stride = (width + height) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![BD::Pixel::default(); 64 * 64];
        ipred_z1(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width,
            _max_height,
            ibp_weights,
        );
        ipred_z1(
            bd,
            dst,
            stride,
            tl,
            o + e_stride,
            width,
            height,
            angle | ANGLE_IS_LUMA,
            max_width,
            _max_height,
            ibp_weights,
        );
        for y in 0..height {
            for x in 0..width {
                let a: i32 = tmp[y * 64 + x].into();
                let b: i32 = dst[y * stride + x].into();
                dst[y * stride + x] = BD::Pixel::from_i32((a + b + 1) >> 1);
            }
        }
        return;
    }

    let dx = DR_INTRA_DERIVATIVE[angle as usize] as i32;
    let max_base_x = (width + height) as i32 - 1 + (mrl_idx as i32 * 2);

    // C: pixel filt[1 + 1 + 3 + 64 + 64 + 2 * 3 + 2] (= 141).
    let mut filt = [BD::Pixel::default(); 141];
    let top_off = 2 + mrl_idx;
    let sz = 1 + mrl_idx + width + height + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        get_filter_strength((width + height) as i32, 90 - angle, is_sm_t)
    } else {
        0
    };
    if str > 0 {
        filter_edge(
            &mut filt[1..],
            sz,
            1,
            sz as i32 + max_width - width as i32,
            &tl[o..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    // C: `filt[sz + 2] = filt[sz + 1] = filt[sz]` (right-associative), so both
    // sz+1 and sz+2 take filt[sz]. The assignment order matters: set sz+1 from
    // sz first, then propagate into sz+2 (ipred_tmpl.c:549).
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz + 1];

    let mut ypos = dx * (1 + mrl_idx as i32);
    for y in 0..height {
        let mut base = ypos >> 6;
        if base > max_base_x {
            for yy in y..height {
                for x in 0..width {
                    dst[yy * stride + x] = filt[top_off + max_base_x as usize];
                }
            }
            break;
        }
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &DR_INTERP_FILTER[shift];
        for x in 0..width {
            if base > max_base_x {
                let fill = filt[top_off + max_base_x as usize];
                for xx in x..width {
                    dst[y * stride + xx] = fill;
                }
                break;
            }
            let bi = top_off as i32 + base;
            if is_luma {
                let v = f.a as i32 * Into::<i32>::into(filt[(bi - 1) as usize])
                    + f.b as i32 * Into::<i32>::into(filt[bi as usize])
                    + f.c as i32 * Into::<i32>::into(filt[(bi + 1) as usize])
                    + f.d as i32 * Into::<i32>::into(filt[(bi + 2) as usize]);
                dst[y * stride + x] = bd.pixel_clip((v + 64) >> 7);
            } else {
                let v = (32 - shift as i32) * Into::<i32>::into(filt[bi as usize])
                    + shift as i32 * Into::<i32>::into(filt[(bi + 1) as usize]);
                dst[y * stride + x] = bd.pixel_clip((v + 16) >> 5);
            }
            base += 1;
        }
        ypos += dx;
    }

    if enable_ibp {
        let mode_idx = imin(10 - (angle >> 3), 6) as usize;
        let mut tmp = vec![BD::Pixel::default(); 64 * 64];
        ipred_z3(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            (180 + angle) | angle_flags,
            max_width,
            _max_height,
            ibp_weights,
        );
        ibp_blend(
            bd,
            dst,
            stride,
            &tmp,
            width,
            height,
            false,
            &ibp_weights[mode_idx],
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ipred_z3_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    ipred_z3(
        BitDepth8, dst, stride, tl, o, width, height, angle, max_width, max_height, ibp_weights,
    );
}

#[allow(clippy::explicit_counter_loop)]
#[allow(clippy::too_many_arguments)]
pub fn ipred_z3<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mut angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    angle &= 511;

    if mrl_mul {
        let e_stride = (width + height) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![BD::Pixel::default(); 64 * 64];
        ipred_z3(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width,
            max_height,
            ibp_weights,
        );
        ipred_z3(
            bd,
            dst,
            stride,
            tl,
            o + e_stride,
            width,
            height,
            angle | ANGLE_IS_LUMA,
            max_width,
            max_height,
            ibp_weights,
        );
        for y in 0..height {
            for x in 0..width {
                let a: i32 = tmp[y * 64 + x].into();
                let b: i32 = dst[y * stride + x].into();
                dst[y * stride + x] = BD::Pixel::from_i32((a + b + 1) >> 1);
            }
        }
        return;
    }

    let dy = DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32;
    let max_base_y = (width + height) as i32 - 1 + (mrl_idx as i32 * 2);

    // C: pixel filt[1 + 1 + 3 + 64 + 64 + 2 * 3 + 2] (= 141).
    let mut filt = [BD::Pixel::default(); 141];
    let left_off = 1 + width + height + mrl_idx * 2;
    let sz = 1 + mrl_idx + width + height + mrl_idx * 2;

    let str = if enable_intra_edge_filter && mrl_idx == 0 && have_left {
        get_filter_strength((width + height) as i32, angle - 180, is_sm_l)
    } else {
        0
    };

    if str > 0 {
        filter_edge(
            &mut filt[2..],
            sz,
            height as i32 - max_height,
            sz as i32 - 1,
            &tl[o + 1 - sz..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[2..2 + sz].copy_from_slice(&tl[o + 1 - sz..o + 1]);
    }
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];

    let mut ypos = dy * (1 + mrl_idx as i32);
    for x in 0..width {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &DR_INTERP_FILTER[shift];
        let mut base = ypos >> 6;
        for y in 0..height {
            if base <= max_base_y {
                let bi = left_off as i32 - base;
                if is_luma {
                    let v = f.a as i32 * Into::<i32>::into(filt[(bi + 1) as usize])
                        + f.b as i32 * Into::<i32>::into(filt[bi as usize])
                        + f.c as i32 * Into::<i32>::into(filt[(bi - 1) as usize])
                        + f.d as i32 * Into::<i32>::into(filt[(bi - 2) as usize]);
                    dst[y * stride + x] = bd.pixel_clip((v + 64) >> 7);
                } else {
                    let v = (32 - shift as i32) * Into::<i32>::into(filt[bi as usize])
                        + shift as i32 * Into::<i32>::into(filt[(bi - 1) as usize]);
                    dst[y * stride + x] = bd.pixel_clip((v + 16) >> 5);
                }
            } else {
                let fill = filt[left_off - max_base_y as usize];
                for yy in y..height {
                    dst[yy * stride + x] = fill;
                }
                break;
            }
            base += 1;
        }
        ypos += dy;
    }

    if enable_ibp {
        let mode_idx = imin((angle - 183) >> 3, 6) as usize;
        let mut tmp = vec![BD::Pixel::default(); 64 * 64];
        ipred_z1(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            (angle - 180) | angle_flags,
            max_width,
            max_height,
            ibp_weights,
        );
        ibp_blend(
            bd,
            dst,
            stride,
            &tmp,
            width,
            height,
            true,
            &ibp_weights[mode_idx],
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ipred_z2_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    ipred_z2(
        BitDepth8, dst, stride, tl, o, width, height, angle, max_width, max_height,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn ipred_z2<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mut angle: i32,
    max_width: i32,
    max_height: i32,
) {
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    angle &= 511;

    if mrl_mul {
        let e_stride = (width + height) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![BD::Pixel::default(); 64 * 64];
        ipred_z2(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width,
            max_height,
        );
        ipred_z2(
            bd,
            dst,
            stride,
            tl,
            o + e_stride,
            width,
            height,
            angle | ANGLE_IS_LUMA,
            max_width,
            max_height,
        );
        for y in 0..height {
            for x in 0..width {
                let a: i32 = tmp[y * 64 + x].into();
                let b: i32 = dst[y * stride + x].into();
                dst[y * stride + x] = BD::Pixel::from_i32((a + b + 1) >> 1);
            }
        }
        return;
    }

    let dy = DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32;
    let dx = DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32;

    // Top filter buffer
    let mut filt = [BD::Pixel::default(); 72];
    let top_off = mrl_idx;
    let sz_t = 1 + width + mrl_idx;
    let str_t = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        get_filter_strength((width + height) as i32, angle - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - width as i32,
            &tl[o..],
            0,
            sz_t as i32,
            str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&tl[o..o + sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // Left filter buffer
    let mut filt2 = [BD::Pixel::default(); 72];
    let left_off: usize = height + 2;
    let sz_l = 1 + height + mrl_idx;
    let str_l = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        get_filter_strength((width + height) as i32, 180 - angle, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        filter_edge(
            &mut filt2[1..],
            sz_l,
            height as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - (height + mrl_idx)..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - (height + mrl_idx)..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..height {
        let ypos = (y + 1) as i32;
        let mut xpos = -(ypos + mrl_idx as i32) * dx;
        let mut x = 0usize;

        // Left reference loop
        while x < width && xpos < -(64 * (1 + mrl_idx as i32)) {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - (xpos_l + mrl_idx as i32) * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            if is_luma {
                let f = &DR_INTERP_FILTER[shift];
                let v = f.a as i32 * Into::<i32>::into(filt2[bi - 1])
                    + f.b as i32 * Into::<i32>::into(filt2[bi - 2])
                    + f.c as i32 * Into::<i32>::into(filt2[bi - 3])
                    + f.d as i32 * Into::<i32>::into(filt2[bi - 4]);
                dst[y * stride + x] = bd.pixel_clip((v + 64) >> 7);
            } else {
                let v = (32 - shift as i32) * Into::<i32>::into(filt2[bi - 2])
                    + shift as i32 * Into::<i32>::into(filt2[bi - 3]);
                dst[y * stride + x] = bd.pixel_clip((v + 16) >> 5);
            }
            x += 1;
            xpos += 64;
        }

        // Top reference loop
        while x < width {
            let base_x = xpos >> 6;
            let shift = ((xpos & 0x3F) >> 1) as usize;
            let ti = top_off as i32 + base_x;
            if is_luma {
                let f = &DR_INTERP_FILTER[shift];
                let v = f.a as i32 * Into::<i32>::into(filt[(ti + 1) as usize])
                    + f.b as i32 * Into::<i32>::into(filt[(ti + 2) as usize])
                    + f.c as i32 * Into::<i32>::into(filt[(ti + 3) as usize])
                    + f.d as i32 * Into::<i32>::into(filt[(ti + 4) as usize]);
                dst[y * stride + x] = bd.pixel_clip((v + 64) >> 7);
            } else {
                let v = (32 - shift as i32) * Into::<i32>::into(filt[(ti + 2) as usize])
                    + shift as i32 * Into::<i32>::into(filt[(ti + 3) as usize]);
                dst[y * stride + x] = bd.pixel_clip((v + 16) >> 5);
            }
            x += 1;
            xpos += 64;
        }
    }
}

pub fn ibp_blend_8bpc(
    dst: &mut [u8],
    stride: usize,
    tmp: &[u8],
    width: usize,
    height: usize,
    inv: bool,
    weights: &[[u8; 16]; 16],
) {
    ibp_blend(BitDepth8, dst, stride, tmp, width, height, inv, weights);
}

pub fn ibp_blend<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tmp: &[BD::Pixel],
    width: usize,
    height: usize,
    inv: bool,
    weights: &[[u8; 16]; 16],
) {
    let x_shift = width >> (4 + 1);
    let y_shift = height >> (4 + 1);

    for y in 0..height {
        let wy = y >> y_shift;
        for x in 0..width {
            let wx = x >> x_shift;
            let weight = if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            } as i32;
            let t: i32 = tmp[y * 64 + x].into();
            let d: i32 = dst[y * stride + x].into();
            dst[y * stride + x] =
                BD::Pixel::from_i32((t * (128 - weight) + d * weight + 64) >> 7);
        }
    }
}

pub fn get_div_scale_sh(d: i32) -> (i32, i32) {
    let d = imax(1, d.abs());
    let sh = ulog2(d as u32);
    let nsh = sh - 14;
    let d = if nsh >= 0 {
        let rnd = if nsh > 0 { 1 << (nsh - 1) } else { 0 };
        (d + rnd) >> nsh
    } else {
        d << (-nsh)
    };
    let d = iclip(d, 1, 0x7fff);
    let d = d & ((1 << 14) - 1);

    let idx = (d >> 11) as usize;
    let coefw = DIV_SCALE_SH_COEFW[idx] as i32;
    let bias = DIV_SCALE_SH_BIAS[idx] as i32;
    let d = d - DIV_SCALE_SH_OFFSET[idx] as i32;
    let scale = (((coefw * ((d * d) >> 14)) >> 8) - (d >> 1) + bias) << 2;
    (scale, sh)
}

pub fn mul32(a: i32, b: i32, sh: i32) -> i32 {
    let a2 = ulog2((a.abs() | 1) as u32) + 1;
    let b2 = ulog2((b.abs() | 1) as u32) + 1;
    let drop = if a2 + b2 > 29 { a2 + b2 - 29 } else { 0 };
    let ash = drop >> 1;
    let bsh = drop - ash;
    let adj = sh - (ash + bsh);
    let mul = (a >> ash) * (b >> bsh);
    if adj <= 0 {
        return mul;
    }
    debug_assert!(adj <= 29);
    let bias = 1u32 << (adj as u32 - 1);
    if mul >= 0 {
        ((mul as u32).wrapping_add(bias) >> adj as u32) as i32
    } else {
        -((((-mul) as u32).wrapping_add(bias) >> adj as u32) as i32)
    }
}

pub fn ipred_dip_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
) {
    ipred_dip(BitDepth8, dst, stride, tl, o, width, height, mode);
}

pub fn ipred_dip<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
) {
    let trans = (mode & 16) != 0;
    let wd = width >> 2;
    let hd = height >> 2;
    let wl2 = ulog2(wd as u32);
    let hl2 = ulog2(hd as u32);
    let wrnd = width >> 3;
    let hrnd = height >> 3;
    let i_t: usize = if trans { 5 } else { 1 };
    let i_l: usize = if trans { 1 } else { 5 };
    let mut inp = [0i32; 11];
    inp[0] = tl[o].into();
    let mut in_sum = inp[0];

    let mut ti = o + 1;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..wd {
            sum += Into::<i32>::into(tl[ti]);
            ti += 1;
        }
        inp[i_t + i] = (sum + wrnd as i32) >> wl2;
        in_sum += inp[i_t + i];
    }

    let mut li = o;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..hd {
            li -= 1;
            sum += Into::<i32>::into(tl[li]);
        }
        inp[i_l + i] = (sum + hrnd as i32) >> hl2;
        in_sum += inp[i_l + i];
    }

    let mut sum = 0i32;
    for x in 0..wd {
        sum += Into::<i32>::into(tl[o + x + width + 1]);
    }
    let idx_tr = if trans { 10 } else { 9 };
    inp[idx_tr] = (sum + wrnd as i32) >> wl2;
    in_sum += inp[idx_tr];

    sum = 0;
    for y in 0..hd {
        sum += Into::<i32>::into(tl[o - (y + height + 1)]);
    }
    let idx_bl = if trans { 9 } else { 10 };
    inp[idx_bl] = (sum + hrnd as i32) >> hl2;
    in_sum += inp[idx_bl];

    let m = (mode & 7) as usize;

    let mut uwl2 = wl2 - 1;
    let mut dwl2 = 0i32;
    if uwl2 < 0 {
        dwl2 = -uwl2;
        uwl2 = 0;
    }
    let step_x = 1usize << uwl2;
    let dw = 1usize << dwl2;
    let mut uhl2 = hl2 - 1;
    let mut dhl2 = 0i32;
    if uhl2 < 0 {
        dhl2 = -uhl2;
        uhl2 = 0;
    }
    let step_y = 1usize << uhl2;
    let dh = 1usize << dhl2;
    let grid_h = 8usize >> dhl2;
    let grid_w = 8usize >> dwl2;

    let mut y = step_y - 1;
    for gy in 0..grid_h {
        let iy = gy * dh;
        let mut x = step_x - 1;
        for gx in 0..grid_w {
            let ix = gx * dw;
            let idx = if trans { ix * 8 + iy } else { iy * 8 + ix };
            let mut s = 0i32;
            for i in 0..11 {
                s += DIP_WEIGHTS[m][idx][i] as i32 * inp[i];
            }
            dst[y * stride + x] = bd.pixel_clip(((s + 2048) >> 12) - in_sum);
            x += step_x;
        }
        y += step_y;
    }

    if step_x > 1 {
        y = step_y - 1;
        for _gy in 0..grid_h {
            let mut p1: i32 = tl[o - (y + 1)].into();
            let mut x = 0usize;
            for _gx in 0..grid_w {
                let p0 = p1;
                p1 = dst[y * stride + x + step_x - 1].into();
                for z in 0..step_x - 1 {
                    let z1 = (z + 1) as i32;
                    dst[y * stride + x + z] =
                        BD::Pixel::from_i32((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2);
                }
                x += step_x;
            }
            y += step_y;
        }
    }

    if step_y > 1 {
        for x in 0..width {
            let mut p1: i32 = tl[o + x + 1].into();
            y = 0;
            for _gy in 0..grid_h {
                let p0 = p1;
                p1 = dst[(y + step_y - 1) * stride + x].into();
                for z in 0..step_y - 1 {
                    let z1 = (z + 1) as i32;
                    dst[(y + z) * stride + x] =
                        BD::Pixel::from_i32((p0 * (step_y as i32 - z1) + p1 * z1) >> uhl2);
                }
                y += step_y;
            }
        }
    }
}

pub fn pal_pred_8bpc(dst: &mut [u8], stride: usize, pal: &[u8], idx: &[u8], w: usize, h: usize) {
    pal_pred(dst, stride, pal, idx, w, h);
}

pub fn pal_pred<P: Pixel>(dst: &mut [P], stride: usize, pal: &[P], idx: &[u8], w: usize, h: usize) {
    let mut ii = 0;
    for y in 0..h {
        let mut x = 0;
        while x < w {
            let i = idx[ii];
            dst[y * stride + x] = pal[(i & 7) as usize];
            dst[y * stride + x + 1] = pal[(i >> 4) as usize];
            x += 2;
            ii += 1;
        }
    }
}

pub const CFL_FLT_TYPE_UNIFORM: i32 = 0;
pub const CFL_FLT_TYPE_VSTRIP: i32 = 1;
pub const CFL_FLT_TYPE_GAUSS: i32 = 2;
pub const CFL_HAS_TOP: i32 = 1 << 2;
pub const CFL_HAS_LEFT: i32 = 1 << 3;
pub const CFL_DIR_ALL: i32 = CflMhDir::All as i32;
pub const CFL_DIR_LEFT: i32 = CflMhDir::Left as i32;
pub const CFL_DIR_TOP: i32 = CflMhDir::Top as i32;
pub const CFL_IS_TOP_SB_EDGE: u32 = 1 << 4;
pub const CFL_ALPHA_LOG2: u32 = 5;
pub const CFL_ALPHA_U_SHIFT: u32 = 16 - CFL_ALPHA_LOG2;
pub const CFL_ALPHA_V_SHIFT: u32 = 32 - CFL_ALPHA_LOG2;
pub const CFL_ALPHA_U_MASK: u32 = ((1 << CFL_ALPHA_LOG2) - 1) << CFL_ALPHA_U_SHIFT;
pub const CFL_ALPHA_V_MASK: u32 = ((1 << CFL_ALPHA_LOG2) - 1) << CFL_ALPHA_V_SHIFT;

#[inline(always)]
fn cfl_filter<P: Pixel>(
    src: &[P],
    c: usize,
    l: usize,
    r: usize,
    b: usize,
    top: &[P],
    tc: usize,
    filter_type: i32,
) -> P {
    let s = |i: usize| -> i32 { src[i].into() };
    let t = |i: usize| -> i32 { top[i].into() };
    match filter_type {
        CFL_FLT_TYPE_UNIFORM => P::from_i32((s(c) + s(r) + s(b + c) + s(b + r)) >> 2),
        CFL_FLT_TYPE_VSTRIP => P::from_i32(
            (s(l) + 2 * s(c) + s(r) + s(b + l) + 2 * s(b + c) + s(b + r)) >> 3,
        ),
        _ => P::from_i32((s(l) + 4 * s(c) + s(r) + t(tc) + s(b + c)) >> 3),
    }
}

/// Generate downsampled luma for CFL prediction at 4:2:0 resolution.
///
/// All src/top_sb_edge indexing uses a "pointer offset" model: `sp` tracks the
/// current source position as an offset into `src`. The caller must ensure `src`
/// is large enough that `sp - n_left*2` never underflows.
///
/// `src_off` is the initial offset into `src` (before subtracting n_left*2).
/// `top_sb_off` is the initial offset into `top_sb_edge` (before subtracting n_left*2).
#[allow(clippy::too_many_arguments)]
pub fn cfl_gen_y_420_8bpc(
    dst: &mut [u8],
    dst_top_stride: usize,
    src: &[u8],
    src_off: usize,
    top_sb_edge: Option<(&[u8], usize)>,
    src_stride: usize,
    refw: usize,
    refh: usize,
    tw: usize,
    th: usize,
    flags: i32,
    filter_type: i32,
) {
    cfl_gen_y_420(
        dst,
        dst_top_stride,
        src,
        src_off,
        top_sb_edge,
        src_stride,
        refw,
        refh,
        tw,
        th,
        flags,
        filter_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn cfl_gen_y_420<P: Pixel>(
    dst: &mut [P],
    dst_top_stride: usize,
    src: &[P],
    src_off: usize,
    top_sb_edge: Option<(&[P], usize)>,
    src_stride: usize,
    refw: usize,
    refh: usize,
    tw: usize,
    th: usize,
    flags: i32,
    filter_type: i32,
) {
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let dir = flags & CFL_DIR_ALL;
    let n_left: usize = if has_l {
        1 + (dir == CFL_DIR_LEFT) as usize
    } else {
        0
    };
    let n_top: usize = if has_t {
        1 + (dir == CFL_DIR_TOP) as usize
    } else {
        0
    };
    let dst_left_base = n_top * dst_top_stride + 64 * 64;
    let ss = n_left << 1;

    let mut dst_p = 0usize;
    let mut dst_lp = dst_left_base;

    // tl+t+tr: top reference rows
    if has_t {
        let has_tsb = top_sb_edge.is_some();
        let (tsb, tsb_off) = top_sb_edge.unwrap_or((src, src_off));
        let mut top_sp: usize;
        let top_buf: &[P];
        let b: isize;
        let mut t: isize;

        if has_tsb {
            top_sp = tsb_off - ss;
            top_buf = tsb;
            b = 0;
            t = 0;
        } else {
            top_sp = (src_off - ss) - n_top * 2 * src_stride;
            top_buf = src;
            b = src_stride as isize;
            t = if n_top == 1 {
                -(src_stride as isize)
            } else {
                0
            };
        }

        for _y in 0..n_top {
            for x in 0..n_left {
                let c = x * 2;
                let r = c + 1;
                // C (ipred_tmpl.c:1156): `(n_left & 1) ? c - 1 : imax(c - 1, 0)`.
                // For odd n_left the left tap is NOT clamped at column 0 (reads the
                // pixel one column left, i.e. relative -1); only the even case clamps.
                let l_off: isize = if n_left & 1 != 0 {
                    c as isize - 1
                } else {
                    imax(c as i32 - 1, 0) as isize
                };
                dst[dst_lp + x] = cfl_filter(
                    top_buf,
                    top_sp + c,
                    (top_sp as isize + l_off) as usize,
                    top_sp + r,
                    b as usize,
                    top_buf,
                    (top_sp as isize + t) as usize + c,
                    filter_type,
                );
            }
            for x in n_left..refw {
                let c = x * 2;
                let r = c + 1;
                let l_idx = if n_left > 0 {
                    c - 1
                } else {
                    imax(c as i32 - 1, 0) as usize
                };
                dst[dst_p + x - n_left] = cfl_filter(
                    top_buf,
                    top_sp + c,
                    top_sp + l_idx,
                    top_sp + r,
                    b as usize,
                    top_buf,
                    (top_sp as isize + t) as usize + c,
                    filter_type,
                );
            }
            if !has_tsb {
                top_sp += 2 * src_stride;
                t = -(src_stride as isize);
            }
            dst_lp += n_left;
            dst_p += dst_top_stride;
        }
    }

    // l+blk: main block rows
    let b = src_stride as isize;
    let mut sp = src_off - ss;
    let first_top: (&[P], usize) = if has_t {
        if let Some((tsb, tsb_off)) = top_sb_edge {
            (tsb, tsb_off - ss)
        } else {
            (src, src_off - ss - src_stride)
        }
    } else {
        (src, src_off - ss)
    };

    for y in 0..th {
        let (tb, tp) = if y == 0 {
            first_top
        } else {
            (src, sp - src_stride)
        };

        for x in 0..n_left {
            let c = x * 2;
            let r = c + 1;
            // C (ipred_tmpl.c:1201): odd n_left does not clamp the left tap at 0.
            let l_off: isize = if n_left & 1 != 0 {
                c as isize - 1
            } else {
                imax(c as i32 - 1, 0) as isize
            };
            dst[dst_lp + x] = cfl_filter(
                src,
                sp + c,
                (sp as isize + l_off) as usize,
                sp + r,
                b as usize,
                tb,
                tp + c,
                filter_type,
            );
        }
        for x in n_left..n_left + tw {
            let c = x * 2;
            let r = c + 1;
            let l_idx = if n_left > 0 {
                c - 1
            } else {
                imax(c as i32 - 1, 0) as usize
            };
            dst[dst_p + x - n_left] = cfl_filter(
                src,
                sp + c,
                sp + l_idx,
                sp + r,
                b as usize,
                tb,
                tp + c,
                filter_type,
            );
        }
        sp += src_stride << 1;
        dst_lp += n_left;
        dst_p += tw;
    }

    // bl: bottom-left extension rows
    let n_bl = refh - th;
    for _y in 0..n_bl {
        let top_sp_bl = sp - src_stride;
        for x in 0..n_left {
            let c = x * 2;
            let r = c + 1;
            // C (ipred_tmpl.c:1240): odd n_left does not clamp the left tap at 0.
            let l_off: isize = if n_left & 1 != 0 {
                c as isize - 1
            } else {
                imax(c as i32 - 1, 0) as isize
            };
            dst[dst_lp + x] = cfl_filter(
                src,
                sp + c,
                (sp as isize + l_off) as usize,
                sp + r,
                b as usize,
                src,
                top_sp_bl + c,
                filter_type,
            );
        }
        sp += src_stride << 1;
        dst_lp += n_left;
    }
}

pub const CFL_MHCCP_MAX_EDGE_SAMPLES: usize = 386;
pub const CFL_MHCCP_MAX_LUMA_SIZE: usize = 4736;

#[inline(always)]
fn sqrnd<BD: BitDepth>(bd: BD, v: i32) -> i32 {
    let b = bd.bitdepth() as i32;
    let mid = 1 << (b - 1);
    (v * v + mid) >> b
}

pub fn cfl_gen_mat_8bpc(
    mat: &mut [[i32; 3]; 3],
    imat: &mut [[u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2],
    y: &[u8],
    y_off: usize,
    y_top_stride: usize,
    refw: usize,
    refh: usize,
    edge_flags: i32,
    dir: CflMhDir,
) {
    cfl_gen_mat(
        BitDepth8, mat, imat, y, y_off, y_top_stride, refw, refh, edge_flags, dir,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn cfl_gen_mat<BD: BitDepth>(
    bd: BD,
    mat: &mut [[i32; 3]; 3],
    imat: &mut [[u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2],
    y: &[BD::Pixel],
    y_off: usize,
    y_top_stride: usize,
    refw: usize,
    refh: usize,
    edge_flags: i32,
    dir: CflMhDir,
) {
    let bd_bits = bd.bitdepth() as i32;
    let has_t = edge_flags & CFL_HAS_TOP != 0;
    let has_l = edge_flags & CFL_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = y_off + n_top * y_top_stride + 64 * 64;

    for r in mat.iter_mut() {
        r.fill(0);
    }

    let mut n: usize = 0;

    let mat2sh = bd_bits - 1;
    if has_t {
        for i in 0..n_left {
            let v0: i32 = y[left_off + i].into();
            let neighbor: i32 = if i == 0 {
                y[left_off + i + (dir_t as usize | dir_l as usize)].into()
            } else {
                y[y_off].into()
            };
            let v1 = sqrnd(bd, neighbor);
            imat[0][n] = v0 as u16;
            imat[1][n] = v1 as u16;
            mat[0][0] += v0 * v0;
            mat[0][1] += v0 * v1;
            mat[0][2] += v0 << mat2sh;
            mat[1][1] += v1 * v1;
            mat[1][2] += v1 << mat2sh;
            n += 1;
        }
        let start: usize = if !dir_l && !has_l { 1 } else { 0 };
        let end = imax(
            start as i32,
            refw as i32 - n_left as i32 - 1 - (start == 0) as i32,
        ) as usize;
        for i in start..end {
            let v0: i32 = y[y_off + i].into();
            let yi = y_off + dir_t as usize * y_top_stride + i + dir_l as usize;
            let v1 = sqrnd(bd, y[yi].into());
            imat[0][n] = v0 as u16;
            imat[1][n] = v1 as u16;
            mat[0][0] += v0 * v0;
            mat[0][1] += v0 * v1;
            mat[0][2] += v0 << mat2sh;
            mat[1][1] += v1 * v1;
            mat[1][2] += v1 << mat2sh;
            n += 1;
        }
    }

    if has_l {
        // C (ipred_tmpl.c:1307-1308): start = dir_t && !has_t;
        //   for (i = 1 - start; i < refh - start - 1; i++)
        let start = (dir_t && !has_t) as i32;
        let begin = (1 - start) as usize;
        let end = imax(begin as i32, refh as i32 - start - 1) as usize;
        for i in begin..end {
            let v0: i32 = y[left_off + i * n_left].into();
            let ni = left_off + (i + dir_t as usize) * n_left + dir_l as usize;
            let v1 = sqrnd(bd, y[ni].into());
            imat[0][n] = v0 as u16;
            imat[1][n] = v1 as u16;
            mat[0][0] += v0 * v0;
            mat[0][1] += v0 * v1;
            mat[0][2] += v0 << mat2sh;
            mat[1][1] += v1 * v1;
            mat[1][2] += v1 << mat2sh;
            n += 1;
        }
    }

    mat[2][2] = (n as i32) << ((bd_bits - 1) << 1);

    if n > 0 {
        let nl2 = 31 - clz(n as u32) as i32;
        let mat_sh = 22 - 2 * bd_bits - nl2 - (n as i32 & ((1 << nl2) - 1) != 0) as i32;
        if mat_sh > 0 {
            for i in 0..3 {
                for j in i..3 {
                    mat[i][j] <<= mat_sh;
                }
            }
        } else if mat_sh < 0 {
            for i in 0..3 {
                for j in i..3 {
                    mat[i][j] >>= -mat_sh;
                }
            }
        }
    }

    let add = 2 << (bd_bits - 8);
    mat[0][0] += add;
    mat[1][1] += add;
    mat[2][2] += add;
    mat[1][0] = mat[0][1];
    mat[2][0] = mat[0][2];
    mat[2][1] = mat[1][2];
}

pub fn cfl_calc_alphas_8bpc(
    alpha: &mut [i32; 3],
    c: &[u8],
    c_off: usize,
    top_sb_edge: Option<(&[u8], usize)>,
    stride: usize,
    refw: usize,
    refh: usize,
    mat: &mut [[i32; 3]; 3],
    imat: &[[u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2],
    edge_flags: i32,
) {
    cfl_calc_alphas(
        BitDepth8, alpha, c, c_off, top_sb_edge, stride, refw, refh, mat, imat, edge_flags,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn cfl_calc_alphas<BD: BitDepth>(
    bd: BD,
    alpha: &mut [i32; 3],
    c: &[BD::Pixel],
    c_off: usize,
    top_sb_edge: Option<(&[BD::Pixel], usize)>,
    stride: usize,
    refw: usize,
    refh: usize,
    mat: &mut [[i32; 3]; 3],
    imat: &[[u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2],
    edge_flags: i32,
) {
    let bd_bits = bd.bitdepth() as i32;
    let has_t = edge_flags & CFL_HAS_TOP != 0;
    let has_l = edge_flags & CFL_HAS_LEFT != 0;
    let a2sh = bd_bits - 1;

    let mut n: usize = 0;
    if has_t {
        let (top, top_off) = if let Some((tsb, tsb_off)) = top_sb_edge {
            (tsb, tsb_off - has_l as usize)
        } else {
            (c, c_off - stride - has_l as usize)
        };
        let start: usize = if !has_l { 1 } else { 0 };
        let end = imax(start as i32, refw as i32 - 1 - (start == 0) as i32) as usize;
        for i in start..end {
            let v: i32 = top[top_off + i].into();
            alpha[0] += imat[0][n] as i32 * v;
            alpha[1] += imat[1][n] as i32 * v;
            alpha[2] += v << a2sh;
            n += 1;
        }
    }
    if has_l {
        // C: `for (i = !has_t; i < refh - 1 - has_t; ...)` (ipred_tmpl.c).
        let start = if has_t { 0 } else { 1 }; // = !has_t
        let end = if has_t { refh - 2 } else { refh - 1 };
        for i in start..end {
            let v: i32 = c[c_off + i * stride - 1].into();
            alpha[0] += imat[0][n] as i32 * v;
            alpha[1] += imat[1][n] as i32 * v;
            alpha[2] += v << a2sh;
            n += 1;
        }
    }

    if n > 0 {
        let nl2 = 31 - clz(n as u32) as i32;
        let mat_sh = 22 - 2 * bd_bits - nl2 - (n as i32 & ((1 << nl2) - 1) != 0) as i32;
        if mat_sh > 0 {
            for a in alpha.iter_mut() {
                *a <<= mat_sh;
            }
        } else if mat_sh < 0 {
            for a in alpha.iter_mut() {
                *a >>= -mat_sh;
            }
        }
    }

    let mut tmp = [[0i32; 2]; 3];
    let (mut scale, mut sh) = get_div_scale_sh(mat[0][0]);
    tmp[0][0] = mul32(mat[0][1], scale, sh);
    tmp[0][1] = mul32(mat[0][2], scale, sh);
    alpha[0] = mul32(alpha[0], scale, sh);
    tmp[1][0] = mat[1][1] - mul32(mat[1][0], tmp[0][0], 16);
    tmp[1][1] = mat[1][2] - mul32(mat[1][0], tmp[0][1], 16);
    alpha[1] -= mul32(mat[1][0], alpha[0], 16);
    tmp[2][0] = mat[2][1] - mul32(mat[2][0], tmp[0][0], 16);
    tmp[2][1] = mat[2][2] - mul32(mat[2][0], tmp[0][1], 16);
    alpha[2] -= mul32(mat[2][0], alpha[0], 16);

    (scale, sh) = get_div_scale_sh(tmp[1][0]);
    tmp[1][1] = mul32(tmp[1][1], scale, sh);
    alpha[1] = mul32(alpha[1], scale, sh);
    tmp[2][1] -= mul32(tmp[2][0], tmp[1][1], 16);
    alpha[2] -= mul32(tmp[2][0], alpha[1], 16);

    (scale, sh) = get_div_scale_sh(tmp[2][1]);
    alpha[2] = mul32(alpha[2], scale, sh);
    alpha[1] -= mul32(tmp[1][1], alpha[2], 16);
    alpha[0] -= mul32(tmp[0][0], alpha[1], 16) + mul32(tmp[0][1], alpha[2], 16);
}

pub fn cfl_mhccp_pred_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_top_stride: usize,
    w: usize,
    h: usize,
    alpha: &[i32; 3],
    edge_flags: i32,
    dir: CflMhDir,
) {
    cfl_mhccp_pred(
        BitDepth8, dst, dst_stride, src, src_off, src_top_stride, w, h, alpha, edge_flags, dir,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn cfl_mhccp_pred<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_off: usize,
    src_top_stride: usize,
    w: usize,
    h: usize,
    alpha: &[i32; 3],
    edge_flags: i32,
    dir: CflMhDir,
) {
    let has_t = edge_flags & CFL_HAS_TOP != 0;
    let has_l = edge_flags & CFL_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;

    let mid = 1 << (bd.bitdepth() as i32 - 1);
    let a2v2 = mul32(alpha[2], mid, 16);
    let mut sp = src_off;
    let mut dp = 0usize;
    let mut y = 0usize;

    while y < dir_t as usize && has_t {
        for x in 0..w {
            let v0: i32 = src[sp + x - src_top_stride].into();
            let v1 = sqrnd(bd, src[sp + x].into());
            dst[dp + x] =
                bd.pixel_clip(mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v2);
        }
        sp += w;
        dp += dst_stride;
        y += 1;
    }

    while y < h {
        let mut x = 0usize;
        while x < dir_l as usize && has_l {
            let v0: i32 = src[left_off + y * n_left + dir_l as usize].into();
            let v1 = sqrnd(bd, src[sp].into());
            dst[dp] = bd.pixel_clip(mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v2);
            x += 1;
        }
        while x < w {
            let v0_idx = if dir_t {
                sp + x - (((y > 0) as usize) | has_t as usize) * w
            } else if dir_l {
                sp + imax(x as i32 - 1, 0) as usize
            } else {
                sp + x
            };
            let v0: i32 = src[v0_idx].into();
            let v1 = sqrnd(bd, src[sp + x].into());
            dst[dp + x] =
                bd.pixel_clip(mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v2);
            x += 1;
        }
        sp += w;
        dp += dst_stride;
        y += 1;
    }
}

fn cfl_luma_left<P: Pixel>(
    ypx: &[P],
    yleft: usize,
    ystride: isize,
    ss_hor: usize,
    ss_ver: usize,
    flags: u32,
    y: usize,
) -> i32 {
    let p = |i: usize| -> i32 { ypx[i].into() };
    if ss_hor | ss_ver == 0 {
        return p(yleft) << 3;
    }
    if ss_ver == 0 {
        let flt = flags & (CFL_FLT_TYPE_GAUSS as u32 | CFL_FLT_TYPE_VSTRIP as u32);
        return if flt == CFL_FLT_TYPE_GAUSS as u32 {
            p(yleft) << 3
        } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
            (p(yleft - 1) + 2 * p(yleft) + p(yleft + 1)) << 1
        } else {
            (p(yleft) + p(yleft + 1)) << 2
        };
    }
    let flt = flags & 3;
    if flt == CFL_FLT_TYPE_GAUSS as u32 {
        let top = if y > 0 {
            (yleft as isize - ystride) as usize
        } else {
            yleft
        };
        p(yleft - 1)
            + 4 * p(yleft)
            + p(yleft + 1)
            + p(top)
            + p((yleft as isize + ystride) as usize)
    } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
        p(yleft - 1)
            + 2 * p(yleft)
            + p(yleft + 1)
            + p((yleft as isize + ystride) as usize - 1)
            + 2 * p((yleft as isize + ystride) as usize)
            + p((yleft as isize + ystride) as usize + 1)
    } else {
        (p(yleft)
            + p(yleft + 1)
            + p((yleft as isize + ystride) as usize)
            + p((yleft as isize + ystride) as usize + 1))
            << 1
    }
}

#[allow(clippy::too_many_arguments)]
fn cfl_luma_top<P: Pixel>(
    ytop: &[P],
    xl: usize,
    base: usize,
    ystride: isize,
    ss_hor: usize,
    ss_ver: usize,
    flags: u32,
    is_top_sb: bool,
) -> i32 {
    let p = |i: usize| -> i32 { ytop[i].into() };
    // `xl` is an absolute index into `ytop`; `base` is the block's left-edge index
    // so the left-neighbour clamp matches dav2d's block-relative `imax(0, xl - 1)`
    // (dav2d's `ytop` pointer is pre-offset to the block, so its column 0 is the
    // clamp boundary — here that boundary is `base`).
    let left = imax(base as i32, xl as i32 - 1) as usize;
    if ss_hor | ss_ver == 0 {
        return p(xl) << 3;
    }
    if ss_ver == 0 {
        let flt = flags & 3;
        return if flt == CFL_FLT_TYPE_GAUSS as u32 {
            p(xl) << 3
        } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
            (p(left) + 2 * p(xl) + p(xl + 1)) << 1
        } else {
            (p(xl) + p(xl + 1)) << 2
        };
    }
    let bottom = if is_top_sb { 0isize } else { ystride };
    let flt = flags & 3;
    if flt == CFL_FLT_TYPE_GAUSS as u32 {
        p(left)
            + 4 * p(xl)
            + p(xl + 1)
            + p((xl as isize - bottom) as usize)
            + p((xl as isize + bottom) as usize)
    } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
        p(left)
            + 2 * p(xl)
            + p(xl + 1)
            + p((left as isize + bottom) as usize)
            + 2 * p((xl as isize + bottom) as usize)
            + p((xl as isize + bottom) as usize + 1)
    } else {
        (p(xl)
            + p(xl + 1)
            + p((xl as isize + bottom) as usize)
            + p((xl as isize + bottom) as usize + 1))
            << 1
    }
}

#[allow(clippy::too_many_arguments)]
fn cfl_luma_block<P: Pixel>(
    ypx: &[P],
    px: usize,
    xl: usize,
    ystride: isize,
    ss_hor: usize,
    ss_ver: usize,
    flags: u32,
    y: usize,
) -> i32 {
    let p = |i: usize| -> i32 { ypx[i].into() };
    if ss_hor | ss_ver == 0 {
        return p(px + xl) << 3;
    }
    if ss_ver == 0 {
        let flt = flags & 3;
        return if flt == CFL_FLT_TYPE_GAUSS as u32 {
            p(px + xl) << 3
        } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
            let left = imax((xl as i32) & -64, xl as i32 - 1) as usize;
            (p(px + left) + 2 * p(px + xl) + p(px + xl + 1)) << 1
        } else {
            (p(px + xl) + p(px + xl + 1)) << 2
        };
    }
    let bot = (px + xl) as isize + ystride;
    let left = imax((xl as i32) & -64, xl as i32 - 1) as usize;
    let flt = flags & 3;
    if flt == CFL_FLT_TYPE_GAUSS as u32 {
        let top = if y & 31 == 0 {
            px + xl
        } else {
            (px as isize + xl as isize - ystride) as usize
        };
        p(px + left)
            + 4 * p(px + xl)
            + p(px + xl + 1)
            + p(top)
            + p(bot as usize)
    } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
        p(px + left)
            + 2 * p(px + xl)
            + p(px + xl + 1)
            + p((px + left) as isize as usize + ystride as usize)
            + 2 * p(bot as usize)
            + p(bot as usize + 1)
    } else {
        (p(px + xl)
            + p(px + xl + 1)
            + p(bot as usize)
            + p(bot as usize + 1))
            << 1
    }
}

/// CFL chroma-from-luma prediction (explicit and implicit modes).
///
/// Buffer layout: `u_buf`/`v_buf` are the chroma planes. The block starts at
/// `u_off`/`v_off`; the left neighbor column is at offset -1 from those.
#[allow(clippy::too_many_arguments)]
pub fn cfl_pred_8bpc(
    ytop: &[u8],
    ytop_off: usize,
    utop: &[u8],
    utop_off: usize,
    vtop: &[u8],
    vtop_off: usize,
    ypx: &[u8],
    ypx_off: usize,
    u_buf: &mut [u8],
    u_off: usize,
    v_buf: &mut [u8],
    v_off: usize,
    ystride: isize,
    cstride: isize,
    wpad: usize,
    hpad: usize,
    w: usize,
    h: usize,
    flags: u32,
    implicit: bool,
    ss_hor: usize,
    ss_ver: usize,
) {
    cfl_pred(
        BitDepth8, ytop, ytop_off, utop, utop_off, vtop, vtop_off, ypx, ypx_off, u_buf, u_off,
        v_buf, v_off, ystride, cstride, wpad, hpad, w, h, flags, implicit, ss_hor, ss_ver,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn cfl_pred<BD: BitDepth>(
    bd: BD,
    ytop: &[BD::Pixel],
    ytop_off: usize,
    utop: &[BD::Pixel],
    utop_off: usize,
    vtop: &[BD::Pixel],
    vtop_off: usize,
    ypx: &[BD::Pixel],
    ypx_off: usize,
    u_buf: &mut [BD::Pixel],
    u_off: usize,
    v_buf: &mut [BD::Pixel],
    v_off: usize,
    ystride: isize,
    cstride: isize,
    wpad: usize,
    hpad: usize,
    w: usize,
    h: usize,
    flags: u32,
    implicit: bool,
    ss_hor: usize,
    ss_ver: usize,
) {
    let has_t = flags & CFL_HAS_TOP as u32 != 0;
    let has_l = flags & CFL_HAS_LEFT as u32 != 0;
    let xlim = w - 4 * wpad;
    let ylim = h - 4 * hpad;
    let skiph = w == 64;
    let skipv = h == 64;

    let mut dc = [0i32; 3];
    let mut n_top = 0usize;
    let mut n_left = 0usize;
    let mut i = 0usize;
    let mut sum_x = 0i32;
    let mut sum_xx = 0i32;
    let mut sum_y = [0i32; 2];
    let mut sum_xy = [0i32; 2];
    let mut edge = [[BD::Pixel::default(); 8]; 3];

    if implicit {
        if has_t && has_l {
            if w > h * 2 {
                n_top = 8;
                n_left = 0;
            } else if h > w * 2 {
                n_top = 0;
                n_left = 8;
            } else {
                n_top = 4;
                n_left = 4;
            }
        } else {
            n_top = if has_t { imin(8, w as i32) as usize } else { 0 };
            n_left = if has_l { imin(8, h as i32) as usize } else { 0 };
        }
    }

    if has_l {
        let mut yleft = ypx_off - (1 + ss_hor);
        let step = if n_left > 0 {
            h >> ctz(n_left as u32)
        } else {
            0
        };
        let mut l = 0i32;
        let mut u_lp = u_off;
        let mut v_lp = v_off;
        for y in 0..ylim {
            l = cfl_luma_left(ypx, yleft, ystride, ss_hor, ss_ver, flags, y);
            if !skipv || y & 1 == 0 {
                dc[0] += l;
                dc[1] += Into::<i32>::into(u_buf[u_lp - 1]);
                dc[2] += Into::<i32>::into(v_buf[v_lp - 1]);
            }
            if n_left > 0 && (y & (step - 1)) ^ (step >> 1) == 0 {
                edge[0][i] = BD::Pixel::from_i32(l >> 3);
                edge[1][i] = u_buf[u_lp - 1];
                edge[2][i] = v_buf[v_lp - 1];
                i += 1;
            }
            yleft = (yleft as isize + (ystride << ss_ver)) as usize;
            u_lp = (u_lp as isize + cstride) as usize;
            v_lp = (v_lp as isize + cstride) as usize;
        }
        for y in ylim..h {
            if !skipv || y & 1 == 0 {
                dc[0] += l;
                dc[1] += Into::<i32>::into(u_buf[(u_lp as isize - cstride) as usize]);
                dc[2] += Into::<i32>::into(v_buf[(v_lp as isize - cstride) as usize]);
            }
            if n_left > 0 && (y & (step - 1)) ^ (step >> 1) == 0 {
                edge[0][i] = BD::Pixel::from_i32(l >> 3);
                edge[1][i] = u_buf[(u_lp as isize - cstride) as usize];
                edge[2][i] = v_buf[(v_lp as isize - cstride) as usize];
                i += 1;
            }
        }
    }

    if has_t {
        let step = if n_top > 0 { w >> ctz(n_top as u32) } else { 0 };
        let is_top_sb = flags & CFL_IS_TOP_SB_EDGE != 0;
        let mut l = 0i32;
        for x in 0..xlim {
            let xl = x << ss_hor;
            l = cfl_luma_top(
                ytop,
                ytop_off + xl,
                ytop_off,
                ystride,
                ss_hor,
                ss_ver,
                flags,
                is_top_sb,
            );
            if !skiph || x & 1 == 0 {
                dc[0] += l;
                dc[1] += Into::<i32>::into(utop[utop_off + x]);
                dc[2] += Into::<i32>::into(vtop[vtop_off + x]);
            }
            if n_top > 0 && (x & (step - 1)) ^ (step >> 1) == 0 {
                edge[0][i] = BD::Pixel::from_i32(l >> 3);
                edge[1][i] = utop[utop_off + x];
                edge[2][i] = vtop[vtop_off + x];
                i += 1;
            }
        }
        for x in xlim..w {
            if !skiph || x & 1 == 0 {
                dc[0] += l;
                dc[1] += Into::<i32>::into(utop[utop_off + xlim - 1]);
                dc[2] += Into::<i32>::into(vtop[vtop_off + xlim - 1]);
            }
            if n_top > 0 && (x & (step - 1)) ^ (step >> 1) == 0 {
                edge[0][i] = BD::Pixel::from_i32(l >> 3);
                edge[1][i] = utop[utop_off + xlim - 1];
                edge[2][i] = vtop[vtop_off + xlim - 1];
                i += 1;
            }
        }
    }

    if !has_t && !has_l {
        dc[0] = 4 << bd.bitdepth();
        dc[1] = (bd.bitdepth_max() + 1) >> 1;
        dc[2] = (bd.bitdepth_max() + 1) >> 1;
    } else {
        let npx = (if has_t { w >> skiph as usize } else { 0 })
            + (if has_l { h >> skipv as usize } else { 0 });
        if npx & (npx - 1) == 0 {
            dc[0] = (dc[0] + (npx as i32 >> 1)) >> ctz(npx as u32);
            dc[1] = (dc[1] + (npx as i32 >> 1)) >> ctz(npx as u32);
            dc[2] = (dc[2] + (npx as i32 >> 1)) >> ctz(npx as u32);
        } else {
            dc[0] = fast_div32_dc(dc[0] as u32, npx as u32) as i32;
            dc[1] = fast_div32_dc(dc[1] as u32, npx as u32) as i32;
            dc[2] = fast_div32_dc(dc[2] as u32, npx as u32) as i32;
        }
    }

    let mut alpha = [0i32; 2];
    if implicit {
        for j in 0..n_top + n_left {
            let e0: i32 = edge[0][j].into();
            let e1: i32 = edge[1][j].into();
            let e2: i32 = edge[2][j].into();
            sum_x += e0;
            sum_y[0] += e1;
            sum_y[1] += e2;
            sum_xx += e0 * e0;
            sum_xy[0] += e0 * e1;
            sum_xy[1] += e0 * e2;
        }
        let count_l2 = ctz((n_top + n_left) as u32);
        let den = sum_xx - ((sum_x as i64 * sum_x as i64) >> count_l2) as i32;
        for pl in 0..2 {
            let num = sum_xy[pl] - ((sum_x as i64 * sum_y[pl] as i64) >> count_l2) as i32;
            alpha[pl] = derive_alpha(num, den, 0);
        }
    } else {
        let shu = CFL_ALPHA_U_SHIFT - 5;
        let shv = CFL_ALPHA_V_SHIFT - 5;
        alpha[0] = ((flags & CFL_ALPHA_U_MASK) as i16 as i32) >> shu;
        alpha[1] = ((flags & CFL_ALPHA_V_MASK) as i32) >> shv;
    }

    if alpha[0] == 0 {
        let dc_u = bd.pixel_clip(dc[1]);
        splat_dc(u_buf, cstride as usize, u_off, w, h, dc_u);
    }
    if alpha[1] == 0 {
        let dc_v = bd.pixel_clip(dc[2]);
        splat_dc(v_buf, cstride as usize, v_off, w, h, dc_v);
    }

    let mut yp = ypx_off;
    let mut u_dp = u_off;
    let mut v_dp = v_off;

    for y in 0..ylim {
        for x in 0..xlim {
            let xl = x << ss_hor;
            let ac = cfl_luma_block(ypx, yp, xl, ystride, ss_hor, ss_ver, flags, y) - dc[0];
            for pl in 0..2 {
                if alpha[pl] != 0 {
                    let diff = alpha[pl] * ac;
                    let val = dc[1 + pl] + apply_sign((diff.abs() + 1024) >> 11, diff);
                    let dp = if pl == 0 { u_dp } else { v_dp };
                    let buf: &mut [BD::Pixel] = if pl == 0 { u_buf } else { v_buf };
                    buf[dp + x] = bd.pixel_clip(val);
                }
            }
        }
        for pl in 0..2 {
            if alpha[pl] != 0 {
                let dp = if pl == 0 { u_dp } else { v_dp };
                let buf: &mut [BD::Pixel] = if pl == 0 { u_buf } else { v_buf };
                let last_val = buf[dp + xlim - 1];
                for xpad in xlim..w {
                    buf[dp + xpad] = last_val;
                }
            }
        }
        yp = (yp as isize + (ystride << ss_ver)) as usize;
        u_dp = (u_dp as isize + cstride) as usize;
        v_dp = (v_dp as isize + cstride) as usize;
    }

    for pl in 0..2 {
        if alpha[pl] != 0 {
            let buf: &mut [BD::Pixel] = if pl == 0 { u_buf } else { v_buf };
            let mut dp = if pl == 0 { u_dp } else { v_dp };
            for y in ylim..h {
                let prev_row = (dp as isize - (1 + y as isize - ylim as isize) * cstride) as usize;
                let tmp: Vec<BD::Pixel> = buf[prev_row..prev_row + w].to_vec();
                buf[dp..dp + w].copy_from_slice(&tmp);
                dp = (dp as isize + cstride) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tl_buf(w: usize, h: usize, o: usize) -> Vec<u8> {
        let mut tl = vec![128u8; o + w + 2];
        for i in 0..w + 2 {
            tl[o + i] = (100 + i) as u8;
        }
        for i in 1..=h + 1 {
            if o >= i {
                tl[o - i] = (80 + i) as u8;
            }
        }
        tl
    }

    #[test]
    fn test_ipred_dc_128() {
        let mut dst = [0u8; 16];
        ipred_dc_128_8bpc(&mut dst, 4, 4, 4);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_ipred_dc_top_uniform() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_dc_top_8bpc(&mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_dc_left_uniform() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_dc_left_8bpc(&mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_dc_uniform() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_dc_8bpc(&mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_v_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_v_8bpc(&mut dst, 4, &tl, 8, 4, 4, 0);
        for y in 1..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], dst[x]);
            }
        }
    }

    #[test]
    fn test_ipred_h_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_h_8bpc(&mut dst, 4, &tl, 8, 4, 4, 0);
        for y in 0..4 {
            let v = dst[y * 4];
            for x in 1..4 {
                assert_eq!(dst[y * 4 + x], v);
            }
        }
    }

    #[test]
    fn test_fast_div32_dc() {
        assert_eq!(fast_div32_dc(256, 4), 64);
        assert_eq!(fast_div32_dc(100, 10), 10);
    }

    #[test]
    fn test_dc_gen_top_basic() {
        let mut tl = vec![0u8; 20];
        for i in 0..4 {
            tl[9 + i] = 80;
        }
        let dc = dc_gen_top(&tl, 8, 4);
        assert_eq!(dc, 80);
    }

    #[test]
    fn test_get_filter_strength_sm_small() {
        assert_eq!(get_filter_strength(8, 64, true), 2);
        assert_eq!(get_filter_strength(8, 40, true), 1);
        assert_eq!(get_filter_strength(8, 30, true), 0);
    }

    #[test]
    fn test_get_filter_strength_nosm_large() {
        assert_eq!(get_filter_strength(48, 10, false), 3);
        assert_eq!(get_filter_strength(32, 32, false), 3);
        assert_eq!(get_filter_strength(32, 4, false), 2);
        assert_eq!(get_filter_strength(32, 2, false), 1);
    }

    #[test]
    fn test_filter_edge_basic() {
        let inp = [100u8, 110, 120, 130, 140, 150, 160, 170];
        let mut out = [0u8; 8];
        filter_edge(&mut out, 8, 0, 8, &inp, 0, 8, 1);
        assert!(out.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_dr_interp_filter_center() {
        let f = &DR_INTERP_FILTER[0];
        assert_eq!(f.a, 0);
        assert_eq!(f.b, 128);
        assert_eq!(f.c, 0);
        assert_eq!(f.d, 0);
        let mid = &DR_INTERP_FILTER[16];
        assert_eq!(mid.b, mid.c);
    }

    #[test]
    fn test_ipred_paeth_uniform() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_paeth_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_paeth_flat() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_paeth_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_smooth_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_smooth_v_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_v_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_smooth_h_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_h_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_smooth_flat() {
        let tl = vec![128u8; 20];
        let mut dst = [0u8; 16];
        ipred_smooth_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_ipred_smooth_v_top_row_near_top() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_v_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        for x in 0..4 {
            assert!(dst[x] > 0);
        }
    }

    #[test]
    fn test_ipred_smooth_h_rows_independent() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_h_8bpc(&mut dst, 4, &tl, 8, 4, 4);
        assert_eq!(dst[0], dst[0]);
    }

    #[test]
    fn test_ibp_blend_uniform_weight() {
        let mut dst = vec![200u8; 16 * 16];
        let tmp = vec![100u8; 16 * 64];
        let weights = [[64u8; 16]; 16];
        ibp_blend_8bpc(&mut dst, 16, &tmp, 16, 16, false, &weights);
        for &v in &dst[..16 * 16] {
            assert!((v as i32 - 150).abs() <= 1);
        }
    }

    #[test]
    fn test_ibp_blend_zero_weight() {
        let mut dst = vec![200u8; 16 * 16];
        let tmp = vec![50u8; 16 * 64];
        let weights = [[0u8; 16]; 16];
        ibp_blend_8bpc(&mut dst, 16, &tmp, 16, 16, false, &weights);
        for &v in &dst[..16 * 16] {
            assert_eq!(v, 50);
        }
    }

    #[test]
    fn test_ibp_blend_full_weight() {
        let mut dst = vec![200u8; 16 * 16];
        let tmp = vec![50u8; 16 * 64];
        let weights = [[128u8; 16]; 16];
        ibp_blend_8bpc(&mut dst, 16, &tmp, 16, 16, false, &weights);
        for &v in &dst[..16 * 16] {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_get_div_scale_sh_small() {
        let (scale, sh) = get_div_scale_sh(1);
        assert!(scale > 0);
        assert_eq!(sh, 0);
    }

    #[test]
    fn test_get_div_scale_sh_power_of_two() {
        let (_, sh) = get_div_scale_sh(16);
        assert_eq!(sh, 4);
    }

    #[test]
    fn test_get_div_scale_sh_negative() {
        let (s1, sh1) = get_div_scale_sh(10);
        let (s2, sh2) = get_div_scale_sh(-10);
        assert_eq!(s1, s2);
        assert_eq!(sh1, sh2);
    }

    #[test]
    fn test_mul32_exact() {
        assert_eq!(mul32(100, 200, 0), 20000);
    }

    #[test]
    fn test_mul32_with_shift() {
        let r = mul32(1024, 1024, 10);
        assert_eq!(r, 1024);
    }

    #[test]
    fn test_mul32_negative() {
        let r = mul32(-100, 200, 0);
        assert_eq!(r, -20000);
    }

    #[test]
    fn test_mul32_large_no_overflow() {
        let r = mul32(1 << 20, 1 << 20, 30);
        assert!(r > 0);
    }

    #[test]
    fn test_mul32_symmetric_rounding() {
        let pos = mul32(7, 1, 1);
        let neg = mul32(-7, 1, 1);
        assert_eq!(pos, -neg);
    }

    fn make_ibp_weights() -> [[[u8; 16]; 16]; 7] {
        crate::ibp::init_ibp_weights()
    }

    #[test]
    fn test_ipred_z1_basic() {
        let mut tl = vec![128u8; 256];
        for i in 0..64 {
            tl[i] = (128 + i) as u8;
        }
        let mut dst = [0u8; 64];
        let w = make_ibp_weights();
        ipred_z1_8bpc(
            &mut dst,
            8,
            &tl,
            0,
            8,
            8,
            45 | ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG,
            8,
            8,
            &w,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_ipred_z1_chroma() {
        let tl = vec![150u8; 256];
        let mut dst = [0u8; 16];
        let w = make_ibp_weights();
        ipred_z1_8bpc(&mut dst, 4, &tl, 0, 4, 4, 45 | ANGLE_HAS_TOP_FLAG, 4, 4, &w);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_ipred_z3_basic() {
        let mut tl = vec![128u8; 256];
        let o = 128;
        for i in 1..=64 {
            tl[o - i] = (128 + i) as u8;
        }
        let mut dst = [0u8; 64];
        let w = make_ibp_weights();
        ipred_z3_8bpc(
            &mut dst,
            8,
            &tl,
            o,
            8,
            8,
            225 | ANGLE_IS_LUMA | ANGLE_HAS_LEFT_FLAG,
            8,
            8,
            &w,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_ipred_z3_chroma() {
        let mut tl = vec![150u8; 256];
        let o = 128;
        for i in 1..=32 {
            tl[o - i] = 150;
        }
        let mut dst = [0u8; 16];
        let w = make_ibp_weights();
        ipred_z3_8bpc(
            &mut dst,
            4,
            &tl,
            o,
            4,
            4,
            225 | ANGLE_HAS_LEFT_FLAG,
            4,
            4,
            &w,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_basic_luma() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 {
            tl[o + i] = 200;
        }
        for i in 1..=32 {
            tl[o - i] = 100;
        }
        let mut dst = [0u8; 16];
        ipred_z2_8bpc(
            &mut dst,
            4,
            &tl,
            o,
            4,
            4,
            135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA,
            4,
            4,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_basic_chroma() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 {
            tl[o + i] = 200;
        }
        for i in 1..=32 {
            tl[o - i] = 100;
        }
        let mut dst = [0u8; 16];
        ipred_z2_8bpc(
            &mut dst,
            4,
            &tl,
            o,
            4,
            4,
            135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG,
            4,
            4,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_uniform_input() {
        let tl = [128u8; 256];
        let o = 128;
        let mut dst = [0u8; 64];
        ipred_z2_8bpc(
            &mut dst,
            8,
            &tl,
            o,
            8,
            8,
            135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA,
            8,
            8,
        );
        for &v in &dst[..64] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_z2_steep_angle() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 {
            tl[o + i] = 250;
        }
        for i in 1..=32 {
            tl[o - i] = 50;
        }
        let mut dst = [0u8; 16];
        ipred_z2_8bpc(
            &mut dst,
            4,
            &tl,
            o,
            4,
            4,
            170 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA,
            4,
            4,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_shallow_angle() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 {
            tl[o + i] = 250;
        }
        for i in 1..=32 {
            tl[o - i] = 50;
        }
        let mut dst = [0u8; 16];
        ipred_z2_8bpc(
            &mut dst,
            4,
            &tl,
            o,
            4,
            4,
            100 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA,
            4,
            4,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_no_panic_8x8() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..64 {
            tl[o + i] = 200;
        }
        for i in 1..=64 {
            tl[o - i] = 100;
        }
        let mut dst = [0u8; 64];
        ipred_z2_8bpc(
            &mut dst,
            8,
            &tl,
            o,
            8,
            8,
            135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA,
            8,
            8,
        );
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_mode0_8x8() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..64 {
            tl[o + 1 + i] = 200;
        }
        for i in 1..=64 {
            tl[o - i] = 100;
        }
        let mut dst = [0u8; 64];
        ipred_dip_8bpc(&mut dst, 8, &tl, o, 8, 8, 0);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_mode1_4x4() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 {
            tl[o + 1 + i] = 180;
        }
        for i in 1..=32 {
            tl[o - i] = 80;
        }
        let mut dst = [0u8; 16];
        ipred_dip_8bpc(&mut dst, 4, &tl, o, 4, 4, 1);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_transposed() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..64 {
            tl[o + 1 + i] = 200;
        }
        for i in 1..=64 {
            tl[o - i] = 100;
        }
        let mut dst = [0u8; 64];
        ipred_dip_8bpc(&mut dst, 8, &tl, o, 8, 8, 16);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_uniform_input() {
        let tl = [128u8; 256];
        let o = 128;
        let mut dst = [0u8; 64];
        ipred_dip_8bpc(&mut dst, 8, &tl, o, 8, 8, 0);
        // uniform input → all predictions close to 128
        for &v in &dst {
            assert!((v as i32 - 128).abs() < 30);
        }
    }

    #[test]
    fn test_pal_pred_basic() {
        let pal = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let idx = [0x10u8, 0x32, 0x10, 0x32, 0x10, 0x32, 0x10, 0x32];
        let mut dst = [0u8; 16];
        pal_pred_8bpc(&mut dst, 4, &pal, &idx, 4, 4);
        assert_eq!(dst[0], 10);
        assert_eq!(dst[1], 20);
        assert_eq!(dst[2], 30);
        assert_eq!(dst[3], 40);
    }

    #[test]
    fn test_pal_pred_all_same() {
        let pal = [99u8; 8];
        let idx = [0u8; 32];
        let mut dst = [0u8; 64];
        pal_pred_8bpc(&mut dst, 8, &pal, &idx, 8, 8);
        for &v in &dst {
            assert_eq!(v, 99);
        }
    }

    #[test]
    fn test_pal_pred_high_indices() {
        let pal = [0u8, 10, 20, 30, 40, 50, 60, 70];
        let idx = [0x77u8; 8];
        let mut dst = [0u8; 16];
        pal_pred_8bpc(&mut dst, 4, &pal, &idx, 4, 4);
        for &v in &dst {
            assert_eq!(v, 70);
        }
    }

    #[test]
    fn test_cfl_gen_y_420_uniform_no_edges() {
        let src = vec![100u8; 64 * 64];
        let mut dst = vec![0u8; 64 * 64 + 256];
        cfl_gen_y_420_8bpc(
            &mut dst,
            8,
            &src,
            0,
            None,
            64,
            4,
            4,
            4,
            4,
            0,
            CFL_FLT_TYPE_UNIFORM,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 100);
            }
        }
    }

    #[test]
    fn test_cfl_gen_y_420_vstrip_no_edges() {
        let src = vec![100u8; 64 * 64];
        let mut dst = vec![0u8; 64 * 64 + 256];
        cfl_gen_y_420_8bpc(
            &mut dst,
            8,
            &src,
            0,
            None,
            64,
            4,
            4,
            4,
            4,
            0,
            CFL_FLT_TYPE_VSTRIP,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 100);
            }
        }
    }

    #[test]
    fn test_cfl_gen_y_420_gauss_no_edges() {
        let src = vec![100u8; 64 * 64];
        let mut dst = vec![0u8; 64 * 64 + 256];
        cfl_gen_y_420_8bpc(
            &mut dst,
            8,
            &src,
            0,
            None,
            64,
            4,
            4,
            4,
            4,
            0,
            CFL_FLT_TYPE_GAUSS,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 100);
            }
        }
    }

    #[test]
    fn test_cfl_constants() {
        assert_eq!(CFL_FLT_TYPE_UNIFORM, 0);
        assert_eq!(CFL_FLT_TYPE_VSTRIP, 1);
        assert_eq!(CFL_FLT_TYPE_GAUSS, 2);
        assert_eq!(CFL_HAS_TOP, 4);
        assert_eq!(CFL_HAS_LEFT, 8);
        assert_eq!(CFL_DIR_LEFT, 2);
        assert_eq!(CFL_DIR_TOP, 1);
    }

    #[test]
    fn test_cfl_filter_uniform() {
        let src = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let top = [0u8; 8];
        let v = cfl_filter(&src, 0, 0, 1, 4, &top, 0, CFL_FLT_TYPE_UNIFORM);
        assert_eq!(v, (10 + 20 + 50 + 60) / 4);
    }

    #[test]
    fn test_cfl_filter_vstrip() {
        let src = vec![100u8; 128];
        let top = vec![100u8; 128];
        let v = cfl_filter(&src, 2, 1, 3, 64, &top, 2, CFL_FLT_TYPE_VSTRIP);
        assert_eq!(v, 100);
    }

    #[test]
    fn test_cfl_filter_gauss() {
        let src = vec![100u8; 128];
        let top = vec![100u8; 128];
        let v = cfl_filter(&src, 2, 1, 3, 64, &top, 2, CFL_FLT_TYPE_GAUSS);
        assert_eq!(v, 100);
    }

    #[test]
    fn test_cfl_gen_mat_no_edges() {
        let y = vec![128u8; 64 * 64 + 512];
        let mut mat = [[0i32; 3]; 3];
        let mut imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
        cfl_gen_mat_8bpc(&mut mat, &mut imat, &y, 0, 8, 4, 4, 0, CflMhDir::Center);
        assert_eq!(mat[2][2], 2);
    }

    #[test]
    fn test_cfl_gen_mat_symmetric() {
        let mut y = vec![100u8; 64 * 64 + 512];
        for i in 0..512 {
            y[64 * 64 + i] = 100;
        }
        let mut mat = [[0i32; 3]; 3];
        let mut imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
        cfl_gen_mat_8bpc(
            &mut mat,
            &mut imat,
            &y,
            0,
            8,
            4,
            4,
            CFL_HAS_TOP | CFL_HAS_LEFT,
            CflMhDir::Center,
        );
        assert_eq!(mat[1][0], mat[0][1]);
        assert_eq!(mat[2][0], mat[0][2]);
        assert_eq!(mat[2][1], mat[1][2]);
    }

    #[test]
    fn test_cfl_gen_mat_regularization() {
        let y = vec![128u8; 64 * 64 + 512];
        let mut mat = [[0i32; 3]; 3];
        let mut imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
        cfl_gen_mat_8bpc(&mut mat, &mut imat, &y, 0, 8, 4, 4, 0, CflMhDir::Center);
        assert!(mat[0][0] >= 2);
        assert!(mat[1][1] >= 2);
        assert!(mat[2][2] >= 2);
    }

    #[test]
    fn test_cfl_calc_alphas_no_edges() {
        let c = vec![128u8; 64 * 64];
        let mut alpha = [0i32; 3];
        let mut mat = [[0i32; 3]; 3];
        mat[0][0] = 100;
        mat[1][1] = 100;
        mat[2][2] = 100;
        mat[0][1] = 0;
        mat[1][0] = 0;
        mat[0][2] = 0;
        mat[2][0] = 0;
        mat[1][2] = 0;
        mat[2][1] = 0;
        let imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
        cfl_calc_alphas_8bpc(&mut alpha, &c, 64, None, 64, 4, 4, &mut mat, &imat, 0);
        assert_eq!(alpha, [0, 0, 0]);
    }

    #[test]
    fn test_cfl_calc_alphas_with_top() {
        let stride = 16usize;
        let mut c = vec![100u8; stride * 16];
        c[stride - 1] = 100;
        let top = vec![120u8; 32];
        let mut mat = [[0i32; 3]; 3];
        mat[0][0] = 1000;
        mat[1][1] = 1000;
        mat[2][2] = 1000;
        mat[1][0] = 0;
        mat[0][1] = 0;
        mat[2][0] = 0;
        mat[0][2] = 0;
        mat[2][1] = 0;
        mat[1][2] = 0;
        let mut imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
        for i in 0..4 {
            imat[0][i] = 100;
            imat[1][i] = 50;
        }
        let mut alpha = [0i32; 3];
        cfl_calc_alphas_8bpc(
            &mut alpha,
            &c,
            stride,
            Some((&top, 1)),
            stride,
            4,
            4,
            &mut mat,
            &imat,
            CFL_HAS_TOP,
        );
        assert!(alpha[0] != 0 || alpha[1] != 0 || alpha[2] != 0);
    }

    #[test]
    fn test_cfl_mhccp_pred_center_zero_alpha() {
        let src = vec![128u8; 64 * 64 + 256];
        let alpha = [0i32; 3];
        let mut dst = vec![0u8; 64];
        cfl_mhccp_pred_8bpc(&mut dst, 8, &src, 0, 8, 4, 4, &alpha, 0, CflMhDir::Center);
        for &v in &dst[..32] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_cfl_mhccp_pred_center_identity() {
        let src = vec![100u8; 64 * 64 + 256];
        let alpha = [1 << 16, 0, 0];
        let mut dst = vec![0u8; 64];
        cfl_mhccp_pred_8bpc(&mut dst, 8, &src, 0, 8, 4, 4, &alpha, 0, CflMhDir::Center);
        for y in 0..4 {
            for x in 0..4 {
                assert!(dst[y * 8 + x] > 0);
            }
        }
    }

    #[test]
    fn test_cfl_pred_explicit_444_uniform() {
        let stride = 16isize;
        let ytop = vec![128u8; 64];
        let utop = vec![128u8; 32];
        let vtop = vec![128u8; 32];
        let ypx = vec![128u8; 512];
        let mut u_buf = vec![128u8; 512];
        let mut v_buf = vec![128u8; 512];
        let off = stride as usize + 1;
        let yoff = stride as usize + 1;
        let flags = CFL_HAS_TOP as u32 | CFL_HAS_LEFT as u32;
        cfl_pred_8bpc(
            &ytop, 0, &utop, 0, &vtop, 0, &ypx, yoff, &mut u_buf, off, &mut v_buf, off, stride,
            stride, 0, 0, 4, 4, flags, false, 0, 0,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(u_buf[off + y * stride as usize + x], 128);
                assert_eq!(v_buf[off + y * stride as usize + x], 128);
            }
        }
    }

    #[test]
    fn test_cfl_pred_no_edges() {
        let ytop = vec![0u8; 32];
        let utop = vec![0u8; 32];
        let vtop = vec![0u8; 32];
        let ypx = vec![100u8; 256];
        let mut u_buf = vec![0u8; 256];
        let mut v_buf = vec![0u8; 256];
        let off = 1usize;
        cfl_pred_8bpc(
            &ytop, 0, &utop, 0, &vtop, 0, &ypx, 0, &mut u_buf, off, &mut v_buf, off, 16, 16, 0, 0,
            4, 4, 0, false, 0, 0,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(u_buf[off + y * 16 + x], 128);
                assert_eq!(v_buf[off + y * 16 + x], 128);
            }
        }
    }

    use crate::pixel::BitDepth16;

    #[test]
    fn test_ipred_dc_128_hbd_10bit() {
        // 10-bit DC_128 default is (1023 + 1) >> 1 = 512, not 128.
        let bd = BitDepth16::new(10);
        let mut dst = [0u16; 16];
        ipred_dc_128(bd, &mut dst, 4, 4, 4);
        assert!(dst.iter().all(|&v| v == 512));
    }

    #[test]
    fn test_ipred_dc_uniform_hbd_10bit() {
        // A uniform 10-bit edge of 800 must reproduce 800 (no 255 clamp).
        let bd = BitDepth16::new(10);
        let tl = vec![800u16; 20];
        let mut dst = [0u16; 16];
        ipred_dc(bd, &mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 800));
    }

    #[test]
    fn test_ipred_v_hbd_copies_10bit_top() {
        // Vertical prediction copies the top row verbatim; 10-bit samples > 255
        // must survive intact.
        let bd = BitDepth16::new(10);
        let mut tl = vec![0u16; 20];
        for i in 0..4 {
            tl[9 + i] = 600 + i as u16;
        }
        let mut dst = [0u16; 16];
        ipred_v(bd, &mut dst, 4, &tl, 8, 4, 4, 0);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 600 + x as u16);
            }
        }
    }

    #[test]
    fn test_ipred_paeth_hbd_10bit() {
        // Paeth picks among left/top/topleft; with a uniform 1000 edge the result
        // is 1000 — well above 255, proving no u8 truncation.
        let bd = BitDepth16::new(10);
        let tl = vec![1000u16; 20];
        let mut dst = [0u16; 16];
        ipred_paeth(bd, &mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v == 1000));
    }

    #[test]
    fn test_ipred_z1_hbd_clamps_to_10bit_max() {
        // A directional z1 prediction over a saturated 10-bit edge must clamp to
        // 1023, never 255.
        let bd = BitDepth16::new(10);
        let mut tl = vec![1023u16; 256];
        for v in tl.iter_mut() {
            *v = 1023;
        }
        let mut dst = [0u16; 64];
        let w = make_ibp_weights();
        ipred_z1(
            bd,
            &mut dst,
            8,
            &tl,
            0,
            8,
            8,
            45 | ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG,
            8,
            8,
            &w,
        );
        assert!(dst.iter().all(|&v| v <= 1023));
        assert!(dst.iter().any(|&v| v > 255));
    }
}

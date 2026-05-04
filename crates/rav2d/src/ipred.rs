use crate::intops::{ctz, iclip, imax, imin, ulog2};
use crate::levels::{
    ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, ANGLE_IS_LUMA,
    ANGLE_MRL_IDX_MASK, ANGLE_MRL_IDX_SHIFT, ANGLE_MULTI_MRL_FLAG,
    ANGLE_SMOOTH_LEFT_EDGE_FLAG, ANGLE_SMOOTH_TOP_EDGE_FLAG, ANGLE_USE_EDGE_FILTER_FLAG,
};
use crate::dip_tables::DIP_WEIGHTS;
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
    DrFilter4Tap { a:   0, b: 128, c:   0, d:   0 },
    DrFilter4Tap { a:  -2, b: 127, c:   4, d:  -1 },
    DrFilter4Tap { a:  -3, b: 125, c:   8, d:  -2 },
    DrFilter4Tap { a:  -5, b: 123, c:  13, d:  -3 },
    DrFilter4Tap { a:  -6, b: 121, c:  17, d:  -4 },
    DrFilter4Tap { a:  -7, b: 118, c:  22, d:  -5 },
    DrFilter4Tap { a:  -9, b: 116, c:  27, d:  -6 },
    DrFilter4Tap { a:  -9, b: 112, c:  32, d:  -7 },
    DrFilter4Tap { a: -10, b: 109, c:  37, d:  -8 },
    DrFilter4Tap { a: -11, b: 106, c:  41, d:  -8 },
    DrFilter4Tap { a: -11, b: 102, c:  46, d:  -9 },
    DrFilter4Tap { a: -12, b:  98, c:  52, d: -10 },
    DrFilter4Tap { a: -12, b:  94, c:  56, d: -10 },
    DrFilter4Tap { a: -12, b:  90, c:  61, d: -11 },
    DrFilter4Tap { a: -12, b:  85, c:  66, d: -11 },
    DrFilter4Tap { a: -12, b:  81, c:  71, d: -12 },
    DrFilter4Tap { a: -12, b:  76, c:  76, d: -12 },
    DrFilter4Tap { a: -12, b:  71, c:  81, d: -12 },
    DrFilter4Tap { a: -11, b:  66, c:  85, d: -12 },
    DrFilter4Tap { a: -11, b:  61, c:  90, d: -12 },
    DrFilter4Tap { a: -10, b:  56, c:  94, d: -12 },
    DrFilter4Tap { a: -10, b:  52, c:  98, d: -12 },
    DrFilter4Tap { a:  -9, b:  46, c: 102, d: -11 },
    DrFilter4Tap { a:  -8, b:  41, c: 106, d: -11 },
    DrFilter4Tap { a:  -8, b:  37, c: 109, d: -10 },
    DrFilter4Tap { a:  -7, b:  32, c: 112, d:  -9 },
    DrFilter4Tap { a:  -6, b:  27, c: 116, d:  -9 },
    DrFilter4Tap { a:  -5, b:  22, c: 118, d:  -7 },
    DrFilter4Tap { a:  -4, b:  17, c: 121, d:  -6 },
    DrFilter4Tap { a:  -3, b:  13, c: 123, d:  -5 },
    DrFilter4Tap { a:  -2, b:   8, c: 125, d:  -3 },
    DrFilter4Tap { a:  -1, b:   4, c: 127, d:  -2 },
];

pub fn get_filter_strength(wh: i32, angle: i32, is_sm: bool) -> i32 {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 { return 2; }
            if angle >= 40 { return 1; }
        } else if wh <= 16 {
            if angle >= 48 { return 2; }
            if angle >= 20 { return 1; }
        } else if wh <= 24 {
            if angle >= 4 { return 3; }
        } else {
            return 3;
        }
    } else {
        if wh <= 8 {
            if angle >= 56 { return 1; }
        } else if wh <= 16 {
            if angle >= 40 { return 1; }
        } else if wh <= 24 {
            if angle >= 32 { return 3; }
            if angle >= 16 { return 2; }
            if angle >= 8 { return 1; }
        } else if wh <= 32 {
            if angle >= 32 { return 3; }
            if angle >= 4 { return 2; }
            return 1;
        } else {
            return 3;
        }
    }
    0
}

pub fn filter_edge(
    out: &mut [u8],
    sz: usize,
    lim_from: usize,
    lim_to: usize,
    inp: &[u8],
    from: i32,
    to: i32,
    strength: usize,
) {
    static KERNEL: [[u8; 5]; 3] = [
        [0, 4, 8, 4, 0],
        [0, 5, 6, 5, 0],
        [2, 4, 4, 4, 2],
    ];

    debug_assert!(strength > 0);
    let mut i = 0;
    while i < imin(sz as i32, lim_from as i32) as usize {
        out[i] = inp[iclip(i as i32, from, to - 1) as usize];
        i += 1;
    }
    while i < imin(lim_to as i32, sz as i32) as usize {
        let mut s = 0i32;
        for j in 0..5 {
            s += inp[iclip(i as i32 - 2 + j, from, to - 1) as usize] as i32
                * KERNEL[strength - 1][j as usize] as i32;
        }
        out[i] = ((s + 8) >> 4) as u8;
        i += 1;
    }
    while i < sz {
        out[i] = inp[iclip(i as i32, from, to - 1) as usize];
        i += 1;
    }
}

fn splat_dc(dst: &mut [u8], stride: usize, off: usize, width: usize, mut height: usize, dc: u8) {
    let mut p = off;
    while height > 0 {
        for x in 0..width {
            dst[p + x] = dc;
        }
        p += stride;
        height -= 1;
    }
}

fn dc_gen_top(tl: &[u8], o: usize, width: usize) -> u32 {
    let mut dc = (width >> 1) as u32;
    for i in 0..width {
        dc += tl[o + 1 + i] as u32;
    }
    dc >> ctz(width as u32)
}

fn dc_gen_left(tl: &[u8], o: usize, height: usize) -> u32 {
    let mut dc = (height >> 1) as u32;
    for i in 0..height {
        dc += tl[o - 1 - i] as u32;
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

fn dc_gen(tl: &[u8], o: usize, width: usize, height: usize) -> u32 {
    let n_pel = width + height;
    let mut dc = 0u32;
    for i in 0..width {
        dc += tl[o + 1 + i] as u32;
    }
    for i in 0..height {
        dc += tl[o - 1 - i] as u32;
    }
    if n_pel & (n_pel - 1) == 0 {
        return (dc + width as u32) >> ctz(n_pel as u32);
    }
    (fast_div32_dc(dc, n_pel as u32)).min(255)
}

pub fn ipred_dc_128(dst: &mut [u8], stride: usize, width: usize, height: usize) {
    splat_dc(dst, stride, 0, width, height, 128);
}

pub fn ipred_dc_top(
    dst: &mut [u8], stride: usize, tl: &[u8], o: usize,
    width: usize, mut height: usize, angle: i32,
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
                dst[off + x] = ((tl[o + 1 + x] as u32 * wy + dc_wy + 64) >> 7) as u8;
            }
            off += stride;
        }
        height -= h;
    }

    splat_dc(dst, stride, off, width, height, dc as u8);
}

pub fn ipred_dc_left(
    dst: &mut [u8], stride: usize, tl: &[u8], o: usize,
    mut width: usize, height: usize, angle: i32,
) {
    let dc = dc_gen_left(tl, o, height);
    let mut off = 0;
    let mut x_off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let w = width >> 2;
        let w_x = &DC_IBP_WEIGHTS[w..];
        for y in 0..height {
            let left = tl[o - 1 - y] as u32;
            for x in 0..w {
                dst[off + x] = ((left * (128 - w_x[x] as u32) + dc * w_x[x] as u32 + 64) >> 7) as u8;
            }
            off += stride;
        }
        off = 0;
        x_off = w;
        width -= w;
    }

    let mut p = off;
    for _ in 0..height {
        for x in 0..width {
            dst[p + x_off + x] = dc as u8;
        }
        p += stride;
    }
}

pub fn ipred_dc(
    dst: &mut [u8], stride: usize, tl: &[u8], o: usize,
    mut width: usize, mut height: usize, angle: i32,
) {
    let dc = dc_gen(tl, o, width, height);
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
                dst[off + x] = ((tl[o + 1 + x] as u32 * wy + dc_wy + 64) >> 7) as u8;
            }
            off += stride;
        }

        let y_start = if width >= height { h } else { 0 };
        off = y_start * stride;
        let w_x = &DC_IBP_WEIGHTS[w..];
        for y in y_start..height {
            let left = tl[o - 1 - y] as u32;
            for x in 0..w {
                dst[off + x] = ((left * (128 - w_x[x] as u32) + dc * w_x[x] as u32 + 64) >> 7) as u8;
            }
            off += stride;
        }
        off = h * stride + w;
        x_off = 0;
        width -= w;
        height -= h;
    }

    let mut p = off;
    for _ in 0..height {
        for x in 0..width {
            dst[p + x_off + x] = dc as u8;
        }
        p += stride;
    }
}

pub fn ipred_v(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, width: usize, height: usize) {
    let mut off = 0;
    for _ in 0..height {
        dst[off..off + width].copy_from_slice(&tl[o + 1..o + 1 + width]);
        off += stride;
    }
}

pub fn ipred_h(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, width: usize, height: usize) {
    let mut off = 0;
    for y in 0..height {
        let v = tl[o - 1 - y];
        for x in 0..width {
            dst[off + x] = v;
        }
        off += stride;
    }
}

pub fn ipred_paeth(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    let topleft = tl[o] as i32;
    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        for x in 0..w {
            let top = tl[o + 1 + x] as i32;
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();
            dst[off + x] = if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            } as u8;
        }
        off += stride;
    }
}

pub fn ipred_smooth(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    let bwl2 = ulog2(w as u32);
    let bhl2 = ulog2(h as u32);
    let rnd_ver = (h >> 1) as i32;
    let rnd_hor = (w >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i32;
    let bottom = tl[o - h - 1] as i32;

    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let diff_hor = left - right;
        let off_ver = (h as i32 - 1 - y as i32) as i32;
        let w_ver = weights[y] as i32;
        for x in 0..w {
            let above = tl[o + 1 + x] as i32;
            let mul_ver = (above - bottom) * off_ver;
            let mul_hor = diff_hor * (w as i32 - 1 - x as i32);
            let mut pred_ver = bottom + ((mul_ver + rnd_ver) >> bhl2);
            let mut pred_hor = right + ((mul_hor + rnd_hor) >> bwl2);
            pred_ver += ((above - pred_ver) * w_ver + 32) >> 6;
            pred_hor += ((left - pred_hor) * weights[x] as i32 + 32) >> 6;
            dst[off + x] = ((pred_ver + pred_hor + 1) >> 1) as u8;
        }
        off += stride;
    }
}

pub fn ipred_smooth_v(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    let bhl2 = ulog2(h as u32);
    let rnd = (h >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom = tl[o - h - 1] as i32;

    let mut off = 0;
    for y in 0..h {
        let off_y = h as i32 - 1 - y as i32;
        let w_ver = weights[y] as i32;
        for x in 0..w {
            let above = tl[o + 1 + x] as i32;
            let mul = (above - bottom) * off_y;
            let pred = bottom + ((mul + rnd) >> bhl2);
            dst[off + x] = (pred + (((above - pred) * w_ver + 32) >> 6)) as u8;
        }
        off += stride;
    }
}

pub fn ipred_smooth_h(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    let bwl2 = ulog2(w as u32);
    let rnd = (w >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right_val = tl[o + w + 1] as i32;

    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let diff = left - right_val;
        for x in 0..w {
            let mul = diff * (w as i32 - 1 - x as i32);
            let pred = right_val + ((mul + rnd) >> bwl2);
            dst[off + x] = (pred + (((left - pred) * weights[x] as i32 + 32) >> 6)) as u8;
        }
        off += stride;
    }
}

pub fn ipred_z1(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
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
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z1(
            &mut tmp, 64, tl, o, width, height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width, _max_height, ibp_weights,
        );
        ipred_z1(
            dst, stride, tl, o + e_stride, width, height,
            angle | ANGLE_IS_LUMA, max_width, _max_height, ibp_weights,
        );
        for y in 0..height {
            for x in 0..width {
                dst[y * stride + x] = ((tmp[y * 64 + x] as u16 + dst[y * stride + x] as u16 + 1) >> 1) as u8;
            }
        }
        return;
    }

    let dx = DR_INTRA_DERIVATIVE[angle as usize] as i32;
    let max_base_x = (width + height) as i32 - 1 + (mrl_idx as i32 * 2);

    let mut filt = vec![0u8; 136];
    let top_off = 2 + mrl_idx;
    let sz = 1 + mrl_idx + width + height + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        get_filter_strength((width + height) as i32, 90 - angle, is_sm_t)
    } else {
        0
    };
    if str > 0 {
        filter_edge(
            &mut filt[1..], sz, 1, sz + max_width as usize - width,
            &tl[o..], 0, sz as i32, str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    filt[sz + 2] = filt[sz + 1];
    if sz + 1 < filt.len() {
        filt[sz + 1] = filt[sz];
    }

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
                let v = f.a as i32 * filt[(bi - 1) as usize] as i32
                    + f.b as i32 * filt[bi as usize] as i32
                    + f.c as i32 * filt[(bi + 1) as usize] as i32
                    + f.d as i32 * filt[(bi + 2) as usize] as i32;
                dst[y * stride + x] = iclip((v + 64) >> 7, 0, 255) as u8;
            } else {
                let v = (32 - shift as i32) * filt[bi as usize] as i32
                    + shift as i32 * filt[(bi + 1) as usize] as i32;
                dst[y * stride + x] = iclip((v + 16) >> 5, 0, 255) as u8;
            }
            base += 1;
        }
        ypos += dx;
    }

    if enable_ibp {
        let mode_idx = imin(10 - (angle >> 3), 6) as usize;
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z3(
            &mut tmp, 64, tl, o, width, height,
            (180 + angle) | angle_flags, max_width, _max_height, ibp_weights,
        );
        ibp_blend_8bpc(dst, stride, &tmp, width, height, false, &ibp_weights[mode_idx]);
    }
}

pub fn ipred_z3(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
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
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z3(
            &mut tmp, 64, tl, o, width, height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width, max_height, ibp_weights,
        );
        ipred_z3(
            dst, stride, tl, o + e_stride, width, height,
            angle | ANGLE_IS_LUMA, max_width, max_height, ibp_weights,
        );
        for y in 0..height {
            for x in 0..width {
                dst[y * stride + x] = ((tmp[y * 64 + x] as u16 + dst[y * stride + x] as u16 + 1) >> 1) as u8;
            }
        }
        return;
    }

    let dy = DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32;
    let max_base_y = (width + height) as i32 - 1 + (mrl_idx as i32 * 2);

    let mut filt = vec![0u8; 136];
    let left_off = 1 + width + height + mrl_idx * 2;
    let sz = 1 + mrl_idx + width + height + mrl_idx * 2;

    let str = if enable_intra_edge_filter && mrl_idx == 0 && have_left {
        get_filter_strength((width + height) as i32, angle - 180, is_sm_l)
    } else {
        0
    };

    if str > 0 {
        filter_edge(
            &mut filt[2..], sz, height - max_height as usize, sz - 1,
            &tl[o + 1 - sz..], 0, sz as i32, str as usize,
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
                    let v = f.a as i32 * filt[(bi + 1) as usize] as i32
                        + f.b as i32 * filt[bi as usize] as i32
                        + f.c as i32 * filt[(bi - 1) as usize] as i32
                        + f.d as i32 * filt[(bi - 2) as usize] as i32;
                    dst[y * stride + x] = iclip((v + 64) >> 7, 0, 255) as u8;
                } else {
                    let v = (32 - shift as i32) * filt[bi as usize] as i32
                        + shift as i32 * filt[(bi - 1) as usize] as i32;
                    dst[y * stride + x] = iclip((v + 16) >> 5, 0, 255) as u8;
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
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z1(
            &mut tmp, 64, tl, o, width, height,
            (angle - 180) | angle_flags, max_width, max_height, ibp_weights,
        );
        ibp_blend_8bpc(dst, stride, &tmp, width, height, true, &ibp_weights[mode_idx]);
    }
}

pub fn ipred_z2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
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
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z2(
            &mut tmp, 64, tl, o, width, height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width, max_height,
        );
        ipred_z2(
            dst, stride, tl, o + e_stride, width, height,
            angle | ANGLE_IS_LUMA,
            max_width, max_height,
        );
        for y in 0..height {
            for x in 0..width {
                dst[y * stride + x] = ((tmp[y * 64 + x] as u16 + dst[y * stride + x] as u16 + 1) >> 1) as u8;
            }
        }
        return;
    }

    let dy = DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32;
    let dx = DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32;

    // Top filter buffer
    let mut filt = vec![0u8; 72];
    let top_off = mrl_idx;
    let sz_t = 1 + width + mrl_idx;
    let str_t = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        get_filter_strength((width + height) as i32, angle - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        filter_edge(
            &mut filt[1..], sz_t, 1, sz_t + max_width as usize - width,
            &tl[o..], 0, sz_t as i32, str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&tl[o..o + sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // Left filter buffer
    let mut filt2 = vec![0u8; 72];
    let left_off: usize = height + 2;
    let sz_l = 1 + height + mrl_idx;
    let str_l = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        get_filter_strength((width + height) as i32, 180 - angle, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        filter_edge(
            &mut filt2[1..], sz_l, height - max_height as usize, sz_l - 1,
            &tl[o - (height + mrl_idx)..], 0, sz_l as i32, str_l as usize,
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
                let v = f.a as i32 * filt2[bi - 1] as i32
                    + f.b as i32 * filt2[bi - 2] as i32
                    + f.c as i32 * filt2[bi - 3] as i32
                    + f.d as i32 * filt2[bi - 4] as i32;
                dst[y * stride + x] = iclip((v + 64) >> 7, 0, 255) as u8;
            } else {
                let v = (32 - shift as i32) * filt2[bi - 2] as i32
                    + shift as i32 * filt2[bi - 3] as i32;
                dst[y * stride + x] = iclip((v + 16) >> 5, 0, 255) as u8;
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
                let v = f.a as i32 * filt[(ti + 1) as usize] as i32
                    + f.b as i32 * filt[(ti + 2) as usize] as i32
                    + f.c as i32 * filt[(ti + 3) as usize] as i32
                    + f.d as i32 * filt[(ti + 4) as usize] as i32;
                dst[y * stride + x] = iclip((v + 64) >> 7, 0, 255) as u8;
            } else {
                let v = (32 - shift as i32) * filt[(ti + 2) as usize] as i32
                    + shift as i32 * filt[(ti + 3) as usize] as i32;
                dst[y * stride + x] = iclip((v + 16) >> 5, 0, 255) as u8;
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
    let x_shift = width >> (4 + 1);
    let y_shift = height >> (4 + 1);

    for y in 0..height {
        let wy = y >> y_shift;
        for x in 0..width {
            let wx = x >> x_shift;
            let weight = if inv { weights[wx][wy] } else { weights[wy][wx] } as i32;
            dst[y * stride + x] =
                ((tmp[y * 64 + x] as i32 * (128 - weight) + dst[y * stride + x] as i32 * weight
                    + 64)
                    >> 7) as u8;
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
    inp[0] = tl[o] as i32;
    let mut in_sum = inp[0];

    let mut ti = o + 1;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..wd {
            sum += tl[ti] as i32;
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
            sum += tl[li] as i32;
        }
        inp[i_l + i] = (sum + hrnd as i32) >> hl2;
        in_sum += inp[i_l + i];
    }

    let mut sum = 0i32;
    for x in 0..wd {
        sum += tl[o + x + width + 1] as i32;
    }
    let idx_tr = if trans { 10 } else { 9 };
    inp[idx_tr] = (sum + wrnd as i32) >> wl2;
    in_sum += inp[idx_tr];

    sum = 0;
    for y in 0..hd {
        sum += tl[o - (y + height + 1)] as i32;
    }
    let idx_bl = if trans { 9 } else { 10 };
    inp[idx_bl] = (sum + hrnd as i32) >> hl2;
    in_sum += inp[idx_bl];

    let m = (mode & 7) as usize;

    let mut uwl2 = wl2 as i32 - 1;
    let mut dwl2 = 0i32;
    if uwl2 < 0 { dwl2 = -uwl2; uwl2 = 0; }
    let step_x = 1usize << uwl2;
    let dw = 1usize << dwl2;
    let mut uhl2 = hl2 as i32 - 1;
    let mut dhl2 = 0i32;
    if uhl2 < 0 { dhl2 = -uhl2; uhl2 = 0; }
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
            dst[y * stride + x] = iclip(((s + 2048) >> 12) - in_sum, 0, 255) as u8;
            x += step_x;
        }
        y += step_y;
    }

    if step_x > 1 {
        y = step_y - 1;
        for _gy in 0..grid_h {
            let mut p1 = tl[o - (y + 1)] as i32;
            let mut x = 0usize;
            for _gx in 0..grid_w {
                let p0 = p1;
                p1 = dst[y * stride + x + step_x - 1] as i32;
                for z in 0..step_x - 1 {
                    let z1 = (z + 1) as i32;
                    dst[y * stride + x + z] =
                        ((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2) as u8;
                }
                x += step_x;
            }
            y += step_y;
        }
    }

    if step_y > 1 {
        for x in 0..width {
            let mut p1 = tl[o + x + 1] as i32;
            y = 0;
            for _gy in 0..grid_h {
                let p0 = p1;
                p1 = dst[(y + step_y - 1) * stride + x] as i32;
                for z in 0..step_y - 1 {
                    let z1 = (z + 1) as i32;
                    dst[(y + z) * stride + x] =
                        ((p0 * (step_y as i32 - z1) + p1 * z1) >> uhl2) as u8;
                }
                y += step_y;
            }
        }
    }
}

pub fn pal_pred_8bpc(
    dst: &mut [u8],
    stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
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
        ipred_dc_128(&mut dst, 4, 4, 4);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_ipred_dc_top_uniform() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_dc_top(&mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_dc_left_uniform() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_dc_left(&mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_dc_uniform() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_dc(&mut dst, 4, &tl, 8, 4, 4, 0);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_v_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_v(&mut dst, 4, &tl, 8, 4, 4);
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
        ipred_h(&mut dst, 4, &tl, 8, 4, 4);
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
        ipred_paeth(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_paeth_flat() {
        let tl = vec![100u8; 20];
        let mut dst = [0u8; 16];
        ipred_paeth(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_ipred_smooth_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_smooth_v_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_v(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_smooth_h_basic() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_h(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_ipred_smooth_flat() {
        let tl = vec![128u8; 20];
        let mut dst = [0u8; 16];
        ipred_smooth(&mut dst, 4, &tl, 8, 4, 4);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_ipred_smooth_v_top_row_near_top() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_v(&mut dst, 4, &tl, 8, 4, 4);
        for x in 0..4 {
            assert!(dst[x] > 0);
        }
    }

    #[test]
    fn test_ipred_smooth_h_rows_independent() {
        let tl = make_tl_buf(4, 4, 8);
        let mut dst = [0u8; 16];
        ipred_smooth_h(&mut dst, 4, &tl, 8, 4, 4);
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
        for i in 0..64 { tl[i] = (128 + i) as u8; }
        let mut dst = [0u8; 64];
        let w = make_ibp_weights();
        ipred_z1(&mut dst, 8, &tl, 0, 8, 8,
                 45 | ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG, 8, 8, &w);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_ipred_z1_chroma() {
        let tl = vec![150u8; 256];
        let mut dst = [0u8; 16];
        let w = make_ibp_weights();
        ipred_z1(&mut dst, 4, &tl, 0, 4, 4,
                 45 | ANGLE_HAS_TOP_FLAG, 4, 4, &w);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_ipred_z3_basic() {
        let mut tl = vec![128u8; 256];
        let o = 128;
        for i in 1..=64 { tl[o - i] = (128 + i) as u8; }
        let mut dst = [0u8; 64];
        let w = make_ibp_weights();
        ipred_z3(&mut dst, 8, &tl, o, 8, 8,
                 225 | ANGLE_IS_LUMA | ANGLE_HAS_LEFT_FLAG, 8, 8, &w);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_ipred_z3_chroma() {
        let mut tl = vec![150u8; 256];
        let o = 128;
        for i in 1..=32 { tl[o - i] = 150; }
        let mut dst = [0u8; 16];
        let w = make_ibp_weights();
        ipred_z3(&mut dst, 4, &tl, o, 4, 4,
                 225 | ANGLE_HAS_LEFT_FLAG, 4, 4, &w);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_basic_luma() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 { tl[o + i] = 200; }
        for i in 1..=32 { tl[o - i] = 100; }
        let mut dst = [0u8; 16];
        ipred_z2(&mut dst, 4, &tl, o, 4, 4,
                 135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA, 4, 4);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_basic_chroma() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 { tl[o + i] = 200; }
        for i in 1..=32 { tl[o - i] = 100; }
        let mut dst = [0u8; 16];
        ipred_z2(&mut dst, 4, &tl, o, 4, 4,
                 135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG, 4, 4);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_uniform_input() {
        let tl = [128u8; 256];
        let o = 128;
        let mut dst = [0u8; 64];
        ipred_z2(&mut dst, 8, &tl, o, 8, 8,
                 135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA, 8, 8);
        for &v in &dst[..64] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_z2_steep_angle() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 { tl[o + i] = 250; }
        for i in 1..=32 { tl[o - i] = 50; }
        let mut dst = [0u8; 16];
        ipred_z2(&mut dst, 4, &tl, o, 4, 4,
                 170 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA, 4, 4);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_shallow_angle() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 { tl[o + i] = 250; }
        for i in 1..=32 { tl[o - i] = 50; }
        let mut dst = [0u8; 16];
        ipred_z2(&mut dst, 4, &tl, o, 4, 4,
                 100 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA, 4, 4);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_z2_no_panic_8x8() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..64 { tl[o + i] = 200; }
        for i in 1..=64 { tl[o - i] = 100; }
        let mut dst = [0u8; 64];
        ipred_z2(&mut dst, 8, &tl, o, 8, 8,
                 135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_IS_LUMA, 8, 8);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_mode0_8x8() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..64 { tl[o + 1 + i] = 200; }
        for i in 1..=64 { tl[o - i] = 100; }
        let mut dst = [0u8; 64];
        ipred_dip_8bpc(&mut dst, 8, &tl, o, 8, 8, 0);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_mode1_4x4() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..32 { tl[o + 1 + i] = 180; }
        for i in 1..=32 { tl[o - i] = 80; }
        let mut dst = [0u8; 16];
        ipred_dip_8bpc(&mut dst, 4, &tl, o, 4, 4, 1);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_dip_transposed() {
        let mut tl = [128u8; 256];
        let o = 128;
        for i in 0..64 { tl[o + 1 + i] = 200; }
        for i in 1..=64 { tl[o - i] = 100; }
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
}

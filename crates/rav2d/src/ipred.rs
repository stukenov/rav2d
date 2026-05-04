use crate::intops::{ctz, iclip, imin, ulog2};
use crate::levels::{ANGLE_IBP_FLAG, ANGLE_MULTI_MRL_FLAG};
use crate::tables::{DC_IBP_WEIGHTS, DIV_RECIP, SM_WEIGHTS};

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
}

use crate::intops::iclip;
use crate::tables::MC_WARP_FILTER;

pub const INTERMEDIATE_BITS_8BPC: i32 = 4;
pub const PREP_BIAS_8BPC: i32 = 0;

pub fn put_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        dst[y * dst_stride..y * dst_stride + w].copy_from_slice(&src[y * src_stride..y * src_stride + w]);
    }
}

pub fn prep_8bpc(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = ((src[y * src_stride + x] as i32) << INTERMEDIATE_BITS_8BPC) as i16;
        }
    }
}

pub fn avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    let sh = INTERMEDIATE_BITS_8BPC + 1;
    let rnd = 1 << INTERMEDIATE_BITS_8BPC;
    for y in 0..h {
        for x in 0..w {
            let ti = y * w + x;
            dst[y * dst_stride + x] =
                iclip((tmp1[ti] as i32 + tmp2[ti] as i32 + rnd) >> sh, 0, 255) as u8;
        }
    }
}

pub fn w_avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    let sh = INTERMEDIATE_BITS_8BPC + 4;
    let rnd = 8 << INTERMEDIATE_BITS_8BPC;
    for y in 0..h {
        for x in 0..w {
            let ti = y * w + x;
            dst[y * dst_stride + x] = iclip(
                (tmp1[ti] as i32 * weight + tmp2[ti] as i32 * (16 - weight) + rnd) >> sh,
                0,
                255,
            ) as u8;
        }
    }
}

pub fn mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    let sh = INTERMEDIATE_BITS_8BPC + 6;
    let rnd = 32 << INTERMEDIATE_BITS_8BPC;
    for y in 0..h {
        for x in 0..w {
            let ti = y * w + x;
            let m = mask[ti] as i32;
            dst[y * dst_stride + x] = iclip(
                (tmp1[ti] as i32 * m + tmp2[ti] as i32 * (64 - m) + rnd) >> sh,
                0,
                255,
            ) as u8;
        }
    }
}

pub fn blend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        for x in 0..w {
            let di = y * dst_stride + x;
            let ti = y * w + x;
            let m = mask[ti] as i32;
            dst[di] = ((dst[di] as i32 * (64 - m) + tmp[ti] as i32 * m + 32) >> 6) as u8;
        }
    }
}

pub fn morph_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    alpha: i32,
    beta: i32,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let di = y * dst_stride + x;
            dst[di] = iclip((alpha * dst[di] as i32 + beta) >> 8, 0, 255) as u8;
        }
    }
}

pub fn emu_edge_8bpc(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: isize,
    y: isize,
    dst: &mut [u8],
    dst_stride: usize,
    r: &[u8],
    ref_stride: usize,
) {
    let ref_y = iclip(y as i32, 0, ih as i32 - 1) as usize;
    let ref_x = iclip(x as i32, 0, iw as i32 - 1) as usize;
    let ref_off = ref_y * ref_stride + ref_x;

    let left_ext = iclip(-x as i32, 0, bw as i32 - 1) as usize;
    let right_ext = iclip((x + bw as isize - iw as isize) as i32, 0, bw as i32 - 1) as usize;
    let top_ext = iclip(-y as i32, 0, bh as i32 - 1) as usize;
    let bottom_ext = iclip((y + bh as isize - ih as isize) as i32, 0, bh as i32 - 1) as usize;

    let center_w = bw - left_ext - right_ext;
    let center_h = bh - top_ext - bottom_ext;

    let mut roff = ref_off;
    for cy in 0..center_h {
        let blk_y = top_ext + cy;
        let blk_off = blk_y * dst_stride;
        dst[blk_off + left_ext..blk_off + left_ext + center_w]
            .copy_from_slice(&r[roff..roff + center_w]);
        if left_ext > 0 {
            let fill = dst[blk_off + left_ext];
            dst[blk_off..blk_off + left_ext].fill(fill);
        }
        if right_ext > 0 {
            let fill = dst[blk_off + left_ext + center_w - 1];
            dst[blk_off + left_ext + center_w..blk_off + bw].fill(fill);
        }
        roff += ref_stride;
    }

    let first_row_off = top_ext * dst_stride;
    for ty in 0..top_ext {
        dst.copy_within(first_row_off..first_row_off + bw, ty * dst_stride);
    }

    if bottom_ext > 0 {
        let last_row_start = (top_ext + center_h - 1) * dst_stride;
        for by in 0..bottom_ext {
            let dst_start = (top_ext + center_h + by) * dst_stride;
            dst.copy_within(last_row_start..last_row_start + bw, dst_start);
        }
    }
}

pub fn sad_nxn_8bpc(
    p0: &[u8],
    p0_stride: usize,
    p1: &[u8],
    p1_stride: usize,
    w: usize,
    h: usize,
) -> i32 {
    let mut sad = 0i32;
    let mut o0 = 0;
    let mut o1 = 0;
    let mut y = 0;
    while y < h {
        for x in 0..w {
            sad += (p0[o0 + x] as i32 - p1[o1 + x] as i32).abs();
        }
        o0 += p0_stride * 2;
        o1 += p1_stride * 2;
        y += 2;
    }
    sad
}

pub fn sad_refine_mv_8bpc(
    p0: &[u8],
    p0_stride: usize,
    p1: &[u8],
    p1_stride: usize,
    w: usize,
    h: usize,
    is_implicit: bool,
) -> (i32, i32) {
    let sadw = w + 4;
    let sadh = h + 4;
    let sad_thr = (sadw * sadh * 2) as i32;
    let mut best_sad = i32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    if is_implicit {
        best_sad = sad_nxn_8bpc(
            &p0[2 * p0_stride + 2..], p0_stride,
            &p1[2 * p1_stride + 2..], p1_stride,
            sadw, sadh,
        );
        best_sad = (best_sad * 7 + 7) >> 3;
        if best_sad < sad_thr {
            return (best_dx, best_dy);
        }
    }

    for y_off in -2i32..=2 {
        for x_off in -2i32..=2 {
            if x_off == 0 && y_off == 0 { continue; }
            let sad = sad_nxn_8bpc(
                &p0[((2 + y_off) as usize) * p0_stride + (2 + x_off) as usize..], p0_stride,
                &p1[((2 - y_off) as usize) * p1_stride + (2 - x_off) as usize..], p1_stride,
                sadw, sadh,
            );
            if sad >= best_sad { continue; }
            best_sad = sad;
            best_dx = x_off;
            best_dy = y_off;
        }
    }
    (best_dx, best_dy)
}

fn filter_warp_rnd(src: &[i16], x: usize, f: &[i8; 8], stride: usize, sh: i32) -> i16 {
    let v = f[0] as i32 * src[x.wrapping_sub(3 * stride)] as i32
        + f[1] as i32 * src[x.wrapping_sub(2 * stride)] as i32
        + f[2] as i32 * src[x.wrapping_sub(stride)] as i32
        + f[3] as i32 * src[x] as i32
        + f[4] as i32 * src[x + stride] as i32
        + f[5] as i32 * src[x + 2 * stride] as i32
        + f[6] as i32 * src[x + 3 * stride] as i32
        + f[7] as i32 * src[x + 4 * stride] as i32
        + ((1 << sh) >> 1);
    (v >> sh) as i16
}

fn filter_warp_rnd_px(src: &[u8], x: usize, f: &[i8; 8], stride: usize, sh: i32) -> i16 {
    let v = f[0] as i32 * src[x.wrapping_sub(3 * stride)] as i32
        + f[1] as i32 * src[x.wrapping_sub(2 * stride)] as i32
        + f[2] as i32 * src[x.wrapping_sub(stride)] as i32
        + f[3] as i32 * src[x] as i32
        + f[4] as i32 * src[x + stride] as i32
        + f[5] as i32 * src[x + 2 * stride] as i32
        + f[6] as i32 * src[x + 3 * stride] as i32
        + f[7] as i32 * src[x + 4 * stride] as i32
        + ((1 << sh) >> 1);
    (v >> sh) as i16
}

pub fn warp_affine_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let mut mid = [0i16; 15 * 8];

    let mut soff = src_off.wrapping_sub(3 * src_stride);
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let fi = (192 + ((tmx + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            mid[y * 8 + x] = filter_warp_rnd_px(src, soff + x, f, 1, 3);
            tmx += abcd[0] as i32;
        }
        soff += src_stride;
        mx += abcd[1] as i32;
    }

    for y in 0..8 {
        let mid_base = (3 + y) * 8;
        let mut tmy = my;
        for x in 0..8 {
            let fi = (192 + ((tmy + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            let v = filter_warp_rnd(&mid, mid_base + x, f, 8, 11);
            dst[y * dst_stride + x] = iclip(v as i32, 0, 255) as u8;
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}

pub fn warp_affine_8x8t_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let mut mid = [0i16; 15 * 8];

    let mut soff = src_off.wrapping_sub(3 * src_stride);
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let fi = (192 + ((tmx + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            mid[y * 8 + x] = filter_warp_rnd_px(src, soff + x, f, 1, 3);
            tmx += abcd[0] as i32;
        }
        soff += src_stride;
        mx += abcd[1] as i32;
    }

    for y in 0..8 {
        let mid_base = (3 + y) * 8;
        let mut tmy = my;
        for x in 0..8 {
            let fi = (192 + ((tmy + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            tmp[y * tmp_stride + x] = filter_warp_rnd(&mid, mid_base + x, f, 8, 7) - PREP_BIAS_8BPC as i16;
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad_identical() {
        let p = vec![128u8; 64];
        assert_eq!(sad_nxn_8bpc(&p, 8, &p, 8, 8, 8), 0);
    }

    #[test]
    fn test_sad_all_different() {
        let p0 = vec![0u8; 64];
        let p1 = vec![10u8; 64];
        let sad = sad_nxn_8bpc(&p0, 8, &p1, 8, 8, 8);
        assert_eq!(sad, 8 * 4 * 10);
    }

    #[test]
    fn test_sad_skips_odd_rows() {
        let mut p0 = vec![0u8; 32];
        let p1 = vec![0u8; 32];
        for i in 0..4 {
            p0[i * 8 + 1] = 5;
        }
        let sad = sad_nxn_8bpc(&p0, 8, &p1, 8, 4, 4);
        assert_eq!(sad, 2 * 5);
    }

    #[test]
    fn test_sad_stride() {
        let mut p0 = vec![100u8; 128];
        let p1 = vec![100u8; 128];
        p0[0] = 200;
        let sad = sad_nxn_8bpc(&p0, 16, &p1, 16, 4, 4);
        assert_eq!(sad, 100);
    }

    #[test]
    fn test_put_copy() {
        let src = vec![42u8; 64];
        let mut dst = vec![0u8; 64];
        put_8bpc(&mut dst, 8, &src, 8, 8, 8);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_put_stride() {
        let src = vec![10u8; 32];
        let mut dst = vec![0u8; 32];
        put_8bpc(&mut dst, 8, &src, 8, 4, 4);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 10);
            }
        }
    }

    #[test]
    fn test_prep_shift() {
        let src = vec![128u8; 16];
        let mut tmp = vec![0i16; 16];
        prep_8bpc(&mut tmp, &src, 4, 4, 4);
        for &v in &tmp {
            assert_eq!(v, (128 << INTERMEDIATE_BITS_8BPC) as i16);
        }
    }

    #[test]
    fn test_avg_midpoint() {
        let tmp1: Vec<i16> = vec![(100 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![(200 << 4) as i16; 16];
        let mut dst = vec![0u8; 16];
        avg_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4);
        for &v in &dst {
            assert_eq!(v, 150);
        }
    }

    #[test]
    fn test_w_avg_full_weight() {
        let tmp1: Vec<i16> = vec![(200 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![0i16; 16];
        let mut dst = vec![0u8; 16];
        w_avg_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, 16);
        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_w_avg_zero_weight() {
        let tmp1: Vec<i16> = vec![0i16; 16];
        let tmp2: Vec<i16> = vec![(100 << 4) as i16; 16];
        let mut dst = vec![0u8; 16];
        w_avg_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, 0);
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_mask_uniform() {
        let tmp1: Vec<i16> = vec![(255 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![0i16; 16];
        let mask_buf = vec![64u8; 16];
        let mut dst = vec![0u8; 16];
        mask_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_mask_half() {
        let tmp1: Vec<i16> = vec![(200 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![(100 << 4) as i16; 16];
        let mask_buf = vec![32u8; 16];
        let mut dst = vec![0u8; 16];
        mask_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 150);
        }
    }

    #[test]
    fn test_blend_zero_mask() {
        let mut dst = vec![200u8; 16];
        let tmp = vec![50u8; 16];
        let mask_buf = vec![0u8; 16];
        blend_8bpc(&mut dst, 4, &tmp, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_blend_full_mask() {
        let mut dst = vec![200u8; 16];
        let tmp = vec![50u8; 16];
        let mask_buf = vec![64u8; 16];
        blend_8bpc(&mut dst, 4, &tmp, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 50);
        }
    }

    #[test]
    fn test_morph_identity() {
        let mut dst = vec![128u8; 16];
        morph_8bpc(&mut dst, 4, 256, 0, 4, 4);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_morph_half() {
        let mut dst = vec![200u8; 16];
        morph_8bpc(&mut dst, 4, 128, 0, 4, 4);
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_morph_clamps() {
        let mut dst = vec![200u8; 16];
        morph_8bpc(&mut dst, 4, 512, 0, 4, 4);
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_emu_edge_inside() {
        let r = vec![42u8; 64];
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, 2, 2, &mut dst, 4, &r, 8);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 42);
            }
        }
    }

    #[test]
    fn test_emu_edge_left_extend() {
        let mut r = vec![0u8; 64];
        for y in 0..8 {
            r[y * 8] = 99;
        }
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, -2, 0, &mut dst, 4, &r, 8);
        for y in 0..4 {
            assert_eq!(dst[y * 4], 99);
            assert_eq!(dst[y * 4 + 1], 99);
        }
    }

    #[test]
    fn test_emu_edge_top_extend() {
        let r = vec![77u8; 64];
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, 0, -2, &mut dst, 4, &r, 8);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 77);
            }
        }
    }

    #[test]
    fn test_emu_edge_corner() {
        let mut r = vec![50u8; 64];
        r[0] = 200;
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, -2, -2, &mut dst, 4, &r, 8);
        assert_eq!(dst[0], 200);
        assert_eq!(dst[1], 200);
    }

    #[test]
    fn test_sad_refine_identical() {
        let p = vec![128u8; 20 * 20];
        let (dx, dy) = sad_refine_mv_8bpc(&p, 20, &p, 20, 8, 8, false);
        assert_eq!((dx, dy), (-2, -2));
    }

    #[test]
    fn test_sad_refine_implicit_low_sad() {
        let p = vec![128u8; 20 * 20];
        let (dx, dy) = sad_refine_mv_8bpc(&p, 20, &p, 20, 8, 8, true);
        assert_eq!((dx, dy), (0, 0));
    }

    #[test]
    fn test_sad_refine_finds_offset() {
        let mut p0 = vec![100u8; 20 * 20];
        let mut p1 = vec![100u8; 20 * 20];
        for y in 0..12 {
            for x in 0..12 {
                p0[(y + 3) * 20 + (x + 3)] = 200;
                p1[(y + 1) * 20 + (x + 1)] = 200;
            }
        }
        let (dx, dy) = sad_refine_mv_8bpc(&p0, 20, &p1, 20, 8, 8, false);
        assert!(dx != 0 || dy != 0);
    }

    #[test]
    fn test_warp_affine_identity() {
        let mut src = vec![0u8; 22 * 22];
        for y in 0..22 {
            for x in 0..22 {
                src[y * 22 + x] = 128;
            }
        }
        let mut dst = [0u8; 64];
        let abcd = [0i16, 0, 0, 0];
        let src_off = 3 * 22 + 3;
        warp_affine_8x8_8bpc(&mut dst, 8, &src, 22, src_off, &abcd, 0, 0);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_warp_affine_no_panic() {
        let src = vec![100u8; 22 * 22];
        let mut dst = [0u8; 64];
        let abcd = [64i16, 0, 0, 64];
        warp_affine_8x8_8bpc(&mut dst, 8, &src, 22, 3 * 22 + 3, &abcd, 0, 0);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_warp_affine_8x8t_identity() {
        let src = vec![128u8; 22 * 22];
        let mut tmp = [0i16; 64];
        let abcd = [0i16, 0, 0, 0];
        warp_affine_8x8t_8bpc(&mut tmp, 8, &src, 22, 3 * 22 + 3, &abcd, 0, 0);
        for &v in &tmp {
            assert_eq!(v, 2048);
        }
    }
}

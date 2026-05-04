use crate::intops::iclip;

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
}

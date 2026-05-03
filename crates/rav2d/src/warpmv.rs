use crate::headers::WarpedMotionParams;
use crate::intops::{apply_sign, apply_sign64, iclip, iclip64to32, u64log2, ulog2};
use crate::levels::MvXY;
use crate::tables::DIV_RECIP;

#[inline(always)]
fn iclip_wmp(v: i32) -> i16 {
    iclip((v + 0x20 - (v < 0) as i32) & !0x3f, -0x8000, 0x7fc0) as i16
}

pub fn resolve_divisor_32(d: u32, shift: &mut i32) -> i32 {
    *shift = ulog2(d);
    let e = d as i32 - (1 << *shift);
    let f = if *shift > 7 {
        (e + (1 << (*shift - 8))) >> (*shift - 7)
    } else {
        e << (7 - *shift)
    };
    debug_assert!(f <= 128);
    *shift += 9;
    DIV_RECIP[f as usize] as i32
}

pub fn get_shear_params(wm: &mut WarpedMotionParams) -> i32 {
    let mat = &wm.matrix;

    if mat[2] <= 0 {
        return 1;
    }

    wm.abcd[0] = iclip_wmp(mat[2] - 0x10000);
    wm.abcd[1] = iclip_wmp(mat[3]);

    let mut shift = 0i32;
    let y = apply_sign(resolve_divisor_32(mat[2].unsigned_abs(), &mut shift), mat[2]);
    let v1 = (mat[4] as i64 * 0x10000) * y as i64;
    let rnd = (1i64 << shift) >> 1;
    wm.abcd[2] = iclip_wmp(apply_sign64((v1.unsigned_abs().wrapping_add(rnd as u64)) as i64 >> shift, v1) as i32);
    let v2 = (mat[3] as i64 * mat[4] as i64) * y as i64;
    wm.abcd[3] = iclip_wmp(
        mat[5]
            - apply_sign64((v2.unsigned_abs().wrapping_add(rnd as u64)) as i64 >> shift, v2) as i32
            - 0x10000,
    );

    wm.affine = ((4 * (wm.abcd[0] as i32).abs() + 7 * (wm.abcd[1] as i32).abs() < 0x30000)
        && (4 * (wm.abcd[2] as i32).abs() + 4 * (wm.abcd[3] as i32).abs() < 0x30000))
        as i32;
    0
}

fn resolve_divisor_64(d: u64, shift: &mut i32) -> i32 {
    *shift = u64log2(d);
    let e = d as i64 - (1i64 << *shift);
    let f = if *shift > 7 {
        (e + (1i64 << (*shift - 8))) >> (*shift - 7)
    } else {
        e << (7 - *shift)
    };
    debug_assert!(f <= 128);
    *shift += 9;
    DIV_RECIP[f as usize] as i32
}

fn get_mult_shift_ndiag(px: i64, idet: i32, rnd: i64, sh: i32) -> i32 {
    let v1 = px * idet as i64;
    let v2 = ((v1 + rnd - (v1 < 0) as i64) >> sh) as i32;
    let v3 = (v2 + 0x20 - (v2 < 0) as i32) & !0x3f;
    iclip(v3, -0x7fc0, 0x7fc0)
}

fn get_mult_shift_diag(px: i64, idet: i32, rnd: i64, sh: i32) -> i32 {
    let v1 = px * idet as i64;
    let v2 = ((v1 + rnd - (v1 < 0) as i64) >> sh) as i32;
    let v3 = (v2 + 0x20 - ((v2 < 0x10000) as i32)) & !0x3f;
    iclip(v3, 0x8040, 0x17fc0)
}

pub fn set_affine_mv2d(
    bw4: i32,
    bh4: i32,
    mv: MvXY,
    wm: &mut WarpedMotionParams,
    bx4: i32,
    by4: i32,
) {
    let rsuy = 2 * bh4 - 1;
    let rsux = 2 * bw4 - 1;
    let isuy = by4 * 4 + rsuy;
    let isux = bx4 * 4 + rsux;

    wm.matrix[0] = iclip64to32(
        mv.x as i64 * 0x2000 - isux as i64 * (wm.matrix[2] as i64 - 0x10000) - isuy as i64 * wm.matrix[3] as i64,
        -0x8000000,
        0x7ffffc0,
    );
    wm.matrix[1] = iclip64to32(
        mv.y as i64 * 0x2000 - isux as i64 * wm.matrix[4] as i64 - isuy as i64 * (wm.matrix[5] as i64 - 0x10000),
        -0x8000000,
        0x7ffffc0,
    );
}

pub fn find_affine_int(
    pts: &[[[i32; 2]; 2]],
    np: usize,
    bw4: i32,
    bh4: i32,
    mv: MvXY,
    wm: &mut WarpedMotionParams,
    bx4: i32,
    by4: i32,
) -> i32 {
    let mut a = [[0i32; 2]; 2];
    let mut bx = [0i32; 2];
    let mut by = [0i32; 2];
    let rsuy = 2 * bh4 - 1;
    let rsux = 2 * bw4 - 1;
    let suy = rsuy * 8;
    let sux = rsux * 8;
    let duy = suy + mv.y;
    let dux = sux + mv.x;

    for i in 0..np {
        let dx = pts[i][1][0] - dux;
        let dy = pts[i][1][1] - duy;
        let sx = pts[i][0][0] - sux;
        let sy = pts[i][0][1] - suy;
        if (sx - dx).abs() < 256 && (sy - dy).abs() < 256 {
            a[0][0] += ((sx * sx) >> 2) + sx * 2 + 8;
            a[0][1] += ((sx * sy) >> 2) + sx + sy + 4;
            a[1][1] += ((sy * sy) >> 2) + sy * 2 + 8;
            bx[0] += ((sx * dx) >> 2) + sx + dx + 8;
            bx[1] += ((sy * dx) >> 2) + sy + dx + 4;
            by[0] += ((sx * dy) >> 2) + sx + dy + 4;
            by[1] += ((sy * dy) >> 2) + sy + dy + 8;
        }
    }

    let det = a[0][0] as i64 * a[1][1] as i64 - a[0][1] as i64 * a[0][1] as i64;
    if det == 0 {
        wm.matrix[2] = 0x10000;
        wm.matrix[5] = 0x10000;
        wm.matrix[3] = 0;
        wm.matrix[4] = 0;
        set_affine_mv2d(bw4, bh4, mv, wm, bx4, by4);
        return 0;
    }

    let mut shift = 0i32;
    let mut idet = apply_sign64(resolve_divisor_64(det.unsigned_abs(), &mut shift) as i64, det) as i32;
    shift -= 16;
    if shift < 0 {
        idet <<= -shift;
        shift = 0;
    }

    let r = (1i64 << shift) >> 1;
    wm.matrix[2] = get_mult_shift_diag(
        a[1][1] as i64 * bx[0] as i64 - a[0][1] as i64 * bx[1] as i64,
        idet, r, shift,
    );
    wm.matrix[3] = get_mult_shift_ndiag(
        a[0][0] as i64 * bx[1] as i64 - a[0][1] as i64 * bx[0] as i64,
        idet, r, shift,
    );
    wm.matrix[4] = get_mult_shift_ndiag(
        a[1][1] as i64 * by[0] as i64 - a[0][1] as i64 * by[1] as i64,
        idet, r, shift,
    );
    wm.matrix[5] = get_mult_shift_diag(
        a[0][0] as i64 * by[1] as i64 - a[0][1] as i64 * by[0] as i64,
        idet, r, shift,
    );

    set_affine_mv2d(bw4, bh4, mv, wm, bx4, by4);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iclip_wmp() {
        assert_eq!(iclip_wmp(0), 0);
        assert_eq!(iclip_wmp(0x100), 0x100);
        assert_eq!(iclip_wmp(-0x100), -0x100);
        assert_eq!(iclip_wmp(0x7fc0), 0x7fc0);
        assert_eq!(iclip_wmp(-0x8000), -0x8000);
        assert_eq!(iclip_wmp(0x10000), 0x7fc0);
        assert_eq!(iclip_wmp(-0x10000), -0x8000);
    }

    #[test]
    fn test_resolve_divisor_32() {
        let mut shift = 0;
        let r = resolve_divisor_32(256, &mut shift);
        assert!(r > 0);
        assert!(shift > 0);
    }

    #[test]
    fn test_get_shear_params_negative_mat2() {
        let mut wm = WarpedMotionParams::default();
        wm.matrix[2] = -1;
        assert_eq!(get_shear_params(&mut wm), 1);
    }

    #[test]
    fn test_get_shear_params_identity() {
        let mut wm = WarpedMotionParams::default();
        wm.matrix[2] = 0x10000;
        wm.matrix[5] = 0x10000;
        assert_eq!(get_shear_params(&mut wm), 0);
        assert_eq!(wm.abcd[0], 0);
        assert_eq!(wm.abcd[1], 0);
    }

    #[test]
    fn test_set_affine_mv2d_zero() {
        let mut wm = WarpedMotionParams::default();
        wm.matrix[2] = 0x10000;
        wm.matrix[5] = 0x10000;
        let mv = MvXY { y: 0, x: 0 };
        set_affine_mv2d(4, 4, mv, &mut wm, 0, 0);
        assert_eq!(wm.matrix[0], 0);
        assert_eq!(wm.matrix[1], 0);
    }

    #[test]
    fn test_find_affine_int_no_points() {
        let mut wm = WarpedMotionParams::default();
        let mv = MvXY { y: 0, x: 0 };
        let pts: &[[[i32; 2]; 2]] = &[];
        assert_eq!(find_affine_int(pts, 0, 4, 4, mv, &mut wm, 0, 0), 0);
        assert_eq!(wm.matrix[2], 0x10000);
        assert_eq!(wm.matrix[5], 0x10000);
    }
}

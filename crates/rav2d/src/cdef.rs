use crate::intops::{apply_sign, imax, imin, ulog2};

#[inline(always)]
pub fn constrain(diff: i32, threshold: i32, shift: i32) -> i32 {
    let adiff = diff.abs();
    apply_sign(imin(adiff, imax(0, threshold - (adiff >> shift))), diff)
}

pub fn fill(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            tmp[y * stride + x] = i16::MIN;
        }
    }
}

pub fn cdef_find_dir(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        for x in 0..8usize {
            let px = img[y * stride + x] as i32 - 128;

            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as u32;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    const DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7usize {
        let d = DIV_TABLE[n];
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n]) as u32)
            * d;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n]) as u32)
            * d;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7]) as u32 * 105;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7]) as u32 * 105;

    for n in 0..4usize {
        let ci = n * 2 + 1;
        for m in 0..5usize {
            cost[ci] +=
                (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as u32;
        }
        cost[ci] *= 105;
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            cost[ci] += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m]) as u32)
                * d;
        }
    }

    let mut best_dir = 0i32;
    let mut best_cost = cost[0];
    for n in 1..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n as i32;
        }
    }

    *var = (best_cost - cost[(best_dir ^ 4) as usize]) >> 10;
    best_dir
}

pub fn cdef_pri_tap(pri_strength: i32) -> i32 {
    4 - ((pri_strength >> 0) & 1)
}

pub fn cdef_apply_constrain(
    px: i32,
    p0: i32,
    p1: i32,
    strength: i32,
    shift: i32,
    tap: i32,
) -> i32 {
    tap * constrain(p0 - px, strength, shift) + tap * constrain(p1 - px, strength, shift)
}

pub fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 {
        imin(ulog2(var >> 6), 12)
    } else {
        0
    };
    (strength * (4 + i) + 8) >> 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrain_zero_diff() {
        assert_eq!(constrain(0, 10, 1), 0);
    }

    #[test]
    fn test_constrain_small_diff() {
        let r = constrain(3, 10, 1);
        assert!(r > 0);
        assert!(r <= 3);
    }

    #[test]
    fn test_constrain_negative_diff() {
        let r = constrain(-5, 10, 1);
        assert!(r < 0);
    }

    #[test]
    fn test_constrain_large_diff_clamped() {
        let r = constrain(100, 10, 1);
        assert!(r <= 10);
    }

    #[test]
    fn test_constrain_zero_threshold() {
        assert_eq!(constrain(5, 0, 1), 0);
    }

    #[test]
    fn test_fill_basic() {
        let mut buf = [0i16; 24];
        fill(&mut buf, 6, 4, 3);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(buf[y * 6 + x], i16::MIN);
            }
            assert_eq!(buf[y * 6 + 4], 0);
            assert_eq!(buf[y * 6 + 5], 0);
        }
    }

    #[test]
    fn test_cdef_find_dir_uniform() {
        let img = [128u8; 64];
        let mut var = 0u32;
        let dir = cdef_find_dir(&img, 8, &mut var);
        assert!(dir >= 0 && dir < 8);
        assert_eq!(var, 0);
    }

    #[test]
    fn test_cdef_find_dir_vertical() {
        let mut img = [0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                img[y * 8 + x] = if x < 4 { 64 } else { 192 };
            }
        }
        let mut var = 0u32;
        let dir = cdef_find_dir(&img, 8, &mut var);
        assert!(dir >= 0 && dir < 8);
        assert!(var > 0);
    }

    #[test]
    fn test_cdef_find_dir_horizontal() {
        let mut img = [0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                img[y * 8 + x] = if y < 4 { 64 } else { 192 };
            }
        }
        let mut var = 0u32;
        let dir = cdef_find_dir(&img, 8, &mut var);
        assert!(dir >= 0 && dir < 8);
        assert!(var > 0);
    }

    #[test]
    fn test_cdef_pri_tap() {
        assert_eq!(cdef_pri_tap(4), 4);
        assert_eq!(cdef_pri_tap(3), 3);
        assert_eq!(cdef_pri_tap(2), 4);
        assert_eq!(cdef_pri_tap(1), 3);
    }

    #[test]
    fn test_cdef_apply_constrain_zero() {
        assert_eq!(cdef_apply_constrain(100, 100, 100, 10, 1, 4), 0);
    }

    #[test]
    fn test_cdef_apply_constrain_symmetric() {
        let r = cdef_apply_constrain(100, 105, 95, 10, 1, 4);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_adjust_strength_zero_var() {
        assert_eq!(adjust_strength(100, 0), 0);
    }

    #[test]
    fn test_adjust_strength_low_var() {
        let r = adjust_strength(100, 32);
        assert_eq!(r, (100 * 4 + 8) >> 4);
    }

    #[test]
    fn test_adjust_strength_high_var() {
        let r = adjust_strength(100, 1 << 18);
        assert!(r > adjust_strength(100, 64));
    }

    #[test]
    fn test_adjust_strength_monotonic() {
        let a = adjust_strength(100, 64);
        let b = adjust_strength(100, 256);
        let c = adjust_strength(100, 4096);
        assert!(a <= b);
        assert!(b <= c);
    }
}

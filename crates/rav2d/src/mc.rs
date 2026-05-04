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
}

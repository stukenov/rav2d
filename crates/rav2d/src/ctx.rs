#[inline(always)]
pub fn set_ctx(buf: &mut [u8], off: usize, val: u8, len: usize) {
    buf[off..off + len].fill(val);
}

#[inline(always)]
pub fn memset_pow2(buf: &mut [u8], off: usize, val: u8, log2_n: u8) {
    let n = 1usize << log2_n;
    buf[off..off + n].fill(val);
}

#[inline(always)]
pub fn memset_likely_pow2(buf: &mut [u8], off: usize, val: u8, n: usize) {
    debug_assert!(n >= 1 && n <= 64);
    buf[off..off + n].fill(val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_ctx() {
        let mut buf = [0u8; 16];
        set_ctx(&mut buf, 2, 0xAB, 4);
        assert_eq!(&buf[0..2], &[0, 0]);
        assert_eq!(&buf[2..6], &[0xAB; 4]);
        assert_eq!(&buf[6..8], &[0, 0]);
    }

    #[test]
    fn test_memset_pow2() {
        for log2 in 0..=6u8 {
            let n = 1usize << log2;
            let mut buf = vec![0u8; 128];
            memset_pow2(&mut buf, 0, 0xFF, log2);
            assert!(buf[..n].iter().all(|&b| b == 0xFF));
            assert!(buf[n..].iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn test_memset_likely_pow2() {
        let mut buf = [0u8; 64];
        memset_likely_pow2(&mut buf, 0, 42, 7);
        assert!(buf[..7].iter().all(|&b| b == 42));
        assert!(buf[7..].iter().all(|&b| b == 0));
    }
}

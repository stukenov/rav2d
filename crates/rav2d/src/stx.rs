use crate::intops::{apply_sign, iclip};

pub fn stxfm(cf_out: &mut [i32], cf: &[i32], kernel: &[i8], sz: usize, eob: usize, bitdepth_max: i32) {
    debug_assert!(sz == 16 || sz == 48);
    debug_assert!(eob < if sz == 16 { 8 } else { 32 });
    let min = -128 * (1 + bitdepth_max);
    let max = 128 * (1 + bitdepth_max) - 1;
    let h = eob + 1;
    for x in 0..sz {
        let mut sum = 0i32;
        for y in 0..h {
            sum += cf[y] * kernel[y * sz + x] as i32;
        }
        sum = apply_sign((sum.abs() + 64) >> 7, sum);
        cf_out[x] = iclip(sum, min, max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stxfm_identity_kernel() {
        let mut cf_out = [0i32; 16];
        let cf = [100i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut kernel = [0i8; 16 * 16];
        kernel[0] = 127;
        stxfm(&mut cf_out, &cf, &kernel, 16, 0, 255);
        assert_eq!(cf_out[0], (100 * 127 + 64) >> 7);
    }

    #[test]
    fn test_stxfm_clamp() {
        let mut cf_out = [0i32; 16];
        let cf = [10000i32; 8];
        let mut kernel = [0i8; 16 * 16];
        for y in 0..7 {
            kernel[y * 16] = 127;
        }
        stxfm(&mut cf_out, &cf, &kernel, 16, 6, 255);
        let max = 128 * 256 - 1;
        let min = -128 * 256;
        assert!(cf_out[0] >= min && cf_out[0] <= max);
    }

    #[test]
    fn test_stxfm_sz48() {
        let mut cf_out = [0i32; 48];
        let cf = [10i32; 32];
        let kernel = [1i8; 48 * 48];
        stxfm(&mut cf_out, &cf, &kernel, 48, 31, 255);
        for &v in &cf_out {
            assert!(v >= -128 * 256 && v <= 128 * 256 - 1);
        }
    }

    #[test]
    fn test_stxfm_negative_coefs() {
        let mut cf_out = [0i32; 16];
        let cf = [-50i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut kernel = [0i8; 16 * 16];
        kernel[0] = 64;
        stxfm(&mut cf_out, &cf, &kernel, 16, 0, 255);
        assert!(cf_out[0] < 0);
    }

    #[test]
    fn test_stxfm_zero_eob() {
        let mut cf_out = [0i32; 16];
        let cf = [42i32; 16];
        let kernel = [0i8; 16 * 16];
        stxfm(&mut cf_out, &cf, &kernel, 16, 0, 255);
        assert!(cf_out.iter().all(|&v| v == 0));
    }
}

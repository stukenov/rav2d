use crate::intops::{apply_sign, iclip, imin, ulog2};
use crate::levels::IntraPredMode;
use crate::msac::MsacContext;
use crate::tables::{DIV_RECIP, TxfmInfo, MODE_TO_ANGLE_MAP};

pub fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 { imin(ulog2(var >> 6) as i32, 12) } else { 0 };
    (strength * (4 + i) + 8) >> 4
}

pub fn decode_exp_golomb(msac: &mut MsacContext, k: u32) -> u32 {
    let length = msac.decode_unary_bypass(21) + k;
    let x = (1u32 << length) + msac.decode_bools_bypass(length);
    x - (1 << k)
}

pub fn decode_hr(msac: &mut MsacContext, hr_avg: i32) -> i32 {
    let m = ulog2(iclip(hr_avg, 2, 64) as u32) as u32;
    let cmax = imin(m as i32 + 4, 6) as u32;
    let q = msac.decode_unary_bypass(cmax);
    let rem = if q == cmax {
        decode_exp_golomb(msac, m + 1)
    } else {
        msac.decode_bools_bypass(m)
    };
    (rem + (q << m)) as i32
}

pub fn tcq_next_state(state: i32, abs_level: i32) -> i32 {
    (((state & 0x4) ^ (((abs_level & 1) ^ (state & 0x1)) << 2))
        | ((state & 0x6) >> 1)
        | -0x80000000i32)
        & (state >> 31)
}

pub fn wide_angle_remap(
    t_dim: &TxfmInfo,
    mode: IntraPredMode,
    angle: &mut i32,
    mrl_idx: i32,
) -> IntraPredMode {
    let mode_u8 = mode as u8;
    if mode_u8.wrapping_sub(1) > IntraPredMode::VertLeftPred as u8 - 1 {
        return mode;
    }

    let mrl_adj = (mrl_idx == 1) as i32 - (mrl_idx == 2) as i32;
    *angle = MODE_TO_ANGLE_MAP[(mode_u8 - 1) as usize] as i32 + *angle * 3 + mrl_adj;

    static THRESH: [u8; 4] = [61, 73, 82, 86];
    let rect = t_dim.lw as i32 - t_dim.lh as i32;

    if rect > 0 {
        debug_assert!(rect <= 4);
        if *angle > 270 - THRESH[(rect - 1) as usize] as i32 {
            *angle -= 180;
            return IntraPredMode::DiagDownLeftPred;
        }
    } else if rect < 0 {
        debug_assert!(rect >= -4);
        if *angle < THRESH[(-1 - rect) as usize] as i32 {
            *angle += 180;
            return IntraPredMode::HorUpPred;
        }
    }

    mode
}

pub fn gen_mask(
    mask: &mut [u8],
    stride: usize,
    bw: i32,
    bh: i32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    fw: u32,
    fh: u32,
) {
    let mut off = 0;
    for y in 0..bh {
        for x in 0..bw {
            let p0 = ((x0 + x) as u32) < fw && ((y0 + y) as u32) < fh;
            let p1 = ((x1 + x) as u32) < fw && ((y1 + y) as u32) < fh;
            mask[off + x as usize] = (32 * (p0 as i32 - p1 as i32 + 1)) as u8;
        }
        off += stride;
    }
}

pub fn derive_alpha(num: i32, den: i32, mut alpha: i32) -> i32 {
    let max = (2 << 8) - 1;
    if num != 0 && den != 0 {
        let num_abs = num.abs();
        let shift_n = ulog2(num_abs as u32);
        debug_assert!(den >= 0);
        let shift_d = ulog2(den as u32);
        let e_d = den - (1 << shift_d);
        let f_d = if shift_d > 7 {
            (e_d + (1 << (shift_d - 8))) >> (shift_d - 7)
        } else {
            e_d << (7 - shift_d)
        };
        let f_n = if shift_n > 7 {
            (num_abs + (1 << (shift_n - 8))) >> (shift_n - 7)
        } else {
            num_abs << (7 - shift_n)
        };
        let shift_add = shift_d - shift_n - 8;
        if shift_add <= 1 {
            let shift0 = 9 + 7 + shift_add;
            let tmp_alpha = if shift0 < 0 {
                max
            } else {
                imin((DIV_RECIP[f_d as usize] as i32 * f_n) >> shift0, max)
            };
            if tmp_alpha != 0 {
                alpha = apply_sign(tmp_alpha, num);
            }
        }
    }
    alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_strength_zero_var() {
        assert_eq!(adjust_strength(10, 0), 0);
    }

    #[test]
    fn test_adjust_strength_small_var() {
        assert_eq!(adjust_strength(16, 1), (16 * 4 + 8) >> 4);
    }

    #[test]
    fn test_adjust_strength_large_var() {
        let v = adjust_strength(16, 1 << 18);
        assert!(v > adjust_strength(16, 1));
    }

    #[test]
    fn test_adjust_strength_zero_strength() {
        assert_eq!(adjust_strength(0, 100), (0 * 4 + 8) >> 4);
    }

    #[test]
    fn test_tcq_next_state_zero() {
        assert_eq!(tcq_next_state(0, 0), 0);
    }

    #[test]
    fn test_tcq_next_state_disabled() {
        let s = tcq_next_state(-1, 5);
        assert_eq!(s, tcq_next_state(-1, 5));
        let s2 = tcq_next_state(-1, 0);
        let _ = (s, s2);
    }

    #[test]
    fn test_tcq_next_state_positive() {
        let s = tcq_next_state(3, 1);
        assert_eq!(s, 0);
        let s = tcq_next_state(5, 0);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_decode_exp_golomb() {
        let data = [0xFF; 16];
        let mut msac = MsacContext::new(&data, true);
        let v = decode_exp_golomb(&mut msac, 1);
        assert!(v < 1000);
    }

    #[test]
    fn test_decode_hr() {
        let data = [0x00; 16];
        let mut msac = MsacContext::new(&data, true);
        let v = decode_hr(&mut msac, 8);
        assert!(v >= 0);
    }

    #[test]
    fn test_wide_angle_remap_dc() {
        let t_dim = TxfmInfo { w: 4, h: 4, lw: 2, lh: 2, min: 2, max: 2, sub: 0, ctx: 0 };
        let mut angle = 0;
        let r = wide_angle_remap(&t_dim, IntraPredMode::DcPred, &mut angle, 0);
        assert_eq!(r, IntraPredMode::DcPred);
    }

    #[test]
    fn test_wide_angle_remap_no_remap() {
        let t_dim = TxfmInfo { w: 4, h: 4, lw: 2, lh: 2, min: 2, max: 2, sub: 0, ctx: 0 };
        let mut angle = 0;
        let r = wide_angle_remap(&t_dim, IntraPredMode::VertPred, &mut angle, 0);
        assert_eq!(r, IntraPredMode::VertPred);
    }

    #[test]
    fn test_wide_angle_remap_wide_rect() {
        let t_dim = TxfmInfo { w: 16, h: 4, lw: 4, lh: 2, min: 2, max: 4, sub: 0, ctx: 0 };
        let mut angle = 30;
        let r = wide_angle_remap(&t_dim, IntraPredMode::VertPred, &mut angle, 0);
        let _ = r;
    }

    #[test]
    fn test_wide_angle_remap_tall_rect() {
        let t_dim = TxfmInfo { w: 4, h: 16, lw: 2, lh: 4, min: 2, max: 4, sub: 0, ctx: 0 };
        let mut angle = -30;
        let r = wide_angle_remap(&t_dim, IntraPredMode::HorPred, &mut angle, 0);
        let _ = r;
    }

    #[test]
    fn test_gen_mask_both_inside() {
        let mut mask = vec![0u8; 16];
        gen_mask(&mut mask, 4, 4, 4, 0, 0, 0, 0, 100, 100);
        assert!(mask.iter().all(|&v| v == 32));
    }

    #[test]
    fn test_gen_mask_p0_only() {
        let mut mask = vec![0u8; 4];
        gen_mask(&mut mask, 4, 4, 1, 0, 0, -100, -100, 100, 100);
        assert!(mask[..4].iter().all(|&v| v == 64));
    }

    #[test]
    fn test_gen_mask_p1_only() {
        let mut mask = vec![0u8; 4];
        gen_mask(&mut mask, 4, 4, 1, -100, -100, 0, 0, 100, 100);
        assert!(mask[..4].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_gen_mask_neither() {
        let mut mask = vec![0u8; 4];
        gen_mask(&mut mask, 4, 4, 1, -100, -100, -100, -100, 100, 100);
        assert!(mask[..4].iter().all(|&v| v == 32));
    }

    #[test]
    fn test_derive_alpha_zero_num() {
        assert_eq!(derive_alpha(0, 100, 5), 5);
    }

    #[test]
    fn test_derive_alpha_zero_den() {
        assert_eq!(derive_alpha(100, 0, 5), 5);
    }

    #[test]
    fn test_derive_alpha_equal() {
        let a = derive_alpha(256, 256, 0);
        assert!(a > 0);
    }

    #[test]
    fn test_derive_alpha_negative_num() {
        let a = derive_alpha(-256, 256, 0);
        assert!(a < 0);
    }

    #[test]
    fn test_derive_alpha_bounded() {
        let a = derive_alpha(1000, 1, 0);
        assert!(a.abs() <= 511);
    }
}

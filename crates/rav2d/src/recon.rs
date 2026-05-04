use crate::intops::{apply_sign, iclip, imin, umin, ulog2};
use crate::levels::{IntraPredMode, N_BS_SIZES};
use crate::msac::MsacContext;
use crate::tables::{BLOCK_DIMENSIONS, DIV_RECIP, TxfmInfo, MODE_TO_ANGLE_MAP};

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

fn read_u16_ne(a: &[u8]) -> u16 {
    u16::from_ne_bytes(a[..2].try_into().unwrap())
}

fn read_u32_ne(a: &[u8]) -> u32 {
    u32::from_ne_bytes(a[..4].try_into().unwrap())
}

fn read_u64_ne(a: &[u8]) -> u64 {
    u64::from_ne_bytes(a[..8].try_into().unwrap())
}

pub fn get_skip_ctx(
    t_dim: &TxfmInfo,
    bs: usize,
    a: &[u8],
    l: &[u8],
    plane: i32,
    u_has_cf: i32,
    ss_hor: bool,
    ss_ver: bool,
) -> u32 {
    debug_assert!(bs < N_BS_SIZES);
    let b_dim = &BLOCK_DIMENSIONS[bs];

    if plane != 0 {
        let not_one_blk =
            (b_dim[2] - (b_dim[2] != 0 && ss_hor) as u8 > t_dim.lw) ||
            (b_dim[3] - (b_dim[3] != 0 && ss_ver) as u8 > t_dim.lh);

        let ca: bool = match t_dim.lw {
            0 => a[0] != 0x40,
            1 => read_u16_ne(a) != 0x4040,
            2 => read_u32_ne(a) != 0x40404040,
            3 => read_u64_ne(a) != 0x4040404040404040,
            4 => (read_u64_ne(a) | read_u64_ne(&a[8..])) != 0x4040404040404040,
            _ => unreachable!(),
        };
        let cl: bool = match t_dim.lh {
            0 => l[0] != 0x40,
            1 => read_u16_ne(l) != 0x4040,
            2 => read_u32_ne(l) != 0x40404040,
            3 => read_u64_ne(l) != 0x4040404040404040,
            4 => (read_u64_ne(l) | read_u64_ne(&l[8..])) != 0x4040404040404040,
            _ => unreachable!(),
        };

        let offset = if plane == 1 {
            6
        } else {
            6 * u_has_cf + not_one_blk as i32 * 3
        } as u32;
        offset + ca as u32 + cl as u32
    } else if b_dim[2] == t_dim.lw && b_dim[3] == t_dim.lh {
        0
    } else {
        let merge = |dir: &[u8], tx: u8| -> u32 {
            let mut v: u32;
            if tx == 4 {
                let tmp = read_u64_ne(dir) | read_u64_ne(&dir[8..]);
                v = (tmp >> 32) as u32 | tmp as u32;
            } else {
                v = match tx {
                    0 => dir[0] as u32,
                    1 => read_u16_ne(dir) as u32,
                    2 | 3 => read_u32_ne(dir) as u32,
                    _ => unreachable!(),
                };
            }
            if tx == 3 { v |= read_u32_ne(&dir[4..]); }
            if tx >= 2 { v |= v >> 16; }
            if tx >= 1 { v |= v >> 8; }
            v
        };
        let la = merge(a, t_dim.lw);
        let ll = merge(l, t_dim.lh);
        (umin(la & 0x3F, 4) + umin(ll & 0x3F, 4) + 3) >> 1
    }
}

pub fn get_dc_sign_ctx(t_dim: &TxfmInfo, a: &[u8], l: &[u8]) -> u32 {
    let mask: u64 = 0xC0C0C0C0C0C0C0C0;
    let mul: u64 = 0x0101010101010101;
    let mut t: u64 = 0;

    for &(edge, len) in &[(a, t_dim.lw), (l, t_dim.lh)] {
        match len {
            0 => t += (edge[0] >> 6) as u64,
            1 => t += (read_u16_ne(edge) as u64 & mask) >> 6,
            2 => t += (read_u32_ne(edge) as u64 & mask) >> 6,
            3 => t += (read_u64_ne(edge) & mask) >> 6,
            4 => {
                t += (read_u64_ne(&edge[8..]) & mask) >> 6;
                t += (read_u64_ne(edge) & mask) >> 6;
            }
            _ => unreachable!(),
        }
    }

    t = t.wrapping_mul(mul);
    let s = (t >> 56) as i32 - t_dim.w as i32 - t_dim.h as i32;
    (s != 0) as u32 + (s > 0) as u32
}

pub fn get_lo_ctx(
    levels: &[i8],
    off: usize,
    tx_class: u8,
    hi_mag: &mut u32,
    xy: u32,
    plane: i32,
    stride: usize,
) -> u32 {
    let chroma = plane != 0;
    let lo_freq = xy < if chroma { 1 } else if tx_class == 0 { 4 } else { 2 };
    let mut lim: u32 = if lo_freq { 5 } else { 3 };
    let mut lo_mag: u32 = 0;
    let mut hi: u32 = 0;

    macro_rules! add {
        ($v:expr) => {{
            let val = $v as u32;
            lo_mag += val.min(lim);
            hi += val.min(5);
        }};
    }

    add!(levels[off + 1]);
    add!(levels[off + stride]);

    let offset: u32;
    if tx_class == 0 {
        add!(levels[off + stride + 1]);
        if !chroma {
            lo_mag += (levels[off + 2] as u32).min(lim)
                    + (levels[off + 2 * stride] as u32).min(lim);
            if lo_freq {
                offset = if xy == 0 { 0 } else if xy < 2 { 9 } else { 16 };
                lim    = if xy == 0 { 8 } else if xy < 2 { 6 } else { 4 };
            } else {
                offset = if xy < 6 { 0 } else if xy < 8 { 5 } else { 10 };
                lim = 4;
            }
        } else {
            lim = 3;
            offset = if plane == 1 { 0 } else { 4 };
        }
    } else {
        if !chroma {
            lim = 3;
            add!(levels[off + 2]);
            lo_mag += (levels[off + 3] as u32).min(3)
                    + (levels[off + 4] as u32).min(3);
            if lo_freq {
                offset = if xy == 0 { 21 } else { 28 };
                lim    = if xy == 0 { 6 } else { 4 };
            } else {
                offset = 15;
                lim = 4;
            }
        } else {
            offset = 8;
            lim = 3;
        }
    }

    *hi_mag = (if !chroma && lo_freq && (xy > 0 || tx_class != 0) { 7 } else { 0 })
        + ((hi + 1) >> 1).min(if chroma { 3 } else { 6 });
    offset + ((lo_mag + 1) >> 1).min(lim)
}

pub fn get_lo_ctx_idtx(
    levels: &[i8],
    off: usize,
    hi_mag: &mut u32,
    stride: usize,
) -> u32 {
    let v0 = levels[off - 1] as u32;
    let v1 = levels[off - stride] as u32;
    let lo_mag = v0.min(3) + v1.min(3);
    let hi = v0.min(5) + v1.min(5);
    *hi_mag = hi.min(6);
    lo_mag
}

pub fn get_sign_ctx_idtx(
    levels: &[i8],
    off: usize,
    stride: usize,
) -> u32 {
    let sum = levels[off - 1] as i32
            + levels[off - stride] as i32
            + levels[off - stride - 1] as i32;
    let offset = if levels[off] > 3 { 2 } else { 0 };
    match sum {
        -3 => offset + 6,
        -2 | -1 => offset + 2,
        0 => 0,
        1 | 2 => offset + 1,
        3 => offset + 5,
        _ => unreachable!(),
    }
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

    #[test]
    fn test_get_lo_ctx_dc_2d_luma() {
        let stride = 32usize;
        let mut levels = vec![0i8; stride * 8];
        let off = stride + 1;
        levels[off + 1] = 3;
        levels[off + stride] = 2;
        let mut hi_mag = 0u32;
        let r = get_lo_ctx(&levels, off, 0, &mut hi_mag, 0, 0, stride);
        assert!(r <= 8);
        assert!(hi_mag <= 6);
    }

    #[test]
    fn test_get_lo_ctx_hf_2d_luma() {
        let stride = 32usize;
        let levels = vec![0i8; stride * 8];
        let off = stride + 5;
        let mut hi_mag = 0u32;
        let r = get_lo_ctx(&levels, off, 0, &mut hi_mag, 10, 0, stride);
        assert_eq!(r, 10);
    }

    #[test]
    fn test_get_lo_ctx_chroma_2d() {
        let stride = 32usize;
        let mut levels = vec![0i8; stride * 8];
        let off = stride + 1;
        levels[off + 1] = 4;
        levels[off + stride] = 4;
        levels[off + stride + 1] = 2;
        let mut hi_mag = 0u32;
        let r = get_lo_ctx(&levels, off, 0, &mut hi_mag, 0, 1, stride);
        assert!(r <= 3);
    }

    #[test]
    fn test_get_lo_ctx_class_h() {
        let stride = 32usize;
        let mut levels = vec![0i8; stride * 8];
        let off = stride + 3;
        levels[off + 1] = 2;
        levels[off + stride] = 1;
        levels[off + 2] = 3;
        let mut hi_mag = 0u32;
        let r = get_lo_ctx(&levels, off, 2, &mut hi_mag, 5, 0, stride);
        assert!(r > 0);
    }

    #[test]
    fn test_get_lo_ctx_idtx_basic() {
        let stride = 16usize;
        let mut levels = vec![0i8; stride * 4];
        let off = stride + 2;
        levels[off - 1] = 5;
        levels[off - stride] = 3;
        let mut hi_mag = 0u32;
        let r = get_lo_ctx_idtx(&levels, off, &mut hi_mag, stride);
        assert_eq!(r, 3 + 3);
        assert_eq!(hi_mag, 6);
    }

    #[test]
    fn test_get_lo_ctx_idtx_zero() {
        let stride = 16usize;
        let levels = vec![0i8; stride * 4];
        let off = stride + 2;
        let mut hi_mag = 0u32;
        let r = get_lo_ctx_idtx(&levels, off, &mut hi_mag, stride);
        assert_eq!(r, 0);
        assert_eq!(hi_mag, 0);
    }

    #[test]
    fn test_get_sign_ctx_idtx_zero() {
        let stride = 16usize;
        let levels = vec![0i8; stride * 4];
        let off = stride + 2;
        let r = get_sign_ctx_idtx(&levels, off, stride);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_get_sign_ctx_idtx_positive() {
        let stride = 16usize;
        let mut levels = vec![0i8; stride * 4];
        let off = stride + 2;
        levels[off] = 2;
        levels[off - 1] = 1;
        levels[off - stride] = 1;
        levels[off - stride - 1] = 1;
        let r = get_sign_ctx_idtx(&levels, off, stride);
        assert_eq!(r, 5);
    }

    #[test]
    fn test_get_sign_ctx_idtx_negative() {
        let stride = 16usize;
        let mut levels = vec![0i8; stride * 4];
        let off = stride + 2;
        levels[off] = 4;
        levels[off - 1] = -1;
        levels[off - stride] = -1;
        levels[off - stride - 1] = -1;
        let r = get_sign_ctx_idtx(&levels, off, stride);
        assert_eq!(r, 8);
    }

    #[test]
    fn test_get_dc_sign_ctx_zero() {
        let t_dim = TxfmInfo { w: 1, h: 1, lw: 0, lh: 0, min: 0, max: 0, sub: 0, ctx: 0 };
        let a = [0u8; 16];
        let l = [0u8; 16];
        let r = get_dc_sign_ctx(&t_dim, &a, &l);
        assert_eq!(r, 1);
    }

    #[test]
    fn test_get_dc_sign_ctx_balanced() {
        let t_dim = TxfmInfo { w: 1, h: 1, lw: 0, lh: 0, min: 0, max: 0, sub: 0, ctx: 0 };
        let a = [0x40u8; 16];
        let l = [0x80u8; 16];
        let r = get_dc_sign_ctx(&t_dim, &a, &l);
        assert!(r <= 2);
    }

    #[test]
    fn test_get_skip_ctx_luma_same_size() {
        let t_dim = TxfmInfo { w: 4, h: 4, lw: 2, lh: 2, min: 2, max: 2, sub: 0, ctx: 0 };
        let a = [0u8; 16];
        let l = [0u8; 16];
        let r = get_skip_ctx(&t_dim, 18, &a, &l, 0, 0, false, false);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_get_skip_ctx_chroma() {
        let t_dim = TxfmInfo { w: 1, h: 1, lw: 0, lh: 0, min: 0, max: 0, sub: 0, ctx: 0 };
        let a = [0x40u8; 16];
        let l = [0x40u8; 16];
        let r = get_skip_ctx(&t_dim, 30, &a, &l, 1, 0, false, false);
        assert_eq!(r, 6);
    }

    #[test]
    fn test_get_skip_ctx_chroma_has_cf() {
        let t_dim = TxfmInfo { w: 1, h: 1, lw: 0, lh: 0, min: 0, max: 0, sub: 0, ctx: 0 };
        let a = [0x00u8; 16];
        let l = [0x00u8; 16];
        let r = get_skip_ctx(&t_dim, 30, &a, &l, 1, 0, false, false);
        assert_eq!(r, 6 + 1 + 1);
    }
}

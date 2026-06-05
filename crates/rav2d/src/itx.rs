use crate::intops::{iclip, imin};
use crate::itx_1d::{TX1D_FNS, inv_wht_wht_4x4, residual_add_8bpc};
use crate::scan::LAST_EOB_PER_COL;
use crate::tables::{TX_SHIFT, TXFM_DIMENSIONS};

const WHT_WHT: u32 = 6 | (6 << 5);

pub fn inv_txfm_add_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    coeff: &mut [i32],
    txtp: u32,
    eob: i32,
    tx: usize,
) {
    if txtp & 0xFF == WHT_WHT {
        assert!(tx == 0);
        let mut tmp = [0i32; 16];
        inv_wht_wht_4x4(&coeff[..16].try_into().unwrap(), &mut tmp);
        coeff[..16].fill(0);
        let dpcm_flag = (txtp >> 8) as u8;
        residual_add_8bpc(&mut dst[dst_off..], stride, &tmp, 4, 4, 0, 0, dpcm_flag);
        return;
    }

    let t_dim = &TXFM_DIMENSIONS[tx];
    let tx_sh = &TX_SHIFT[tx];
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let is_rect2 = ((t_dim.lw + t_dim.lh) & 1) != 0;

    if eob + txtp as i32 == 0 {
        let shift_p1 = tx_sh[0] as i32;
        let shift = shift_p1 + tx_sh[1] as i32 - 12;
        let rnd = (1 << (shift - 1)) + shift_p1 - 6;
        let mut dc = coeff[0];
        coeff[0] = 0;
        if is_rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc + rnd) >> shift;
        for y in 0..h {
            for x in 0..w {
                dst[dst_off + y * stride + x] =
                    iclip(dst[dst_off + y * stride + x] as i32 + dc, 0, 255) as u8;
            }
        }
        return;
    }

    let first_1d_fn = TX1D_FNS[t_dim.lw as usize][(txtp & 7) as usize].unwrap();
    let second_1d_fn = TX1D_FNS[t_dim.lh as usize][((txtp >> 5) & 7) as usize].unwrap();
    let sh = imin(h as i32, 32) as usize;
    let sw = imin(w as i32, 32) as usize;
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;

    let mut tmp = [0i32; 32 * 32];
    let mut col = 0usize;
    let tx_class = (txtp >> 3) & 0x3;

    if tx_class == 0 {
        let off = LAST_EOB_PER_COL.offset[tx] as usize;
        let last_eob = &LAST_EOB_PER_COL.table[off..];
        let mut ei = 0usize;
        loop {
            for x in 0..sw {
                let v = coeff[col + x * sh];
                tmp[col * sw + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            }
            first_1d_fn(&mut tmp[col * sw..], 1);
            col += 1;
            if col & 3 == 0 {
                if eob > last_eob[ei] as i32 {
                    ei += 1;
                } else {
                    break;
                }
            }
        }
    } else {
        let last_nz_col = if tx_class == 2 {
            imin(sh as i32 - 1, eob) as usize
        } else if tx_class == 3 {
            (eob as usize) >> (t_dim.lw as usize + 2)
        } else {
            sh - 1
        };
        loop {
            for x in 0..sw {
                let v = coeff[col + x * sh];
                tmp[col * sw + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            }
            first_1d_fn(&mut tmp[col * sw..], 1);
            col += 1;
            if col > last_nz_col {
                break;
            }
        }
    }

    if col < sh {
        tmp[col * sw..sh * sw].fill(0);
    }
    coeff[..sw * sh].fill(0);

    if std::env::var("RAV2D_ITXTMP").is_ok() && tx == 1 && txtp == 165 {
        let mut s = String::from("ITXTMP-pass1");
        for i in 0..64 {
            s.push_str(&format!(" {}", tmp[i]));
        }
        eprintln!("{s}");
    }

    let shift0 = tx_sh[0] as i32;
    let rnd0 = (1 << shift0) >> 1;
    for i in 0..sw * sh {
        tmp[i] = iclip((tmp[i] + rnd0) >> shift0, row_clip_min, row_clip_max);
    }

    for x in 0..sw {
        second_1d_fn(&mut tmp[x..], sw);
    }

    if std::env::var("RAV2D_ITXTMP").is_ok() && tx == 1 && txtp == 165 {
        let mut s = format!("ITXTMP-pass2 eob={} cf0={}", eob, coeff[0]);
        for i in 0..64 {
            s.push_str(&format!(" {}", tmp[i]));
        }
        eprintln!("{s}");
    }

    let shift1 = tx_sh[1] as i32;
    let rnd1 = (1 << shift1) >> 1;

    if w > sw {
        if h > sh {
            let mut ci = 0;
            for y in (0..h).step_by(2) {
                for x in (0..w).step_by(2) {
                    let cf = (tmp[ci] + rnd1) >> shift1;
                    ci += 1;
                    let d0 = dst_off + y * stride + x;
                    let d1 = dst_off + (y + 1) * stride + x;
                    dst[d0] = iclip(dst[d0] as i32 + cf, 0, 255) as u8;
                    dst[d0 + 1] = iclip(dst[d0 + 1] as i32 + cf, 0, 255) as u8;
                    dst[d1] = iclip(dst[d1] as i32 + cf, 0, 255) as u8;
                    dst[d1 + 1] = iclip(dst[d1 + 1] as i32 + cf, 0, 255) as u8;
                }
            }
        } else {
            let mut ci = 0;
            for y in 0..h {
                for x in (0..w).step_by(2) {
                    let cf = (tmp[ci] + rnd1) >> shift1;
                    ci += 1;
                    let d = dst_off + y * stride + x;
                    dst[d] = iclip(dst[d] as i32 + cf, 0, 255) as u8;
                    dst[d + 1] = iclip(dst[d + 1] as i32 + cf, 0, 255) as u8;
                }
            }
        }
    } else if h > sh {
        let mut ci = 0;
        for y in (0..h).step_by(2) {
            for x in 0..w {
                let cf = (tmp[ci] + rnd1) >> shift1;
                ci += 1;
                let d0 = dst_off + y * stride + x;
                let d1 = dst_off + (y + 1) * stride + x;
                dst[d0] = iclip(dst[d0] as i32 + cf, 0, 255) as u8;
                dst[d1] = iclip(dst[d1] as i32 + cf, 0, 255) as u8;
            }
        }
    } else {
        let dpcm_flag = (txtp >> 8) as u8;
        residual_add_8bpc(
            &mut dst[dst_off..],
            stride,
            &tmp,
            w,
            h,
            rnd1,
            shift1,
            dpcm_flag,
        );
    }
}

pub fn cctx_8bpc(u: &mut [i32], v: &mut [i32], angle: &[i16; 3], sz: usize) {
    use crate::itx_1d::cctx;
    cctx(u, v, angle, sz, 8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_txfm_add_dc_only_4x4() {
        let mut dst = vec![128u8; 16 * 4];
        let mut coeff = vec![0i32; 16];
        coeff[0] = 64;
        inv_txfm_add_8bpc(&mut dst, 0, 16, &mut coeff, 0, 0, 0);
        assert_eq!(coeff[0], 0);
        assert_ne!(dst[0], 128);
    }

    #[test]
    fn test_inv_txfm_add_zero_coeff() {
        let mut dst = vec![128u8; 16 * 8];
        let mut coeff = vec![0i32; 64];
        inv_txfm_add_8bpc(&mut dst, 0, 16, &mut coeff, 0, 0, 1);
        for &v in &dst[..8 * 16] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_inv_txfm_add_wht_4x4() {
        let mut dst = vec![128u8; 16 * 4];
        let mut coeff = vec![0i32; 16];
        coeff[0] = 256;
        inv_txfm_add_8bpc(&mut dst, 0, 16, &mut coeff, WHT_WHT, 0, 0);
        assert!(coeff[..16].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_inv_txfm_add_8x8() {
        let mut dst = vec![128u8; 16 * 8];
        let mut coeff = vec![0i32; 64];
        coeff[0] = 100;
        inv_txfm_add_8bpc(&mut dst, 0, 16, &mut coeff, 0, 0, 1);
        assert_ne!(dst[0], 128);
    }

    #[test]
    fn test_inv_txfm_add_clamp() {
        let mut dst = vec![250u8; 16 * 4];
        let mut coeff = vec![0i32; 16];
        coeff[0] = 10000;
        inv_txfm_add_8bpc(&mut dst, 0, 16, &mut coeff, 0, 0, 0);
        assert_eq!(dst[0], 255);
    }
}

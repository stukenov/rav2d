use crate::headers::PixelLayout;
use crate::intops::{apply_sign, iclip, imax, imin, ulog2, umin};
use crate::levels::{BlockSize, IntraPredMode, Mv, N_BS_SIZES, RefPair, txtp};
use crate::mc::OpflRegressionData;
use crate::msac::MsacContext;
use crate::refmvs::{self, INVALID_TRAJ, TemporalBlock};
use crate::scan::SCANS;
use crate::tables::{
    BLOCK_DIMENSIONS, DIV_RECIP, MODE_TO_ANGLE_MAP, TXFM_DIMENSIONS, TXTP_FROM_UVMODE, TxfmInfo,
};
use crate::warpmv::resolve_divisor_32;

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
        // `den` is non-negative for valid streams, but hostile input can wrap
        // the CfL sums negative. Release-build dav2d feeds the negative int
        // through ulog2's unsigned conversion (derivation.h asserts only in
        // debug); `den as u32` reproduces that deterministic behaviour.
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
        let not_one_blk = (b_dim[2] - (b_dim[2] != 0 && ss_hor) as u8 > t_dim.lw)
            || (b_dim[3] - (b_dim[3] != 0 && ss_ver) as u8 > t_dim.lh);

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
                    2 | 3 => read_u32_ne(dir),
                    _ => unreachable!(),
                };
            }
            if tx == 3 {
                v |= read_u32_ne(&dir[4..]);
            }
            if tx >= 2 {
                v |= v >> 16;
            }
            if tx >= 1 {
                v |= v >> 8;
            }
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
    let lo_freq = xy
        < if chroma {
            1
        } else if tx_class == 0 {
            4
        } else {
            2
        };
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
            lo_mag +=
                (levels[off + 2] as u32).min(lim) + (levels[off + 2 * stride] as u32).min(lim);
            if lo_freq {
                offset = if xy == 0 {
                    0
                } else if xy < 2 {
                    9
                } else {
                    16
                };
                lim = if xy == 0 {
                    8
                } else if xy < 2 {
                    6
                } else {
                    4
                };
            } else {
                offset = if xy < 6 {
                    0
                } else if xy < 8 {
                    5
                } else {
                    10
                };
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
            lo_mag += (levels[off + 3] as u32).min(3) + (levels[off + 4] as u32).min(3);
            if lo_freq {
                offset = if xy == 0 { 21 } else { 28 };
                lim = if xy == 0 { 6 } else { 4 };
            } else {
                offset = 15;
                lim = 4;
            }
        } else {
            offset = 8;
            lim = 3;
        }
    }

    *hi_mag = (if !chroma && lo_freq && (xy > 0 || tx_class != 0) {
        7
    } else {
        0
    }) + ((hi + 1) >> 1).min(if chroma { 3 } else { 6 });
    offset + ((lo_mag + 1) >> 1).min(lim)
}

pub fn get_lo_ctx_idtx(levels: &[i8], off: usize, hi_mag: &mut u32, stride: usize) -> u32 {
    let v0 = levels[off - 1] as u32;
    let v1 = levels[off - stride] as u32;
    let lo_mag = v0.min(3) + v1.min(3);
    let hi = v0.min(5) + v1.min(5);
    *hi_mag = hi.min(6);
    lo_mag
}

pub fn get_sign_ctx_idtx(levels: &[i8], off: usize, stride: usize) -> u32 {
    let sum =
        levels[off - 1] as i32 + levels[off - stride] as i32 + levels[off - stride - 1] as i32;
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

pub fn get_mask(
    mask: &mut [u8],
    stride: usize,
    bx4: i32,
    x4: i32,
    by4: i32,
    y4: i32,
    mv: &[Mv; 2],
    h_subpel_bits: i32,
    v_subpel_bits: i32,
    bw4: i32,
    bh4: i32,
    iw: i32,
    ih: i32,
) -> bool {
    let (mv0, mv1) = unsafe { (mv[0].c, mv[1].c) };
    let x0 = (bx4 + x4) * 4 + (mv0.x >> h_subpel_bits);
    let y0 = (by4 + y4) * 4 + (mv0.y >> v_subpel_bits);
    let x1 = (bx4 + x4) * 4 + (mv1.x >> h_subpel_bits);
    let y1 = (by4 + y4) * 4 + (mv1.y >> v_subpel_bits);
    if x0 < 0
        || x1 < 0
        || y0 < 0
        || y1 < 0
        || x0 + bw4 * 4 >= iw
        || x1 + bw4 * 4 >= iw
        || y0 + bh4 * 4 >= ih
        || y1 + bh4 * 4 >= ih
    {
        let off = (y4 as usize * stride + x4 as usize) * 4;
        gen_mask(
            &mut mask[off..],
            stride,
            bw4 * 4,
            bh4 * 4,
            x0,
            y0,
            x1,
            y1,
            iw as u32,
            ih as u32,
        );
        return true;
    }
    false
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct OpflMvDelta {
    pub x: i8,
    pub y: i8,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct OpflMvDeltaBlock {
    pub d: [OpflMvDelta; 2],
}

pub fn opfl_mv_adj(r: &OpflRegressionData, dd: &mut OpflMvDeltaBlock, d: [i8; 2]) {
    let mut su2 = r.su2;
    let mut suv = r.suv;
    let mut sv2 = r.sv2;
    let mut suw = r.suw;
    let mut svw = r.svw;
    let nbits_su2 = 1 + ulog2((su2 + (su2 == 0) as i32) as u32);
    let nbits_sv2 = 1 + ulog2((sv2 + (sv2 == 0) as i32) as u32);
    let nbits_suv = 1 + ulog2((suv.abs() + (suv == 0) as i32) as u32);
    let nbits_suw = 1 + ulog2((suw.abs() + (suw == 0) as i32) as u32);
    let nbits_svw = 1 + ulog2((svw.abs() + (svw == 0) as i32) as u32);
    let nbits_max = imax(
        nbits_su2 + nbits_sv2,
        imax(
            imax(nbits_sv2 + nbits_suw, nbits_suv + nbits_svw),
            imax(nbits_su2 + nbits_svw, nbits_suv + nbits_suw),
        ),
    );
    let rbits = imax(0, nbits_max - 23) >> 1;
    if rbits != 0 {
        let rnd = (1 << rbits) >> 1;
        su2 = (su2 + rnd) >> rbits;
        sv2 = (sv2 + rnd) >> rbits;
        suv = (suv + rnd - (suv < 0) as i32) >> rbits;
        suw = (suw + rnd - (suw < 0) as i32) >> rbits;
        svw = (svw + rnd - (svw < 0) as i32) >> rbits;
    }
    let det = su2 * sv2 - suv * suv;
    if det > 0 {
        let mut s = [sv2 * suw - suv * svw, su2 * svw - suv * suw];
        let mut shift = 0i32;
        let idet = resolve_divisor_32(det as u32, &mut shift);
        let idet_bits = ulog2(idet as u32);
        for i in 0..2 {
            if s[i] == 0 {
                continue;
            }
            let mut abss = s[i].abs();
            let rb = imax(0, ulog2(abss as u32) + idet_bits - 22);
            if rb > 0 {
                abss = (abss + ((1 << rb) >> 1)) >> rb;
            }
            let ibits = 3 + rb - shift;
            if ibits >= 0 {
                abss = abss * idet * (1 << ibits);
            } else {
                abss = (abss * idet + ((1 << -ibits) >> 1)) >> -ibits;
            }
            s[i] = apply_sign(abss, s[i]);
        }
        dd.d[0].x = -iclip(d[0] as i32 * s[0], -16, 16) as i8;
        dd.d[0].y = -iclip(d[0] as i32 * s[1], -16, 16) as i8;
        dd.d[1].x = iclip(d[1] as i32 * s[0], -16, 16) as i8;
        dd.d[1].y = iclip(d[1] as i32 * s[1], -16, 16) as i8;
    } else {
        *dd = OpflMvDeltaBlock::default();
    }
}

pub fn scaledown_16pel_mv_for_chroma(mv: &mut [Mv; 2], layout: PixelLayout) {
    match layout {
        PixelLayout::I420 => {
            for i in 0..2 {
                unsafe {
                    let y = mv[i].c.y;
                    mv[i].c.y = (y + (y > 0) as i32) >> 1;
                }
            }
            for i in 0..2 {
                unsafe {
                    let x = mv[i].c.x;
                    mv[i].c.x = (x + (x > 0) as i32) >> 1;
                }
            }
        }
        PixelLayout::I422 => {
            for i in 0..2 {
                unsafe {
                    let x = mv[i].c.x;
                    mv[i].c.x = (x + (x > 0) as i32) >> 1;
                }
            }
        }
        _ => {}
    }
}

pub fn scaleup_8pel_mv_for_chroma(mv: &mut [Mv; 2], layout: PixelLayout) {
    match layout {
        PixelLayout::I444 => {
            for i in 0..2 {
                unsafe {
                    mv[i].c.x <<= 1;
                }
            }
            for i in 0..2 {
                unsafe {
                    mv[i].c.y <<= 1;
                }
            }
        }
        PixelLayout::I422 => {
            for i in 0..2 {
                unsafe {
                    mv[i].c.y <<= 1;
                }
            }
        }
        _ => {}
    }
}

pub fn update_temporal(
    t_dst: &mut [TemporalBlock],
    t_stride: usize,
    w8: usize,
    h8: usize,
    r: RefPair,
    mv: &[Mv; 2],
    swap: bool,
) {
    let s0 = swap as usize;
    let s1 = (!swap) as usize;
    let mut t_src = TemporalBlock::default();
    unsafe {
        t_src.r#ref.r[0] = r.r[s0];
        t_src.r#ref.r[1] = r.r[s1];
    }
    t_src.mv = refmvs::TemporalBlockMv {
        mv: [refmvs::quantize_mv(mv[s0]), refmvs::quantize_mv(mv[s1])],
    };
    unsafe {
        let mv0_n = t_src.mv.mv[0].n;
        let mv1_n = t_src.mv.mv[1].n;
        if mv0_n == INVALID_TRAJ {
            if mv1_n == INVALID_TRAJ {
                t_src.r#ref.pair = -1;
            } else {
                t_src.mv.mv[0] = t_src.mv.mv[1];
                t_src.r#ref.r[0] = t_src.r#ref.r[1];
            }
        } else if mv1_n == INVALID_TRAJ {
            t_src.mv.mv[1] = t_src.mv.mv[0];
            t_src.r#ref.r[1] = t_src.r#ref.r[0];
        }
    }
    for y in 0..h8 {
        let row = &mut t_dst[y * t_stride..y * t_stride + w8];
        for x in 0..w8 {
            row[x] = t_src;
        }
    }
}

pub struct DecodeCoefParams<'a> {
    pub tx: usize,
    pub bs: usize,
    pub plane: i32,
    pub intra: bool,
    pub fsc: bool,
    pub lossless: bool,
    pub sdp_active: bool,
    pub y_mode: usize,
    pub uv_mode: usize,
    pub seg_id: usize,
    pub seq_fsc: bool,
    pub seq_ist: [bool; 2],
    pub seq_cctx: bool,
    pub chroma_dctonly: bool,
    pub reduced_txtp_set: i32,
    pub tcq_enabled: bool,
    pub layout: PixelLayout,
    pub u_has_cf: i32,
    pub cbx: i32,
    pub cby: i32,
    pub luma_fsc_map: &'a [u8],
    pub dq_tbl: [u32; 2],
    pub bitdepth: u32,
    pub qm: Option<&'a [u8]>,
    pub ss_hor: bool,
    pub ss_ver: bool,
}

use crate::cdf::{CdfCoefContext, CdfModeContext};

pub fn decode_coefs(
    msac: &mut MsacContext,
    coef: &mut CdfCoefContext,
    mode: &mut CdfModeContext,
    a: &[u8],
    l: &[u8],
    p: &DecodeCoefParams,
    cf: &mut [i32],
    txtp: &mut u16,
    res_ctx: &mut u8,
) -> i32 {
    let t_dim = &TXFM_DIMENSIONS[p.tx];
    let chroma = p.plane != 0;
    let cf_max = !((!127u32) << p.bitdepth) as i32;

    if env_flag!("RAV2D_CF") {
        eprintln!("CF pl={} ENTER tx={} rng={}", p.plane, p.tx, msac.dbg_rng());
    }

    // skip detection
    let sctx = if p.fsc && !chroma && p.seq_fsc {
        9
    } else {
        get_skip_ctx(t_dim, p.bs, a, l, p.plane, p.u_has_cf, p.ss_hor, p.ss_ver) as usize
    };
    let all_skip = if p.plane == 2 {
        if env_flag!("RAV2D_CF") {
            eprintln!(
                "CF pl=2 SKIPCDF cdf0={} rng={}",
                coef.skip_v(sctx)[0],
                msac.dbg_rng()
            );
        }
        msac.decode_bool_adapt(coef.skip_v(sctx))
    } else {
        let i = if !p.intra || p.fsc { 1 } else { 0 };
        if env_flag!("RAV2D_CF") {
            eprintln!(
                "CF pl={} SKIPCDF i={} txctx={} sctx={} cdf0={} rng={}",
                p.plane,
                i,
                t_dim.ctx,
                sctx,
                coef.skip(i, t_dim.ctx as usize, sctx)[0],
                msac.dbg_rng()
            );
        }
        msac.decode_bool_adapt(coef.skip(i, t_dim.ctx as usize, sctx))
    };
    if env_flag!("RAV2D_CF") {
        eprintln!(
            "CF pl={} tx={} sctx={} all_skip={} rng={}",
            p.plane,
            p.tx,
            sctx,
            all_skip,
            msac.dbg_rng()
        );
    }
    if all_skip != 0 {
        *res_ctx = 0x40;
        *txtp = if !chroma && p.fsc {
            txtp::IDTX as u16
        } else {
            (p.lossless as u16) * txtp::WHT_WHT as u16
        };
        return -1;
    }

    // EOB bin decoding
    let slw = imin(t_dim.lw as i32, 3) as usize;
    let slh = imin(t_dim.lh as i32, 3) as usize;
    let tx2dszctx = slw + slh;
    let eob_ctx = if chroma { 2 } else { (!p.intra) as usize };

    let mut eob: i32 = match tx2dszctx {
        0 => msac.decode_symbol_adapt(coef.eob_bin_16(eob_ctx), 4) as i32,
        1 => msac.decode_symbol_adapt(coef.eob_bin_32(eob_ctx), 5) as i32,
        2 => msac.decode_symbol_adapt(coef.eob_bin_64(eob_ctx), 6) as i32,
        3 => msac.decode_symbol_adapt(coef.eob_bin_128(eob_ctx), 7) as i32,
        4 => {
            let mut e = msac.decode_symbol_adapt(coef.eob_bin_256(eob_ctx), 7) as i32;
            if e == 7 {
                e += msac.decode_bools_bypass(1) as i32;
            }
            e
        }
        5 => {
            let mut e = msac.decode_symbol_adapt(coef.eob_bin_512(eob_ctx), 7) as i32;
            if e == 7 {
                e += msac.decode_bools_bypass(2) as i32;
                if e == 10 {
                    return i32::MIN;
                }
            }
            e
        }
        _ => {
            let mut e = msac.decode_symbol_adapt(coef.eob_bin_1024(eob_ctx), 7) as i32;
            if e == 7 {
                e += msac.decode_bools_bypass(2) as i32;
            }
            e
        }
    };

    if eob > 1 {
        let eob_hi_bit = msac.decode_bool_adapt(coef.eob_hi_bit()) as i32;
        let eob_bin = eob - 2;
        eob = eob_hi_bit | 2;
        if eob_bin != 0 {
            eob = (eob << eob_bin) | msac.decode_bools_bypass(eob_bin as u32) as i32;
        }
    }

    if env_flag!("RAV2D_CF") {
        eprintln!("CF pl={} eob={} rng={}", p.plane, eob, msac.dbg_rng());
    }
    if env_flag!("RAV2D_CF_TRACE") {
        eprintln!(
            "DCF cby={} cbx={} pl={} tx={} sctx={} uhc={} skip={} eob={} rng={}",
            p.cby,
            p.cbx,
            p.plane,
            p.tx,
            sctx,
            p.u_has_cf,
            all_skip,
            eob,
            msac.dbg_rng()
        );
    }

    // transform type selection
    static TXTP_LONG_TBL: [[[u8; 4]; 2]; 2] = [
        [
            [txtp::V_DCT, txtp::V_ADST, txtp::V_FLIPADST, txtp::IDTX],
            [txtp::H_DCT, txtp::H_ADST, txtp::H_FLIPADST, txtp::IDTX],
        ],
        [
            [
                txtp::DCT_DCT,
                txtp::ADST_DCT,
                txtp::FLIPADST_DCT,
                txtp::H_DCT,
            ],
            [
                txtp::DCT_DCT,
                txtp::DCT_ADST,
                txtp::DCT_FLIPADST,
                txtp::V_DCT,
            ],
        ],
    ];

    if p.lossless {
        if chroma {
            if p.intra {
                let y_fsc = if !p.sdp_active {
                    p.fsc
                } else {
                    let idx = (p.cby & 15) as usize * 16 + (p.cbx & 15) as usize;
                    p.luma_fsc_map[idx] != 0
                };
                *txtp = if y_fsc {
                    txtp::IDTX as u16
                } else {
                    txtp::WHT_WHT as u16
                };
            } else {
                *txtp &= 0xe7; // IDTX_INV -> IDTX
            }
        } else if p.intra {
            *txtp = if p.fsc {
                txtp::IDTX as u16
            } else {
                txtp::WHT_WHT as u16
            };
        } else if t_dim.max == 0 {
            *txtp = if msac.decode_bool_adapt(mode.txtp_lossless()) != 0 {
                txtp::IDTX as u16
            } else {
                txtp::WHT_WHT as u16
            };
        } else {
            *txtp = txtp::IDTX as u16;
        }
    } else if chroma {
        if p.chroma_dctonly {
            *txtp = txtp::DCT_DCT as u16;
        } else {
            if p.intra {
                *txtp = TXTP_FROM_UVMODE[p.uv_mode] as u16;
            }
            let t = *txtp as u8;
            if (t_dim.w >= 8 && t & 0x02 != 0)
                || (t_dim.h >= 8 && t & 0x40 != 0)
                || (p.tx == 2 /* TX_16X16 */
                    && ((t & 0x47 == 0x41) || (t & 0xe2 == 0x22)))
            {
                *txtp = txtp::DCT_DCT as u16;
            } else if t == txtp::IDTX_INV {
                *txtp = txtp::IDTX as u16;
            }
        }
    } else if p.intra {
        if t_dim.sub == 3 {
            *txtp = txtp::DCT_DCT as u16;
        } else if p.fsc {
            *txtp = txtp::IDTX as u16;
        } else if eob == 0 || p.tx == 3 {
            *txtp = txtp::DCT_DCT as u16;
        } else if t_dim.max >= 3 {
            let long_dct = t_dim.max == 4 || msac.decode_bool_adapt(mode.txtp_long32_dct(0)) != 0;
            let short_idx =
                msac.decode_symbol_adapt(mode.txtp_intra_short_1d(t_dim.min as usize), 3) as usize;
            let wh = (t_dim.w < t_dim.h) as usize;
            *txtp = TXTP_LONG_TBL[long_dct as usize][wh][short_idx] as u16;
        } else if p.reduced_txtp_set == 2 {
            *txtp = txtp::DCT_DCT as u16;
        } else {
            let sz_ctx = ((t_dim.lw + t_dim.lh) >> 1) as usize;
            let tx_idx = if p.reduced_txtp_set != 0 {
                msac.decode_bool_adapt(mode.txtp_ext_reduced(t_dim.min as usize)) as usize
            } else {
                msac.decode_symbol_adapt(mode.txtp_ext(t_dim.min as usize), 6) as usize
            };
            static MD_IDX2TYPE: [[[u8; 7]; 13]; 3] = [
                [
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::V_DCT,
                        txtp::V_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_DCT,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::V_ADST,
                        txtp::V_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_DCT,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::V_DCT,
                        txtp::V_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::V_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::DCT_ADST,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::V_ADST,
                        txtp::H_ADST,
                    ],
                ],
                [
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::FLIPADST_ADST,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::H_ADST,
                    ],
                ],
                [
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::FLIPADST_ADST,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::FLIPADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::V_ADST,
                    ],
                ],
            ];
            *txtp = MD_IDX2TYPE[sz_ctx][p.y_mode][tx_idx] as u16;
        }
    } else {
        // inter
        if t_dim.sub == 3 {
            *txtp = txtp::DCT_DCT as u16;
        } else {
            let y = eob >> (2 + slw as i32);
            let x = eob & ((4 << slw) - 1);
            let xy = x + y;
            let ww = imin(8, t_dim.w as i32);
            let hh = imin(8, t_dim.h as i32);
            let ctx = if xy < 2 {
                1usize
            } else if xy > 4 * (ww + hh) - 4 {
                2
            } else {
                0
            };
            if p.tx == 3 {
                *txtp = if msac.decode_bool_adapt(mode.txtp_inter_dct_idtx(ctx, 3)) != 0 {
                    txtp::DCT_DCT as u16
                } else {
                    txtp::IDTX as u16
                };
            } else if t_dim.max >= 3 {
                let long_dct =
                    t_dim.max == 4 || msac.decode_bool_adapt(mode.txtp_long32_dct(1)) != 0;
                let short_idx = msac
                    .decode_symbol_adapt(mode.txtp_inter_short_1d(ctx, t_dim.min as usize), 3)
                    as usize;
                let wh = (t_dim.w < t_dim.h) as usize;
                *txtp = TXTP_LONG_TBL[long_dct as usize][wh][short_idx] as u16;
            } else if p.reduced_txtp_set == 1 || p.reduced_txtp_set == 2 {
                *txtp = if msac.decode_bool_adapt(mode.txtp_inter_dct_idtx(ctx, t_dim.min as usize))
                    != 0
                {
                    txtp::DCT_DCT as u16
                } else {
                    txtp::IDTX as u16
                };
            } else if p.reduced_txtp_set == 3 {
                let tx_idx = msac
                    .decode_symbol_adapt(mode.txtp_inter_dct_idtx_iddct(ctx, t_dim.min as usize), 3)
                    as usize;
                static TXTP_DCT_IDTX_IDDCT: [u8; 4] =
                    [txtp::DCT_DCT, txtp::V_DCT, txtp::H_DCT, txtp::IDTX];
                *txtp = TXTP_DCT_IDTX_IDDCT[tx_idx] as u16;
            } else {
                let setidx = (p.tx == 2) as usize;
                let set =
                    msac.decode_bool_adapt(mode.txtp_inter_tx_set(setidx, ctx, t_dim.min as usize))
                        as usize;
                let t = if set == 0 {
                    msac.decode_symbol_adapt(mode.txtp_inter_set0(setidx, ctx), 7) as usize
                } else if setidx != 0 {
                    msac.decode_symbol_adapt(mode.txtp_inter_set2(ctx), 3) as usize + 8
                } else {
                    msac.decode_symbol_adapt(mode.txtp_inter_set1(ctx), 7) as usize + 8
                };
                static TXTP_INV_TBL: [[u8; 16]; 2] = [
                    [
                        txtp::IDTX,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::V_ADST,
                        txtp::H_ADST,
                        txtp::V_FLIPADST,
                        txtp::H_FLIPADST,
                        txtp::DCT_DCT,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::IDTX,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::DCT_DCT,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        0,
                        0,
                        0,
                        0,
                    ],
                ];
                *txtp = TXTP_INV_TBL[setidx][t] as u16;
            }
        }
    }

    let tx_class = txtp::class(*txtp as u8);

    // secondary transform (IST)
    let mut stx_type: u32 = 0;
    if p.seq_ist[(!p.intra) as usize] && !chroma {
        if p.intra {
            if eob >= 1
                && p.y_mode != IntraPredMode::PaethPred as usize
                && (*txtp as u8 == txtp::DCT_DCT || *txtp as u8 == txtp::ADST_ADST)
            {
                let lim = if p.tx == 1 && *txtp as u8 == txtp::DCT_DCT {
                    20
                } else if t_dim.min >= 1 {
                    if *txtp as u8 == txtp::DCT_DCT { 32 } else { 20 }
                } else {
                    8
                };
                stx_type = (eob < lim) as u32;
            }
        } else {
            stx_type =
                (t_dim.min >= 2 && *txtp as u8 == txtp::DCT_DCT && (3..32).contains(&eob)) as u32;
        }
        if stx_type != 0 {
            stx_type =
                msac.decode_symbol_adapt(mode.stx((!p.intra) as usize, t_dim.min as usize), 3);
            if stx_type != 0 && p.intra {
                let mut stx_set: u32;
                if t_dim.min >= 1 && *txtp as u8 == txtp::ADST_ADST {
                    static INV_MAP_ADST: [[u8; 4]; 12] = [
                        [3, 1, 0, 2],
                        [1, 3, 0, 2],
                        [1, 3, 0, 2],
                        [1, 3, 0, 2],
                        [0, 2, 3, 1],
                        [2, 1, 0, 3],
                        [2, 1, 0, 3],
                        [1, 0, 3, 2],
                        [1, 0, 3, 2],
                        [3, 1, 0, 2],
                        [1, 3, 0, 2],
                        [1, 3, 0, 2],
                    ];
                    let s = msac.decode_symbol_adapt(mode.stx_set_adst(), 3) as usize;
                    stx_set = INV_MAP_ADST[p.y_mode][s] as u32;
                } else {
                    static INV_MAP: [[u8; 7]; 12] = [
                        [6, 1, 0, 5, 4, 3, 2],
                        [1, 6, 0, 4, 2, 5, 3],
                        [1, 6, 0, 4, 2, 5, 3],
                        [2, 6, 0, 5, 1, 4, 3],
                        [3, 4, 6, 1, 0, 2, 5],
                        [4, 1, 3, 6, 0, 5, 2],
                        [4, 1, 3, 6, 0, 5, 2],
                        [5, 0, 6, 2, 1, 4, 3],
                        [5, 0, 6, 2, 1, 4, 3],
                        [6, 1, 0, 5, 4, 3, 2],
                        [1, 6, 0, 4, 2, 5, 3],
                        [1, 6, 0, 4, 2, 5, 3],
                    ];
                    let s = msac.decode_symbol_adapt(mode.stx_set(), 6) as usize;
                    stx_set = INV_MAP[p.y_mode][s] as u32;
                }
                stx_set += 7 * (*txtp as u8 == txtp::ADST_ADST) as u32;
                *txtp |= (stx_set << 10) as u16;
            }
            *txtp |= (stx_type << 8) as u16;
        }
    } else if p.seq_cctx
        && p.plane == 1
        && eob >= p.intra as i32
        && !p.lossless
        && (p.layout == PixelLayout::I420 || t_dim.max < 8)
    {
        let cctx = msac.decode_symbol_adapt(mode.cctx(), 6);
        *txtp |= (cctx << 8) as u16;
    }

    // base tokens
    let mut cul_level: u32 = 0;
    let mut dc_tok: i32;
    let tcq_en = p.tcq_enabled && !chroma && tx_class == 0 && !p.lossless;
    let mut hr_avg: i32 = 0;
    let mut tcq_state: i32 = if tcq_en { -0x80000000i32 } else { 0 };
    let has_qm = p.qm.is_some() && (*txtp as u8) < txtp::IDTX;
    let mut dq_shift = tcq_en as i32 + 3 + imax(0, t_dim.ctx as i32 - 2);
    let mut dc_sign_level: u32 = 1 << 6;

    let scan = SCANS[p.tx];

    // IDTX/FSC path
    if p.seq_fsc && (!p.intra || p.fsc) && *txtp as u8 == txtp::IDTX && !chroma {
        *txtp = txtp::IDTX_INV as u16;
        let stride = 1 + (4 << slh);
        let mut levels = vec![0i8; stride * ((4 << slw) + 1)];
        let sz_ctx = imin(t_dim.ctx as i32, 2) as usize;
        let sz = (16 << tx2dszctx) - 1;
        let bob = sz - eob;
        let ctx = ((bob > 2 << tx2dszctx) as usize) + ((bob > 4 << tx2dszctx) as usize);
        let mut tok = 1 + msac.decode_symbol_adapt(coef.bob_base_y_tok(sz_ctx, ctx), 2) as i32;
        if tok == 3 {
            tok += msac.decode_symbol_adapt(coef.br_y_tok_idtx(sz_ctx, 0), 3) as i32;
        }
        let shift = slh + 2;
        let mask = (4 << slh) - 1;
        let rc = scan[bob as usize] as usize;
        let x = rc >> shift;
        let y = rc & mask;
        cf[rc] = tok;
        levels[(1 + x) * stride + (y + 1)] = tok as i8;

        for i in (bob + 1)..=sz {
            let rc = scan[i as usize] as usize;
            let x = rc >> shift;
            let y = rc & mask;
            let off = (1 + x) * stride + (1 + y);
            let mut hr_ctx = 0u32;
            let ctx = get_lo_ctx_idtx(&levels, off, &mut hr_ctx, stride);
            let mut tok =
                msac.decode_symbol_adapt(coef.base_y_tok_idtx(sz_ctx, ctx as usize), 3) as i32;
            if tok == 3 {
                tok +=
                    msac.decode_symbol_adapt(coef.br_y_tok_idtx(sz_ctx, hr_ctx as usize), 3) as i32;
            }
            cf[rc] = tok;
            levels[off] = tok as i8;
        }

        let dq = p.dq_tbl[1];
        dq_shift -= tcq_en as i32;
        for i in bob..=sz {
            let rc = scan[i as usize] as usize;
            let tok_val = cf[rc];
            if tok_val == 0 {
                continue;
            }
            let x = rc >> shift;
            let y = rc & mask;
            let off = (1 + x) * stride + (1 + y);
            let ctx = get_sign_ctx_idtx(&levels, off, stride);
            let sign = msac.decode_bool_adapt(coef.sign_idtx(sz_ctx, ctx as usize));
            if i == 0 {
                dc_sign_level = ((sign as i32 - 1) & (2 << 6)) as u32;
            }
            levels[off] = 1 - 2 * sign as i8;

            let mut tok = tok_val;
            let val: i32;
            if tok >= 6 {
                let hr = decode_hr(msac, hr_avg);
                tok += hr;
                hr_avg = (hr_avg + hr) >> 1;
                tok &= 0xfffff;
                val = imin(
                    ((((tok as u32).wrapping_mul(dq)) & 0xffffff).wrapping_add(4) >> dq_shift)
                        as i32,
                    cf_max + sign as i32,
                );
            } else {
                val = ((tok as u32).wrapping_mul(dq).wrapping_add(4) >> dq_shift) as i32;
            }
            cul_level += tok as u32;
            cf[rc] = if sign != 0 { -val } else { val };
        }

        if env_flag!("RAV2D_CF") {
            eprintln!(
                "CF pl={} RET-fsc eob={} rng={}",
                p.plane,
                eob,
                msac.dbg_rng()
            );
        }
        *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
        return eob;
    }

    if eob != 0 {
        let mut levels = vec![0i8; 1089];
        let is_stx = stx_type != 0 && tx_class == 0;

        macro_rules! decode_coefs_class {
            ($tx_cl:expr, $stride:expr, $shift:expr, $shift2:expr, $mask:expr, $hi_to_low:expr, $xy_expr:ident) => {{
                let hi_to_low_tx: i32 = $hi_to_low;
                let stride: usize = $stride;
                let shift: usize = $shift;
                let shift2: usize = $shift2;
                let mask: usize = $mask;

                // eob token
                let (mut lim, mut tok): (i32, i32);
                let (mut hi_base, mut hi_stride): (usize, usize);
                let (mut lo_base, mut lo_stride, mut lo_nsym): (usize, usize, usize);
                let mut hi_cdf_valid: bool = true;

                let ctx_init = 1 + (eob > 2 << tx2dszctx) as u32 + (eob > 4 << tx2dszctx) as u32;
                if eob >= hi_to_low_tx {
                    lim = 3;
                    if !chroma {
                        tok = 1 + msac.decode_symbol_adapt(
                            coef.eob_base_y_tok_hf(t_dim.ctx as usize, ctx_init as usize),
                            2,
                        ) as i32;
                        hi_base = 1252;
                        hi_stride = 4;
                        lo_base = 452 + (t_dim.ctx as usize) * 160;
                        lo_stride = 4;
                        lo_nsym = 3;
                    } else {
                        tok = 1 + msac
                            .decode_symbol_adapt(coef.eob_base_uv_tok_hf(ctx_init as usize), 2)
                            as i32;
                        hi_base = 4508;
                        hi_stride = 4;
                        lo_base = 4460;
                        lo_stride = 4;
                        lo_nsym = 3;
                    }
                    hi_cdf_valid = true;
                } else {
                    lim = 5;
                    if !chroma {
                        tok = 1 + msac.decode_symbol_adapt(
                            coef.eob_base_y_tok_lf(t_dim.ctx as usize, ctx_init as usize),
                            4,
                        ) as i32;
                        hi_base = 4080;
                        hi_stride = 4;
                        lo_base = 1440 + (t_dim.ctx as usize) * 528;
                        lo_stride = 8;
                        lo_nsym = 5;
                    } else {
                        tok = 1 + msac
                            .decode_symbol_adapt(coef.eob_base_uv_tok_lf(ctx_init as usize), 4)
                            as i32;
                        hi_base = 0;
                        hi_stride = 0;
                        lo_base = 4560;
                        lo_stride = 8;
                        lo_nsym = 5;
                        hi_cdf_valid = false;
                    }
                    if chroma {
                        hi_cdf_valid = false;
                    }
                }

                let (mut rc, mut x, mut y): (usize, usize, usize);
                if $tx_cl == 0 {
                    rc = scan[eob as usize] as usize;
                    x = rc >> shift;
                    y = rc & mask;
                } else if $tx_cl == 1 {
                    x = eob as usize & mask;
                    y = eob as usize >> shift;
                    rc = eob as usize;
                } else {
                    x = eob as usize & mask;
                    y = eob as usize >> shift;
                    rc = (x << shift2) | y;
                }
                if tok == lim && hi_cdf_valid {
                    let hi_idx = if lim == 5 { 7 } else { 0 };
                    let o = hi_base + hi_idx * hi_stride;
                    tok += msac.decode_symbol_adapt(&mut coef.data[o..o + 4], 3) as i32;
                }
                tcq_state = tcq_next_state(tcq_state, tok);
                cf[if is_stx { eob as usize } else { rc }] = tok;
                if $tx_cl == 0 {
                    levels[rc] = tok as i8;
                } else {
                    levels[x * stride + y] = tok as i8;
                }

                // ac tokens (eob-1 down to 1)
                let mut i = eob - 1;
                loop {
                    if i == hi_to_low_tx - 1 {
                        lim = 5;
                        if !chroma {
                            hi_base = 4080;
                            hi_stride = 4;
                            lo_base = 1440 + (t_dim.ctx as usize) * 528;
                            lo_stride = 8;
                            lo_nsym = 5;
                            hi_cdf_valid = true;
                        } else {
                            hi_base = 0;
                            hi_stride = 0;
                            lo_base = 4560;
                            lo_stride = 8;
                            lo_nsym = 5;
                            hi_cdf_valid = false;
                        }
                    }
                    if i == 0 {
                        break;
                    }
                    if $tx_cl == 0 {
                        rc = scan[i as usize] as usize;
                        x = rc >> shift;
                        y = rc & mask;
                    } else if $tx_cl == 1 {
                        x = i as usize & mask;
                        y = i as usize >> shift;
                        rc = i as usize;
                    } else {
                        x = i as usize & mask;
                        y = i as usize >> shift;
                        rc = (x << shift2) | y;
                    }
                    let off = if $tx_cl == 0 { rc } else { x * stride + y };
                    let mut hr_ctx = 0u32;
                    let xy_val: u32 = if $tx_cl == 0 {
                        (x + y) as u32
                    } else {
                        y as u32
                    };
                    let ctx =
                        get_lo_ctx(&levels, off, $tx_cl, &mut hr_ctx, xy_val, p.plane, stride);
                    let tcq_bit = ((tcq_state & 2) >> 1) as u32;
                    let lo_cdf_idx = (ctx * (2 - chroma as u32) + tcq_bit) as usize;
                    let o = lo_base + lo_cdf_idx * lo_stride;
                    let mut tok =
                        msac.decode_symbol_adapt(&mut coef.data[o..o + lo_stride], lo_nsym) as i32;
                    if tok == lim && hi_cdf_valid {
                        let o2 = hi_base + hr_ctx as usize * hi_stride;
                        tok += msac.decode_symbol_adapt(&mut coef.data[o2..o2 + 4], 3) as i32;
                    }
                    tcq_state = tcq_next_state(tcq_state, tok);
                    levels[off] = tok as i8;
                    cf[if is_stx { i as usize } else { rc }] = tok;
                    i -= 1;
                }

                // dc token
                let mut hr_ctx = 0u32;
                let ctx = get_lo_ctx(&levels, 0, $tx_cl, &mut hr_ctx, 0, p.plane, stride);
                let tcq_bit = ((tcq_state & 2) >> 1) as u32;
                let lo_cdf_idx = (ctx * (2 - chroma as u32) + tcq_bit) as usize;
                let o = lo_base + lo_cdf_idx * lo_stride;
                dc_tok = msac.decode_symbol_adapt(&mut coef.data[o..o + lo_stride], lo_nsym) as i32;
                if dc_tok == lim && hi_cdf_valid {
                    let o2 = hi_base + hr_ctx as usize * hi_stride;
                    dc_tok += msac.decode_symbol_adapt(&mut coef.data[o2..o2 + 4], 3) as i32;
                }

                // sign & dequant for AC
                tcq_state = if tcq_en { -0x80000000i32 } else { 0 };
                let ac_dq = p.dq_tbl[1];
                for i in (1..=eob).rev() {
                    if $tx_cl == 0 {
                        rc = if is_stx {
                            i as usize
                        } else {
                            scan[i as usize] as usize
                        };
                    } else if $tx_cl == 1 {
                        y = i as usize >> shift;
                        rc = i as usize;
                    } else {
                        x = i as usize & mask;
                        y = i as usize >> shift;
                        rc = (x << shift2) | y;
                    }
                    let tok_val = cf[rc];
                    if tok_val == 0 {
                        tcq_state = tcq_next_state(tcq_state, 0);
                        continue;
                    }
                    let sign: u32;
                    if $tx_cl == 0 || y > 0 || chroma {
                        sign = msac.decode_bool_bypass();
                    } else {
                        sign = msac.decode_bool_adapt(coef.dc_sign(chroma as usize, 0, 0));
                    }
                    let tcq_bit = ((tcq_state & 2) >> 1) as i32;
                    tcq_state = tcq_next_state(tcq_state, tok_val);
                    let max_br = if i < hi_to_low_tx {
                        if chroma { 5 } else { 8 }
                    } else {
                        6
                    };
                    let mut tok = tok_val;
                    let ac_val: i32;
                    if tok >= max_br - tcq_en as i32 {
                        let hr = decode_hr(msac, hr_avg);
                        tok += hr << tcq_en as i32;
                        hr_avg = (hr_avg + hr) >> 1;
                        tok &= 0xfffff;
                        let v = (tok << tcq_en as i32) - tcq_bit;
                        ac_val = imin(
                            ((((v as u32).wrapping_mul(ac_dq)) & 0xffffff).wrapping_add(4)
                                >> dq_shift) as i32,
                            cf_max + sign as i32,
                        );
                    } else {
                        let v = (tok << tcq_en as i32) - tcq_bit;
                        ac_val =
                            (((v as u32).wrapping_mul(ac_dq)).wrapping_add(4) >> dq_shift) as i32;
                    }
                    cul_level += tok as u32;
                    cf[rc] = if sign != 0 { -ac_val } else { ac_val };
                }
            }};
        }

        // tx_class values follow the dav2d enum: 0=2D, 1=2D_INV, 2=H, 3=V
        // (recon_tmpl.c:990-1022, which dispatches on 2D/H/V; 2D_INV is not
        // reached here). The class!() macro arg mirrors that scan orientation.
        match tx_class {
            0 => {
                let stride = (4 << slh) as usize;
                let shift = slh + 2;
                let mask = (4 << slh) - 1;
                levels[..stride * ((4 << slw) + 2)].fill(0);
                let hi_to_low = if chroma { 1i32 } else { 10 };
                decode_coefs_class!(0, stride, shift, 0, mask, hi_to_low, xy_2d);
            }
            2 => {
                let stride = 32usize;
                let shift = slh + 2;
                let mask = (4 << slh) - 1;
                levels[..stride * ((4 << slh) + 2)].fill(0);
                let hi_to_low = (8 << slh) >> chroma as usize;
                decode_coefs_class!(1, stride, shift, 0, mask, hi_to_low, xy_h);
            }
            3 => {
                let stride = 32usize;
                let shift = slw + 2;
                let shift2 = slh + 2;
                let mask = (4 << slw) - 1;
                levels[..stride * ((4 << slw) + 2)].fill(0);
                let hi_to_low = (8 << slw) >> chroma as usize;
                decode_coefs_class!(2, stride, shift, shift2, mask, hi_to_low, xy_v);
            }
            _ => unreachable!(),
        }
    } else if chroma {
        dc_tok = 1 + msac.decode_symbol_adapt(coef.eob_base_uv_tok_lf(0), 4) as i32;
    } else {
        dc_tok =
            1 + msac.decode_symbol_adapt(coef.eob_base_y_tok_lf(t_dim.ctx as usize, 0), 4) as i32;
        if dc_tok == 5 {
            let hi_idx = if tx_class == 0 { 0 } else { 7 };
            dc_tok += msac.decode_symbol_adapt(coef.br_y_tok_lf(hi_idx), 3) as i32;
        }
    }

    if dc_tok == 0 {
        if env_flag!("RAV2D_CF") {
            eprintln!(
                "CF pl={} RET eob={} dc_tok={} rng={}",
                p.plane,
                eob,
                dc_tok,
                msac.dbg_rng()
            );
        }
        *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
        return eob;
    }

    // dc sign & residual
    let dc_sign: u32;
    if chroma {
        dc_sign = msac.decode_bool_bypass();
    } else {
        let dc_sign_ctx = get_dc_sign_ctx(t_dim, a, l) as usize;
        dc_sign = msac.decode_bool_adapt(coef.dc_sign(chroma as usize, 0, dc_sign_ctx));
    }

    let mut dc_dq = p.dq_tbl[0] as i32;
    dc_sign_level = ((dc_sign as i32 - 1) & (2 << 6)) as u32;

    if has_qm {
        let qm_tbl = p.qm.unwrap();
        dc_dq = (dc_dq * qm_tbl[0] as i32 + 16) >> 5;
        if dc_tok == 15 {
            dc_tok = 0;
            dc_tok &= 0xfffff;
            let dq_val = ((dc_dq * dc_tok) & 0xffffff) >> dq_shift;
            let dq_val = imin(dq_val, cf_max + dc_sign as i32);
            cul_level = dc_tok as u32;
            cf[0] = if dc_sign != 0 { -dq_val } else { dq_val };
        } else {
            let dq_val = dc_dq * dc_tok;
            cul_level = dc_tok as u32;
            let dq_val = dq_val >> dq_shift;
            let dq_val = imin(dq_val, cf_max + dc_sign as i32);
            cf[0] = if dc_sign != 0 { -dq_val } else { dq_val };
        }
    } else {
        let max_br = if chroma { 5 } else { 8 };
        let tcq_bit = (tcq_state & 2) >> 1;
        let dc_val: i32;
        if dc_tok >= max_br - tcq_en as i32 {
            let hr = decode_hr(msac, hr_avg);
            dc_tok += hr << tcq_en as i32;
            dc_tok &= 0xfffff;
            let v = (dc_tok << tcq_en as i32) - tcq_bit;
            dc_val = imin(
                ((((v as u32).wrapping_mul(dc_dq as u32)) & 0xffffff).wrapping_add(4) >> dq_shift)
                    as i32,
                cf_max + dc_sign as i32,
            );
        } else {
            let v = (dc_tok << tcq_en as i32) - tcq_bit;
            dc_val = (((v as u32).wrapping_mul(dc_dq as u32)).wrapping_add(4) >> dq_shift) as i32;
        }
        cul_level += dc_tok as u32;
        cf[0] = if dc_sign != 0 { -dc_val } else { dc_val };
    }

    if env_flag!("RAV2D_CF") {
        eprintln!(
            "CF pl={} RET eob={} dc_tok={} rng={}",
            p.plane,
            eob,
            dc_tok,
            msac.dbg_rng()
        );
    }
    *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
    eob
}

pub struct McParams {
    pub bw4: i32,
    pub bh4: i32,
    pub bx: i32,
    pub by: i32,
    pub pl: usize,
    pub mvx: i32,
    pub mvy: i32,
    pub filter: usize,
    pub ss_hor: i32,
    pub ss_ver: i32,
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
}

/// IntraBC block-copy prediction for one plane (8bpc). Mirrors `mc(... &f->cur,
/// DAV2D_FILTER_BILINEAR ...)` in `dav2d_recon_b` (recon_tmpl.c:3150 / 3591):
/// copies an already-decoded region of the *current* plane (src == dst buffer)
/// at the block-vector offset into the block at (bx,by), applying the AV2
/// bilinear subpel filter. `bx`,`by` are in 4x4 units; `mvx`,`mvy` are the
/// luma-resolution block vector (1/8-pel). `right`/`bottom` are `f->bw*4 >>
/// ss_hor` / `f->bh*4 >> ss_ver` (the clamping edges); `left`/`top` are 0.
#[allow(clippy::too_many_arguments)]
pub fn intrabc_pred_8bpc(
    plane: &mut [u8],
    stride: usize,
    bw4: i32,
    bh4: i32,
    bx: i32,
    by: i32,
    mvx: i32,
    mvy: i32,
    ss_hor: i32,
    ss_ver: i32,
    right: i32,
    bottom: i32,
) {
    intrabc_pred(
        crate::pixel::BitDepth8,
        plane,
        stride,
        bw4,
        bh4,
        bx,
        by,
        mvx,
        mvy,
        ss_hor,
        ss_ver,
        right,
        bottom,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn intrabc_pred<BD: crate::pixel::BitDepth>(
    bd: BD,
    plane: &mut [BD::Pixel],
    stride: usize,
    bw4: i32,
    bh4: i32,
    bx: i32,
    by: i32,
    mvx: i32,
    mvy: i32,
    ss_hor: i32,
    ss_ver: i32,
    right: i32,
    bottom: i32,
) {
    use crate::pixel::Pixel;
    let left = 0i32;
    let top = 0i32;
    let h_mul = 4 >> ss_hor;
    let v_mul = 4 >> ss_ver;
    // dav2d mc(): mx/my are the 1/16-pel subpel positions actually passed to the
    // put filter (`mx << !ss_hor`). For an unsubsampled plane that is
    // (mvx & 7) << 1; for a subsampled chroma plane it is (mvx & 15).
    let mx_lo = mvx & (15 >> (ss_hor == 0) as i32);
    let my_lo = mvy & (15 >> (ss_ver == 0) as i32);
    let mx = mx_lo << (ss_hor == 0) as i32;
    let my = my_lo << (ss_ver == 0) as i32;
    // Source (reference) position within the current plane, in samples.
    let sx = bx * h_mul + (mvx >> (3 + ss_hor));
    let sy = by * v_mul + (mvy >> (3 + ss_ver));
    // Destination position (block origin), in samples.
    let dpx = bx * h_mul;
    let dpy = by * v_mul;

    let w = (bw4 * h_mul) as usize;
    let h = (bh4 * v_mul) as usize;

    // Gather the source region (plus one extra row/col of subpel context) into a
    // contiguous scratch buffer (as i32), with edge clamping identical to
    // mc.emu_edge. This also lets src and dst share the same plane buffer safely.
    let src_w = w + (mx_lo != 0) as usize; // extra column for the +stride tap
    let src_h = h + (my_lo != 0) as usize; // extra row for the +stride tap
    let src_stride = src_w;
    let mut srcbuf = vec![0i32; src_stride * src_h];
    for ry in 0..src_h {
        let cy = (sy + ry as i32).clamp(top, bottom - 1) as usize;
        for rx in 0..src_w {
            let cx = (sx + rx as i32).clamp(left, right - 1) as usize;
            srcbuf[ry * src_stride + rx] = plane[cy * stride + cx].into();
        }
    }

    let dst_off = (dpy as usize) * stride + dpx as usize;
    let bdmax = bd.bitdepth_max();

    // put_bilin_c (mc_tmpl.c): intermediate_bits = 4 (8bpc) / 14-bd (HBD).
    let ib = crate::mc::intermediate_bits(bd);
    if mx != 0 {
        if my != 0 {
            // 2-pass: horizontal into mid (16-bit), then vertical.
            let mut mid = vec![0i32; src_w * (h + 1)];
            for ry in 0..(h + 1) {
                for x in 0..w {
                    let s = ry * src_stride + x;
                    let v = 16 * srcbuf[s] + mx * (srcbuf[s + 1] - srcbuf[s]);
                    mid[ry * w + x] = (v + ((1 << (4 - ib)) >> 1)) >> (4 - ib);
                }
            }
            for ry in 0..h {
                for x in 0..w {
                    let m0 = mid[ry * w + x];
                    let m1 = mid[(ry + 1) * w + x];
                    let v = 16 * m0 + my * (m1 - m0);
                    let px = (v + ((1 << (4 + ib)) >> 1)) >> (4 + ib);
                    plane[dst_off + ry * stride + x] = BD::Pixel::from_i32(iclip(px, 0, bdmax));
                }
            }
        } else {
            let rnd = (1 << ib) >> 1;
            for ry in 0..h {
                for x in 0..w {
                    let s = ry * src_stride + x;
                    let v = 16 * srcbuf[s] + mx * (srcbuf[s + 1] - srcbuf[s]);
                    let px = (v + ((1 << (4 - ib)) >> 1)) >> (4 - ib);
                    plane[dst_off + ry * stride + x] =
                        BD::Pixel::from_i32(iclip((px + rnd) >> ib, 0, bdmax));
                }
            }
        }
    } else if my != 0 {
        for ry in 0..h {
            for x in 0..w {
                let s0 = ry * src_stride + x;
                let s1 = (ry + 1) * src_stride + x;
                let v = 16 * srcbuf[s0] + my * (srcbuf[s1] - srcbuf[s0]);
                let px = (v + ((1 << 4) >> 1)) >> 4;
                plane[dst_off + ry * stride + x] = BD::Pixel::from_i32(iclip(px, 0, bdmax));
            }
        }
    } else {
        // integer copy
        for ry in 0..h {
            for x in 0..w {
                plane[dst_off + ry * stride + x] = BD::Pixel::from_i32(srcbuf[ry * src_stride + x]);
            }
        }
    }
}

pub fn mc_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    ref_data: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    cur_w: i32,
    cur_h: i32,
    p: &McParams,
    emu_edge_buf: &mut [u8],
    is_compound: bool,
    dst16: Option<&mut [i16]>,
) {
    let h_mul = 4 >> p.ss_hor;
    let v_mul = 4 >> p.ss_ver;

    if ref_w == cur_w && ref_h == cur_h {
        let mx = p.mvx & (15 >> (p.ss_hor == 0) as i32);
        let my = p.mvy & (15 >> (p.ss_ver == 0) as i32);
        let dx = p.bx * h_mul + (p.mvx >> (3 + p.ss_hor));
        let dy = p.by * v_mul + (p.mvy >> (3 + p.ss_ver));

        let need_emu = dx - (mx != 0) as i32 * 3 < p.left
            || dy - (my != 0) as i32 * 3 < p.top
            || dx + p.bw4 * h_mul + (mx != 0) as i32 * 4 > p.right
            || dy + p.bh4 * v_mul + (my != 0) as i32 * 4 > p.bottom;

        let (src, src_stride) = if need_emu {
            let emu_w = (p.bw4 * h_mul + (mx != 0) as i32 * 7) as usize;
            let emu_h = (p.bh4 * v_mul + (my != 0) as i32 * 7) as usize;
            let emu_stride = 192usize;
            emu_edge(
                emu_edge_buf,
                emu_stride,
                ref_data,
                ref_stride,
                emu_w,
                emu_h,
                (p.right - p.left) as usize,
                (p.bottom - p.top) as usize,
                dx - (mx != 0) as i32 * 3 - p.left,
                dy - (my != 0) as i32 * 3 - p.top,
            );
            let ref_off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
            (ref_off, emu_stride)
        } else {
            let ref_off = (dy as usize) * ref_stride + dx as usize;
            (ref_off, ref_stride)
        };

        let w = p.bw4 * h_mul;
        let h = p.bh4 * v_mul;

        if !is_compound {
            mc_subpel_8bpc(
                dst,
                dst_stride,
                if need_emu { emu_edge_buf } else { ref_data },
                src,
                src_stride,
                w as usize,
                h as usize,
                mx << (p.ss_hor == 0) as i32,
                my << (p.ss_ver == 0) as i32,
                p.filter,
            );
        } else if let Some(d16) = dst16 {
            mct_subpel_8bpc(
                d16,
                dst_stride,
                if need_emu { emu_edge_buf } else { ref_data },
                src,
                src_stride,
                w as usize,
                h as usize,
                mx << (p.ss_hor == 0) as i32,
                my << (p.ss_ver == 0) as i32,
                p.filter,
            );
        }
    }
}

fn emu_edge(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
) {
    let _src_y = y.max(0) as usize;
    for dst_y in 0..bh {
        let actual_y = ((y + dst_y as i32).max(0) as usize).min(ih.saturating_sub(1));
        let dst_row = dst_y * dst_stride;
        let src_row = actual_y * src_stride;
        for dst_x in 0..bw {
            let actual_x = ((x + dst_x as i32).max(0) as usize).min(iw.saturating_sub(1));
            dst[dst_row + dst_x] = if src_row + actual_x < src.len() {
                src[src_row + actual_x]
            } else {
                128
            };
        }
    }
}

fn mc_subpel_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter: usize,
) {
    if mx == 0 && my == 0 {
        for y in 0..h {
            let d = y * dst_stride;
            let s = src_off + y * src_stride;
            for x in 0..w {
                if s + x < src.len() && d + x < dst.len() {
                    dst[d + x] = src[s + x];
                }
            }
        }
        return;
    }

    let filter_taps = get_subpel_filter(filter, mx, my);

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            for ky in 0..8i32 {
                for kx in 0..8i32 {
                    let sy = (y as i32 + ky - 3) as usize;
                    let sx = (x as i32 + kx - 3) as usize;
                    let idx = src_off + sy * src_stride + sx;
                    let pix = if idx < src.len() {
                        src[idx] as i32
                    } else {
                        128
                    };
                    sum += pix * filter_taps[ky as usize] as i32 * filter_taps[kx as usize] as i32
                        / 128;
                }
            }
            let d = y * dst_stride + x;
            if d < dst.len() {
                dst[d] = iclip((sum + (1 << 13)) >> 14, 0, 255) as u8;
            }
        }
    }
}

fn mct_subpel_8bpc(
    dst: &mut [i16],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter: usize,
) {
    let filter_taps = get_subpel_filter(filter, mx, my);

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            for ky in 0..8i32 {
                for kx in 0..8i32 {
                    let sy = (y as i32 + ky - 3) as usize;
                    let sx = (x as i32 + kx - 3) as usize;
                    let idx = src_off + sy * src_stride + sx;
                    let pix = if idx < src.len() {
                        src[idx] as i32
                    } else {
                        128
                    };
                    sum += pix * filter_taps[ky as usize] as i32 * filter_taps[kx as usize] as i32
                        / 128;
                }
            }
            let d = y * dst_stride + x;
            if d < dst.len() {
                dst[d] = ((sum + (1 << 6)) >> 7) as i16;
            }
        }
    }
}

fn get_subpel_filter(filter: usize, mx: i32, my: i32) -> [i16; 8] {
    static BILINEAR_FILTER: [[i16; 8]; 16] = [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 0, 120, 8, 0, 0, 0],
        [0, 0, 0, 112, 16, 0, 0, 0],
        [0, 0, 0, 104, 24, 0, 0, 0],
        [0, 0, 0, 96, 32, 0, 0, 0],
        [0, 0, 0, 88, 40, 0, 0, 0],
        [0, 0, 0, 80, 48, 0, 0, 0],
        [0, 0, 0, 72, 56, 0, 0, 0],
        [0, 0, 0, 64, 64, 0, 0, 0],
        [0, 0, 0, 56, 72, 0, 0, 0],
        [0, 0, 0, 48, 80, 0, 0, 0],
        [0, 0, 0, 40, 88, 0, 0, 0],
        [0, 0, 0, 32, 96, 0, 0, 0],
        [0, 0, 0, 24, 104, 0, 0, 0],
        [0, 0, 0, 16, 112, 0, 0, 0],
        [0, 0, 0, 8, 120, 0, 0, 0],
    ];

    static REGULAR_FILTER: [[i16; 8]; 16] = [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 1, -3, 126, 5, -1, 0, 0],
        [0, 1, -5, 123, 11, -3, 1, 0],
        [0, 2, -8, 119, 18, -5, 2, 0],
        [0, 2, -10, 114, 25, -7, 3, 1],
        [-1, 3, -12, 108, 32, -8, 3, 3],
        [-1, 3, -13, 101, 40, -10, 4, 4],
        [-1, 4, -15, 94, 48, -11, 4, 5],
        [-1, 4, -16, 86, 56, -12, 4, 7],
        [-1, 4, -16, 78, 64, -12, 4, 7],
        [-1, 4, -15, 69, 71, -12, 4, 8],
        [-1, 4, -14, 60, 78, -11, 3, 9],
        [-1, 3, -12, 51, 85, -10, 3, 9],
        [-1, 3, -10, 42, 90, -8, 2, 10],
        [-1, 2, -8, 33, 95, -6, 1, 12],
        [0, 2, -6, 24, 99, -4, 1, 12],
    ];

    let idx = if my != 0 { my } else { mx } as usize;
    if filter == 0 {
        REGULAR_FILTER[idx & 15]
    } else {
        BILINEAR_FILTER[idx & 15]
    }
}

pub fn warp_affine_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    ref_data: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bw4: i32,
    bh4: i32,
    mat: &[i32; 6],
    _pl: usize,
    ss_hor: i32,
    ss_ver: i32,
    _emu_edge_buf: &mut [u8],
) {
    let h_mul = 4 >> ss_hor;
    let v_mul = 4 >> ss_ver;
    let w = bw4 * h_mul;
    let h = bh4 * v_mul;

    for y in 0..h {
        for x in 0..w {
            let src_x = mat[0] * x + mat[1] * y + mat[2];
            let src_y = mat[3] * x + mat[4] * y + mat[5];

            let ix = src_x >> 16;
            let iy = src_y >> 16;

            let cx = ix.max(0).min(ref_w - 1) as usize;
            let cy = iy.max(0).min(ref_h - 1) as usize;

            let d = y as usize * dst_stride + x as usize;
            let s = cy * ref_stride + cx;
            if d < dst.len() && s < ref_data.len() {
                dst[d] = ref_data[s];
            }
        }
    }
}

pub struct BawpContext {
    pub alpha: i32,
    pub beta: i32,
}

pub fn bawp_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    ctx: &BawpContext,
) {
    for y in 0..h {
        for x in 0..w {
            let d = y * dst_stride + x;
            let s = y * src_stride + x;
            if d < dst.len() && s < src.len() {
                let p = dst[d] as i32;
                let q = src[s] as i32;
                let alpha = ctx.alpha;
                let beta = ctx.beta;
                let val = (alpha * p + beta * q + (1 << 5)) >> 6;
                dst[d] = iclip(val, 0, 255) as u8;
            }
        }
    }
}

pub fn iiblend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    inter_pred: &[u8],
    inter_stride: usize,
    intra_pred: &[u8],
    intra_stride: usize,
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        for x in 0..w {
            let d = y * dst_stride + x;
            let ip = y * inter_stride + x;
            let ia = y * intra_stride + x;
            if d < dst.len() && ip < inter_pred.len() && ia < intra_pred.len() {
                let m_idx = y * w + x;
                let m = if m_idx < mask.len() {
                    mask[m_idx] as i32
                } else {
                    32
                };
                let val = (m * intra_pred[ia] as i32 + (64 - m) * inter_pred[ip] as i32 + 32) >> 6;
                dst[d] = iclip(val, 0, 255) as u8;
            }
        }
    }
}

pub fn avg_8bpc(dst: &mut [u8], dst_stride: usize, tmp1: &[i16], tmp2: &[i16], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            let d = y * dst_stride + x;
            let s = y * w + x;
            if d < dst.len() && s < tmp1.len() && s < tmp2.len() {
                let val = ((tmp1[s] as i32 + tmp2[s] as i32 + (1 << 4)) >> 5).clamp(0, 255);
                dst[d] = val as u8;
            }
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
    for y in 0..h {
        for x in 0..w {
            let d = y * dst_stride + x;
            let s = y * w + x;
            if d < dst.len() && s < tmp1.len() && s < tmp2.len() {
                let val =
                    (tmp1[s] as i32 * weight + tmp2[s] as i32 * (16 - weight) + (1 << 8)) >> 9;
                dst[d] = iclip(val, 0, 255) as u8;
            }
        }
    }
}

pub fn mask_blend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        for x in 0..w {
            let d = y * dst_stride + x;
            let s = y * w + x;
            if d < dst.len() && s < tmp1.len() && s < tmp2.len() && s < mask.len() {
                let m = mask[s] as i32;
                let val = (tmp1[s] as i32 * m + tmp2[s] as i32 * (64 - m) + (1 << 9)) >> 10;
                dst[d] = iclip(val, 0, 255) as u8;
            }
        }
    }
}

pub fn opfl_mv_refinement(
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    dx: &mut [i32],
    dy: &mut [i32],
    n_blocks: usize,
) {
    let blk_w = w / n_blocks.max(1);
    let blk_h = h;
    for blk in 0..n_blocks {
        let off_x = blk * blk_w;
        let mut sum_dx = 0i64;
        let mut sum_dy = 0i64;
        let mut sum_d = 0i64;

        for y in 1..blk_h.saturating_sub(1) {
            for x in 1..blk_w.saturating_sub(1) {
                let px = off_x + x;
                let idx = y * w + px;
                if idx < tmp1.len() && idx < tmp2.len() {
                    let diff = tmp1[idx] as i64 - tmp2[idx] as i64;
                    let gx = (tmp1[idx + 1] as i64 - tmp1[idx - 1] as i64 + tmp2[idx + 1] as i64
                        - tmp2[idx - 1] as i64)
                        >> 1;
                    let gy = (tmp1[idx + w] as i64 - tmp1[idx - w] as i64 + tmp2[idx + w] as i64
                        - tmp2[idx - w] as i64)
                        >> 1;
                    sum_dx += gx * diff;
                    sum_dy += gy * diff;
                    sum_d += gx * gx + gy * gy;
                }
            }
        }

        if sum_d != 0 {
            dx[blk] = ((-sum_dx * 64) / sum_d) as i32;
            dy[blk] = ((-sum_dy * 64) / sum_d) as i32;
        } else {
            dx[blk] = 0;
            dy[blk] = 0;
        }
    }
}

pub struct ReconLumaTxParams {
    pub tx: usize,
    pub bx4: i32,
    pub by4: i32,
    pub tw: i32,
    pub th: i32,
}

pub fn recon_b_luma_tx_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    cf: &[i32],
    eob: i32,
    txtp: u16,
    _tx: usize,
    qm: Option<&[u8]>,
    dq: &[u32; 2],
    bitdepth_max: i32,
    p: &ReconLumaTxParams,
) {
    if eob < 0 {
        return;
    }

    let tw = p.tw as usize;
    let th = p.th as usize;

    if eob > 0 || cf[0] != 0 {
        let mut dequant = vec![0i32; tw * th];
        dequant_coeffs(cf, &mut dequant, tw, th, eob as usize, dq, qm, bitdepth_max);
        itxfm_add_8bpc(dst, dst_stride, &dequant, tw, th, txtp, bitdepth_max);
    }
}

fn dequant_coeffs(
    cf: &[i32],
    out: &mut [i32],
    w: usize,
    h: usize,
    eob: usize,
    dq: &[u32; 2],
    qm: Option<&[u8]>,
    bitdepth_max: i32,
) {
    let dc_q = dq[0] as i32;
    let ac_q = dq[1] as i32;

    for i in 0..=eob.min(w * h - 1) {
        let q = if i == 0 { dc_q } else { ac_q };
        let qm_val = qm.map_or(1 << 5, |m| if i < m.len() { m[i] as i32 } else { 1 << 5 });
        let val = (cf[i] * q * qm_val + (1 << 4)) >> 5;
        out[i] = iclip(val, -(bitdepth_max + 1) * (1 << 6), bitdepth_max * (1 << 6));
    }
}

fn itxfm_add_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    cf: &[i32],
    w: usize,
    h: usize,
    _txtp: u16,
    _bitdepth_max: i32,
) {
    for y in 0..h {
        for x in 0..w {
            let d = y * dst_stride + x;
            let c = y * w + x;
            if d < dst.len() && c < cf.len() {
                let val = dst[d] as i32 + ((cf[c] + (1 << 3)) >> 4);
                dst[d] = iclip(val, 0, 255) as u8;
            }
        }
    }
}

pub struct ReadCoefBlocksParams {
    pub has_luma: bool,
    pub has_chroma: bool,
    pub skip_txfm: bool,
    pub tx_part: u8,
    pub bw4: i32,
    pub bh4: i32,
    pub ss_hor: i32,
    pub ss_ver: i32,
}

pub fn read_coef_blocks(
    msac: &mut MsacContext,
    coef_cdf: &mut CdfCoefContext,
    mode_cdf: &mut CdfModeContext,
    cf_buf: &mut [i32],
    cbi_out: &mut [(i16, u16)],
    a_coef: &mut [u8],
    l_coef: &mut [u8],
    rp: &ReadCoefBlocksParams,
    decode_p: &DecodeCoefParams,
) -> i32 {
    if rp.skip_txfm {
        return 0;
    }

    let mut txtp = 0u16;
    let mut res_ctx = 0u8;
    let eob = decode_coefs(
        msac,
        coef_cdf,
        mode_cdf,
        a_coef,
        l_coef,
        decode_p,
        cf_buf,
        &mut txtp,
        &mut res_ctx,
    );

    if eob == i32::MIN {
        return -1;
    }

    if !cbi_out.is_empty() {
        cbi_out[0] = (eob as i16, txtp);
    }

    0
}

pub enum PredictionMode {
    Intra,
    Inter,
    IntraBC,
    Compound,
    Palette,
}

pub struct ReconBlockContext {
    pub bx: i32,
    pub by: i32,
    pub bw4: i32,
    pub bh4: i32,
    pub bs: BlockSize,
    pub intra: bool,
    pub intrabc: bool,
    pub skip_txfm: bool,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub comp_type: u8,
    pub ref_idx: [i8; 2],
    pub motion_mode: u8,
    pub fsc: u8,
}

pub fn recon_b_8bpc(
    dst_y: &mut [u8],
    y_stride: usize,
    _dst_u: &mut [u8],
    _dst_v: &mut [u8],
    _uv_stride: usize,
    ctx: &ReconBlockContext,
    cf: &[i32],
    cbi: &[(i16, u16)],
    ref_planes: &[&[u8]; 7],
    ref_strides: &[usize; 7],
    ref_w: &[i32; 7],
    ref_h: &[i32; 7],
    cur_w: i32,
    cur_h: i32,
    dq: &[u32; 2],
    ss_hor: i32,
    ss_ver: i32,
    bitdepth_max: i32,
) {
    let w = ctx.bw4 * 4;
    let h = ctx.bh4 * 4;
    let _cw = w >> ss_hor;
    let _ch = h >> ss_ver;

    if ctx.skip_txfm {
        return;
    }

    if ctx.intra && !ctx.intrabc {
        if !cbi.is_empty() && cbi[0].0 >= 0 {
            let _tx_w = w as usize;
            let _tx_h = h as usize;
            let txtp = cbi[0].1;
            let eob = cbi[0].0 as i32;

            let off = ctx.by as usize * y_stride + ctx.bx as usize * 4;
            if off < dst_y.len() {
                let params = ReconLumaTxParams {
                    tx: 0,
                    bx4: ctx.bx,
                    by4: ctx.by,
                    tw: w,
                    th: h,
                };
                recon_b_luma_tx_8bpc(
                    &mut dst_y[off..],
                    y_stride,
                    cf,
                    eob,
                    txtp,
                    0,
                    None,
                    dq,
                    bitdepth_max,
                    &params,
                );
            }
        }
        return;
    }

    if ctx.intrabc {
        return;
    }

    let ref0 = ctx.ref_idx[0] as usize;
    if ctx.comp_type == 0 {
        let mut emu_buf = vec![0u8; 192 * 192];
        let mc_params = McParams {
            bw4: ctx.bw4,
            bh4: ctx.bh4,
            bx: ctx.bx,
            by: ctx.by,
            pl: 0,
            mvx: 0,
            mvy: 0,
            filter: 0,
            ss_hor: 0,
            ss_ver: 0,
            left: 0,
            right: cur_w * 4,
            top: 0,
            bottom: cur_h * 4,
        };

        mc_8bpc(
            &mut dst_y[ctx.by as usize * y_stride + ctx.bx as usize * 4..],
            y_stride,
            ref_planes[ref0],
            ref_strides[ref0],
            ref_w[ref0],
            ref_h[ref0],
            cur_w,
            cur_h,
            &mc_params,
            &mut emu_buf,
            false,
            None,
        );

        if !cbi.is_empty() && cbi[0].0 > 0 {
            let off = ctx.by as usize * y_stride + ctx.bx as usize * 4;
            if off < dst_y.len() {
                let params = ReconLumaTxParams {
                    tx: 0,
                    bx4: ctx.bx,
                    by4: ctx.by,
                    tw: w,
                    th: h,
                };
                recon_b_luma_tx_8bpc(
                    &mut dst_y[off..],
                    y_stride,
                    cf,
                    cbi[0].0 as i32,
                    cbi[0].1,
                    0,
                    None,
                    dq,
                    bitdepth_max,
                    &params,
                );
            }
        }
    } else {
        let ref1 = ctx.ref_idx[1] as usize;
        let mut emu_buf = vec![0u8; 192 * 192];
        let mut tmp1 = vec![0i16; 128 * 128];
        let mut tmp2 = vec![0i16; 128 * 128];

        let mc_params0 = McParams {
            bw4: ctx.bw4,
            bh4: ctx.bh4,
            bx: ctx.bx,
            by: ctx.by,
            pl: 0,
            mvx: 0,
            mvy: 0,
            filter: 0,
            ss_hor: 0,
            ss_ver: 0,
            left: 0,
            right: cur_w * 4,
            top: 0,
            bottom: cur_h * 4,
        };

        mc_8bpc(
            dst_y,
            y_stride,
            ref_planes[ref0],
            ref_strides[ref0],
            ref_w[ref0],
            ref_h[ref0],
            cur_w,
            cur_h,
            &mc_params0,
            &mut emu_buf,
            true,
            Some(&mut tmp1),
        );

        mc_8bpc(
            dst_y,
            y_stride,
            ref_planes[ref1],
            ref_strides[ref1],
            ref_w[ref1],
            ref_h[ref1],
            cur_w,
            cur_h,
            &mc_params0,
            &mut emu_buf,
            true,
            Some(&mut tmp2),
        );

        let off = ctx.by as usize * y_stride + ctx.bx as usize * 4;
        avg_8bpc(
            &mut dst_y[off..],
            y_stride,
            &tmp1,
            &tmp2,
            w as usize,
            h as usize,
        );

        if !cbi.is_empty() && cbi[0].0 > 0 && off < dst_y.len() {
            let params = ReconLumaTxParams {
                tx: 0,
                bx4: ctx.bx,
                by4: ctx.by,
                tw: w,
                th: h,
            };
            recon_b_luma_tx_8bpc(
                &mut dst_y[off..],
                y_stride,
                cf,
                cbi[0].0 as i32,
                cbi[0].1,
                0,
                None,
                dq,
                bitdepth_max,
                &params,
            );
        }
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
        let t_dim = TxfmInfo {
            w: 4,
            h: 4,
            lw: 2,
            lh: 2,
            min: 2,
            max: 2,
            sub: 0,
            ctx: 0,
        };
        let mut angle = 0;
        let r = wide_angle_remap(&t_dim, IntraPredMode::DcPred, &mut angle, 0);
        assert_eq!(r, IntraPredMode::DcPred);
    }

    #[test]
    fn test_wide_angle_remap_no_remap() {
        let t_dim = TxfmInfo {
            w: 4,
            h: 4,
            lw: 2,
            lh: 2,
            min: 2,
            max: 2,
            sub: 0,
            ctx: 0,
        };
        let mut angle = 0;
        let r = wide_angle_remap(&t_dim, IntraPredMode::VertPred, &mut angle, 0);
        assert_eq!(r, IntraPredMode::VertPred);
    }

    #[test]
    fn test_wide_angle_remap_wide_rect() {
        let t_dim = TxfmInfo {
            w: 16,
            h: 4,
            lw: 4,
            lh: 2,
            min: 2,
            max: 4,
            sub: 0,
            ctx: 0,
        };
        let mut angle = 30;
        let r = wide_angle_remap(&t_dim, IntraPredMode::VertPred, &mut angle, 0);
        let _ = r;
    }

    #[test]
    fn test_wide_angle_remap_tall_rect() {
        let t_dim = TxfmInfo {
            w: 4,
            h: 16,
            lw: 2,
            lh: 4,
            min: 2,
            max: 4,
            sub: 0,
            ctx: 0,
        };
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
        let t_dim = TxfmInfo {
            w: 1,
            h: 1,
            lw: 0,
            lh: 0,
            min: 0,
            max: 0,
            sub: 0,
            ctx: 0,
        };
        let a = [0u8; 16];
        let l = [0u8; 16];
        let r = get_dc_sign_ctx(&t_dim, &a, &l);
        assert_eq!(r, 1);
    }

    #[test]
    fn test_get_dc_sign_ctx_balanced() {
        let t_dim = TxfmInfo {
            w: 1,
            h: 1,
            lw: 0,
            lh: 0,
            min: 0,
            max: 0,
            sub: 0,
            ctx: 0,
        };
        let a = [0x40u8; 16];
        let l = [0x80u8; 16];
        let r = get_dc_sign_ctx(&t_dim, &a, &l);
        assert!(r <= 2);
    }

    #[test]
    fn test_get_skip_ctx_luma_same_size() {
        let t_dim = TxfmInfo {
            w: 4,
            h: 4,
            lw: 2,
            lh: 2,
            min: 2,
            max: 2,
            sub: 0,
            ctx: 0,
        };
        let a = [0u8; 16];
        let l = [0u8; 16];
        let r = get_skip_ctx(&t_dim, 18, &a, &l, 0, 0, false, false);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_get_skip_ctx_chroma() {
        let t_dim = TxfmInfo {
            w: 1,
            h: 1,
            lw: 0,
            lh: 0,
            min: 0,
            max: 0,
            sub: 0,
            ctx: 0,
        };
        let a = [0x40u8; 16];
        let l = [0x40u8; 16];
        let r = get_skip_ctx(&t_dim, 30, &a, &l, 1, 0, false, false);
        assert_eq!(r, 6);
    }

    #[test]
    fn test_get_skip_ctx_chroma_has_cf() {
        let t_dim = TxfmInfo {
            w: 1,
            h: 1,
            lw: 0,
            lh: 0,
            min: 0,
            max: 0,
            sub: 0,
            ctx: 0,
        };
        let a = [0x00u8; 16];
        let l = [0x00u8; 16];
        let r = get_skip_ctx(&t_dim, 30, &a, &l, 1, 0, false, false);
        assert_eq!(r, 6 + 1 + 1);
    }

    use crate::levels::MvXY;

    fn make_mv(x: i32, y: i32) -> Mv {
        Mv { c: MvXY { y, x } }
    }

    #[test]
    fn test_get_mask_inside_frame() {
        let mut mask = vec![0u8; 256];
        let mv = [make_mv(0, 0), make_mv(0, 0)];
        let r = get_mask(&mut mask, 8, 0, 0, 0, 0, &mv, 0, 0, 2, 2, 1000, 1000);
        assert!(!r);
    }

    #[test]
    fn test_get_mask_outside_frame() {
        let mut mask = vec![0u8; 256];
        let mv = [make_mv(-100, 0), make_mv(0, 0)];
        let r = get_mask(&mut mask, 8, 0, 0, 0, 0, &mv, 0, 0, 2, 2, 100, 100);
        assert!(r);
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn test_get_mask_both_outside() {
        let mut mask = vec![0u8; 256];
        let mv = [make_mv(-1000, -1000), make_mv(-1000, -1000)];
        let r = get_mask(&mut mask, 8, 0, 0, 0, 0, &mv, 0, 0, 1, 1, 100, 100);
        assert!(r);
        for i in 0..4 {
            assert_eq!(mask[i], 32);
        }
    }

    #[test]
    fn test_get_mask_mv0_only_visible() {
        let mut mask = vec![0u8; 256];
        let mv = [make_mv(0, 0), make_mv(-1000, -1000)];
        let r = get_mask(&mut mask, 4, 0, 0, 0, 0, &mv, 0, 0, 1, 1, 100, 100);
        assert!(r);
        for i in 0..4 {
            assert_eq!(mask[i], 64);
        }
    }

    #[test]
    fn test_opfl_mv_adj_zero_det() {
        let r = OpflRegressionData {
            su2: 0,
            suv: 0,
            sv2: 0,
            suw: 10,
            svw: 10,
        };
        let mut dd = OpflMvDeltaBlock::default();
        dd.d[0].x = 99;
        opfl_mv_adj(&r, &mut dd, [1, 1]);
        assert_eq!(dd.d[0].x, 0);
        assert_eq!(dd.d[0].y, 0);
        assert_eq!(dd.d[1].x, 0);
        assert_eq!(dd.d[1].y, 0);
    }

    #[test]
    fn test_opfl_mv_adj_identity_like() {
        let r = OpflRegressionData {
            su2: 100,
            suv: 0,
            sv2: 100,
            suw: 0,
            svw: 0,
        };
        let mut dd = OpflMvDeltaBlock::default();
        opfl_mv_adj(&r, &mut dd, [0, 0]);
        assert_eq!(dd.d[0].x, 0);
        assert_eq!(dd.d[0].y, 0);
    }

    #[test]
    fn test_opfl_mv_adj_clamps() {
        let r = OpflRegressionData {
            su2: 1000,
            suv: 0,
            sv2: 1000,
            suw: 1000000,
            svw: 1000000,
        };
        let mut dd = OpflMvDeltaBlock::default();
        opfl_mv_adj(&r, &mut dd, [127, 127]);
        assert!(dd.d[0].x >= -16 && dd.d[0].x <= 16);
        assert!(dd.d[0].y >= -16 && dd.d[0].y <= 16);
        assert!(dd.d[1].x >= -16 && dd.d[1].x <= 16);
        assert!(dd.d[1].y >= -16 && dd.d[1].y <= 16);
    }

    #[test]
    fn test_scaledown_16pel_i420() {
        let mut mv = [make_mv(100, 200), make_mv(-100, -200)];
        scaledown_16pel_mv_for_chroma(&mut mv, PixelLayout::I420);
        unsafe {
            assert_eq!(mv[0].c.x, 50);
            assert_eq!(mv[0].c.y, 100);
            assert_eq!(mv[1].c.x, -50);
            assert_eq!(mv[1].c.y, -100);
        }
    }

    #[test]
    fn test_scaledown_16pel_i422() {
        let mut mv = [make_mv(100, 200), make_mv(-100, -200)];
        scaledown_16pel_mv_for_chroma(&mut mv, PixelLayout::I422);
        unsafe {
            assert_eq!(mv[0].c.x, 50);
            assert_eq!(mv[0].c.y, 200); // y unchanged
            assert_eq!(mv[1].c.x, -50);
            assert_eq!(mv[1].c.y, -200);
        }
    }

    #[test]
    fn test_scaledown_16pel_i444_noop() {
        let mut mv = [make_mv(100, 200), make_mv(-100, -200)];
        scaledown_16pel_mv_for_chroma(&mut mv, PixelLayout::I444);
        unsafe {
            assert_eq!(mv[0].c.x, 100);
            assert_eq!(mv[0].c.y, 200);
        }
    }

    #[test]
    fn test_scaledown_rounding() {
        let mut mv = [make_mv(3, 5), make_mv(-3, -5)];
        scaledown_16pel_mv_for_chroma(&mut mv, PixelLayout::I420);
        unsafe {
            // positive: (3+1)>>1=2, (5+1)>>1=3
            assert_eq!(mv[0].c.x, 2);
            assert_eq!(mv[0].c.y, 3);
            // negative: (-3+0)>>1=-2, (-5+0)>>1=-3
            assert_eq!(mv[1].c.x, -2);
            assert_eq!(mv[1].c.y, -3);
        }
    }

    #[test]
    fn test_scaleup_8pel_i444() {
        let mut mv = [make_mv(10, 20), make_mv(-5, -10)];
        scaleup_8pel_mv_for_chroma(&mut mv, PixelLayout::I444);
        unsafe {
            assert_eq!(mv[0].c.x, 20);
            assert_eq!(mv[0].c.y, 40);
            assert_eq!(mv[1].c.x, -10);
            assert_eq!(mv[1].c.y, -20);
        }
    }

    #[test]
    fn test_scaleup_8pel_i422() {
        let mut mv = [make_mv(10, 20), make_mv(-5, -10)];
        scaleup_8pel_mv_for_chroma(&mut mv, PixelLayout::I422);
        unsafe {
            assert_eq!(mv[0].c.x, 10); // x unchanged
            assert_eq!(mv[0].c.y, 40);
            assert_eq!(mv[1].c.x, -5);
            assert_eq!(mv[1].c.y, -20);
        }
    }

    #[test]
    fn test_scaleup_8pel_i420_noop() {
        let mut mv = [make_mv(10, 20), make_mv(-5, -10)];
        scaleup_8pel_mv_for_chroma(&mut mv, PixelLayout::I420);
        unsafe {
            assert_eq!(mv[0].c.x, 10);
            assert_eq!(mv[0].c.y, 20);
        }
    }

    #[test]
    fn test_update_temporal_basic() {
        let mut dst = vec![TemporalBlock::default(); 4];
        let r = RefPair { r: [1, 2] };
        let mv = [make_mv(10, 20), make_mv(30, 40)];
        update_temporal(&mut dst, 2, 2, 2, r, &mv, false);
        unsafe {
            assert_eq!(dst[0].r#ref.r[0], 1);
            assert_eq!(dst[0].r#ref.r[1], 2);
            assert_eq!(dst[3].r#ref.r[0], 1);
        }
    }

    #[test]
    fn test_update_temporal_swap() {
        let mut dst = vec![TemporalBlock::default(); 2];
        let r = RefPair { r: [1, 2] };
        let mv = [make_mv(10, 20), make_mv(30, 40)];
        update_temporal(&mut dst, 2, 1, 1, r, &mv, true);
        unsafe {
            assert_eq!(dst[0].r#ref.r[0], 2);
            assert_eq!(dst[0].r#ref.r[1], 1);
        }
    }

    #[test]
    fn test_update_temporal_both_invalid() {
        let mut dst = vec![TemporalBlock::default(); 1];
        let r = RefPair { r: [1, 2] };
        let mv0 = Mv { n: 0x80008000 };
        let mv = [mv0, mv0];
        update_temporal(&mut dst, 1, 1, 1, r, &mv, false);
        unsafe {
            assert_eq!(dst[0].r#ref.pair, -1);
        }
    }

    #[test]
    fn test_update_temporal_first_invalid() {
        let mut dst = vec![TemporalBlock::default(); 1];
        let r = RefPair { r: [1, 2] };
        let mv_inv = Mv { n: 0x80008000 };
        let mv_ok = make_mv(10, 20);
        let mv = [mv_inv, mv_ok];
        update_temporal(&mut dst, 1, 1, 1, r, &mv, false);
        unsafe {
            assert_eq!(dst[0].r#ref.r[0], 2);
            assert_eq!(dst[0].r#ref.r[1], 2);
        }
    }

    #[test]
    fn test_decode_coefs_all_skip_intra() {
        use crate::cdf::{CdfCoefContext, CdfModeContext};
        // Set skip CDF to force all_skip=1 (high probability for symbol 1)
        let mut coef = CdfCoefContext::default();
        // skip[0][0][0..2] - set high prob for "true" (all skip)
        coef.data[0] = 100;
        coef.data[1] = 0; // low threshold → always picks 1
        let mut mode = CdfModeContext::default();
        let buf = [0xFFu8; 16];
        let mut msac = MsacContext::new(&buf, false);
        let a = [0x40u8; 16];
        let l = [0x40u8; 16];
        let params = DecodeCoefParams {
            tx: 0,
            bs: 6,
            plane: 0,
            intra: true,
            fsc: false,
            lossless: false,
            sdp_active: false,
            seg_id: 0,
            y_mode: 0,
            uv_mode: 0,
            seq_fsc: false,
            seq_ist: [false, false],
            seq_cctx: false,
            chroma_dctonly: false,
            reduced_txtp_set: 0,
            tcq_enabled: false,
            layout: PixelLayout::I420,
            u_has_cf: 0,
            cbx: 0,
            cby: 0,
            luma_fsc_map: &[0; 256],
            dq_tbl: [100, 100],
            bitdepth: 8,
            qm: None,
            ss_hor: true,
            ss_ver: true,
        };
        let mut cf = [0i32; 16];
        let mut txtp_val = 0u16;
        let mut res_ctx = 0u8;
        let ret = decode_coefs(
            &mut msac,
            &mut coef,
            &mut mode,
            &a,
            &l,
            &params,
            &mut cf,
            &mut txtp_val,
            &mut res_ctx,
        );
        assert_eq!(ret, -1);
        assert_eq!(res_ctx, 0x40);
        assert_eq!(txtp_val, txtp::DCT_DCT as u16);
    }

    #[test]
    fn test_decode_coefs_all_skip_fsc() {
        use crate::cdf::{CdfCoefContext, CdfModeContext};
        let mut coef = CdfCoefContext::default();
        // skip[1][0][9] for fsc+intra sctx=9 path
        // offset: 1*100 + 0*20 + 9*2 = 118
        coef.data[118] = 100;
        coef.data[119] = 0;
        let mut mode = CdfModeContext::default();
        let buf = [0xFFu8; 16];
        let mut msac = MsacContext::new(&buf, false);
        let a = [0x40u8; 16];
        let l = [0x40u8; 16];
        let params = DecodeCoefParams {
            tx: 0,
            bs: 6,
            plane: 0,
            intra: true,
            fsc: true,
            lossless: false,
            sdp_active: false,
            seg_id: 0,
            y_mode: 0,
            uv_mode: 0,
            seq_fsc: true,
            seq_ist: [false, false],
            seq_cctx: false,
            chroma_dctonly: false,
            reduced_txtp_set: 0,
            tcq_enabled: false,
            layout: PixelLayout::I420,
            u_has_cf: 0,
            cbx: 0,
            cby: 0,
            luma_fsc_map: &[0; 256],
            dq_tbl: [100, 100],
            bitdepth: 8,
            qm: None,
            ss_hor: true,
            ss_ver: true,
        };
        let mut cf = [0i32; 16];
        let mut txtp_val = 0u16;
        let mut res_ctx = 0u8;
        let ret = decode_coefs(
            &mut msac,
            &mut coef,
            &mut mode,
            &a,
            &l,
            &params,
            &mut cf,
            &mut txtp_val,
            &mut res_ctx,
        );
        assert_eq!(ret, -1);
        assert_eq!(res_ctx, 0x40);
        assert_eq!(txtp_val, txtp::IDTX as u16);
    }
}

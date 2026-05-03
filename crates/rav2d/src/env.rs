use crate::dsp::N_SWITCHABLE_FILTERS;
use crate::headers::{FrameHeader, WarpedMotionParams, WarpedMotionType};
use crate::intops::{apply_sign64, iclip, imax};
use crate::levels::{MvXY, RefPair, TIP_FRAME};

#[derive(Clone)]
pub struct BlockContext {
    pub fsc: [u8; 64],
    pub mode: [u8; 64],
    pub midx: [u8; 64],
    pub mrl: [u8; 64],
    pub multi_mrl: [u8; 64],
    pub dip: [u8; 64],
    pub lcoef: [u8; 64],
    pub ccoef: [[u8; 64]; 2],
    pub seg_pred: [u8; 64],
    pub skip_txfm: [u8; 64],
    pub skip_mode: [u8; 64],
    pub intra: [u8; 64],
    pub intrabc: [u8; 64],
    pub morph_pred: [u8; 64],
    pub comp_type: [u8; 64],
    pub r#ref: [[i8; 64]; 2],
    pub motion_mode: [u8; 64],
    pub amvd: [u8; 64],
    pub mvprec: [u8; 64],
    pub filter: [u8; 64],
    pub tx_lpf_y: [u8; 64],
    pub tx_lpf_uv: [u8; 64],
    pub partition: [[u8; 64]; 2],
    pub uvmode: [u8; 64],
    pub pal_sz: [u8; 64],
}

impl Default for BlockContext {
    fn default() -> Self {
        Self {
            fsc: [0; 64],
            mode: [0; 64],
            midx: [0; 64],
            mrl: [0; 64],
            multi_mrl: [0; 64],
            dip: [0; 64],
            lcoef: [0; 64],
            ccoef: [[0; 64]; 2],
            seg_pred: [0; 64],
            skip_txfm: [0; 64],
            skip_mode: [0; 64],
            intra: [0; 64],
            intrabc: [0; 64],
            morph_pred: [0; 64],
            comp_type: [0; 64],
            r#ref: [[0; 64]; 2],
            motion_mode: [0; 64],
            amvd: [0; 64],
            mvprec: [0; 64],
            filter: [0; 64],
            tx_lpf_y: [0; 64],
            tx_lpf_uv: [0; 64],
            partition: [[0; 64]; 2],
            uvmode: [0; 64],
            pal_sz: [0; 64],
        }
    }
}

#[derive(Clone)]
pub struct SBEdgeCtx {
    pub r#ref: [[i8; 64]; 2],
    pub motion_mode: [u8; 64],
}

impl Default for SBEdgeCtx {
    fn default() -> Self {
        Self {
            r#ref: [[0; 64]; 2],
            motion_mode: [0; 64],
        }
    }
}

#[inline(always)]
pub fn get_intra_ctx(
    nx: [&BlockContext; 2],
    xoff: [usize; 2],
    n_ctx: i32,
) -> i32 {
    if n_ctx == 0 {
        return 0;
    }
    let i = (n_ctx - 1) as usize;
    let sum = (nx[0].intra[xoff[0]] != 0 && nx[0].intrabc[xoff[0]] == 0) as i32
        + (nx[1].intra[xoff[1] + i] != 0 && nx[1].intrabc[xoff[1] + i] == 0) as i32;
    sum + n_ctx
}

#[inline(always)]
pub fn sm_flag(b: &BlockContext, idx: usize) -> i32 {
    let m = b.mode[idx];
    (m == 9 || m == 10 || m == 11) as i32
}

#[inline(always)]
pub fn sm_uv_flag(b: &BlockContext, idx: usize) -> i32 {
    let m = b.uvmode[idx];
    (m == 9 || m == 10 || m == 11) as i32
}

#[inline(always)]
pub fn get_poc_diff(order_hint_n_bits: i32, poc0: i32, poc1: i32) -> i32 {
    if order_hint_n_bits == 0 {
        return 0;
    }
    let mask = 1 << (order_hint_n_bits - 1);
    let diff = poc0 - poc1;
    (diff & (mask - 1)) - (diff & mask)
}

#[inline(always)]
pub fn fix_int_mv_precision(mv: &mut MvXY) {
    mv.x = ((mv.x - (mv.x >> 15) + 3) as u32 & !7u32) as i32;
    mv.y = ((mv.y - (mv.y >> 15) + 3) as u32 & !7u32) as i32;
}

#[inline(always)]
pub fn fix_mv_precision(hdr: &FrameHeader, mv: &mut MvXY) {
    if hdr.force_integer_mv != 0 {
        fix_int_mv_precision(mv);
    } else if hdr.mv_precision < 3 {
        mv.x = ((mv.x - (mv.x >> 15)) as u32 & !1u32) as i32;
        mv.y = ((mv.y - (mv.y >> 15)) as u32 & !1u32) as i32;
    }
}

#[inline(always)]
pub fn mv_reduce_prec(mv: &mut MvXY, mv_prec: i32) {
    if mv_prec == 6 {
        return;
    }
    let rnd = 32 >> mv_prec;
    mv.x = mv.x + rnd - (mv.x > 0) as i32;
    mv.y = mv.y + rnd - (mv.y > 0) as i32;
    let mask = !(rnd as u32 * 2 - 1);
    mv.x = (mv.x as u32 & mask) as i32;
    mv.y = (mv.y as u32 & mask) as i32;
}

#[inline(always)]
pub fn get_warpmv_2d(
    matrix: &[i32; 6],
    bx4: i32,
    by4: i32,
    bw4: i32,
    bh4: i32,
    iw4: i32,
    ih4: i32,
    mv_precision: i32,
) -> MvXY {
    let x = bx4 * 4 + bw4 * 2 - 1;
    let y = by4 * 4 + bh4 * 2 - 1;
    let xc = (matrix[2] as i64 - (1 << 16)) * x as i64
        + matrix[3] as i64 * y as i64
        + matrix[0] as i64;
    let yc = (matrix[5] as i64 - (1 << 16)) * y as i64
        + matrix[4] as i64 * x as i64
        + matrix[1] as i64;
    let not_epel = (mv_precision < 6) as i32;
    let shift = 13 + not_epel;
    let rnd = (1i64 << shift) >> 1;
    let max = 0xffff - not_epel;

    let mut res = MvXY {
        y: iclip(
            apply_sign64(((yc.unsigned_abs() as i64 + rnd) >> shift) << not_epel, yc) as i32,
            -max,
            max,
        ),
        x: iclip(
            apply_sign64(((xc.unsigned_abs() as i64 + rnd) >> shift) << not_epel, xc) as i32,
            -max,
            max,
        ),
    };
    res.y = iclip(res.y, -(by4 + bh4 + 4) * 32, (ih4 - by4 + 4) * 32);
    res.x = iclip(res.x, -(bx4 + bw4 + 4) * 32, (iw4 - bx4 + 4) * 32);
    res
}

#[inline(always)]
pub fn get_gmv_2d(
    gmv: &WarpedMotionParams,
    bx4: i32,
    by4: i32,
    bw4: i32,
    bh4: i32,
    iw4: i32,
    ih4: i32,
    hdr: &FrameHeader,
) -> MvXY {
    match gmv.wm_type {
        WarpedMotionType::Affine | WarpedMotionType::RotZoom => {
            let mut res = get_warpmv_2d(
                &gmv.matrix,
                bx4, by4, bw4, bh4, iw4, ih4,
                hdr.mv_precision as i32 + 3,
            );
            if hdr.force_integer_mv != 0 {
                fix_int_mv_precision(&mut res);
            }
            res
        }
        WarpedMotionType::Translation => {
            let mut res = MvXY {
                y: gmv.matrix[0] >> 13,
                x: gmv.matrix[1] >> 13,
            };
            res.y = iclip(res.y, -(by4 + bh4 + 4) * 32, (ih4 - by4 + 4) * 32);
            res.x = iclip(res.x, -(bx4 + bw4 + 4) * 32, (iw4 - bx4 + 4) * 32);
            if hdr.force_integer_mv != 0 {
                fix_int_mv_precision(&mut res);
            }
            res
        }
        WarpedMotionType::Identity | WarpedMotionType::Invalid => MvXY { x: 0, y: 0 },
    }
}

#[inline(always)]
pub fn warp_type(mtx: &[i32; 6]) -> WarpedMotionType {
    if mtx[2] != mtx[5] || mtx[3] != -mtx[4] {
        return WarpedMotionType::Affine;
    }
    if mtx[2] != 0x10000 || mtx[3] != 0 {
        return WarpedMotionType::RotZoom;
    }
    if mtx[0] | mtx[1] != 0 {
        WarpedMotionType::Translation
    } else {
        WarpedMotionType::Identity
    }
}

#[inline(always)]
pub fn get_partition_ctx(
    a: &BlockContext,
    l: &BlockContext,
    b_dim: &[u8],
    plane: usize,
    yb4: usize,
    xb4: usize,
) -> i32 {
    ((a.partition[plane][xb4] >> imax(b_dim[2] as i32 - 1, 0)) & 1) as i32
        + (((l.partition[plane][yb4] >> imax(b_dim[3] as i32 - 1, 0)) & 1) as i32) * 2
}

#[inline(always)]
pub fn get_partition2_ctx(
    a: &BlockContext,
    l: &BlockContext,
    b_dim: &[u8],
    plane: usize,
    dir: i32,
    yb4: usize,
    xb4: usize,
) -> i32 {
    if dir == 0 {
        let hh4 = (b_dim[1] >> 1) as usize;
        ((l.partition[plane][yb4 + hh4] >> (b_dim[3] - 2)) & 1) as i32
            + (((l.partition[plane][yb4] >> (b_dim[3] - 2)) & 1) as i32) * 2
    } else {
        let hw4 = (b_dim[0] >> 1) as usize;
        ((a.partition[plane][xb4 + hw4] >> (b_dim[2] - 2)) & 1) as i32
            + (((a.partition[plane][xb4] >> (b_dim[2] - 2)) & 1) as i32) * 2
    }
}

#[inline(always)]
pub fn get_filter_ctx(
    nb: [&BlockContext; 2],
    boff: [i32; 2],
    r: RefPair,
) -> i32 {
    let (r_ref, comp) = unsafe {
        (r.r[0], (r.r[1] != -1) as i32)
    };
    let flt0 = if boff[0] != -1 {
        let i = boff[0] as usize;
        if nb[0].r#ref[0][i] == r_ref || nb[0].r#ref[1][i] == r_ref {
            nb[0].filter[i] as i32
        } else {
            N_SWITCHABLE_FILTERS as i32
        }
    } else {
        N_SWITCHABLE_FILTERS as i32
    };
    let flt1 = if boff[1] != -1 {
        let i = boff[1] as usize;
        if nb[1].r#ref[0][i] == r_ref || nb[1].r#ref[1][i] == r_ref {
            nb[1].filter[i] as i32
        } else {
            N_SWITCHABLE_FILTERS as i32
        }
    } else {
        N_SWITCHABLE_FILTERS as i32
    };

    if flt0 == flt1 || flt1 == N_SWITCHABLE_FILTERS as i32 {
        comp * 4 + flt0
    } else if flt0 == N_SWITCHABLE_FILTERS as i32 {
        comp * 4 + flt1
    } else {
        comp * 4 + N_SWITCHABLE_FILTERS as i32
    }
}

#[inline(always)]
pub fn get_comp_ctx(
    nx: [&BlockContext; 2],
    xoff: [usize; 2],
    n_ctx: i32,
    refdir: &[i8],
) -> i32 {
    match n_ctx {
        2 => {
            let refa2 = nx[0].r#ref[1][xoff[0]];
            let refb2 = nx[1].r#ref[1][xoff[1]];
            if refa2 == -1 {
                let refa1 = nx[0].r#ref[0][xoff[0]] as usize;
                if refb2 == -1 {
                    let refb1 = nx[1].r#ref[0][xoff[1]] as usize;
                    ((refdir[refa1] == 1) ^ (refdir[refb1] == 1)) as i32
                } else {
                    2 + (nx[0].intrabc[xoff[0]] == 0 && refdir[refa1] != 0) as i32
                }
            } else if refb2 == -1 {
                let refb1 = nx[1].r#ref[0][xoff[1]] as usize;
                2 + (nx[1].intrabc[xoff[1]] == 0 && refdir[refb1] != 0) as i32
            } else {
                4
            }
        }
        1 => {
            let ref2 = nx[0].r#ref[1][xoff[0]];
            if ref2 == -1 {
                let ref1 = nx[0].r#ref[0][xoff[0]] as usize;
                (nx[0].intrabc[xoff[0]] == 0 && refdir[ref1] != 0) as i32
            } else {
                3
            }
        }
        0 => 1,
        _ => unreachable!(),
    }
}

const NEWMV0_MODE_MASK: u32 =
    (1 << 15) | (1 << 20) | (1 << 22) | (1 << 23) |
    (1 << 26) | (1 << 27) | (1 << 28);

const NEWMV1_MODE_MASK: u32 =
    (1 << 19) | (1 << 22) | (1 << 25) | (1 << 27);

const NEWMV_COMP_MODE_MASK: u32 =
    (1 << 15) | (1 << 19) | (1 << 20) | (1 << 22) | (1 << 23) |
    (1 << 25) | (1 << 26) | (1 << 27) | (1 << 28);

#[inline(always)]
pub fn get_snglref_ctx(
    a: &BlockContext,
    l: &BlockContext,
    yb4: usize,
    xb4: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    b_dim: &[u8],
    r#ref: i8,
) -> i32 {
    let mut row = 0i32;
    let mut col = 0i32;
    let mut newmv = 0i32;

    macro_rules! add_matching {
        ($dir:expr, $cnt:expr, $idx:expr) => {
            if $dir.r#ref[0][$idx] == r#ref {
                $cnt += 1;
                newmv += (((1u32 << $dir.mode[$idx]) & NEWMV0_MODE_MASK) != 0) as i32;
            } else if $dir.r#ref[1][$idx] == r#ref {
                $cnt += 1;
                newmv += (((1u32 << $dir.mode[$idx]) & NEWMV1_MODE_MASK) != 0) as i32;
            }
        };
    }

    if have_top {
        add_matching!(a, col, xb4);
        if have_top_right {
            add_matching!(a, col, xb4 + b_dim[0] as usize - 1);
        }
    }
    if have_left {
        add_matching!(l, row, yb4);
        if have_bottom_left {
            add_matching!(l, row, yb4 + b_dim[1] as usize - 1);
        }
    }

    (row != 0) as i32 + (col != 0) as i32 + 2 * (newmv != 0) as i32
}

#[inline(always)]
pub fn get_compref_ctx(
    a: &BlockContext,
    l: &BlockContext,
    yb4: usize,
    xb4: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    b_dim: &[u8],
    r#ref: RefPair,
    tip: RefPair,
) -> i32 {
    let mut row = 0i32;
    let mut col = 0i32;
    let mut newmv = 0i32;
    let (ref0, ref1) = unsafe { (r#ref.r[0], r#ref.r[1]) };
    let (tip0, tip1) = unsafe { (tip.r[0], tip.r[1]) };

    macro_rules! add_matching {
        ($dir:expr, $cnt:expr, $idx:expr) => {
            if $dir.r#ref[0][$idx] == TIP_FRAME as i8
                && tip0 == ref0 && tip1 == ref1
            {
                $cnt += 1;
                newmv += ($dir.mode[$idx] == 15) as i32; // NEWMV
            } else if $dir.r#ref[0][$idx] == ref0 && $dir.r#ref[1][$idx] == ref1 {
                $cnt += 1;
                newmv += (((1u32 << $dir.mode[$idx]) & NEWMV_COMP_MODE_MASK) != 0) as i32;
            }
        };
    }

    if have_top {
        add_matching!(a, col, xb4);
        if have_top_right {
            add_matching!(a, col, xb4 + b_dim[0] as usize - 1);
        }
    }
    if have_left {
        add_matching!(l, row, yb4);
        if have_bottom_left {
            add_matching!(l, row, yb4 + b_dim[1] as usize - 1);
        }
    }

    (row != 0) as i32 + (col != 0) as i32 + 2 * (newmv != 0) as i32
}

#[inline(always)]
pub fn get_cur_frame_segid(
    by: i32,
    bx: i32,
    have_top: bool,
    have_left: bool,
    seg_ctx: &mut i32,
    cur_seg_map: &[u8],
    stride: isize,
) -> u32 {
    let off = (bx as isize + by as isize * stride) as usize;
    if have_left && have_top {
        let l = cur_seg_map[off - 1] as u32;
        let a = cur_seg_map[(off as isize - stride) as usize] as u32;
        let al = cur_seg_map[(off as isize - stride - 1) as usize] as u32;

        if l == a && al == l {
            *seg_ctx = 2;
        } else if l == a || al == l || a == al {
            *seg_ctx = 1;
        } else {
            *seg_ctx = 0;
        }
        if a == al { a } else { l }
    } else {
        *seg_ctx = 0;
        if have_left {
            cur_seg_map[off - 1] as u32
        } else if have_top {
            cur_seg_map[(off as isize - stride) as usize] as u32
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_poc_diff() {
        assert_eq!(get_poc_diff(0, 10, 5), 0);
        assert_eq!(get_poc_diff(4, 10, 5), 5);
        assert_eq!(get_poc_diff(4, 5, 10), -5);
    }

    #[test]
    fn test_fix_int_mv_precision() {
        let mut mv = MvXY { x: 13, y: -5 };
        fix_int_mv_precision(&mut mv);
        assert_eq!(mv.x & 7, 0);
        assert_eq!(mv.y & 7, 0);
    }

    #[test]
    fn test_mv_reduce_prec_noop() {
        let mut mv = MvXY { x: 100, y: -200 };
        let orig = mv;
        mv_reduce_prec(&mut mv, 6);
        assert_eq!(mv.x, orig.x);
        assert_eq!(mv.y, orig.y);
    }

    #[test]
    fn test_warp_type_identity() {
        let mtx = [0, 0, 0x10000, 0, 0, 0x10000];
        assert_eq!(warp_type(&mtx), WarpedMotionType::Identity);
    }

    #[test]
    fn test_warp_type_translation() {
        let mtx = [100, 200, 0x10000, 0, 0, 0x10000];
        assert_eq!(warp_type(&mtx), WarpedMotionType::Translation);
    }

    #[test]
    fn test_warp_type_rotzoom() {
        let mtx = [0, 0, 0x10100, 0x100, -0x100, 0x10100];
        assert_eq!(warp_type(&mtx), WarpedMotionType::RotZoom);
    }

    #[test]
    fn test_warp_type_affine() {
        let mtx = [0, 0, 0x10100, 0x100, 0x50, 0x10200];
        assert_eq!(warp_type(&mtx), WarpedMotionType::Affine);
    }

    #[test]
    fn test_block_context_default() {
        let bc = BlockContext::default();
        assert_eq!(bc.intra[0], 0);
        assert_eq!(bc.r#ref[0][0], 0);
    }
}

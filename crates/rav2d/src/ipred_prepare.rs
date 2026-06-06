use crate::intops::imin;
use crate::levels::*;

pub static MODE_CONV: [[[u8; 2]; 2]; 2] = [
    // DC_PRED
    [
        [DC_128_PRED, TOP_DC_PRED],
        [LEFT_DC_PRED, IntraPredMode::DcPred as u8],
    ],
    // PAETH_PRED
    [
        [DC_128_PRED, IntraPredMode::VertPred as u8],
        [IntraPredMode::HorPred as u8, IntraPredMode::PaethPred as u8],
    ],
];

#[derive(Clone, Copy, Default)]
pub struct EdgeMask {
    pub needs_left: bool,
    pub needs_top: bool,
    pub needs_topleft: bool,
    pub needs_topright: bool,
    pub needs_bottomleft: bool,
}

impl EdgeMask {
    const fn new(left: bool, top: bool, tl: bool, tr: bool, bl: bool) -> Self {
        Self {
            needs_left: left,
            needs_top: top,
            needs_topleft: tl,
            needs_topright: tr,
            needs_bottomleft: bl,
        }
    }
}

pub fn intra_prediction_edge(mode: u8) -> EdgeMask {
    match mode {
        0  /* DcPred */       => EdgeMask::new(true,  true,  false, false, false),
        1  /* VertPred */     => EdgeMask::new(false, true,  false, false, false),
        2  /* HorPred */      => EdgeMask::new(true,  false, false, false, false),
        _ if mode == LEFT_DC_PRED  => EdgeMask::new(true,  false, false, false, false),
        _ if mode == TOP_DC_PRED   => EdgeMask::new(false, true,  false, false, false),
        _ if mode == DC_128_PRED   => EdgeMask::new(false, false, false, false, false),
        _ if mode == Z1_PRED       => EdgeMask::new(false, true,  true,  true,  false),
        _ if mode == Z2_PRED       => EdgeMask::new(true,  true,  true,  false, false),
        _ if mode == Z3_PRED       => EdgeMask::new(true,  false, true,  false, true),
        9  /* SmoothPred */   => EdgeMask::new(true,  true,  false, true,  true),
        10 /* SmoothVPred */  => EdgeMask::new(false, true,  false, false, true),
        11 /* SmoothHPred */  => EdgeMask::new(true,  false, false, true,  false),
        12 /* PaethPred */    => EdgeMask::new(true,  true,  true,  false, false),
        _ if mode == DIP_PRED      => EdgeMask::new(true,  true,  true,  true,  true),
        _ => EdgeMask::default(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_intra_edges_8bpc(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    n_tr: i32,
    n_bl: i32,
    dst: &[u8],
    dst_off: usize,
    stride: usize,
    prefilter_toplevel_sb_edge: Option<&[u8]>,
    mode: u8,
    tw4: i32,
    th4: i32,
    intra_flags: i32,
    tl: &mut [u8],
    tl_o: usize,
) -> u8 {
    prepare_intra_edges(
        crate::pixel::BitDepth8,
        x,
        y,
        w,
        h,
        n_tr,
        n_bl,
        dst,
        dst_off,
        stride,
        prefilter_toplevel_sb_edge,
        mode,
        tw4,
        th4,
        intra_flags,
        tl,
        tl_o,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_intra_edges<BD: crate::pixel::BitDepth>(
    bd: BD,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    n_tr: i32,
    n_bl: i32,
    dst: &[BD::Pixel],
    dst_off: usize,
    stride: usize,
    prefilter_toplevel_sb_edge: Option<&[BD::Pixel]>,
    mode: u8,
    tw4: i32,
    th4: i32,
    intra_flags: i32,
    tl: &mut [BD::Pixel],
    tl_o: usize,
) -> u8 {
    use crate::pixel::Pixel;
    debug_assert!(y < h && x < w);
    // dav2d ipred_prepare_tmpl.c: edge fills use ((1 << bd) >> 1) {+1,-1,+0}.
    let mid = (bd.bitdepth_max() + 1) >> 1;
    let fill_left = BD::Pixel::from_i32(mid + 1); // 129 @ 8bpc
    let fill_top = BD::Pixel::from_i32(mid - 1); // 127 @ 8bpc
    let fill_tl = BD::Pixel::from_i32(mid); // 128 @ 8bpc

    let mut is_dir = false;
    let enable_edge_filter = (intra_flags & ANGLE_USE_EDGE_FILTER_FLAG) != 0;
    let angle = intra_flags & 511;
    let apply_dip = (intra_flags & ANGLE_DIP_FLAG) != 0;
    let apply_ibp = (intra_flags & ANGLE_IBP_FLAG) != 0;
    let mrl_idx = ((intra_flags & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let mrl_mul = (intra_flags & ANGLE_MULTI_MRL_FLAG) != 0;
    let have_left = (intra_flags & ANGLE_HAS_LEFT_FLAG) != 0;
    let have_top = (intra_flags & ANGLE_HAS_TOP_FLAG) != 0;

    let mut mode = mode;
    let mut tl_filter = false;

    match mode {
        1..=8 => {
            is_dir = true;
            if angle <= 90 {
                mode = if angle < 90 && (have_top || apply_ibp) {
                    Z1_PRED
                } else {
                    IntraPredMode::VertPred as u8
                };
            } else if angle < 180 {
                mode = Z2_PRED;
            } else {
                mode = if angle > 180 && (have_left || apply_ibp) {
                    Z3_PRED
                } else {
                    IntraPredMode::HorPred as u8
                };
            }
            tl_filter = (Z1_PRED..=Z3_PRED).contains(&mode)
                && have_left
                && have_top
                && mrl_idx == 0
                && enable_edge_filter
                && tw4 + th4 >= 6;
        }
        0 => {
            mode = if apply_dip {
                DIP_PRED
            } else {
                MODE_CONV[0][have_left as usize][have_top as usize]
            };
        }
        12 => {
            debug_assert!(!apply_dip);
            mode = MODE_CONV[1][have_left as usize][have_top as usize];
        }
        _ => {}
    }
    debug_assert!(mrl_idx == 0 || is_dir);

    let mut e = intra_prediction_edge(mode);
    if (mode == Z1_PRED || mode == Z3_PRED) && apply_ibp {
        e = intra_prediction_edge(DIP_PRED);
    }

    let mut top_buf: &[BD::Pixel] = dst;
    let mut dst_top_off: usize = 0;
    let mut dst_top2_off: usize = 0;
    let mut top_stride_val: usize = stride;

    if have_top
        && ((e.needs_top || e.needs_topleft || e.needs_topright)
            || ((e.needs_left || e.needs_bottomleft) && !have_left))
    {
        if let Some(prefilter) = prefilter_toplevel_sb_edge {
            top_buf = prefilter;
            dst_top_off = x as usize * 4;
            dst_top2_off = x as usize * 4;
            top_stride_val = 0;
        } else {
            dst_top_off = dst_off - (mrl_idx + 1) * stride;
            dst_top2_off = dst_off - stride;
        }
    }

    let tw = (tw4 as usize) << 2;
    let th = (th4 as usize) << 2;
    let diag_mrl_idx = if (Z1_PRED..=Z3_PRED).contains(&mode) {
        mrl_idx
    } else {
        0
    };
    let e_stride = (tw + th) * 2 + diag_mrl_idx * 3 + 1;
    let o = tl_o as isize;

    // Left edge
    if e.needs_left || tl_filter {
        let mut sz = if e.needs_left { th } else { 1 };
        let mut sz2 = th;
        if e.needs_bottomleft {
            sz += if apply_dip {
                th >> 2
            } else if is_dir {
                tw + 2 * diag_mrl_idx
            } else {
                1
            };
            sz2 = sz - 2 * diag_mrl_idx;
        }
        let left_base = o - diag_mrl_idx as isize - 1;
        let left2_base = o + e_stride as isize - 1;

        if have_left {
            let left_src = dst_off - 1 - mrl_idx;
            let mut px_have = if e.needs_left {
                imin(th as i32, (h - y) << 2) as usize
            } else {
                1
            };
            let mut i = 0usize;
            while i < px_have {
                tl[(left_base - i as isize) as usize] = dst[left_src + stride * i];
                i += 1;
            }
            if e.needs_bottomleft && n_bl > 0 {
                px_have += imin(n_bl << 2, (sz - th) as i32) as usize;
                while i < px_have {
                    tl[(left_base - i as isize) as usize] = dst[left_src + stride * i];
                    i += 1;
                }
            }
            if px_have < sz {
                let fill_val = tl[(left_base + 1 - i as isize) as usize];
                let start = (left_base + 1 - sz as isize) as usize;
                tl[start..start + sz - px_have].fill(fill_val);
            }
            if mrl_mul {
                let left2_src = dst_off - 1;
                let px2 = imin(i as i32, sz2 as i32) as usize;
                for j in 0..px2 {
                    tl[(left2_base - j as isize) as usize] = dst[left2_src + stride * j];
                }
                if px2 < sz2 {
                    let fill_val = tl[(left2_base + 1 - px2 as isize) as usize];
                    let start = (left2_base + 1 - sz2 as isize) as usize;
                    tl[start..start + sz2 - px2].fill(fill_val);
                }
            }
        } else {
            let fill_val = if have_top { top_buf[dst_top_off] } else { fill_left };
            let start = (left_base + 1 - sz as isize) as usize;
            tl[start..start + sz].fill(fill_val);
            if mrl_mul {
                let fill_val2 = if have_top { top_buf[dst_top2_off] } else { fill_left };
                let start2 = (left2_base + 1 - sz2 as isize) as usize;
                tl[start2..start2 + sz2].fill(fill_val2);
            }
        }
    } else if e.needs_bottomleft {
        debug_assert!(mode == IntraPredMode::SmoothVPred as u8);
        let bl_idx = (o - 1 - th as isize) as usize;
        if !have_left {
            tl[bl_idx] = if have_top { top_buf[dst_top_off] } else { fill_left };
        } else if n_bl <= 0 {
            let row = imin(th as i32, (h - y) << 2) as usize - 1;
            tl[bl_idx] = dst[dst_off + stride * row - 1];
        } else {
            tl[bl_idx] = dst[dst_off + stride * th - 1];
        }
    }

    // Top edge
    if e.needs_top || tl_filter {
        let mut sz = if e.needs_top { tw } else { 1 };
        let mut sz2 = tw;
        if e.needs_topright {
            sz += if apply_dip {
                tw >> 2
            } else if is_dir {
                th + 2 * diag_mrl_idx
            } else {
                1
            };
            sz2 = sz - 2 * diag_mrl_idx;
        }
        let top_base = (o + diag_mrl_idx as isize + 1) as usize;
        let top2_base = (o + e_stride as isize + 1) as usize;

        if have_top {
            let mut px_have = if e.needs_top {
                imin(tw as i32, (w - x) << 2) as usize
            } else {
                1
            };
            tl[top_base..top_base + px_have]
                .copy_from_slice(&top_buf[dst_top_off..dst_top_off + px_have]);
            if e.needs_topright && n_tr > 0 {
                px_have += imin(n_tr << 2, (sz - tw) as i32) as usize;
                tl[top_base + tw..top_base + px_have]
                    .copy_from_slice(&top_buf[dst_top_off + tw..dst_top_off + px_have]);
            }
            if px_have < sz {
                let fill_val = tl[top_base + px_have - 1];
                tl[top_base + px_have..top_base + sz].fill(fill_val);
            }
            if mrl_mul {
                let px2 = imin(px_have as i32, sz2 as i32) as usize;
                tl[top2_base..top2_base + px2]
                    .copy_from_slice(&top_buf[dst_top2_off..dst_top2_off + px2]);
                if px2 < sz2 {
                    let fill_val = tl[top2_base + px2 - 1];
                    tl[top2_base + px2..top2_base + sz2].fill(fill_val);
                }
            }
        } else {
            let fill_val = if have_left {
                dst[dst_off - 1 - mrl_idx]
            } else {
                fill_top
            };
            tl[top_base..top_base + sz].fill(fill_val);
            if mrl_mul {
                let fill_val2 = if have_left { dst[dst_off - 1] } else { fill_top };
                tl[top2_base..top2_base + sz2].fill(fill_val2);
            }
        }
    } else if e.needs_topright {
        debug_assert!(mode == IntraPredMode::SmoothHPred as u8);
        let tr_idx = (o + 1) as usize + tw;
        if !have_top {
            tl[tr_idx] = if have_left { dst[dst_off - 1] } else { fill_top };
        } else if n_tr <= 0 {
            let col = imin(tw as i32, (w - x) << 2) as usize - 1;
            tl[tr_idx] = top_buf[dst_top_off + col];
        } else {
            tl[tr_idx] = top_buf[dst_top_off + tw];
        }
    }

    // Topleft pixel
    if e.needs_topleft {
        debug_assert!(diag_mrl_idx == mrl_idx);
        if have_top && have_left {
            for i in (-(mrl_idx as isize))..0 {
                tl[(o + i) as usize] = top_buf[(dst_top_off as isize - mrl_idx as isize - 1
                    + (-i) * top_stride_val as isize)
                    as usize];
            }
            for i in 0..=mrl_idx as isize {
                tl[(o + i) as usize] =
                    top_buf[(dst_top_off as isize - mrl_idx as isize - 1 + i) as usize];
            }
        } else {
            let v = if have_left {
                dst[dst_off - 1 - mrl_idx]
            } else if have_top {
                top_buf[dst_top_off]
            } else {
                fill_tl
            };
            let start = (o - mrl_idx as isize) as usize;
            tl[start..start + 2 * mrl_idx + 1].fill(v);
        }
        tl[(o + e_stride as isize) as usize] = if have_left {
            if have_top {
                top_buf[dst_top2_off - 1]
            } else {
                dst[dst_off - 1]
            }
        } else if have_top {
            top_buf[dst_top2_off]
        } else {
            fill_tl
        };

        if tl_filter {
            let c0: i32 = tl[tl_o].into();
            let cm: i32 = tl[tl_o - 1].into();
            let cp: i32 = tl[tl_o + 1].into();
            let c = c0 + (cm + c0 + cp) * 5;
            tl[tl_o] = BD::Pixel::from_i32((c + 8) >> 4);
        }
    }

    mode
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_conv_dc() {
        assert_eq!(MODE_CONV[0][0][0], DC_128_PRED);
        assert_eq!(MODE_CONV[0][0][1], TOP_DC_PRED);
        assert_eq!(MODE_CONV[0][1][0], LEFT_DC_PRED);
        assert_eq!(MODE_CONV[0][1][1], IntraPredMode::DcPred as u8);
    }

    #[test]
    fn test_mode_conv_paeth() {
        assert_eq!(MODE_CONV[1][0][0], DC_128_PRED);
        assert_eq!(MODE_CONV[1][0][1], IntraPredMode::VertPred as u8);
        assert_eq!(MODE_CONV[1][1][0], IntraPredMode::HorPred as u8);
        assert_eq!(MODE_CONV[1][1][1], IntraPredMode::PaethPred as u8);
    }

    #[test]
    fn test_edge_mask_dc_pred() {
        let e = intra_prediction_edge(IntraPredMode::DcPred as u8);
        assert!(e.needs_left && e.needs_top);
        assert!(!e.needs_topleft && !e.needs_topright && !e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_vert_pred() {
        let e = intra_prediction_edge(IntraPredMode::VertPred as u8);
        assert!(e.needs_top);
        assert!(!e.needs_left);
    }

    #[test]
    fn test_edge_mask_paeth_pred() {
        let e = intra_prediction_edge(IntraPredMode::PaethPred as u8);
        assert!(e.needs_left && e.needs_top && e.needs_topleft);
        assert!(!e.needs_topright && !e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_z1() {
        let e = intra_prediction_edge(Z1_PRED);
        assert!(e.needs_top && e.needs_topright && e.needs_topleft);
        assert!(!e.needs_left && !e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_dip() {
        let e = intra_prediction_edge(DIP_PRED);
        assert!(e.needs_left && e.needs_top && e.needs_topleft);
        assert!(e.needs_topright && e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_dc128() {
        let e = intra_prediction_edge(DC_128_PRED);
        assert!(!e.needs_left && !e.needs_top && !e.needs_topleft);
    }

    #[test]
    fn test_edge_mask_unknown() {
        let e = intra_prediction_edge(255);
        assert!(!e.needs_left && !e.needs_top);
    }

    fn make_test_image() -> (Vec<u8>, usize) {
        let stride = 16usize;
        let mut dst = vec![0u8; stride * 16];
        for r in 0..16 {
            for c in 0..16 {
                dst[r * stride + c] = (r * 16 + c) as u8;
            }
        }
        (dst, stride)
    }

    #[test]
    fn test_prepare_dc_pred_have_both() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_LEFT_FLAG | ANGLE_HAS_TOP_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 0, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, IntraPredMode::DcPred as u8);
        assert_eq!(tl[o - 1], dst[dst_off - 1]);
        assert_eq!(tl[o - 2], dst[dst_off + stride - 1]);
        assert_eq!(tl[o - 3], dst[dst_off + stride * 2 - 1]);
        assert_eq!(tl[o - 4], dst[dst_off + stride * 3 - 1]);
        assert_eq!(tl[o + 1], dst[dst_off - stride]);
        assert_eq!(tl[o + 2], dst[dst_off - stride + 1]);
        assert_eq!(tl[o + 3], dst[dst_off - stride + 2]);
        assert_eq!(tl[o + 4], dst[dst_off - stride + 3]);
    }

    #[test]
    fn test_prepare_dc_pred_no_left() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_TOP_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 0, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, TOP_DC_PRED);
        assert_eq!(tl[o + 1], dst[dst_off - stride]);
    }

    #[test]
    fn test_prepare_dc_pred_no_top() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_LEFT_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 0, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, LEFT_DC_PRED);
        assert_eq!(tl[o - 1], dst[dst_off - 1]);
    }

    #[test]
    fn test_prepare_dc_pred_no_edges() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 0, 1, 1, 0, &mut tl, o,
        );
        assert_eq!(result, DC_128_PRED);
    }

    #[test]
    fn test_prepare_z1_pred() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = 45 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 2, 0, &dst, dst_off, stride, None, 1, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, Z1_PRED);
        assert_eq!(tl[o + 1], dst[dst_off - stride]);
        assert_eq!(tl[o], dst[dst_off - stride - 1]);
    }

    #[test]
    fn test_prepare_z2_pred() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = 135 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 1, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, Z2_PRED);
    }

    #[test]
    fn test_prepare_z3_pred() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = 200 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 2, &dst, dst_off, stride, None, 1, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, Z3_PRED);
        assert_eq!(tl[o - 1], dst[dst_off - 1]);
        assert_eq!(tl[o], dst[dst_off - stride - 1]);
    }

    #[test]
    fn test_prepare_dip_mode() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_DIP_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 2, 2, &dst, dst_off, stride, None, 0, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, DIP_PRED);
        assert_eq!(tl[o - 1], dst[dst_off - 1]);
        assert_eq!(tl[o + 1], dst[dst_off - stride]);
        assert_eq!(tl[o], dst[dst_off - stride - 1]);
    }

    #[test]
    fn test_prepare_no_left_fills_top_val() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_TOP_FLAG;
        prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 0, 1, 1, flags, &mut tl, o,
        );
        let top_val = dst[dst_off - stride];
        assert_eq!(tl[o + 1], top_val);
    }

    #[test]
    fn test_prepare_tl_filter() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = 45 | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG | ANGLE_USE_EDGE_FILTER_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 1, 3, 3, flags, &mut tl, o,
        );
        assert_eq!(result, Z1_PRED);
        let raw_tl = dst[dst_off - stride - 1] as i32;
        let left_val = tl[o - 1] as i32;
        let top_val = tl[o + 1] as i32;
        let expected = ((raw_tl + (left_val + raw_tl + top_val) * 5 + 8) >> 4) as u8;
        assert_eq!(tl[o], expected);
    }

    #[test]
    fn test_prepare_paeth_have_both() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        let result = prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 0, &dst, dst_off, stride, None, 12, 1, 1, flags, &mut tl, o,
        );
        assert_eq!(result, IntraPredMode::PaethPred as u8);
        assert_eq!(tl[o], dst[dst_off - stride - 1]);
    }

    #[test]
    fn test_prepare_smooth_v_bottomleft() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        prepare_intra_edges_8bpc(
            1, 1, 4, 4, 0, 1, &dst, dst_off, stride, None, 10, 1, 1, flags, &mut tl, o,
        );
        let th = 4usize;
        assert_eq!(tl[o - 1 - th], dst[dst_off + stride * th - 1]);
    }

    #[test]
    fn test_prepare_smooth_h_topright() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        prepare_intra_edges_8bpc(
            1, 1, 4, 4, 1, 0, &dst, dst_off, stride, None, 11, 1, 1, flags, &mut tl, o,
        );
        let tw = 4usize;
        assert_eq!(tl[o + 1 + tw], dst[dst_off - stride + tw]);
    }

    #[test]
    fn test_prepare_left_extension() {
        let (dst, stride) = make_test_image();
        let dst_off = 4 * stride + 4;
        let mut tl = vec![0u8; 512];
        let o = 256usize;
        let flags = ANGLE_HAS_LEFT_FLAG | ANGLE_HAS_TOP_FLAG;
        prepare_intra_edges_8bpc(
            1, 3, 4, 4, 0, 0, &dst, dst_off, stride, None, 0, 1, 1, flags, &mut tl, o,
        );
        let px_have = imin(4, (4 - 3) << 2) as usize;
        assert_eq!(px_have, 4);
        assert_eq!(tl[o - 1], dst[dst_off - 1]);
    }
}

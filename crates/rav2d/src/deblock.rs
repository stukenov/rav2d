use crate::headers::FrameHeader;
use crate::intops::{iclip, imin};
use crate::lf_mask::{deblock_quant_thr, deblock_side_thr};

pub static MAX_WIDTH_Y: [i8; 4] = [1, 3, 6, 8];
pub static MAX_WIDTH_UV: [i8; 3] = [1, 3, 4];

pub static Q_FIRST: [i8; 3] = [45, 40, 32];
pub static Q_THRESH_MULTS: [i8; 8] = [32, 25, 19, 19, 0, 18, 0, 17];
pub static W_MULT: [i8; 8] = [85, 51, 37, 28, 0, 20, 0, 15];

pub fn init_deblock_thr_lut_y(
    frame_hdr: &FrameHeader,
    hbd: i32,
    dir: usize,
    qidx: i32,
    lut: &mut [[u32; 16]; 2],
) {
    let qmax = 255 + 48 * hbd;
    let seg = &frame_hdr.segmentation;
    let n = if seg.enabled != 0 { 8 } else { 1 };
    for i in 0..n {
        let yac = if seg.enabled != 0 {
            iclip(qidx + seg.d.delta_q[i] as i32, 0, qmax)
        } else {
            qidx
        };
        let dir_yac = yac + 8 * frame_hdr.deblock.delta_q_y[dir] as i32;
        lut[0][i] = deblock_quant_thr(hbd, dir_yac);
        lut[1][i] = deblock_side_thr(hbd, dir_yac);
    }
}

pub fn init_deblock_thr_lut_uv(
    frame_hdr: &FrameHeader,
    hbd: i32,
    qidx: i32,
    lut: &mut [[[u32; 16]; 2]; 2],
) {
    let qmax = 255 + 48 * hbd;
    let seg = &frame_hdr.segmentation;
    let n = if seg.enabled != 0 { 8 } else { 1 };
    for i in 0..n {
        let yac = if seg.enabled != 0 {
            iclip(qidx + seg.d.delta_q[i] as i32, 0, qmax)
        } else {
            qidx
        };
        let uac = yac + frame_hdr.quant.uac_delta as i32
            + 8 * frame_hdr.deblock.delta_q_u as i32;
        lut[0][0][i] = deblock_quant_thr(hbd, uac);
        lut[0][1][i] = deblock_side_thr(hbd, uac);
        let vac = yac + frame_hdr.quant.vac_delta as i32
            + 8 * frame_hdr.deblock.delta_q_v as i32;
        lut[1][0][i] = deblock_quant_thr(hbd, vac);
        lut[1][1][i] = deblock_side_thr(hbd, vac);
    }
}

fn filter_choice_8bpc(
    buf: &[u8],
    s: isize,
    t: isize,
    stride: isize,
    max_width_neg: i32,
    max_width_pos: i32,
    q_thr: u32,
    side_thr: u32,
) -> i32 {
    let mut sd = [0u32; 4];
    for dist in -2i32..2 {
        let d = dist as isize;
        let ds = (buf[(s + (d - 1) * stride) as usize] as i32
            - (buf[(s + d * stride) as usize] as i32) * 2
            + buf[(s + (d + 1) * stride) as usize] as i32).unsigned_abs();
        let dt = (buf[(t + (d - 1) * stride) as usize] as i32
            - (buf[(t + d * stride) as usize] as i32) * 2
            + buf[(t + (d + 1) * stride) as usize] as i32).unsigned_abs();
        sd[(dist + 2) as usize] = (ds + dt + 1) >> 1;
    }

    let high_deriv = sd[0].max(sd[3]);
    if high_deriv > side_thr { return 0; }
    if max_width_pos == 1 { return 1; }

    let side_thr2 = side_thr >> 2;
    let mut transition = sd[1] + sd[2];
    if high_deriv > side_thr2 { return 1; }
    if transition > q_thr * 4 { return 1; }

    let side_thr3 = side_thr >> 3;
    if high_deriv > side_thr3 { return 2; }
    if transition > q_thr * 3 { return 2; }

    let end_thr = (side_thr * 3) >> 4;

    if max_width_neg >= 3 {
        let ds = (buf[(s - stride) as usize] as i32
            - buf[(s - 4 * stride) as usize] as i32
            - 3 * (buf[(s - stride) as usize] as i32
                - buf[(s - 2 * stride) as usize] as i32)).unsigned_abs();
        let dt = (buf[(t - stride) as usize] as i32
            - buf[(t - 4 * stride) as usize] as i32
            - 3 * (buf[(t - stride) as usize] as i32
                - buf[(t - 2 * stride) as usize] as i32)).unsigned_abs();
        if ((ds + dt + 1) >> 1) > end_thr { return 2; }
    }

    let ds = (buf[s as usize] as i32 - buf[(s + 3 * stride) as usize] as i32
        - 3 * (buf[s as usize] as i32 - buf[(s + stride) as usize] as i32)).unsigned_abs();
    let dt = (buf[t as usize] as i32 - buf[(t + 3 * stride) as usize] as i32
        - 3 * (buf[t as usize] as i32 - buf[(t + stride) as usize] as i32)).unsigned_abs();
    if ((ds + dt + 1) >> 1) > end_thr { return 2; }
    if max_width_pos == 3 { return 3; }

    transition <<= 4;
    let mut prev_dist = 3i32;
    let mut dist = 4i32;
    while dist <= max_width_pos {
        let q_thr4 = q_thr * Q_FIRST[((dist - 4) >> 1) as usize] as u32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        if transition > q_thr4 { return prev_dist; }
        let dist2 = imin(7, dist);

        if max_width_neg >= dist2 {
            let ds = (buf[(s - stride) as usize] as i32
                - buf[(s + (-dist2 as isize - 1) * stride) as usize] as i32
                - dist2 * (buf[(s - stride) as usize] as i32
                    - buf[(s - 2 * stride) as usize] as i32)).unsigned_abs();
            let dt = (buf[(t - stride) as usize] as i32
                - buf[(t + (-dist2 as isize - 1) * stride) as usize] as i32
                - dist2 * (buf[(t - stride) as usize] as i32
                    - buf[(t - 2 * stride) as usize] as i32)).unsigned_abs();
            if ((ds + dt + 1) >> 1) > end_thr4 { return prev_dist; }
        }

        let ds = (buf[s as usize] as i32
            - buf[(s + dist2 as isize * stride) as usize] as i32
            - dist2 * (buf[s as usize] as i32
                - buf[(s + stride) as usize] as i32)).unsigned_abs();
        let dt = (buf[t as usize] as i32
            - buf[(t + dist2 as isize * stride) as usize] as i32
            - dist2 * (buf[t as usize] as i32
                - buf[(t + stride) as usize] as i32)).unsigned_abs();
        if ((ds + dt + 1) >> 1) > end_thr4 { return prev_dist; }

        prev_dist = dist;
        dist += 2;
    }

    max_width_pos
}

fn deblock_8bpc(
    dst: &mut [u8],
    off: isize,
    q_thr: u32,
    side_thr: u32,
    stridea: isize,
    strideb: isize,
    max_width_pos: i32,
    max_width_neg: i32,
    pos_lossless: bool,
    neg_lossless: bool,
) {
    let width = filter_choice_8bpc(
        dst, off, off + 3 * stridea, strideb,
        max_width_neg, max_width_pos, q_thr, side_thr,
    );
    let width_neg = imin(width, max_width_neg);
    let width_pos = width;

    if width_pos < 1 { return; }

    let q_thr_clamp = q_thr as i32 * Q_THRESH_MULTS[(width - 1) as usize] as i32;
    let mut dp = off;
    for _ in 0..4 {
        let d0 = dst[dp as usize] as i32;
        let dm1 = dst[(dp - strideb) as usize] as i32;
        let dp1 = dst[(dp + strideb) as usize] as i32;
        let dm2 = dst[(dp - 2 * strideb) as usize] as i32;
        let delta_m2 = iclip(
            4 * (3 * (d0 - dm1) - (dp1 - dm2)),
            -q_thr_clamp, q_thr_clamp,
        );

        if !neg_lossless {
            let delta_m2_neg = delta_m2 * W_MULT[(width_neg - 1) as usize] as i32;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * strideb) as usize;
                let diff = (delta_m2_neg * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = iclip(dst[idx] as i32 + diff, 0, 255) as u8;
            }
        }

        if !pos_lossless {
            let delta_m2_pos = delta_m2 * W_MULT[(width_pos - 1) as usize] as i32;
            for j in 0..width_pos {
                let idx = (dp + j as isize * strideb) as usize;
                let diff = (delta_m2_pos * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = iclip(dst[idx] as i32 - diff, 0, 255) as u8;
            }
        }

        dp += stridea;
    }
}

pub fn deblock_h_sb64y_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32 | vmask[3] as u32;
    let mut y: u32 = 1;
    let mut dp = dst_off;
    let mut qi: usize = 0;
    while (vm & !(y - 1)) != 0 {
        if (vm & y) != 0 {
            let idx = if (vmask[3] as u32 & y) != 0 { 3usize }
                else if (vmask[2] as u32 & y) != 0 { 2 }
                else { ((vmask[1] as u32 & y) != 0) as usize };
            let max_width_pos = MAX_WIDTH_Y[idx] as i32;
            let max_width_neg = if edge { imin(6, max_width_pos) } else { max_width_pos };
            deblock_8bpc(
                dst, dp as isize,
                q_thr[qi] as u32, side_thr[qi] as u32,
                stride as isize, 1,
                max_width_pos, max_width_neg,
                (ll_mask[1] as u32 & y) != 0,
                (ll_mask[0] as u32 & y) != 0,
            );
        }
        y <<= 1;
        dp += 4 * stride;
        qi += 1;
    }
}

pub fn deblock_v_sb64y_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32 | vmask[3] as u32;
    let mut x: u32 = 1;
    let mut dp = dst_off;
    let mut qi: usize = 0;
    while (vm & !(x - 1)) != 0 {
        if (vm & x) != 0 {
            let idx = if (vmask[3] as u32 & x) != 0 { 3usize }
                else if (vmask[2] as u32 & x) != 0 { 2 }
                else { ((vmask[1] as u32 & x) != 0) as usize };
            let max_width_pos = MAX_WIDTH_Y[idx] as i32;
            let max_width_neg = if edge { imin(6, max_width_pos) } else { max_width_pos };
            deblock_8bpc(
                dst, dp as isize,
                q_thr[qi] as u32, side_thr[qi] as u32,
                1, stride as isize,
                max_width_pos, max_width_neg,
                (ll_mask[1] as u32 & x) != 0,
                (ll_mask[0] as u32 & x) != 0,
            );
        }
        x <<= 1;
        dp += 4;
        qi += 1;
    }
}

pub fn deblock_h_sb64uv_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32;
    let mut y: u32 = 1;
    let mut dp = dst_off;
    let mut qi: usize = 0;
    while (vm & !(y - 1)) != 0 {
        if (vm & y) != 0 {
            let idx = if (vmask[2] as u32 & y) != 0 { 2usize }
                else { ((vmask[1] as u32 & y) != 0) as usize };
            let max_width_pos = MAX_WIDTH_UV[idx] as i32;
            let max_width_neg = if edge { imin(2, max_width_pos) } else { max_width_pos };
            deblock_8bpc(
                dst, dp as isize,
                q_thr[qi] as u32, side_thr[qi] as u32,
                stride as isize, 1,
                max_width_pos, max_width_neg,
                (ll_mask[1] as u32 & y) != 0,
                (ll_mask[0] as u32 & y) != 0,
            );
        }
        y <<= 1;
        dp += 4 * stride;
        qi += 1;
    }
}

pub fn deblock_v_sb64uv_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32;
    let mut x: u32 = 1;
    let mut dp = dst_off;
    let mut qi: usize = 0;
    while (vm & !(x - 1)) != 0 {
        if (vm & x) != 0 {
            let idx = if (vmask[2] as u32 & x) != 0 { 2usize }
                else { ((vmask[1] as u32 & x) != 0) as usize };
            let max_width_pos = MAX_WIDTH_UV[idx] as i32;
            let max_width_neg = if edge { imin(2, max_width_pos) } else { max_width_pos };
            deblock_8bpc(
                dst, dp as isize,
                q_thr[qi] as u32, side_thr[qi] as u32,
                1, stride as isize,
                max_width_pos, max_width_neg,
                (ll_mask[1] as u32 & x) != 0,
                (ll_mask[0] as u32 & x) != 0,
            );
        }
        x <<= 1;
        dp += 4;
        qi += 1;
    }
}

pub fn transpose_lossless_mask(
    dst_mask: &mut [u16],
    src_mask: &[[u16; 4]],
    x64: usize,
    ss_hor: u32,
    ss_ver: u32,
) {
    let w = (16 >> ss_hor) as usize;
    dst_mask[0] = dst_mask[w];
    let h = 16u32 >> ss_ver;
    for x in 0..w {
        let mut col_mask: u32 = 0;
        for y in 0..h {
            col_mask |= ((src_mask[y as usize][x64] >> x) as u32 & 1) << y;
        }
        dst_mask[x + 1] = col_mask as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::FrameHeader;

    fn make_frame_hdr() -> FrameHeader {
        FrameHeader::default()
    }

    #[test]
    fn test_max_width_y() {
        assert_eq!(MAX_WIDTH_Y[0], 1);
        assert_eq!(MAX_WIDTH_Y[3], 8);
    }

    #[test]
    fn test_max_width_uv() {
        assert_eq!(MAX_WIDTH_UV[0], 1);
        assert_eq!(MAX_WIDTH_UV[2], 4);
    }

    #[test]
    fn test_q_first() {
        assert!(Q_FIRST[0] > Q_FIRST[2]);
    }

    #[test]
    fn test_q_thresh_mults_zero_entries() {
        assert_eq!(Q_THRESH_MULTS[4], 0);
        assert_eq!(Q_THRESH_MULTS[6], 0);
    }

    #[test]
    fn test_w_mult_zero_entries() {
        assert_eq!(W_MULT[4], 0);
        assert_eq!(W_MULT[6], 0);
    }

    #[test]
    fn test_init_deblock_thr_lut_y_no_seg() {
        let hdr = make_frame_hdr();
        let mut lut = [[0u32; 16]; 2];
        init_deblock_thr_lut_y(&hdr, 0, 0, 128, &mut lut);
        assert!(lut[0][0] > 0 || lut[1][0] > 0);
    }

    #[test]
    fn test_init_deblock_thr_lut_y_with_seg() {
        let mut hdr = make_frame_hdr();
        hdr.segmentation.enabled = 1;
        hdr.segmentation.d.delta_q[0] = 0;
        hdr.segmentation.d.delta_q[1] = 10;
        hdr.segmentation.d.delta_q[2] = -10;
        let mut lut = [[0u32; 16]; 2];
        init_deblock_thr_lut_y(&hdr, 0, 0, 128, &mut lut);
        let mut lut2 = [[0u32; 16]; 2];
        init_deblock_thr_lut_y(&hdr, 0, 0, 138, &mut lut2);
        assert_eq!(lut[0][1], lut2[0][0]);
    }

    #[test]
    fn test_init_deblock_thr_lut_y_dir() {
        let mut hdr = make_frame_hdr();
        hdr.deblock.delta_q_y[0] = 2;
        hdr.deblock.delta_q_y[1] = -1;
        let mut lut0 = [[0u32; 16]; 2];
        let mut lut1 = [[0u32; 16]; 2];
        init_deblock_thr_lut_y(&hdr, 0, 0, 128, &mut lut0);
        init_deblock_thr_lut_y(&hdr, 0, 1, 128, &mut lut1);
        assert_ne!(lut0[0][0], lut1[0][0]);
    }

    #[test]
    fn test_init_deblock_thr_lut_uv_no_seg() {
        let hdr = make_frame_hdr();
        let mut lut = [[[0u32; 16]; 2]; 2];
        init_deblock_thr_lut_uv(&hdr, 0, 128, &mut lut);
        assert_eq!(lut[0][0][0], lut[1][0][0]);
    }

    #[test]
    fn test_init_deblock_thr_lut_uv_delta() {
        let mut hdr = make_frame_hdr();
        hdr.quant.uac_delta = 5;
        hdr.quant.vac_delta = -5;
        let mut lut = [[[0u32; 16]; 2]; 2];
        init_deblock_thr_lut_uv(&hdr, 0, 128, &mut lut);
        assert_ne!(lut[0][0][0], lut[1][0][0]);
    }

    #[test]
    fn test_init_deblock_thr_lut_y_clamps() {
        let mut hdr = make_frame_hdr();
        hdr.segmentation.enabled = 1;
        hdr.segmentation.d.delta_q[0] = 500;
        let mut lut = [[0u32; 16]; 2];
        init_deblock_thr_lut_y(&hdr, 0, 0, 200, &mut lut);
        let mut lut_max = [[0u32; 16]; 2];
        init_deblock_thr_lut_y(&hdr, 0, 0, 255, &mut lut_max);
        assert!(lut[0][0] <= lut_max[0][0] || lut[0][0] >= lut_max[0][0]);
    }

    #[test]
    fn test_filter_choice_uniform() {
        let buf = vec![128u8; 64];
        let stride = 1isize;
        let w = filter_choice_8bpc(&buf, 16, 19, stride, 3, 3, 10, 100);
        assert_eq!(w, 3);
    }

    #[test]
    fn test_filter_choice_sharp_edge() {
        let mut buf = vec![0u8; 64];
        for i in 0..32 { buf[i] = 50; }
        for i in 32..64 { buf[i] = 200; }
        let s = 32isize;
        let t = 35;
        let w = filter_choice_8bpc(&buf, s, t, 1, 3, 3, 10, 20);
        assert!(w <= 1, "sharp edge should limit filter width");
    }

    #[test]
    fn test_deblock_uniform_unchanged() {
        let stride = 32usize;
        let mut dst = vec![128u8; stride * 8];
        let off = (stride * 2 + 8) as isize;
        let orig = dst.clone();
        deblock_8bpc(
            &mut dst, off, 10, 20,
            stride as isize, 1, 3, 3, false, false,
        );
        assert_eq!(dst, orig, "uniform buffer should be unchanged");
    }

    #[test]
    fn test_deblock_sharp_edge_modifies() {
        let stride = 32usize;
        let mut dst = vec![0u8; stride * 8];
        let edge_col = 10;
        for y in 0..8 {
            for x in 0..edge_col { dst[y * stride + x] = 50; }
            for x in edge_col..32 { dst[y * stride + x] = 200; }
        }
        let off = (stride * 2 + edge_col) as isize;
        let orig_at_edge = dst[off as usize];
        deblock_8bpc(
            &mut dst, off, 200, 200,
            stride as isize, 1, 1, 1, false, false,
        );
        assert_ne!(dst[off as usize], orig_at_edge,
            "deblock should modify sharp edge pixel");
    }

    #[test]
    fn test_deblock_lossless_skip() {
        let stride = 32usize;
        let mut dst = vec![0u8; stride * 8];
        let edge_col = 10;
        for y in 0..8 {
            for x in 0..edge_col { dst[y * stride + x] = 50; }
            for x in edge_col..32 { dst[y * stride + x] = 200; }
        }
        let off = (stride * 2 + edge_col) as isize;
        let orig = dst.clone();
        deblock_8bpc(
            &mut dst, off, 200, 200,
            stride as isize, 1, 1, 1, true, true,
        );
        assert_eq!(dst, orig, "both-lossless should not modify pixels");
    }

    #[test]
    fn test_deblock_h_sb64y_no_vmask() {
        let stride = 32;
        let mut dst = vec![128u8; stride * 8];
        let vmask = [0u16; 4];
        let ll_mask = [0u16; 2];
        let q_thr = [10u8; 16];
        let side_thr = [20u8; 16];
        let orig = dst.clone();
        deblock_h_sb64y_8bpc(
            &mut dst, 8, stride, &vmask, &ll_mask,
            &q_thr, &side_thr, false,
        );
        assert_eq!(dst, orig);
    }

    #[test]
    fn test_deblock_h_sb64y_uniform() {
        let stride = 32;
        let mut dst = vec![128u8; stride * 8];
        let vmask = [1u16, 0, 0, 0];
        let ll_mask = [0u16; 2];
        let q_thr = [10u8; 16];
        let side_thr = [20u8; 16];
        let orig = dst.clone();
        deblock_h_sb64y_8bpc(
            &mut dst, 8, stride, &vmask, &ll_mask,
            &q_thr, &side_thr, false,
        );
        assert_eq!(dst, orig, "uniform input should not change");
    }

    #[test]
    fn test_deblock_v_sb64y_uniform() {
        let stride = 32;
        let mut dst = vec![128u8; stride * 16];
        let vmask = [1u16, 0, 0, 0];
        let ll_mask = [0u16; 2];
        let q_thr = [10u8; 16];
        let side_thr = [20u8; 16];
        let orig = dst.clone();
        deblock_v_sb64y_8bpc(
            &mut dst, stride * 4, stride, &vmask, &ll_mask,
            &q_thr, &side_thr, false,
        );
        assert_eq!(dst, orig, "uniform input should not change");
    }

    #[test]
    fn test_deblock_h_sb64uv_no_vmask() {
        let stride = 32;
        let mut dst = vec![128u8; stride * 8];
        let vmask = [0u16; 3];
        let ll_mask = [0u16; 2];
        let q_thr = [10u8; 16];
        let side_thr = [20u8; 16];
        let orig = dst.clone();
        deblock_h_sb64uv_8bpc(
            &mut dst, 8, stride, &vmask, &ll_mask,
            &q_thr, &side_thr, false,
        );
        assert_eq!(dst, orig);
    }

    #[test]
    fn test_deblock_v_sb64uv_no_vmask() {
        let stride = 32;
        let mut dst = vec![128u8; stride * 8];
        let vmask = [0u16; 3];
        let ll_mask = [0u16; 2];
        let q_thr = [10u8; 16];
        let side_thr = [20u8; 16];
        let orig = dst.clone();
        deblock_v_sb64uv_8bpc(
            &mut dst, stride * 4, stride, &vmask, &ll_mask,
            &q_thr, &side_thr, false,
        );
        assert_eq!(dst, orig);
    }

    #[test]
    fn test_transpose_lossless_mask_basic() {
        let src_mask = [[0xAAAAu16; 4]; 16];
        let mut dst_mask = [0u16; 17];
        transpose_lossless_mask(&mut dst_mask, &src_mask, 0, 0, 0);
        for x in 0..16u32 {
            let bit = (0xAAAAu16 >> x) & 1;
            let expected = if bit != 0 { 0xFFFF } else { 0 };
            assert_eq!(dst_mask[x as usize + 1], expected);
        }
    }

    #[test]
    fn test_transpose_lossless_mask_ss() {
        let src_mask = [[0xFFu16; 4]; 8];
        let mut dst_mask = [0u16; 17];
        transpose_lossless_mask(&mut dst_mask, &src_mask, 0, 1, 1);
        for x in 0..8 {
            assert_eq!(dst_mask[x + 1], 0xFF);
        }
    }

    #[test]
    fn test_transpose_lossless_mask_prev_col() {
        let src_mask = [[0u16; 4]; 16];
        let mut dst_mask = [0u16; 17];
        dst_mask[16] = 42;
        transpose_lossless_mask(&mut dst_mask, &src_mask, 0, 0, 0);
        assert_eq!(dst_mask[0], 42);
    }
}

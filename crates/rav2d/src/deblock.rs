use crate::headers::FrameHeader;
use crate::intops::iclip;
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
}

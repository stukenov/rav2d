use crate::intops::iclip;

pub static CCSO_POS: [[i8; 2]; 7] = [
    [-1, 0],
    [0, -1],
    [-1, -1],
    [-1, 1],
    [-1, -2],
    [1, -2],
    [0, 2],
];

#[inline(always)]
pub fn ccso_score(diff: i32, quant_step: i32, edge_classifier: u32) -> u32 {
    if diff > quant_step && edge_classifier == 0 {
        return 2;
    }
    if diff < -quant_step {
        return 0;
    }
    1
}

pub fn ccso_add_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
) {
    for yy in (0..h).step_by(4) {
        let mi = yy >> 2;
        let mut bx = 0usize;
        let mut xx = 0usize;
        while xx < w {
            if ll_mask[mi][0] & (1 << bx) == 0 {
                for y in yy..yy + 4 {
                    for x in xx..xx + 4 {
                        let i = idx_buf[y * idx_stride + x];
                        let byte_idx = (i >> 1) as usize;
                        let half_idx = (i & 1) as usize;
                        let offset_idx = (7 & (offset_idxs[byte_idx] >> (4 * half_idx))) as usize;
                        dst[y * dst_stride + x] = iclip(
                            dst[y * dst_stride + x] as i32 + offset_lut[offset_idx] as i32,
                            0, 255,
                        ) as u8;
                    }
                }
            }
            xx += 4;
            bx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccso_pos_table() {
        assert_eq!(CCSO_POS[0], [-1, 0]);
        assert_eq!(CCSO_POS[6], [0, 2]);
        assert_eq!(CCSO_POS.len(), 7);
    }

    #[test]
    fn test_ccso_score_large_positive() {
        assert_eq!(ccso_score(20, 10, 0), 2);
    }

    #[test]
    fn test_ccso_score_large_positive_with_edge() {
        assert_eq!(ccso_score(20, 10, 1), 1);
    }

    #[test]
    fn test_ccso_score_large_negative() {
        assert_eq!(ccso_score(-20, 10, 0), 0);
    }

    #[test]
    fn test_ccso_score_within_range() {
        assert_eq!(ccso_score(5, 10, 0), 1);
        assert_eq!(ccso_score(-5, 10, 0), 1);
    }

    #[test]
    fn test_ccso_add_basic() {
        let mut dst = vec![128u8; 16];
        let idx = vec![0u8; 16];
        let offset_idxs = vec![0x21u8; 8];
        let offset_lut = [0i8, 5, -5, 10, -10, 15, -15, 20];
        let ll_mask = vec![[0u16; 4]; 1];
        ccso_add_8bpc(&mut dst, 4, &idx, 4, &offset_idxs, &offset_lut, 4, 4, &ll_mask);
        assert_eq!(dst[0], 133);
    }

    #[test]
    fn test_ccso_add_skip_mask() {
        let mut dst = vec![128u8; 16];
        let idx = vec![0u8; 16];
        let offset_idxs = vec![0u8; 8];
        let offset_lut = [10i8; 8];
        let ll_mask = vec![[0xFFFFu16; 4]; 1];
        ccso_add_8bpc(&mut dst, 4, &idx, 4, &offset_idxs, &offset_lut, 4, 4, &ll_mask);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_ccso_score_boundary() {
        assert_eq!(ccso_score(10, 10, 0), 1);
        assert_eq!(ccso_score(-10, 10, 0), 1);
        assert_eq!(ccso_score(11, 10, 0), 2);
        assert_eq!(ccso_score(-11, 10, 0), 0);
    }
}

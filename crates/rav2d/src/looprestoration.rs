use crate::intops::{apply_sign, iclip, imax, imin};

pub const LR_HAVE_LEFT: u8 = 1 << 0;
pub const LR_HAVE_RIGHT: u8 = 1 << 1;
pub const LR_HAVE_TOP: u8 = 1 << 2;
pub const LR_HAVE_BOTTOM: u8 = 1 << 3;
pub const LR_HAVE_TOP_INTEGRATED: u8 = 1 << 4;
pub const LR_HAVE_BOTTOM_INTEGRATED: u8 = 1 << 5;

pub const PC_WIENER_NORMALIZER: [u16; 4] = [3739, 3273, 3074, 7];

static MODE_WEIGHTS: [[i16; 3]; 4] = [
    [-527, 15325, 321],
    [26436, -17705, 17905],
    [366, -147, -194],
    [202, -267, -179],
];

static MODE_OFFSETS: [i16; 4] = [-547, -21565, -573, -680];

pub fn get_qval_given_tskip(mut qstep: i32, tskip: i32, i: usize, bitdepth_min_8: i32) -> i32 {
    qstep = (qstep + ((1 << bitdepth_min_8) >> 1)) >> bitdepth_min_8;
    let prod = (tskip * qstep + 128) >> 8;
    let qval = MODE_WEIGHTS[i][0] as i32 * (tskip << 5)
        + MODE_WEIGHTS[i][1] as i32 * qstep
        + MODE_WEIGHTS[i][2] as i32 * prod;
    let abs_qval = qval.abs();
    let qval = apply_sign((abs_qval + (1 << 12)) >> 13, qval);
    255 * (MODE_OFFSETS[i] as i32 + qval)
}

/// Backup a row with edge extension. `dst` and `src` are indexed with `o` as the
/// origin (position 0 in C). Left edge fills dst[o-edge_len..o], right fills
/// dst[o+w..o+w+edge_len].
pub fn backup_row_8bpc(
    dst: &mut [u8],
    o: usize,
    src: &[u8],
    src_o: usize,
    left: &[u8],
    left_off: usize,
    w: usize,
    edge_len: usize,
    edges: u8,
) {
    if edges & LR_HAVE_LEFT != 0 {
        for x in 0..edge_len {
            dst[o - edge_len + x] = left[left_off - edge_len + x];
        }
    } else {
        dst[o - edge_len..o].fill(src[src_o]);
    }

    dst[o..o + w].copy_from_slice(&src[src_o..src_o + w]);

    if edges & LR_HAVE_RIGHT != 0 {
        dst[o + w..o + w + edge_len].copy_from_slice(&src[src_o + w..src_o + w + edge_len]);
    } else {
        dst[o + w..o + w + edge_len].fill(src[src_o + w - 1]);
    }
}

/// Backup a row from LPF buffer with edge extension.
pub fn backup_row_lpf_8bpc(
    dst: &mut [u8],
    o: usize,
    src: &[u8],
    src_o: usize,
    w: usize,
    edge_len: usize,
    edges: u8,
) {
    if edges & LR_HAVE_LEFT != 0 {
        for x in 0..edge_len {
            dst[o - edge_len + x] = src[src_o - edge_len + x];
        }
    } else {
        dst[o - edge_len..o].fill(src[src_o]);
    }

    dst[o..o + w].copy_from_slice(&src[src_o..src_o + w]);

    if edges & LR_HAVE_RIGHT != 0 {
        dst[o + w..o + w + edge_len].copy_from_slice(&src[src_o + w..src_o + w + edge_len]);
    } else {
        dst[o + w..o + w + edge_len].fill(src[src_o + w - 1]);
    }
}

/// Compute 2x2 gradient features for 4 directions.
/// `rows[row_off]` is the first center row. `col_off` is the column origin
/// in each row (matching the C convention where row pointers are pre-offset).
pub fn compute_gradient_row_8bpc(
    dst: &mut [[u16; 4]],
    rows: &[&[u8]],
    row_off: usize,
    col_off: usize,
    w: usize,
    shift: u32,
) {
    let offs: [[i32; 2]; 4] = [[1, 0], [0, 1], [1, 1], [-1, 1]];
    let mut x1 = 0usize;
    while x1 < w + 2 {
        for d in 0..4 {
            let mut grad = 0i32;
            for x2 in 0..2usize {
                let x = col_off + x1 + x2;
                for y in 0..2 {
                    let dy = offs[d][0];
                    let dx = offs[d][1];
                    let ry = row_off + y;
                    let a = (rows[(ry as i32 - 1 - dy) as usize][(x as i32 - 1 - dx) as usize] >> shift) as i32;
                    let b = (rows[ry - 1][x - 1] >> shift) as i32;
                    let c = (rows[(ry as i32 - 1 + dy) as usize][(x as i32 - 1 + dx) as usize] >> shift) as i32;
                    grad += (b * 2 - a - c).abs();
                }
            }
            dst[x1 >> 1][d] = grad as u16;
        }
        x1 += 2;
    }
}

/// Compute PC-Wiener class LUT index from gradient features and skip mask.
pub fn get_class_lut_idx_8bpc(
    rows: &[&[u8]],
    row_center: usize,
    noskip_mask: &[u16],
    base_q: i32,
    bx: usize,
    by: usize,
    bh: usize,
) -> i32 {
    let mut f = [0i32; 3];
    let mut s = 0i32;

    for dy in -1i32..=4 {
        for dx in -1i32..=4 {
            let x = (bx as i32 * 4 + dx) as usize;
            let y = (row_center as i32 + dy) as usize;
            let m = rows[y][x] as i32;
            let up = rows[y - 1][x] as i32;
            let down = rows[y + 1][x] as i32;
            let vert = up - 2 * m + down;

            let up_right = rows[y - 1][x + 1] as i32;
            let down_left = rows[y + 1][x.wrapping_sub(1)] as i32;
            let anti_diag = up_right - 2 * m + down_left;

            let down_right = rows[y + 1][x + 1] as i32;
            let up_left = rows[y - 1][x.wrapping_sub(1)] as i32;
            let diag = up_left - 2 * m + down_right;

            f[0] += vert.abs();
            f[1] += anti_diag.abs();
            f[2] += diag.abs();
        }
    }

    let num_pixels: [u8; 3] = [16, 4, 1];
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let edge = (dy != 0) as usize + (dx != 0) as usize;
            let fx = iclip((bx & 15) as i32 + dx, 0, 15) as usize;
            let fy = iclip(by as i32 + dy, 0, bh as i32 - 1) as usize;
            s += num_pixels[edge] as i32 * (((noskip_mask[fy] >> fx) & 1) == 0) as i32;
        }
    }

    for i in 0..3 {
        f[i] = (f[i] * PC_WIENER_NORMALIZER[i] as i32 + 0) >> 0;
    }
    s = s * PC_WIENER_NORMALIZER[3] as i32;

    let mut qval = (imax(0, get_qval_given_tskip(base_q, s, 0, 0)) + (1 << 13)) >> 14;
    qval = imin(qval, 255) >> 5;
    let mut lut_idx = qval << 9;
    for i in 0..3 {
        qval = (imax(0, f[i] + get_qval_given_tskip(base_q, s, i + 1, 0)) + (1 << 13)) >> 14;
        qval = imin(qval, 255) >> 5;
        lut_idx |= qval << (3 * (2 - i));
    }
    lut_idx
}

pub fn gdf_add_8bpc(
    p: &mut [u8],
    stride: usize,
    err: &[i8],
    err_stride: usize,
    w: usize,
    h: usize,
    scale: i32,
    ll_mask: &[[u16; 4]],
) {
    let shift = 4;
    let rnd = 1 << (shift - 1);
    for by in 0..h >> 2 {
        for bx in 0..w >> 2 {
            if ll_mask[by][0] & (1 << bx) != 0 { continue; }
            for y in by * 4..by * 4 + 4 {
                for x in bx * 4..bx * 4 + 4 {
                    let diff = err[y * err_stride + x] as i32 * scale;
                    let adj = apply_sign((diff.abs() + rnd) >> shift, diff);
                    p[y * stride + x] = iclip(p[y * stride + x] as i32 + adj, 0, 255) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc_wiener_normalizer() {
        assert_eq!(PC_WIENER_NORMALIZER[3], 7);
    }

    #[test]
    fn test_get_qval_given_tskip_8bpc() {
        let v = get_qval_given_tskip(16, 100, 0, 0);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_get_qval_given_tskip_all_modes() {
        for i in 0..4 {
            let v = get_qval_given_tskip(32, 200, i, 0);
            let _ = v;
        }
    }

    #[test]
    fn test_get_qval_given_tskip_10bpc() {
        let v0 = get_qval_given_tskip(64, 300, 1, 0);
        let v2 = get_qval_given_tskip(64, 300, 1, 2);
        assert_ne!(v0, v2);
    }

    #[test]
    fn test_get_qval_given_tskip_zero() {
        let v = get_qval_given_tskip(0, 0, 0, 0);
        assert_eq!(v, 255 * MODE_OFFSETS[0] as i32);
    }

    #[test]
    fn test_lr_edge_flags() {
        assert_eq!(LR_HAVE_LEFT, 1);
        assert_eq!(LR_HAVE_RIGHT, 2);
        assert_eq!(LR_HAVE_TOP, 4);
        assert_eq!(LR_HAVE_BOTTOM, 8);
    }

    #[test]
    fn test_backup_row_no_edges() {
        let src = vec![10u8; 32];
        let left = vec![0u8; 32];
        let mut dst = vec![0u8; 32];
        let o = 6;
        backup_row_8bpc(&mut dst, o, &src, 0, &left, 0, 16, 6, 0);
        assert!(dst[0..o].iter().all(|&v| v == 10));
        assert_eq!(&dst[o..o + 16], &src[0..16]);
        assert!(dst[o + 16..o + 16 + 6].iter().all(|&v| v == src[15]));
    }

    #[test]
    fn test_backup_row_both_edges() {
        let mut src = vec![50u8; 32];
        src[0] = 10;
        src[15] = 90;
        let left = vec![20u8; 16];
        let mut dst = vec![0u8; 32];
        let o = 6;
        backup_row_8bpc(&mut dst, o, &src, 0, &left, 6, 16, 6, LR_HAVE_LEFT | LR_HAVE_RIGHT);
        assert_eq!(&dst[o..o + 16], &src[0..16]);
    }

    #[test]
    fn test_backup_row_lpf_no_edges() {
        let src = vec![42u8; 32];
        let mut dst = vec![0u8; 32];
        let o = 6;
        backup_row_lpf_8bpc(&mut dst, o, &src, 0, 16, 6, 0);
        assert!(dst[0..o].iter().all(|&v| v == 42));
        assert_eq!(&dst[o..o + 16], &src[0..16]);
        assert!(dst[o + 16..o + 16 + 6].iter().all(|&v| v == src[15]));
    }

    #[test]
    fn test_compute_gradient_row_flat() {
        let row = vec![128u8; 32];
        let rows: Vec<&[u8]> = vec![&row; 5];
        let mut dst = [[0u16; 4]; 16];
        compute_gradient_row_8bpc(&mut dst, &rows, 2, 2, 4, 0);
        for d in &dst[..3] {
            for &v in d {
                assert_eq!(v, 0);
            }
        }
    }

    #[test]
    fn test_compute_gradient_row_edge() {
        let flat = vec![100u8; 32];
        let bright = vec![200u8; 32];
        let rows: Vec<&[u8]> = vec![&flat, &bright, &flat, &bright, &flat];
        let mut dst = [[0u16; 4]; 16];
        compute_gradient_row_8bpc(&mut dst, &rows, 2, 2, 4, 0);
        assert!(dst[0][0] > 0);
    }

    #[test]
    fn test_gdf_add_basic() {
        let mut p = vec![128u8; 64];
        let err = vec![10i8; 64];
        let ll_mask = vec![[0u16; 4]; 2];
        gdf_add_8bpc(&mut p, 8, &err, 8, 8, 8, 16, &ll_mask);
        assert!(p[0] > 128);
    }

    #[test]
    fn test_gdf_add_skip_mask() {
        let mut p = vec![128u8; 64];
        let err = vec![10i8; 64];
        let ll_mask = vec![[0xFFFFu16; 4]; 2];
        gdf_add_8bpc(&mut p, 8, &err, 8, 8, 8, 16, &ll_mask);
        assert!(p.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_gdf_add_negative_err() {
        let mut p = vec![128u8; 64];
        let err = vec![-10i8; 64];
        let ll_mask = vec![[0u16; 4]; 2];
        gdf_add_8bpc(&mut p, 8, &err, 8, 8, 8, 16, &ll_mask);
        assert!(p[0] < 128);
    }

    #[test]
    fn test_get_class_lut_idx_flat() {
        let row = vec![128u8; 64];
        let rows: Vec<&[u8]> = vec![&row; 16];
        let noskip = vec![0xFFFFu16; 16];
        let idx = get_class_lut_idx_8bpc(&rows, 6, &noskip, 32, 1, 1, 8);
        let _ = idx;
    }
}

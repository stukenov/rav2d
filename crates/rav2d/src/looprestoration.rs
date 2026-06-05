use crate::gdf_tables::{GDF_ALPHA, GDF_BIAS, GDF_INTER_ERROR, GDF_INTRA_ERROR, GDF_WEIGHT};
use crate::intops::{apply_sign, iclip, imax, imin};
use crate::tables::PC_WIENER_LUT_TO_CLASS;

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

/// Backup N pixels per row for U rows, right-aligned in a [u8; 6] buffer.
/// Copies src[off-n..off] into dst[row][6-n..6] per row, advancing by stride.
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
                    let a = (rows[(ry as i32 - 1 - dy) as usize][(x as i32 - 1 - dx) as usize]
                        >> shift) as i32;
                    let b = (rows[ry - 1][x - 1] >> shift) as i32;
                    let c = (rows[(ry as i32 - 1 + dy) as usize][(x as i32 - 1 + dx) as usize]
                        >> shift) as i32;
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
    col_off: usize,
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
            let x = (col_off as i32 + bx as i32 * 4 + dx) as usize;
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
        f[i] *= PC_WIENER_NORMALIZER[i] as i32;
    }
    s *= PC_WIENER_NORMALIZER[3] as i32;

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
            if ll_mask[by][0] & (1 << bx) != 0 {
                continue;
            }
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

pub const REST_UNIT_STRIDE: usize = 76;
const ROW_ORIGIN: usize = 6;

pub static GDF_COORDS: [[i8; 2]; 18] = [
    [6, 0],
    [5, 0],
    [4, 0],
    [3, 0],
    [2, 1],
    [2, 0],
    [2, -1],
    [1, 2],
    [1, 1],
    [1, 0],
    [1, -1],
    [1, -2],
    [0, 6],
    [0, 5],
    [0, 4],
    [0, 3],
    [0, 2],
    [0, 1],
];

const GRADIENT_BUF_STRIDE: usize = 33;

pub static WIENER_NS_CONFIG_UV: [[i8; 2]; 6] = [[1, 0], [0, 1], [1, 1], [-1, 1], [2, 0], [0, 2]];

pub static WIENER_NS_CONFIG_UV_FROM_Y: [[i8; 2]; 12] = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [-1, -1],
    [-1, 1],
    [1, -1],
    [2, 0],
    [-2, 0],
    [0, 2],
    [0, -2],
];

pub static PC_WIENER_CONFIG: [[i8; 2]; 12] = [
    [1, 0],
    [0, 1],
    [2, 0],
    [0, 2],
    [1, 1],
    [-1, 1],
    [2, 1],
    [2, -1],
    [1, 2],
    [1, -2],
    [3, 0],
    [0, 3],
];

pub static WIENER_NS_CONFIG_Y: [[i8; 2]; 16] = [
    [1, 0],
    [0, 1],
    [2, 0],
    [0, 2],
    [1, 1],
    [-1, 1],
    [2, 1],
    [2, -1],
    [1, 2],
    [1, -2],
    [3, 0],
    [0, 3],
    [4, 0],
    [0, 4],
    [3, 3],
    [3, -3],
];

pub fn backup_row_luma_8bpc(
    dst: &mut [u8],
    o: usize,
    src: &[u8],
    src_o: usize,
    src_stride: usize,
    w: usize,
    edges: u8,
    ss_hor: usize,
    ss_ver: usize,
    cfl_ds_flt: i32,
) {
    if ss_ver == 0 {
        backup_row_lpf_8bpc(dst, o, src, src_o, w, 4, edges);
        return;
    }

    let src2_o = src_o + src_stride;

    match cfl_ds_flt {
        0 => {
            let mut x = 0;
            while x < w {
                dst[o + x] = ((src[src_o + x] as u16
                    + src[src2_o + x] as u16
                    + src[src_o + x + 1] as u16
                    + src[src2_o + x + 1] as u16)
                    >> 2) as u8;
                x += 1 + ss_hor;
            }
        }
        1 => {
            for x in 0..w {
                dst[o + x] = ((src[src_o + x] as u16 + src[src2_o + x] as u16) >> 1) as u8;
            }
        }
        _ => {
            dst[o..o + w].copy_from_slice(&src[src_o..src_o + w]);
        }
    }

    if edges & LR_HAVE_LEFT != 0 {
        match cfl_ds_flt {
            0 => {
                for i in 0..4usize {
                    let si = src_o - 4 + i;
                    let s2i = src2_o - 4 + i;
                    dst[o - 4 + i] = ((src[si] as u16
                        + src[s2i] as u16
                        + src[si + 1] as u16
                        + src[s2i + 1] as u16)
                        >> 2) as u8;
                }
            }
            1 => {
                for i in 0..4usize {
                    dst[o - 4 + i] =
                        ((src[src_o - 4 + i] as u16 + src[src2_o - 4 + i] as u16) >> 1) as u8;
                }
            }
            _ => {
                for i in 0..4usize {
                    dst[o - 4 + i] = src[src_o - 4 + i];
                }
            }
        }
    } else {
        let fill_val = dst[o];
        dst[o - 4..o].fill(fill_val);
    }

    if edges & LR_HAVE_RIGHT != 0 {
        match cfl_ds_flt {
            0 => {
                for i in 0..4usize {
                    let si = src_o + w + i;
                    let s2i = src2_o + w + i;
                    dst[o + w + i] = ((src[si] as u16
                        + src[s2i] as u16
                        + src[si + 1] as u16
                        + src[s2i + 1] as u16)
                        >> 2) as u8;
                }
            }
            1 => {
                for i in 0..4usize {
                    dst[o + w + i] =
                        ((src[src_o + w + i] as u16 + src[src2_o + w + i] as u16) >> 1) as u8;
                }
            }
            _ => {
                for i in 0..4usize {
                    dst[o + w + i] = src[src_o + w + i];
                }
            }
        }
    } else {
        let fill_val = dst[o + w - 2];
        dst[o + w..o + w + 4].fill(fill_val);
    }
}

pub fn ns_wiener_single_y_8bpc(
    p: &mut [u8],
    p_off: usize,
    stride: usize,
    left: &[[u8; 6]],
    lpf: &[u8],
    lpf_off: usize,
    lpf_bottom: &[u8],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    filter: &[i8; 16],
    edges: u8,
    ll_mask: &[[u16; 4]],
) {
    let mut row_buffers = [[0u8; REST_UNIT_STRIDE]; 9];
    let mut ptrs: [usize; 9] = [0; 9];
    let o = ROW_ORIGIN;

    backup_row_8bpc(&mut row_buffers[4], o, &*p, p_off, &left[0], 6, w, 4, edges);
    ptrs[4] = 4;

    if edges & LR_HAVE_TOP_INTEGRATED != 0 {
        let mut loff = lpf_off;
        for i in 0..4 {
            backup_row_lpf_8bpc(&mut row_buffers[i], o, lpf, loff, w, 4, edges);
            loff += stride;
            ptrs[i] = i;
        }
    } else if edges & LR_HAVE_TOP != 0 {
        backup_row_lpf_8bpc(&mut row_buffers[2], o, lpf, lpf_off, w, 4, edges);
        ptrs[2] = 2;
        backup_row_lpf_8bpc(&mut row_buffers[3], o, lpf, lpf_off + stride, w, 4, edges);
        ptrs[3] = 3;
        ptrs[0] = 2;
        ptrs[1] = 2;
    } else {
        ptrs[0] = 4;
        ptrs[1] = 4;
        ptrs[2] = 4;
        ptrs[3] = 4;
    }

    backup_row_8bpc(
        &mut row_buffers[5],
        o,
        &*p,
        p_off + stride,
        &left[1],
        6,
        w,
        4,
        edges,
    );
    ptrs[5] = 5;
    backup_row_8bpc(
        &mut row_buffers[6],
        o,
        &*p,
        p_off + 2 * stride,
        &left[2],
        6,
        w,
        4,
        edges,
    );
    ptrs[6] = 6;
    backup_row_8bpc(
        &mut row_buffers[7],
        o,
        &*p,
        p_off + 3 * stride,
        &left[3],
        6,
        w,
        4,
        edges,
    );
    ptrs[7] = 7;

    let mut bak_idx: usize = 8;

    for y in 0..h {
        if y + 4 < h {
            backup_row_8bpc(
                &mut row_buffers[bak_idx],
                o,
                &*p,
                p_off + (y + 4) * stride,
                &left[y + 4],
                6,
                w,
                4,
                edges,
            );
            ptrs[8] = bak_idx;
        } else if edges & LR_HAVE_BOTTOM_INTEGRATED != 0 {
            backup_row_lpf_8bpc(
                &mut row_buffers[bak_idx],
                o,
                &*p,
                p_off + (y + 4) * stride,
                w,
                4,
                edges,
            );
            ptrs[8] = bak_idx;
        } else if y + 2 < h && edges & LR_HAVE_BOTTOM != 0 {
            let offset_y = y + 4 - h;
            backup_row_lpf_8bpc(
                &mut row_buffers[bak_idx],
                o,
                lpf_bottom,
                lpf_bottom_off + offset_y * stride,
                w,
                4,
                edges,
            );
            ptrs[8] = bak_idx;
        } else {
            ptrs[8] = ptrs[7];
        }

        bak_idx += 1;
        if bak_idx == 9 {
            bak_idx = 0;
        }

        for bx in 0..(w >> 2) {
            if ll_mask[y >> 2][0] & (1 << bx) != 0 {
                continue;
            }
            for x in bx * 4..bx * 4 + 4 {
                let m = row_buffers[ptrs[4]][o + x] as i32;
                let mut s = m << 7;
                for i in 0..16 {
                    let dy = WIENER_NS_CONFIG_Y[i][0] as i32;
                    let dx = WIENER_NS_CONFIG_Y[i][1] as i32;
                    let a = row_buffers[ptrs[(4 + dy) as usize]]
                        [(o as i32 + x as i32 + dx) as usize] as i32;
                    let b = row_buffers[ptrs[(4 - dy) as usize]]
                        [(o as i32 + x as i32 - dx) as usize] as i32;
                    let diff = a + b - 2 * m;
                    s += diff * filter[i] as i32;
                }
                let v = (s + 64) >> 7;
                p[p_off + y * stride + x] = iclip(v, 0, 255) as u8;
            }
        }

        for r in 0..8 {
            ptrs[r] = ptrs[r + 1];
        }
    }
}

fn wiener_multi_8bpc(
    p: &mut [u8],
    p_off: usize,
    stride: usize,
    left: &[[u8; 6]],
    lpf: &[u8],
    lpf_off: usize,
    lpf_bottom: &[u8],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    filters_user: Option<&[[i8; 18]]>,
    filters_pretrained: Option<&[[i16; 13]]>,
    subclass_lut: &[u8],
    noskip_mask: &[u16],
    base_q: i32,
    edges: u8,
    ll_mask: &[[u16; 4]],
) {
    let mut classes = [0u8; 16];
    let mut row_buffers = [[0u8; REST_UNIT_STRIDE]; 10];
    let mut ptrs: [usize; 10] = [0; 10];
    let o = ROW_ORIGIN;

    backup_row_8bpc(&mut row_buffers[4], o, &*p, p_off, &left[0], 6, w, 4, edges);
    ptrs[4] = 4;

    if edges & LR_HAVE_TOP_INTEGRATED != 0 {
        let mut loff = lpf_off;
        for i in 0..4 {
            backup_row_lpf_8bpc(&mut row_buffers[i], o, lpf, loff, w, 4, edges);
            loff += stride;
            ptrs[i] = i;
        }
    } else if edges & LR_HAVE_TOP != 0 {
        backup_row_lpf_8bpc(&mut row_buffers[2], o, lpf, lpf_off, w, 4, edges);
        ptrs[2] = 2;
        backup_row_lpf_8bpc(&mut row_buffers[3], o, lpf, lpf_off + stride, w, 4, edges);
        ptrs[3] = 3;
        ptrs[0] = 2;
        ptrs[1] = 2;
    } else {
        ptrs[0] = 4;
        ptrs[1] = 4;
        ptrs[2] = 4;
        ptrs[3] = 4;
    }

    backup_row_8bpc(
        &mut row_buffers[5],
        o,
        &*p,
        p_off + stride,
        &left[1],
        6,
        w,
        4,
        edges,
    );
    ptrs[5] = 5;
    backup_row_8bpc(
        &mut row_buffers[6],
        o,
        &*p,
        p_off + 2 * stride,
        &left[2],
        6,
        w,
        4,
        edges,
    );
    ptrs[6] = 6;
    backup_row_8bpc(
        &mut row_buffers[7],
        o,
        &*p,
        p_off + 3 * stride,
        &left[3],
        6,
        w,
        4,
        edges,
    );
    ptrs[7] = 7;

    let mut bak_idx: usize = 8;
    let bh = h >> 2;
    let bw = w >> 2;

    for by in 0..bh {
        let by4 = by << 2;

        if by + 1 < bh {
            backup_row_8bpc(
                &mut row_buffers[bak_idx],
                o,
                &*p,
                p_off + (by4 + 4) * stride,
                &left[by4 + 4],
                6,
                w,
                4,
                edges,
            );
            ptrs[8] = bak_idx;
            backup_row_8bpc(
                &mut row_buffers[9],
                o,
                &*p,
                p_off + (by4 + 5) * stride,
                &left[by4 + 5],
                6,
                w,
                4,
                edges,
            );
            ptrs[9] = 9;
        } else if edges & LR_HAVE_BOTTOM_INTEGRATED != 0 {
            backup_row_lpf_8bpc(
                &mut row_buffers[bak_idx],
                o,
                &*p,
                p_off + (by4 + 4) * stride,
                w,
                4,
                edges,
            );
            ptrs[8] = bak_idx;
            backup_row_lpf_8bpc(
                &mut row_buffers[9],
                o,
                &*p,
                p_off + (by4 + 5) * stride,
                w,
                4,
                edges,
            );
            ptrs[9] = 9;
        } else if edges & LR_HAVE_BOTTOM != 0 {
            backup_row_lpf_8bpc(
                &mut row_buffers[bak_idx],
                o,
                lpf_bottom,
                lpf_bottom_off,
                w,
                4,
                edges,
            );
            ptrs[8] = bak_idx;
            backup_row_lpf_8bpc(
                &mut row_buffers[9],
                o,
                lpf_bottom,
                lpf_bottom_off + stride,
                w,
                4,
                edges,
            );
            ptrs[9] = 9;
        } else {
            ptrs[8] = ptrs[7];
            ptrs[9] = ptrs[7];
        }

        {
            let refs: [&[u8]; 10] = core::array::from_fn(|i| &row_buffers[ptrs[i]] as &[u8]);
            for bx in 0..bw {
                let lut_idx = get_class_lut_idx_8bpc(&refs, 4, o, noskip_mask, base_q, bx, by, bh);
                let cls = PC_WIENER_LUT_TO_CLASS[lut_idx as usize];
                classes[bx] = subclass_lut[cls as usize];
            }
        }

        for y in by4..by4 + 4 {
            if y + 4 < h {
                backup_row_8bpc(
                    &mut row_buffers[bak_idx],
                    o,
                    &*p,
                    p_off + (y + 4) * stride,
                    &left[y + 4],
                    6,
                    w,
                    4,
                    edges,
                );
                ptrs[8] = bak_idx;
            } else if edges & LR_HAVE_BOTTOM_INTEGRATED != 0 {
                backup_row_lpf_8bpc(
                    &mut row_buffers[bak_idx],
                    o,
                    &*p,
                    p_off + (y + 4) * stride,
                    w,
                    4,
                    edges,
                );
                ptrs[8] = bak_idx;
            } else if y + 2 < h && edges & LR_HAVE_BOTTOM != 0 {
                let offset_y = y + 4 - h;
                backup_row_lpf_8bpc(
                    &mut row_buffers[bak_idx],
                    o,
                    lpf_bottom,
                    lpf_bottom_off + offset_y * stride,
                    w,
                    4,
                    edges,
                );
                ptrs[8] = bak_idx;
            } else {
                ptrs[8] = ptrs[7];
            }

            bak_idx += 1;
            if bak_idx == 9 {
                bak_idx = 0;
            }

            for bx in 0..bw {
                if ll_mask[y >> 2][0] & (1 << bx) != 0 {
                    continue;
                }

                if let Some(fu) = filters_user {
                    let filter = &fu[classes[bx] as usize];
                    for x in (bx << 2)..(bx << 2) + 4 {
                        let m = row_buffers[ptrs[4]][o + x] as i32;
                        let mut s = m << 7;
                        for i in 0..16 {
                            let dy = WIENER_NS_CONFIG_Y[i][0] as i32;
                            let dx = WIENER_NS_CONFIG_Y[i][1] as i32;
                            let a = row_buffers[ptrs[(4 + dy) as usize]]
                                [(o as i32 + x as i32 + dx) as usize]
                                as i32;
                            let b = row_buffers[ptrs[(4 - dy) as usize]]
                                [(o as i32 + x as i32 - dx) as usize]
                                as i32;
                            s += (a + b - 2 * m) * filter[i] as i32;
                        }
                        p[p_off + y * stride + x] = iclip((s + 64) >> 7, 0, 255) as u8;
                    }
                } else if let Some(fp) = filters_pretrained {
                    let filter = &fp[classes[bx] as usize];
                    for x in (bx << 2)..(bx << 2) + 4 {
                        let mut s = row_buffers[ptrs[4]][o + x] as i32 * filter[12] as i32;
                        for i in 0..12 {
                            let dy = PC_WIENER_CONFIG[i][0] as i32;
                            let dx = PC_WIENER_CONFIG[i][1] as i32;
                            let a = row_buffers[ptrs[(4 + dy) as usize]]
                                [(o as i32 + x as i32 + dx) as usize]
                                as i32;
                            let b = row_buffers[ptrs[(4 - dy) as usize]]
                                [(o as i32 + x as i32 - dx) as usize]
                                as i32;
                            s += filter[i] as i32 * (a + b);
                        }
                        p[p_off + y * stride + x] = iclip((s + 64) >> 7, 0, 255) as u8;
                    }
                }
            }

            for r in 0..8 {
                ptrs[r] = ptrs[r + 1];
            }
        }
    }
}

pub fn ns_wiener_multi_8bpc(
    p: &mut [u8],
    p_off: usize,
    stride: usize,
    left: &[[u8; 6]],
    lpf: &[u8],
    lpf_off: usize,
    lpf_bottom: &[u8],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    filters_user: &[[i8; 18]],
    subclass_lut: &[u8],
    noskip_mask: &[u16],
    base_q: i32,
    edges: u8,
    ll_mask: &[[u16; 4]],
) {
    wiener_multi_8bpc(
        p,
        p_off,
        stride,
        left,
        lpf,
        lpf_off,
        lpf_bottom,
        lpf_bottom_off,
        w,
        h,
        Some(filters_user),
        None,
        subclass_lut,
        noskip_mask,
        base_q,
        edges,
        ll_mask,
    );
}

pub fn pc_wiener_8bpc(
    p: &mut [u8],
    p_off: usize,
    stride: usize,
    left: &[[u8; 6]],
    lpf: &[u8],
    lpf_off: usize,
    lpf_bottom: &[u8],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    filters_pretrained: &[[i16; 13]],
    subclass_lut: &[u8],
    noskip_mask: &[u16],
    base_q: i32,
    edges: u8,
    ll_mask: &[[u16; 4]],
) {
    wiener_multi_8bpc(
        p,
        p_off,
        stride,
        left,
        lpf,
        lpf_off,
        lpf_bottom,
        lpf_bottom_off,
        w,
        h,
        None,
        Some(filters_pretrained),
        subclass_lut,
        noskip_mask,
        base_q,
        edges,
        ll_mask,
    );
}

const LUMA_BUF_STRIDE: usize = REST_UNIT_STRIDE + 64;

pub fn ns_wiener_single_uv_8bpc(
    p: &mut [u8],
    p_off: usize,
    stride: usize,
    left: &[[u8; 6]],
    lpf: &[u8],
    lpf_off: usize,
    lpf_bottom: &[u8],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    filter: &[i8; 18],
    luma: &[u8],
    luma_off: usize,
    lstride: usize,
    luma_top: &[u8],
    luma_top_off: usize,
    luma_bottom: &[u8],
    luma_bottom_off: usize,
    ss_hor: usize,
    ss_ver: usize,
    ds_flt: i32,
    edges: u8,
    ll_mask: &[[u16; 4]],
) {
    let mut c_buffers = [[0u8; REST_UNIT_STRIDE]; 5];
    let mut l_buffers = [[0u8; LUMA_BUF_STRIDE]; 5];
    let mut c_ptrs: [usize; 5] = [0; 5];
    let mut l_ptrs: [usize; 5] = [0; 5];
    let o = ROW_ORIGIN;
    let luma_w = w << ss_hor;

    backup_row_8bpc(&mut c_buffers[2], o, &*p, p_off, &left[0], 6, w, 2, edges);
    c_ptrs[2] = 2;

    if edges & (LR_HAVE_TOP_INTEGRATED | LR_HAVE_TOP) != 0 {
        let mut loff = lpf_off;
        for i in 0..2 {
            backup_row_lpf_8bpc(&mut c_buffers[i], o, lpf, loff, w, 2, edges);
            c_ptrs[i] = i;
            loff += stride;
        }
    } else {
        c_ptrs[0] = 2;
        c_ptrs[1] = 2;
    }

    backup_row_8bpc(
        &mut c_buffers[3],
        o,
        &*p,
        p_off + stride,
        &left[1],
        6,
        w,
        2,
        edges,
    );
    c_ptrs[3] = 3;
    let mut bak_idx: usize = 4;

    backup_row_luma_8bpc(
        &mut l_buffers[2],
        o,
        luma,
        luma_off,
        lstride,
        luma_w,
        edges,
        ss_hor,
        ss_ver,
        ds_flt,
    );
    l_ptrs[2] = 2;

    if edges & LR_HAVE_TOP_INTEGRATED != 0 {
        backup_row_luma_8bpc(
            &mut l_buffers[0],
            o,
            luma,
            luma_off - 4 * lstride,
            lstride,
            luma_w,
            edges,
            ss_hor,
            ss_ver,
            ds_flt,
        );
        l_ptrs[0] = 0;
        backup_row_luma_8bpc(
            &mut l_buffers[1],
            o,
            luma,
            luma_off - 2 * lstride,
            lstride,
            luma_w,
            edges,
            ss_hor,
            ss_ver,
            ds_flt,
        );
        l_ptrs[1] = 1;
    } else if edges & LR_HAVE_TOP != 0 {
        backup_row_luma_8bpc(
            &mut l_buffers[0],
            o,
            luma_top,
            luma_top_off,
            0,
            luma_w,
            edges,
            ss_hor,
            ss_ver,
            ds_flt,
        );
        l_ptrs[0] = 0;
        backup_row_luma_8bpc(
            &mut l_buffers[1],
            o,
            luma_top,
            luma_top_off,
            lstride,
            luma_w,
            edges,
            ss_hor,
            ss_ver,
            ds_flt,
        );
        l_ptrs[1] = 1;
    } else {
        l_ptrs[0] = 2;
        l_ptrs[1] = 2;
    }

    backup_row_luma_8bpc(
        &mut l_buffers[3],
        o,
        luma,
        luma_off + (1 << ss_ver) * lstride,
        lstride,
        luma_w,
        edges,
        ss_hor,
        ss_ver,
        ds_flt,
    );
    l_ptrs[3] = 3;
    let mut lbak_idx: usize = 4;
    let mut luma_pos = luma_off;

    for y in 0..h {
        if y + 2 < h {
            backup_row_8bpc(
                &mut c_buffers[bak_idx],
                o,
                &*p,
                p_off + (y + 2) * stride,
                &left[y + 2],
                6,
                w,
                2,
                edges,
            );
            c_ptrs[4] = bak_idx;
        } else if edges & LR_HAVE_BOTTOM_INTEGRATED != 0 {
            backup_row_lpf_8bpc(
                &mut c_buffers[bak_idx],
                o,
                &*p,
                p_off + (y + 2) * stride,
                w,
                2,
                edges,
            );
            c_ptrs[4] = bak_idx;
        } else if edges & LR_HAVE_BOTTOM != 0 {
            let offset_y = y + 2 - h;
            backup_row_lpf_8bpc(
                &mut c_buffers[bak_idx],
                o,
                lpf_bottom,
                lpf_bottom_off + offset_y * stride,
                w,
                2,
                edges,
            );
            c_ptrs[4] = bak_idx;
        } else {
            c_ptrs[4] = c_ptrs[3];
        }
        bak_idx += 1;
        if bak_idx == 5 {
            bak_idx = 0;
        }

        if c_ptrs[4] == c_ptrs[3] {
            l_ptrs[4] = l_ptrs[3];
        } else if y + 2 == h && edges & LR_HAVE_BOTTOM_INTEGRATED == 0 {
            backup_row_luma_8bpc(
                &mut l_buffers[lbak_idx],
                o,
                luma_bottom,
                luma_bottom_off,
                lstride,
                luma_w,
                edges,
                ss_hor,
                ss_ver,
                ds_flt,
            );
            l_ptrs[4] = lbak_idx;
        } else if y + 1 == h && edges & LR_HAVE_BOTTOM_INTEGRATED == 0 {
            backup_row_luma_8bpc(
                &mut l_buffers[lbak_idx],
                o,
                luma_bottom,
                luma_bottom_off + lstride,
                0,
                luma_w,
                edges,
                ss_hor,
                ss_ver,
                ds_flt,
            );
            l_ptrs[4] = lbak_idx;
        } else {
            backup_row_luma_8bpc(
                &mut l_buffers[lbak_idx],
                o,
                luma,
                luma_pos + (2 << ss_ver) * lstride,
                lstride,
                luma_w,
                edges,
                ss_hor,
                ss_ver,
                ds_flt,
            );
            l_ptrs[4] = lbak_idx;
        }
        lbak_idx += 1;
        if lbak_idx == 5 {
            lbak_idx = 0;
        }

        for bx in 0..(w >> 2) {
            if ll_mask[y >> 2][0] & (1 << bx) != 0 {
                continue;
            }
            for x in bx * 4..bx * 4 + 4 {
                let m = c_buffers[c_ptrs[2]][o + x] as i32;
                let mut s = m << 7;
                for i in 0..6 {
                    let dy = WIENER_NS_CONFIG_UV[i][0] as i32;
                    let dx = WIENER_NS_CONFIG_UV[i][1] as i32;
                    let a = c_buffers[c_ptrs[(2 + dy) as usize]]
                        [(o as i32 + x as i32 + dx) as usize] as i32;
                    let b = c_buffers[c_ptrs[(2 - dy) as usize]]
                        [(o as i32 + x as i32 - dx) as usize] as i32;
                    s += (a + b - 2 * m) * filter[i] as i32;
                }
                let l = l_buffers[l_ptrs[2]][o + (x << ss_hor)] as i32;
                for i in 0..12 {
                    let dy = WIENER_NS_CONFIG_UV_FROM_Y[i][0] as i32;
                    let dx = WIENER_NS_CONFIG_UV_FROM_Y[i][1] as i32;
                    let lx = (o as i32 + (x as i32 + dx) * (1i32 << ss_hor)) as usize;
                    let lval = l_buffers[l_ptrs[(2 + dy) as usize]][lx] as i32;
                    s += (lval - l) * filter[6 + i] as i32;
                }
                p[p_off + y * stride + x] = iclip((s + 64) >> 7, 0, 255) as u8;
            }
        }

        for r in 0..4 {
            c_ptrs[r] = c_ptrs[r + 1];
        }
        for r in 0..4 {
            l_ptrs[r] = l_ptrs[r + 1];
        }
        luma_pos += lstride << ss_ver;
    }
}

pub fn gdf_prep_8bpc(
    dst: &mut [i8],
    dst_stride: usize,
    p: &[u8],
    p_off: usize,
    stride: usize,
    left: &[[u8; 6]],
    lpf: &[u8],
    lpf_off: usize,
    lpf_bottom: &[u8],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    ref_dst_idx: usize,
    qp_idx: usize,
    edges: u8,
) {
    let mut row_buffers = [[0u8; REST_UNIT_STRIDE]; 13];
    let mut ptrs: [usize; 13] = [0; 13];
    let o = ROW_ORIGIN;

    backup_row_8bpc(&mut row_buffers[6], o, p, p_off, &left[0], 6, w, 6, edges);
    ptrs[6] = 6;

    if edges & LR_HAVE_TOP_INTEGRATED != 0 {
        for n in 0..6 {
            backup_row_lpf_8bpc(
                &mut row_buffers[n],
                o,
                lpf,
                lpf_off + n * stride,
                w,
                6,
                edges,
            );
            ptrs[n] = n;
        }
    } else if edges & LR_HAVE_TOP != 0 {
        backup_row_lpf_8bpc(&mut row_buffers[4], o, lpf, lpf_off, w, 6, edges);
        ptrs[4] = 4;
        backup_row_lpf_8bpc(&mut row_buffers[5], o, lpf, lpf_off + stride, w, 6, edges);
        ptrs[5] = 5;
        ptrs[0] = 4;
        ptrs[1] = 4;
        ptrs[2] = 4;
        ptrs[3] = 4;
    } else {
        for n in 0..6 {
            ptrs[n] = 6;
        }
    }

    let mut bak_idx = 7usize;
    for y in 1..6 {
        backup_row_8bpc(
            &mut row_buffers[bak_idx],
            o,
            p,
            p_off + y * stride,
            &left[y],
            6,
            w,
            6,
            edges,
        );
        ptrs[bak_idx] = bak_idx;
        bak_idx += 1;
    }

    let alpha_base = ref_dst_idx * 528 + qp_idx * 88;
    let weight_base = ref_dst_idx * 1584 + qp_idx * 264;
    let bias_idx = ref_dst_idx * 6 + qp_idx;

    let (error_lut_base, scale) = if ref_dst_idx == 0 {
        (qp_idx * 4096, 8i32)
    } else {
        ((ref_dst_idx - 1) * 6000 + qp_idx * 1000, 5i32)
    };

    let mut grad = [[[0u16; 4]; GRADIENT_BUF_STRIDE]; 2];
    {
        let refs: [&[u8]; 13] = core::array::from_fn(|i| &row_buffers[ptrs[i]] as &[u8]);
        compute_gradient_row_8bpc(&mut grad[0], &refs, 6, o, w, 0);
    }
    let mut grad_bit: usize = 1;

    for y in 0..h {
        if y + 6 < h {
            backup_row_8bpc(
                &mut row_buffers[bak_idx],
                o,
                p,
                p_off + (y + 6) * stride,
                &left[y + 6],
                6,
                w,
                6,
                edges,
            );
            ptrs[12] = bak_idx;
        } else if edges & LR_HAVE_BOTTOM_INTEGRATED != 0 {
            backup_row_lpf_8bpc(
                &mut row_buffers[bak_idx],
                o,
                p,
                p_off + (y + 6) * stride,
                w,
                6,
                edges,
            );
            ptrs[12] = bak_idx;
        } else if y + 4 < h && edges & LR_HAVE_BOTTOM != 0 {
            let offset_y = y + 6 - h;
            backup_row_lpf_8bpc(
                &mut row_buffers[bak_idx],
                o,
                lpf_bottom,
                lpf_bottom_off + offset_y * stride,
                w,
                6,
                edges,
            );
            ptrs[12] = bak_idx;
        } else {
            ptrs[12] = ptrs[11];
        }
        bak_idx += 1;
        if bak_idx == 13 {
            bak_idx = 0;
        }

        if y & 1 == 0 {
            let refs: [&[u8]; 13] = core::array::from_fn(|i| &row_buffers[ptrs[i]] as &[u8]);
            compute_gradient_row_8bpc(&mut grad[grad_bit], &refs, 8, o, w, 0);
            grad_bit ^= 1;
        }

        let mut x1 = 0usize;
        while x1 < w {
            let mut grad_sums = [0i32; 4];
            let hx = x1 >> 1;
            for d in 0..4 {
                grad_sums[d] = grad[0][hx][d] as i32
                    + grad[0][hx + 1][d] as i32
                    + grad[1][hx][d] as i32
                    + grad[1][hx + 1][d] as i32;
            }
            let cls = ((grad_sums[0] <= grad_sums[1]) as usize)
                | (((grad_sums[2] <= grad_sums[3]) as usize) << 1);

            let mut shared_vals = [0i32; 3];
            for idx in 0..3 {
                shared_vals[idx] = GDF_BIAS[bias_idx][idx] as i32;
            }
            for d in 0..4 {
                let k = d + 18;
                let alpha = GDF_ALPHA[alpha_base + k * 4 + cls] as i32;
                let v = imin(grad_sums[d] >> 2, alpha);
                for idx in 0..3 {
                    shared_vals[idx] += v * GDF_WEIGHT[weight_base + idx * 88 + k * 4 + cls] as i32;
                }
            }

            for x2 in 0..2 {
                let x = x1 + x2;
                let mut idx_vals = shared_vals;
                let m = row_buffers[ptrs[6]][o + x] as i32;
                for k in 0..18 {
                    let alpha = GDF_ALPHA[alpha_base + k * 4 + cls] as i32;
                    let dy = GDF_COORDS[k][0] as i32;
                    let dx = GDF_COORDS[k][1] as i32;
                    let a = row_buffers[ptrs[(6 - dy) as usize]]
                        [(o as i32 + x as i32 - dx) as usize] as i32;
                    let b = row_buffers[ptrs[(6 + dy) as usize]]
                        [(o as i32 + x as i32 + dx) as usize] as i32;
                    let above = iclip((a - m) << 2, -alpha, alpha);
                    let below = iclip((b - m) << 2, -alpha, alpha);
                    let v = iclip(above + below, -512, 511);
                    for idx in 0..3 {
                        idx_vals[idx] +=
                            v * GDF_WEIGHT[weight_base + idx * 88 + k * 4 + cls] as i32;
                    }
                }

                let mut full_idx = 0usize;
                for idx in 0..3 {
                    let sv = idx_vals[idx] * scale;
                    let v = apply_sign((sv.abs() + (1 << 14)) >> 15, sv);
                    let sub_idx = (iclip(v, -scale, scale - 1) + scale) as usize;
                    full_idx = full_idx * (scale as usize * 2) + sub_idx;
                }
                if ref_dst_idx == 0 {
                    dst[y * dst_stride + x] = GDF_INTRA_ERROR[error_lut_base + full_idx];
                } else {
                    dst[y * dst_stride + x] = GDF_INTER_ERROR[error_lut_base + full_idx];
                }
            }
            x1 += 2;
        }

        for r in 0..12 {
            ptrs[r] = ptrs[r + 1];
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LrEdgeFlags(pub u8);

impl LrEdgeFlags {
    pub const HAVE_TOP: Self = Self(1 << 0);
    pub const HAVE_BOTTOM: Self = Self(1 << 1);
    pub const HAVE_LEFT: Self = Self(1 << 2);
    pub const HAVE_RIGHT: Self = Self(1 << 3);
    pub const HAVE_TOP_INTEGRATED: Self = Self(1 << 4);
    pub const HAVE_BOTTOM_INTEGRATED: Self = Self(1 << 5);

    pub fn has(self, flag: Self) -> bool {
        self.0 & flag.0 != 0
    }

    pub fn with(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }

    pub fn without(self, flag: Self) -> Self {
        Self(self.0 & !flag.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RestorationType {
    None = 0,
    PcWiener = 1,
    NsWiener = 2,
    Switchable = 3,
}

#[derive(Debug, Clone)]
pub struct RestorationUnit {
    pub restoration_type: RestorationType,
    pub ns_filter: [[i8; 32]; 16],
}

impl Default for RestorationUnit {
    fn default() -> Self {
        Self {
            restoration_type: RestorationType::None,
            ns_filter: [[0; 32]; 16],
        }
    }
}

pub const LR_RESTORE_Y: i32 = 1 << 0;
pub const LR_RESTORE_U: i32 = 1 << 1;
pub const LR_RESTORE_V: i32 = 1 << 2;

pub const INLOOPFILTER_DEBLOCK: u32 = 1 << 0;
pub const INLOOPFILTER_CDEF: u32 = 1 << 1;
pub const INLOOPFILTER_CCSO: u32 = 1 << 2;
pub const INLOOPFILTER_WIENER: u32 = 1 << 3;
pub const INLOOPFILTER_GDF: u32 = 1 << 4;
pub const INLOOPFILTER_ALL: u32 = INLOOPFILTER_DEBLOCK
    | INLOOPFILTER_CDEF
    | INLOOPFILTER_CCSO
    | INLOOPFILTER_WIENER
    | INLOOPFILTER_GDF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FirstSbInTileRow {
    None = 0,
    Top = 1,
    Bottom = 2,
}

#[repr(C)]
pub struct WienerParamsSingle {
    pub filter: *const i8,
    pub luma: *const u8,
    pub luma_top: *const u8,
    pub luma_bottom: *const u8,
    pub stride: isize,
    pub ss_ver: i32,
    pub ss_hor: i32,
    pub ds_flt: i32,
}

#[repr(C)]
pub struct WienerParamsMulti {
    pub filters: *const u8,
    pub subclass_lut: *const u8,
    pub noskip_mask: *const u16,
    pub base_q: i32,
}

#[repr(C)]
pub union WienerParamsC {
    pub single: std::mem::ManuallyDrop<WienerParamsSingle>,
    pub multi: std::mem::ManuallyDrop<WienerParamsMulti>,
}

type WienerFilterFn = unsafe extern "C" fn(
    dst: *mut u8,
    dst_stride: isize,
    left: *const [u8; 6],
    top: *const u8,
    bottom: *const u8,
    w: i32,
    h: i32,
    params: *const WienerParamsC,
    edges: u8,
    ll_mask: *const [u16; 4],
);

type GdfPrepFn = unsafe extern "C" fn(
    dst: *mut i8,
    dst_stride: isize,
    p: *const u8,
    stride: isize,
    left: *const [u8; 6],
    top: *const u8,
    bottom: *const u8,
    w: i32,
    h: i32,
    ref_dst_idx: i32,
    qp_idx: i32,
    edges: u8,
);

type GdfAddFn = unsafe extern "C" fn(
    p: *mut u8,
    dst_stride: isize,
    err: *const i8,
    err_stride: isize,
    w: i32,
    h: i32,
    scale: i32,
    ll_mask: *const [u16; 4],
);

use crate::dsp::LoopRestorationDSPContext;
use crate::headers::{FhRestorationPlane, PixelLayout};
use crate::lf_mask::{Av2Filter, Av2Restoration};

pub struct LrContext<'a> {
    pub dsp_lr: &'a LoopRestorationDSPContext,
    pub restoration_p: &'a [FhRestorationPlane; 3],
    pub gdf_qp_idx: i32,
    pub gdf_scale: i32,
    pub sb128: bool,
    pub cfl_ds_filter_index: i32,
    pub layout: PixelLayout,
    pub bw: i32,
    pub bh: i32,
    pub sb256w: i32,
    pub sbh: i32,
    pub mask: &'a [Av2Filter],
    pub lr_mask: &'a [Av2Restoration],
    pub lr_db_line: &'a [Vec<u8>; 3],
    pub lr_cdef_line: &'a [Vec<u8>; 3],
    /// Luma plane (read-only) for chroma cross-component single-Wiener; dav2d's
    /// `f->lf.p[0]`. The deferred filter pass processes chroma before luma, so
    /// this is the pre-luma-LR luma for the current sbrow.
    pub lf_p_luma: &'a [u8],
    pub base_q: i32,
    pub gdf_ref_dst_idx: i32,
    pub start_of_tile_row: &'a [u8],
    pub ns_subclass_lut: &'a [u8],
    pub pc_subclass_lut: &'a [u8],
    pub pc_filters: &'a [[i16; 13]],
    pub n_tc: i32,
    pub inloop_filters: u32,
    pub cur_stride: [isize; 2],
    pub unit_size: [u8; 2],
    pub restore_planes: i32,
}

fn backup_nxu(
    dst: &mut [[u8; 6]],
    src: &[u8],
    src_off: usize,
    src_stride: isize,
    mut u: i32,
    n: usize,
) {
    let abs_stride = src_stride.unsigned_abs();
    let mut di = 0usize;
    let mut si = src_off;
    while u > 0 {
        if di < dst.len() && si >= n && si + n <= src.len() {
            dst[di][6 - n..6].copy_from_slice(&src[si - n..si]);
        }
        di += 1;
        si += abs_stride;
        u -= 1;
    }
}

fn copy_n_lines(dst: &mut [u8], src: &[u8], stride: isize, n: usize) {
    let len = stride.unsigned_abs() * n;
    let copy_len = len.min(dst.len()).min(src.len());
    dst[..copy_len].copy_from_slice(&src[..copy_len]);
}

fn lr_stripe_8bpc(
    ctx: &LrContext,
    p: &mut [u8],
    p_off: usize,
    left: &[[u8; 6]],
    x: i32,
    mut y: i32,
    plane: i32,
    w: i32,
    row_h: i32,
    lr: &crate::lf_mask::Av2RestorationUnit,
    mut edges: u8,
    first_sby_in_tile_row: FirstSbInTileRow,
    tile_row_m1: i32,
) {
    let chroma = (plane != 0) as i32;
    let ss_ver = chroma & (ctx.layout == PixelLayout::I420) as i32;
    let ss_hor = chroma & (ctx.layout != PixelLayout::I444) as i32;
    let stride = ctx.cur_stride[chroma as usize];
    let abs_stride = stride.unsigned_abs();
    let sby = (y + if y != 0 { 8 << ss_ver } else { 0 }) >> (6 - ss_ver + ctx.sb128 as i32);
    let have_tt = (ctx.n_tc > 1) as i32;
    let sb256x = (x << ss_hor) >> 8;
    let sb64x_idx = ((x << ss_hor) >> 6) & 3;

    let mut stripe_h = imin(
        (64 - 8 * (first_sby_in_tile_row != FirstSbInTileRow::None) as i32) >> ss_ver,
        row_h - y,
    );

    let ref_dst_idx = ctx.gdf_ref_dst_idx;
    let qp_idx = ctx.gdf_qp_idx;
    let gdf_scale = ctx.gdf_scale;

    let wiener_type: u8 = if ctx.inloop_filters & INLOOPFILTER_WIENER != 0 {
        lr.restoration_type
    } else {
        0 // RestorationType::None
    };

    // Rust-native LR kernel selection (replaces the dav2d fn-pointer dispatch:
    // the asm/dsp_lr pointers are all None, so we call the slice kernels directly,
    // mirroring lr_apply_tmpl.c:79-101).
    #[derive(PartialEq)]
    enum WKind {
        None,
        NsSingle,
        NsMulti,
        PcWiener,
    }
    let mut multi_wiener = false;
    let mut noskip_mask = [0u16; 66];

    let pd = &ctx.restoration_p[plane as usize].ns;

    let wkind = if wiener_type == RestorationType::NsWiener as u8 {
        if pd.frame_filters_on != 0 {
            if pd.num_classes == 1 {
                WKind::NsSingle
            } else {
                multi_wiener = true;
                WKind::NsMulti
            }
        } else {
            WKind::NsSingle
        }
    } else if wiener_type == RestorationType::PcWiener as u8 {
        multi_wiener = true;
        WKind::PcWiener
    } else {
        WKind::None
    };

    if multi_wiener {
        let mut r = 0usize;
        let mut by = y >> 2;
        while by < row_h >> 2 {
            let by_idx = (by & 63) as usize;
            let sb256_idx = ctx.sb256w as usize * (by as usize >> 6) + sb256x as usize;
            if sb256_idx < ctx.mask.len() {
                let noskip_row = &ctx.mask[sb256_idx].lr_noskip_mask[by_idx];
                noskip_mask[r] = noskip_row[sb64x_idx as usize];
                if edges & LR_HAVE_RIGHT == 0 && w & 63 != 0 {
                    let shift = ((w >> 2) & 15) - 1;
                    let mask_val = noskip_mask[r];
                    let edge_bit = mask_val >> shift as u16;
                    noskip_mask[r] |= edge_bit << ((shift + 1) as u16);
                }
            }
            r += 1;
            by += 1;
        }
    }

    let lstride = ctx.cur_stride[0];
    let lstride_u = lstride.unsigned_abs();
    let stride_u = abs_stride;

    let mut gdf_err = [0i8; 64 * 64];
    let mut left_idx = 0usize;
    let mut noskip_offset = 0usize;
    // Running pixel offset of the current stripe's first row (dav2d advances `p`
    // by `stripe_h * stride` each iteration). The slice kernels index `p` with a
    // local stripe-relative `y`, so this must track the stripe origin.
    let mut cur_off = p_off;
    // Running deblocked-line (lpf) offset. dav2d advances lpf by 4*stride per
    // stripe (lr_apply_tmpl.c:188); the single-thread base term is just `x`.
    let mut lpf_off =
        (have_tt as isize * (sby as isize * (4 << ctx.sb128 as i32) - 4) * stride_u as isize
            + x as isize) as usize;
    let mut llpf_off =
        (have_tt as isize * (sby as isize * (4 << ctx.sb128 as i32) - 4) * lstride_u as isize
            + (x * 2) as isize) as usize;

    while y + stripe_h <= row_h {
        edges ^= ((-(((sby + 1) != ctx.sbh) as i32 | (y + stripe_h != row_h) as i32)) as u8
            ^ edges)
            & LR_HAVE_BOTTOM;

        let inc = if (edges & (LR_HAVE_TOP | LR_HAVE_TOP_INTEGRATED | LR_HAVE_BOTTOM_INTEGRATED))
            == LR_HAVE_TOP
            && y + 8 < ((ctx.bh * 4) >> ss_ver)
        {
            8
        } else {
            0
        };

        let sb256_idx =
            ctx.sb256w as usize * ((((y << ss_ver) + inc) as usize) >> 8) + sb256x as usize;

        let gdf_enabled = plane == 0
            && ctx.inloop_filters & INLOOPFILTER_GDF != 0
            && sb256_idx < ctx.mask.len()
            && ctx.mask[sb256_idx].gdf[((((y + inc) >> 4) & 12) + sb64x_idx) as usize] != 0;

        let plane_u = plane as usize;
        let stripe_u = stripe_h as usize;
        let w_u = w as usize;

        // `top` source: the cross-tile-row CDEF line when both HAVE_TOP and
        // HAVE_TOP_INTEGRATED are set; otherwise the deblocked-line store.
        let cdef_top: Option<usize> = if (edges & (LR_HAVE_TOP | LR_HAVE_TOP_INTEGRATED))
            == (LR_HAVE_TOP | LR_HAVE_TOP_INTEGRATED)
        {
            let cdef_off = tile_row_m1 as usize * (6 - 4 * chroma as usize) * stride_u + x as usize;
            let off = cdef_off + (2 * (1 - chroma) as usize) * stride_u;
            if off < ctx.lr_cdef_line[plane_u].len() {
                Some(off)
            } else {
                None
            }
        } else {
            None
        };

        // GDF prep: compute the per-pixel error into gdf_err.
        if gdf_enabled {
            let (top_slice, top_off): (&[u8], usize) = match cdef_top {
                Some(off) => (&ctx.lr_cdef_line[plane_u], off),
                None => (&ctx.lr_db_line[plane_u], lpf_off),
            };
            gdf_prep_8bpc(
                &mut gdf_err,
                64,
                &*p,
                cur_off,
                stride_u,
                &left[left_idx..],
                top_slice,
                top_off,
                &ctx.lr_db_line[plane_u],
                lpf_off + 6 * stride_u,
                w_u,
                stripe_u,
                ref_dst_idx as usize,
                qp_idx as usize,
                edges,
            );
        }

        let y4 = (((y << ss_ver) & 255) >> 2) as usize;

        // Build the lossless-mask slice for this stripe. dav2d packs the chroma
        // mask (ss_hor) into a single byte per 4-row; for luma / non-ss_hor it is
        // the raw lossless mask offset by sb64x_idx (lr_apply_tmpl.c:154-169).
        let ll_rows = (stripe_h >> 2).max(1) as usize + 1;
        let mut ll_mask_buf = vec![[0u16; 4]; ll_rows];
        if sb256_idx < ctx.mask.len() {
            let m = &ctx.mask[sb256_idx];
            if plane == 0 || ss_hor == 0 {
                let src = if plane == 0 {
                    &m.lossless_mask_y
                } else {
                    &m.lossless_mask_uv
                };
                for (r, slot) in ll_mask_buf.iter_mut().enumerate() {
                    let yy = y4 + r;
                    if yy < 64 {
                        let row = &src[yy];
                        for k in 0..4 {
                            if sb64x_idx as usize + k < 4 {
                                slot[k] = row[sb64x_idx as usize + k];
                            }
                        }
                    }
                }
            } else {
                // chroma, ss_hor: pack two adjacent columns into the low/high byte.
                let init_y = (y >> 2) as usize;
                let end_y = ((y + stripe_h) >> 2) as usize;
                let base = (y4 >> ss_ver) as usize;
                for (r, yy) in (init_y..end_y).enumerate() {
                    let src_y = base + (yy - init_y);
                    if src_y < 64 && r < ll_mask_buf.len() {
                        let row = &m.lossless_mask_uv[src_y];
                        let lo = row[sb64x_idx as usize];
                        let hi = if sb64x_idx as usize + 1 < 4 {
                            row[sb64x_idx as usize + 1]
                        } else {
                            0
                        };
                        ll_mask_buf[r][0] = (lo & 0xff) | ((hi & 0xff) << 8);
                    }
                }
            }
        }

        if wkind != WKind::None {
            let (top_slice, top_off): (&[u8], usize) = match cdef_top {
                Some(off) => (&ctx.lr_cdef_line[plane_u], off),
                None => (&ctx.lr_db_line[plane_u], lpf_off),
            };
            // Take read-only copies of the line stores so the kernel can borrow
            // `p` mutably while reading lpf/luma (they are distinct allocations).
            let db_line = ctx.lr_db_line[plane_u].clone();
            let bottom_off = lpf_off + 6 * stride_u;
            match wkind {
                WKind::NsSingle if chroma == 0 => {
                    let mut filter = [0i8; 16];
                    let src = if pd.frame_filters_on != 0 {
                        &pd.filter[0][..16]
                    } else {
                        &lr.ns_filter[0][..16]
                    };
                    filter.copy_from_slice(src);
                    let top_owned = if cdef_top.is_some() {
                        Some(top_slice.to_vec())
                    } else {
                        None
                    };
                    let (ts, to): (&[u8], usize) = match &top_owned {
                        Some(v) => (v, top_off),
                        None => (&db_line, lpf_off),
                    };
                    ns_wiener_single_y_8bpc(
                        p,
                        cur_off,
                        stride_u,
                        &left[left_idx..],
                        ts,
                        to,
                        &db_line,
                        bottom_off,
                        w_u,
                        stripe_u,
                        &filter,
                        edges,
                        &ll_mask_buf,
                    );
                }
                WKind::NsSingle => {
                    // chroma single wiener (uses luma for cross-component refine).
                    let mut filter = [0i8; 18];
                    let src = if pd.frame_filters_on != 0 {
                        &pd.filter[0][..18]
                    } else {
                        &lr.ns_filter[0][..18]
                    };
                    filter.copy_from_slice(src);
                    let luma = ctx.lf_p_luma.to_vec();
                    let luma_off = (x << ss_hor) as usize + (y << ss_ver) as usize * lstride_u;
                    let luma_db = ctx.lr_db_line[0].clone();
                    let top_owned = if cdef_top.is_some() {
                        Some(top_slice.to_vec())
                    } else {
                        None
                    };
                    let (ts, to): (&[u8], usize) = match &top_owned {
                        Some(v) => (v, top_off),
                        None => (&db_line, lpf_off),
                    };
                    ns_wiener_single_uv_8bpc(
                        p,
                        cur_off,
                        stride_u,
                        &left[left_idx..],
                        ts,
                        to,
                        &db_line,
                        bottom_off,
                        w_u,
                        stripe_u,
                        &filter,
                        &luma,
                        luma_off,
                        lstride_u,
                        &luma_db,
                        llpf_off,
                        &luma_db,
                        llpf_off + 6 * lstride_u,
                        ss_hor as usize,
                        ss_ver as usize,
                        ctx.cfl_ds_filter_index,
                        edges,
                        &ll_mask_buf,
                    );
                }
                WKind::NsMulti => {
                    let filters: &[[i8; 18]] = if pd.frame_filters_on != 0 {
                        &pd.filter
                    } else {
                        // per-unit ns_filter is [i8;32]; the multi kernel reads the
                        // first 18 taps per class.
                        // SAFETY: reinterpret the [i8;32] rows as [i8;18] prefixes.
                        unsafe {
                            std::slice::from_raw_parts(
                                lr.ns_filter.as_ptr() as *const [i8; 18],
                                lr.ns_filter.len(),
                            )
                        }
                    };
                    let top_owned = if cdef_top.is_some() {
                        Some(top_slice.to_vec())
                    } else {
                        None
                    };
                    let (ts, to): (&[u8], usize) = match &top_owned {
                        Some(v) => (v, top_off),
                        None => (&db_line, lpf_off),
                    };
                    ns_wiener_multi_8bpc(
                        p,
                        cur_off,
                        stride_u,
                        &left[left_idx..],
                        ts,
                        to,
                        &db_line,
                        bottom_off,
                        w_u,
                        stripe_u,
                        filters,
                        ctx.ns_subclass_lut,
                        &noskip_mask[noskip_offset..],
                        ctx.base_q,
                        edges,
                        &ll_mask_buf,
                    );
                    noskip_offset += (stripe_h >> 2) as usize;
                }
                WKind::PcWiener => {
                    let top_owned = if cdef_top.is_some() {
                        Some(top_slice.to_vec())
                    } else {
                        None
                    };
                    let (ts, to): (&[u8], usize) = match &top_owned {
                        Some(v) => (v, top_off),
                        None => (&db_line, lpf_off),
                    };
                    pc_wiener_8bpc(
                        p,
                        cur_off,
                        stride_u,
                        &left[left_idx..],
                        ts,
                        to,
                        &db_line,
                        bottom_off,
                        w_u,
                        stripe_u,
                        ctx.pc_filters,
                        ctx.pc_subclass_lut,
                        &noskip_mask[noskip_offset..],
                        ctx.base_q,
                        edges,
                        &ll_mask_buf,
                    );
                    noskip_offset += (stripe_h >> 2) as usize;
                }
                WKind::None => {}
            }
        }

        if gdf_enabled {
            gdf_add_8bpc(
                &mut p[cur_off..],
                stride_u,
                &gdf_err,
                64,
                w_u,
                stripe_u,
                gdf_scale,
                &ll_mask_buf,
            );
        }

        edges &= !(LR_HAVE_BOTTOM_INTEGRATED | LR_HAVE_TOP_INTEGRATED);
        left_idx += stripe_h as usize;
        cur_off += stripe_u * stride_u;
        y += stripe_h;
        edges |= LR_HAVE_TOP;
        stripe_h = imin(64 >> ss_ver, row_h - y);
        if stripe_h == 0 {
            break;
        }
        // dav2d advances lpf/llpf by 4 rows per stripe (lr_apply_tmpl.c:188-189).
        lpf_off += 4 * stride_u;
        llpf_off += 4 * lstride_u;
    }
}

fn lr_sbrow(
    ctx: &LrContext,
    p: &mut [u8],
    p_off: usize,
    y: i32,
    w: i32,
    h: i32,
    row_h: i32,
    plane: i32,
    first_sby_in_tile_row: FirstSbInTileRow,
    tile_row_m1: i32,
) {
    let chroma = (plane != 0) as i32;
    let ss_ver = chroma & (ctx.layout == PixelLayout::I420) as i32;
    let ss_hor = chroma & (ctx.layout != PixelLayout::I444) as i32;
    let p_stride = ctx.cur_stride[chroma as usize];
    let abs_stride = p_stride.unsigned_abs();

    let unit_size_log2 = ctx.unit_size[chroma.min(1) as usize] as i32;
    let unit_size = 1 << unit_size_log2;
    let half_unit_size = unit_size >> 1;

    let row_y = y + ((8 >> ss_ver) * (first_sby_in_tile_row != FirstSbInTileRow::None) as i32);
    let shift_hor = 8 - ss_hor;

    let mut pre_lr_border = [[[0u8; 6]; 264]; 2];

    let mut aligned_unit_pos = row_y & !(unit_size - 1);
    if aligned_unit_pos != 0 && aligned_unit_pos + half_unit_size > h {
        aligned_unit_pos -= unit_size;
    }
    aligned_unit_pos <<= ss_ver;
    let sb_idx = (aligned_unit_pos >> 8) * ctx.sb256w;
    let unit_idx_base = ((aligned_unit_pos >> 6) & 0x3) << 2;

    let mut edges: u8 = if y > 0 { LR_HAVE_TOP } else { 0 } | LR_HAVE_RIGHT;
    if first_sby_in_tile_row == FirstSbInTileRow::Top {
        edges |= LR_HAVE_BOTTOM_INTEGRATED;
    }
    if first_sby_in_tile_row == FirstSbInTileRow::Bottom && y > 0 {
        edges |= LR_HAVE_TOP_INTEGRATED;
    }

    let mut lr_idx_0 = sb_idx as usize;
    let mut lr_unit_0 = unit_idx_base as usize;
    let mut bit = 0usize;
    let mut x = 0i32;
    let mut cur_p_off = p_off;

    while x + 64 < w {
        let next_x = x + 64;
        let mut next_iter_lru_start_x = next_x & !(unit_size - 1);
        if next_iter_lru_start_x != 0 && w - next_iter_lru_start_x < half_unit_size {
            next_iter_lru_start_x -= unit_size;
        }
        let next_u_idx = unit_idx_base + ((next_iter_lru_start_x >> (shift_hor - 2)) & 3);
        let next_sb_idx = sb_idx + (next_iter_lru_start_x >> shift_hor);

        let n = if plane != 0 { 2 } else { 6 };
        let border_src_off = cur_p_off + 64;
        if border_src_off + n <= p.len() {
            for row in 0..(row_h - y) as usize {
                let src_row = border_src_off + row * abs_stride;
                if src_row >= n && src_row < p.len() {
                    let start = src_row - n;
                    for i in 0..n.min(6) {
                        pre_lr_border[bit][row][6 - n + i] = p[start + i];
                    }
                }
            }
        }

        let lr_sb = lr_idx_0;
        let lr_u = lr_unit_0;
        if lr_sb < ctx.lr_mask.len() && lr_u < ctx.lr_mask[lr_sb].lr[plane as usize].len() {
            let lr_unit = &ctx.lr_mask[lr_sb].lr[plane as usize][lr_u];
            lr_stripe_8bpc(
                ctx,
                p,
                cur_p_off,
                &pre_lr_border[1 - bit],
                x,
                y,
                plane,
                64,
                row_h,
                lr_unit,
                edges,
                first_sby_in_tile_row,
                tile_row_m1,
            );
        }

        lr_idx_0 = next_sb_idx as usize;
        lr_unit_0 = next_u_idx as usize;
        x = next_x;
        cur_p_off += 64;
        edges |= LR_HAVE_LEFT;
        bit ^= 1;
    }

    edges &= !LR_HAVE_RIGHT;
    let end_w = w - x;
    let lr_sb = lr_idx_0;
    let lr_u = lr_unit_0;
    if lr_sb < ctx.lr_mask.len() && lr_u < ctx.lr_mask[lr_sb].lr[plane as usize].len() {
        let lr_unit = &ctx.lr_mask[lr_sb].lr[plane as usize][lr_u];
        lr_stripe_8bpc(
            ctx,
            p,
            cur_p_off,
            &pre_lr_border[1 - bit],
            x,
            y,
            plane,
            end_w,
            row_h,
            lr_unit,
            edges,
            first_sby_in_tile_row,
            tile_row_m1,
        );
    }
}

pub fn lr_sbrow_8bpc(ctx: &LrContext, dst: &mut [&mut [u8]; 3], sby: i32) {
    let dst_stride = ctx.cur_stride;
    let restore_planes = ctx.restore_planes;
    let not_last = (sby + 1 < ctx.sbh) as i32;
    let start_tile_val = if (sby as usize) < ctx.start_of_tile_row.len() {
        ctx.start_of_tile_row[sby as usize]
    } else {
        0
    };
    let tile_row = (start_tile_val >> 1) as i32;
    let first_sby_in_tile_row_flag = start_tile_val & 1;

    if restore_planes & (LR_RESTORE_U | LR_RESTORE_V) != 0 {
        let ss_ver = (ctx.layout == PixelLayout::I420) as i32;
        let ss_hor = (ctx.layout != PixelLayout::I444) as i32;
        let h = (ctx.bh * 4) >> ss_ver;
        let w = (ctx.bw * 4) >> ss_hor;
        let next_row_y = (sby + 1) << ((6 - ss_ver) + ctx.sb128 as i32);
        let row_h = imin(next_row_y - (8 >> ss_ver) * not_last, h);
        let offset_uv = (8 * (sby != 0) as i32) >> ss_ver;
        let mut y_stripe = (sby << ((6 - ss_ver) + ctx.sb128 as i32)) - offset_uv;

        if sby != 0 && first_sby_in_tile_row_flag != 0 {
            let first_top = if first_sby_in_tile_row_flag != 0 {
                FirstSbInTileRow::Top
            } else {
                FirstSbInTileRow::None
            };
            if restore_planes & LR_RESTORE_U != 0 {
                let abs_stride_uv = dst_stride[1].unsigned_abs();
                lr_sbrow(
                    ctx,
                    dst[1],
                    (y_stripe.max(0) as usize) * abs_stride_uv,
                    y_stripe,
                    w,
                    h,
                    y_stripe + (8 >> ss_ver),
                    1,
                    first_top,
                    tile_row - 1,
                );
            }
            if restore_planes & LR_RESTORE_V != 0 {
                let abs_stride_uv = dst_stride[1].unsigned_abs();
                lr_sbrow(
                    ctx,
                    dst[2],
                    (y_stripe.max(0) as usize) * abs_stride_uv,
                    y_stripe,
                    w,
                    h,
                    y_stripe + (8 >> ss_ver),
                    2,
                    first_top,
                    tile_row - 1,
                );
            }
            y_stripe += 8 >> ss_ver;
        }

        let first_bot = if first_sby_in_tile_row_flag != 0 {
            FirstSbInTileRow::Bottom
        } else {
            FirstSbInTileRow::None
        };
        if restore_planes & LR_RESTORE_U != 0 {
            let abs_stride_uv = dst_stride[1].unsigned_abs();
            lr_sbrow(
                ctx,
                dst[1],
                (y_stripe.max(0) as usize) * abs_stride_uv,
                y_stripe,
                w,
                h,
                row_h,
                1,
                first_bot,
                tile_row - 1,
            );
        }
        if restore_planes & LR_RESTORE_V != 0 {
            let abs_stride_uv = dst_stride[1].unsigned_abs();
            lr_sbrow(
                ctx,
                dst[2],
                (y_stripe.max(0) as usize) * abs_stride_uv,
                y_stripe,
                w,
                h,
                row_h,
                2,
                first_bot,
                tile_row - 1,
            );
        }
    }

    if restore_planes & LR_RESTORE_Y != 0 {
        let h = ctx.bh * 4;
        let w = ctx.bw * 4;
        let next_row_y = (sby + 1) << (6 + ctx.sb128 as i32);
        let row_h = imin(next_row_y - 8 * not_last, h);
        let offset_y = 8 * (sby != 0) as i32;
        let mut y_stripe = (sby << (6 + ctx.sb128 as i32)) - offset_y;

        if sby != 0 && first_sby_in_tile_row_flag != 0 {
            let abs_stride_y = dst_stride[0].unsigned_abs();
            lr_sbrow(
                ctx,
                dst[0],
                (y_stripe.max(0) as usize) * abs_stride_y,
                y_stripe,
                w,
                h,
                y_stripe + 8,
                0,
                FirstSbInTileRow::Top,
                tile_row - 1,
            );
            y_stripe += 8;
        }
        let abs_stride_y = dst_stride[0].unsigned_abs();
        lr_sbrow(
            ctx,
            dst[0],
            (y_stripe.max(0) as usize) * abs_stride_y,
            y_stripe,
            w,
            h,
            row_h,
            0,
            if first_sby_in_tile_row_flag != 0 {
                FirstSbInTileRow::Bottom
            } else {
                FirstSbInTileRow::None
            },
            tile_row - 1,
        );
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
        backup_row_8bpc(
            &mut dst,
            o,
            &src,
            0,
            &left,
            6,
            16,
            6,
            LR_HAVE_LEFT | LR_HAVE_RIGHT,
        );
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
        let idx = get_class_lut_idx_8bpc(&rows, 6, 0, &noskip, 32, 1, 1, 8);
        let _ = idx;
    }

    #[test]
    fn test_backup_row_luma_no_ss_ver() {
        let src = vec![42u8; 64];
        let mut dst = vec![0u8; 32];
        let o = 6;
        backup_row_luma_8bpc(&mut dst, o, &src, 0, 16, 16, 0, 0, 0, 0);
        assert_eq!(&dst[o..o + 16], &src[0..16]);
        assert!(dst[o - 4..o].iter().all(|&v| v == 42));
    }

    #[test]
    fn test_backup_row_luma_box_filter() {
        let stride = 32;
        let mut src = vec![0u8; stride * 2 + 16];
        for x in 0..16 {
            src[4 + x] = 100;
            src[4 + stride + x] = 200;
        }
        let mut dst = vec![0u8; 32];
        let o = 4;
        backup_row_luma_8bpc(&mut dst, o, &src, 4, stride, 8, 0, 1, 1, 0);
        assert_eq!(dst[o], ((100 + 200 + 100 + 200) >> 2) as u8);
    }

    #[test]
    fn test_backup_row_luma_vert_avg() {
        let stride = 32;
        let mut src = vec![0u8; stride * 2 + 16];
        for x in 0..16 {
            src[4 + x] = 100;
            src[4 + stride + x] = 200;
        }
        let mut dst = vec![0u8; 32];
        let o = 4;
        backup_row_luma_8bpc(&mut dst, o, &src, 4, stride, 8, 0, 0, 1, 1);
        for x in 0..8 {
            assert_eq!(dst[o + x], 150);
        }
    }

    #[test]
    fn test_backup_row_luma_copy() {
        let stride = 32;
        let mut src = vec![0u8; stride * 2 + 16];
        for x in 0..16 {
            src[4 + x] = 100;
            src[4 + stride + x] = 200;
        }
        let mut dst = vec![0u8; 32];
        let o = 4;
        backup_row_luma_8bpc(&mut dst, o, &src, 4, stride, 8, 0, 0, 1, 2);
        for x in 0..8 {
            assert_eq!(dst[o + x], 100);
        }
    }

    #[test]
    fn test_backup_row_luma_edges() {
        let stride = 32;
        let src = vec![80u8; stride * 2 + 16];
        let mut dst = vec![0u8; 32];
        let o = 6;
        backup_row_luma_8bpc(
            &mut dst,
            o,
            &src,
            6,
            stride,
            8,
            LR_HAVE_LEFT | LR_HAVE_RIGHT,
            0,
            1,
            2,
        );
        assert_eq!(&dst[o..o + 8], &[80u8; 8]);
        assert_eq!(&dst[o - 4..o], &[80u8; 4]);
        assert_eq!(&dst[o + 8..o + 12], &[80u8; 4]);
    }

    #[test]
    fn test_ns_wiener_single_y_identity() {
        let stride = 16;
        let h = 8;
        let w = 8;
        let p_off = stride + 4;
        let mut p = vec![128u8; stride * (h + 8)];
        for y in 0..h + 4 {
            for x in 0..w + 8 {
                p[p_off - 4 + y * stride + x] = 128;
            }
        }
        let left = vec![[128u8; 6]; h + 4];
        let lpf = vec![128u8; stride * 4 + 8];
        let lpf_bottom = vec![128u8; stride * 2 + 8];
        let filter = [0i8; 16];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let ll_mask = vec![[0u16; 4]; (h >> 2) + 1];
        ns_wiener_single_y_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            4,
            &lpf_bottom,
            4,
            w,
            h,
            &filter,
            edges,
            &ll_mask,
        );
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    p[p_off + y * stride + x],
                    128,
                    "pixel ({x},{y}) changed from 128"
                );
            }
        }
    }

    #[test]
    fn test_ns_wiener_single_y_skip_mask() {
        let stride = 16;
        let h = 8;
        let w = 8;
        let p_off = stride + 4;
        let mut p = vec![100u8; stride * (h + 8)];
        p[p_off + 3 * stride + 3] = 80;
        let orig = p.clone();
        let left = vec![[100u8; 6]; h + 4];
        let lpf = vec![100u8; stride * 4 + 8];
        let lpf_bottom = vec![100u8; stride * 2 + 8];
        let filter = [4i8; 16];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let ll_mask = vec![[0xFFFFu16; 4]; (h >> 2) + 1];
        ns_wiener_single_y_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            4,
            &lpf_bottom,
            4,
            w,
            h,
            &filter,
            edges,
            &ll_mask,
        );
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    p[p_off + y * stride + x],
                    orig[p_off + y * stride + x],
                    "pixel ({x},{y}) changed despite full skip mask"
                );
            }
        }
    }

    #[test]
    fn test_ns_wiener_single_y_filters_noise() {
        let stride = 16;
        let h = 8;
        let w = 8;
        let p_off = stride + 4;
        let mut p = vec![128u8; stride * (h + 8)];
        p[p_off + 4 * stride + 4] = 110;
        let left = vec![[128u8; 6]; h + 4];
        let lpf = vec![128u8; stride * 4 + 8];
        let lpf_bottom = vec![128u8; stride * 2 + 8];
        let filter = [8i8, 6, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let ll_mask = vec![[0u16; 4]; (h >> 2) + 1];
        ns_wiener_single_y_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            4,
            &lpf_bottom,
            4,
            w,
            h,
            &filter,
            edges,
            &ll_mask,
        );
        let filtered = p[p_off + 4 * stride + 4];
        assert!(
            filtered > 110 && filtered < 128,
            "noisy pixel should move toward neighbors: got {filtered}"
        );
    }

    #[test]
    fn test_wiener_ns_config_y_table() {
        assert_eq!(WIENER_NS_CONFIG_Y.len(), 16);
        assert_eq!(WIENER_NS_CONFIG_Y[0], [1, 0]);
        assert_eq!(WIENER_NS_CONFIG_Y[15], [3, -3]);
    }

    #[test]
    fn test_wiener_ns_config_uv() {
        assert_eq!(WIENER_NS_CONFIG_UV.len(), 6);
        assert_eq!(WIENER_NS_CONFIG_UV_FROM_Y.len(), 12);
    }

    #[test]
    fn test_pc_wiener_config() {
        assert_eq!(PC_WIENER_CONFIG.len(), 12);
        assert_eq!(PC_WIENER_CONFIG[0], [1, 0]);
        assert_eq!(PC_WIENER_CONFIG[11], [0, 3]);
    }

    fn make_wiener_multi_test_data(
        _w: usize,
        h: usize,
        stride: usize,
    ) -> (
        Vec<u8>,
        usize,
        Vec<[u8; 6]>,
        Vec<u8>,
        usize,
        Vec<u8>,
        usize,
        Vec<[u16; 4]>,
        Vec<u16>,
    ) {
        let p_off = stride + 4;
        let p = vec![128u8; stride * (h + 8)];
        let left = vec![[128u8; 6]; h + 8];
        let lpf = vec![128u8; stride * 6 + 8];
        let lpf_off = 4usize;
        let lpf_bottom = vec![128u8; stride * 4 + 8];
        let lpf_bottom_off = 4usize;
        let ll_mask = vec![[0u16; 4]; (h >> 2) + 1];
        let noskip_mask = vec![0xFFFFu16; (h >> 2) + 1];
        (
            p,
            p_off,
            left,
            lpf,
            lpf_off,
            lpf_bottom,
            lpf_bottom_off,
            ll_mask,
            noskip_mask,
        )
    }

    #[test]
    fn test_wiener_multi_user_identity() {
        let w = 8;
        let h = 8;
        let stride = 16;
        let (mut p, p_off, left, lpf, lpf_off, lpf_bottom, lpf_bottom_off, ll_mask, noskip_mask) =
            make_wiener_multi_test_data(w, h, stride);
        let subclass_lut = vec![0u8; 256];
        let filters_user = [[0i8; 18]; 1];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;

        ns_wiener_multi_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            lpf_off,
            &lpf_bottom,
            lpf_bottom_off,
            w,
            h,
            &filters_user,
            &subclass_lut,
            &noskip_mask,
            32,
            edges,
            &ll_mask,
        );

        for y in 0..h {
            for x in 0..w {
                assert_eq!(p[p_off + y * stride + x], 128, "mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_wiener_multi_pretrained_identity() {
        let w = 8;
        let h = 8;
        let stride = 16;
        let (mut p, p_off, left, lpf, lpf_off, lpf_bottom, lpf_bottom_off, ll_mask, noskip_mask) =
            make_wiener_multi_test_data(w, h, stride);
        let subclass_lut = vec![0u8; 256];
        let mut filters_pt = [[0i16; 13]; 1];
        filters_pt[0][12] = 128;
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;

        pc_wiener_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            lpf_off,
            &lpf_bottom,
            lpf_bottom_off,
            w,
            h,
            &filters_pt,
            &subclass_lut,
            &noskip_mask,
            32,
            edges,
            &ll_mask,
        );

        for y in 0..h {
            for x in 0..w {
                assert_eq!(p[p_off + y * stride + x], 128, "mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_wiener_multi_skip_mask() {
        let w = 8;
        let h = 8;
        let stride = 16;
        let (mut p, p_off, left, lpf, lpf_off, lpf_bottom, lpf_bottom_off, _, noskip_mask) =
            make_wiener_multi_test_data(w, h, stride);
        for i in 0..p.len() {
            p[i] = 100;
        }
        let subclass_lut = vec![0u8; 256];
        let filters_user = [[127i8; 18]; 1];
        let ll_mask = vec![[0xFFFFu16; 4]; (h >> 2) + 1];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;

        ns_wiener_multi_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            lpf_off,
            &lpf_bottom,
            lpf_bottom_off,
            w,
            h,
            &filters_user,
            &subclass_lut,
            &noskip_mask,
            32,
            edges,
            &ll_mask,
        );

        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    p[p_off + y * stride + x],
                    100,
                    "skip mask should prevent changes"
                );
            }
        }
    }

    #[test]
    fn test_ns_wiener_single_uv_identity() {
        let w = 8;
        let h = 8;
        let stride = 16;
        let p_off = stride + 4;
        let mut p = vec![128u8; stride * (h + 6)];
        let left = vec![[128u8; 6]; h + 4];
        let lpf = vec![128u8; stride * 4 + 8];
        let lpf_bottom = vec![128u8; stride * 4 + 8];
        let lstride = 32usize;
        let luma_off = 4 * lstride + 8;
        let luma = vec![128u8; lstride * (h * 2 + 12)];
        let luma_top = vec![128u8; lstride * 4 + 8];
        let luma_bottom = vec![128u8; lstride * 4 + 8];
        let filter = [0i8; 18];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let ll_mask = vec![[0u16; 4]; (h >> 2) + 1];

        ns_wiener_single_uv_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            4,
            &lpf_bottom,
            4,
            w,
            h,
            &filter,
            &luma,
            luma_off,
            lstride,
            &luma_top,
            8,
            &luma_bottom,
            8,
            1,
            1,
            2,
            edges,
            &ll_mask,
        );

        for y in 0..h {
            for x in 0..w {
                assert_eq!(p[p_off + y * stride + x], 128, "mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_ns_wiener_single_uv_no_subsample() {
        let w = 8;
        let h = 8;
        let stride = 16;
        let p_off = stride + 4;
        let mut p = vec![128u8; stride * (h + 6)];
        let left = vec![[128u8; 6]; h + 4];
        let lpf = vec![128u8; stride * 4 + 8];
        let lpf_bottom = vec![128u8; stride * 4 + 8];
        let lstride = 16usize;
        let luma_off = 4 * lstride + 4;
        let luma = vec![128u8; lstride * (h + 12)];
        let luma_top = vec![128u8; lstride * 4 + 8];
        let luma_bottom = vec![128u8; lstride * 4 + 8];
        let filter = [0i8; 18];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let ll_mask = vec![[0u16; 4]; (h >> 2) + 1];

        ns_wiener_single_uv_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            4,
            &lpf_bottom,
            4,
            w,
            h,
            &filter,
            &luma,
            luma_off,
            lstride,
            &luma_top,
            4,
            &luma_bottom,
            4,
            0,
            0,
            2,
            edges,
            &ll_mask,
        );

        for y in 0..h {
            for x in 0..w {
                assert_eq!(p[p_off + y * stride + x], 128, "mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_ns_wiener_single_uv_skip_mask() {
        let w = 8;
        let h = 8;
        let stride = 16;
        let p_off = stride + 4;
        let mut p = vec![100u8; stride * (h + 6)];
        let left = vec![[100u8; 6]; h + 4];
        let lpf = vec![100u8; stride * 4 + 8];
        let lpf_bottom = vec![100u8; stride * 4 + 8];
        let lstride = 16usize;
        let luma_off = 4 * lstride + 4;
        let luma = vec![100u8; lstride * (h + 12)];
        let luma_top = vec![100u8; lstride * 4 + 8];
        let luma_bottom = vec![100u8; lstride * 4 + 8];
        let filter = [127i8; 18];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let ll_mask = vec![[0xFFFFu16; 4]; (h >> 2) + 1];

        ns_wiener_single_uv_8bpc(
            &mut p,
            p_off,
            stride,
            &left,
            &lpf,
            4,
            &lpf_bottom,
            4,
            w,
            h,
            &filter,
            &luma,
            luma_off,
            lstride,
            &luma_top,
            4,
            &luma_bottom,
            4,
            0,
            0,
            2,
            edges,
            &ll_mask,
        );

        for y in 0..h {
            for x in 0..w {
                assert_eq!(p[p_off + y * stride + x], 100, "skip mask failed");
            }
        }
    }

    #[test]
    fn test_gdf_coords() {
        assert_eq!(GDF_COORDS.len(), 18);
        assert_eq!(GDF_COORDS[0], [6, 0]);
        assert_eq!(GDF_COORDS[17], [0, 1]);
    }

    #[test]
    fn test_gdf_prep_uniform() {
        let w = 8;
        let h = 8;
        let stride = 24;
        let p_off = 6 * stride + 6;
        let p = vec![128u8; stride * (h + 14)];
        let left = vec![[128u8; 6]; h + 8];
        let lpf = vec![128u8; stride * 8 + 8];
        let lpf_bottom = vec![128u8; stride * 4 + 8];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let mut dst = vec![0i8; h * w];

        gdf_prep_8bpc(
            &mut dst,
            w,
            &p,
            p_off,
            stride,
            &left,
            &lpf,
            6,
            &lpf_bottom,
            6,
            w,
            h,
            0,
            0,
            edges,
        );

        for y in 0..h {
            for x in 0..w {
                let _ = dst[y * w + x];
            }
        }
    }

    #[test]
    fn test_gdf_prep_inter() {
        let w = 8;
        let h = 8;
        let stride = 24;
        let p_off = 6 * stride + 6;
        let p = vec![128u8; stride * (h + 14)];
        let left = vec![[128u8; 6]; h + 8];
        let lpf = vec![128u8; stride * 8 + 8];
        let lpf_bottom = vec![128u8; stride * 4 + 8];
        let edges = LR_HAVE_TOP | LR_HAVE_BOTTOM | LR_HAVE_LEFT | LR_HAVE_RIGHT;
        let mut dst = vec![0i8; h * w];

        gdf_prep_8bpc(
            &mut dst,
            w,
            &p,
            p_off,
            stride,
            &left,
            &lpf,
            6,
            &lpf_bottom,
            6,
            w,
            h,
            1,
            0,
            edges,
        );

        for y in 0..h {
            for x in 0..w {
                let _ = dst[y * w + x];
            }
        }
    }

    #[test]
    fn test_backup_nxu_3_pixels() {
        let src: Vec<u8> = (0..32).collect();
        let stride: isize = 8;
        let mut dst = [[0u8; 6]; 4];
        backup_nxu(&mut dst, &src, 5, stride, 4, 3);
        assert_eq!(dst[0][3..6], src[2..5]);
        assert_eq!(dst[1][3..6], src[10..13]);
        assert_eq!(dst[2][3..6], src[18..21]);
        assert_eq!(dst[3][3..6], src[26..29]);
    }

    #[test]
    fn test_backup_nxu_6_pixels() {
        let src: Vec<u8> = (0..24).collect();
        let stride: isize = 8;
        let mut dst = [[0u8; 6]; 2];
        backup_nxu(&mut dst, &src, 6, stride, 2, 6);
        assert_eq!(dst[0], [0, 1, 2, 3, 4, 5]);
        assert_eq!(dst[1], [8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn test_copy_n_lines() {
        let src: Vec<u8> = (0..30).collect();
        let mut dst = vec![0u8; 30];
        copy_n_lines(&mut dst, &src, 10, 3);
        assert_eq!(&dst[..30], &src[..30]);
    }

    #[test]
    fn test_copy_n_lines_single_row() {
        let src = [1u8, 2, 3, 4, 5];
        let mut dst = [0u8; 5];
        copy_n_lines(&mut dst, &src, 5, 1);
        assert_eq!(dst, [1, 2, 3, 4, 5]);
    }
}

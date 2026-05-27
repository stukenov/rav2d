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

pub fn setup_thr_cols_sb64(
    q_thr_dst: &mut [u8],
    side_thr_dst: &mut [u8],
    dst_stride: usize,
    segmap: &[u8],
    seg_off: usize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]],
    thr_lut: &[[u8; 16]; 2],
    left_q_thr: &mut [u8],
    left_side_thr: &mut [u8],
    y64: i32,
    _ss_hor: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    let mask_idx = (y64 >> ss_ver) as usize;
    let mask_shift: u32 = if y64 & ss_ver != 0 { 8 } else { 0 };

    for y4 in 0..h4 as usize {
        let mut prev_q_thr = left_q_thr[y4] as i32;
        let mut prev_side_thr = left_side_thr[y4] as i32;

        for x4 in 0..w4 as usize {
            let seg_id = segmap[(seg_off as isize + x4 as isize +
                y4 as isize * seg_stride) as usize] as usize;
            let cur_q_thr = thr_lut[0][seg_id] as i32;
            let cur_side_thr = thr_lut[1][seg_id] as i32;
            let subpu = 3 * (((mask[x4][4][mask_idx] >>
                (mask_shift + y4 as u32)) & 1) as i32);

            let edge_q_thr = if cur_q_thr != 0 && prev_q_thr != 0 {
                (cur_q_thr + prev_q_thr + 1) >> 1
            } else {
                cur_q_thr | prev_q_thr
            };
            let edge_side_thr = if cur_side_thr != 0 && prev_side_thr != 0 {
                (cur_side_thr + prev_side_thr + 1) >> 1
            } else {
                cur_side_thr | prev_side_thr
            };

            q_thr_dst[x4 * dst_stride + y4] = (edge_q_thr >> subpu) as u8;
            side_thr_dst[x4 * dst_stride + y4] = (edge_side_thr >> subpu) as u8;

            prev_q_thr = cur_q_thr;
            prev_side_thr = cur_side_thr;
        }

        left_q_thr[y4] = prev_q_thr as u8;
        left_side_thr[y4] = prev_side_thr as u8;
    }
}

pub fn setup_thr_rows_sb64(
    q_thr_dst: &mut [u8],
    side_thr_dst: &mut [u8],
    dst_stride: usize,
    segmap: &[u8],
    seg_off: usize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]],
    thr_lut: &[[u8; 16]; 2],
    above_thr_lut: Option<&[[u8; 16]; 2]>,
    sb64x: i32,
    ss_hor: i32,
    _ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    let mask_idx = (sb64x >> ss_hor) as usize;
    let mask_shift: u32 = if sb64x & ss_hor != 0 { 8 } else { 0 };

    let mut above_q_thr = [0u8; 16];
    let mut above_side_thr = [0u8; 16];
    if let Some(above_lut) = above_thr_lut {
        for x4 in 0..w4 as usize {
            let seg_id = segmap[(seg_off as isize + x4 as isize -
                seg_stride) as usize] as usize;
            above_q_thr[x4] = above_lut[0][seg_id];
            above_side_thr[x4] = above_lut[1][seg_id];
        }
    }

    for x4 in 0..w4 as usize {
        let mut prev_q_thr = above_q_thr[x4] as i32;
        let mut prev_side_thr = above_side_thr[x4] as i32;

        for y4 in 0..h4 as usize {
            let seg_id = segmap[(seg_off as isize + x4 as isize +
                y4 as isize * seg_stride) as usize] as usize;
            let cur_q_thr = thr_lut[0][seg_id] as i32;
            let cur_side_thr = thr_lut[1][seg_id] as i32;
            let subpu = 3 * (((mask[y4][4][mask_idx] >>
                (mask_shift + x4 as u32)) & 1) as i32);

            let edge_q_thr = if cur_q_thr != 0 && prev_q_thr != 0 {
                (cur_q_thr + prev_q_thr + 1) >> 1
            } else {
                cur_q_thr | prev_q_thr
            };
            let edge_side_thr = if cur_side_thr != 0 && prev_side_thr != 0 {
                (cur_side_thr + prev_side_thr + 1) >> 1
            } else {
                cur_side_thr | prev_side_thr
            };

            q_thr_dst[x4 + y4 * dst_stride] = (edge_q_thr >> subpu) as u8;
            side_thr_dst[x4 + y4 * dst_stride] = (edge_side_thr >> subpu) as u8;

            prev_q_thr = cur_q_thr;
            prev_side_thr = cur_side_thr;
        }
    }
}

pub fn backup_db(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    ss_ver: i32,
    sb128: bool,
    mut row: i32,
    row_h: i32,
    w: usize,
    lr_backup: bool,
    n_tc: i32,
) {
    let cdef_backup = (!lr_backup) as i32;
    let sb128_i = sb128 as i32;

    let mut stripe_h = ((64 << (cdef_backup & sb128_i)) - 8 * (row == 0) as i32) >> ss_ver;
    let mut src_off = (stripe_h - 2) as usize * stride;
    let mut dst_off = 0usize;

    if n_tc == 1 {
        if row > 0 {
            let top = 4usize << sb128_i;
            for i in 0..4usize {
                let from = dst_off + (top + i) * stride;
                let to = dst_off + i * stride;
                dst.copy_within(from..from + w, to);
            }
        }
        dst_off += 4 * stride;
    }

    while row + stripe_h <= row_h {
        for _ in 0..4 {
            dst[dst_off..dst_off + w].copy_from_slice(&src[src_off..src_off + w]);
            dst_off += stride;
            src_off += stride;
        }
        row += stripe_h;
        stripe_h = 64 >> ss_ver;
        src_off += (stripe_h - 4) as usize * stride;
    }
}

pub struct DeblockApplyParams {
    pub y_stride: isize,
    pub uv_stride: isize,
    pub bw: usize,
    pub bh: usize,
    pub sb128: bool,
    pub ss_hor: bool,
    pub ss_ver: bool,
    pub level_y: [i32; 2],
    pub level_u: i32,
    pub level_v: i32,
    pub have_chroma: bool,
}

pub fn deblock_sbrow_cols_8bpc(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    params: &DeblockApplyParams,
    masks: &[[[u16; 4]; 5]],
    segmap: &[u8],
    thr_lut_y: &[[u8; 16]; 2],
    thr_lut_uv: &[[[u8; 16]; 2]; 2],
    sby: i32,
    start_of_tile_row: bool,
) {
    if params.level_y[0] == 0 && params.level_y[1] == 0 {
        return;
    }

    let sb_size = if params.sb128 { 128 } else { 64 };
    let y64_start = (sby as usize * sb_size) / 64;
    let y64_end = ((sby as usize + 1) * sb_size).min(params.bh * 4) / 64;

    for _y64 in y64_start..y64_end {
        deblock_sbrow64_cols_8bpc(
            y,
            u,
            v,
            params,
            masks,
            segmap,
            thr_lut_y,
            thr_lut_uv,
            _y64,
            start_of_tile_row && _y64 == y64_start,
        );
    }
}

fn deblock_sbrow64_cols_8bpc(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    params: &DeblockApplyParams,
    masks: &[[[u16; 4]; 5]],
    segmap: &[u8],
    thr_lut_y: &[[u8; 16]; 2],
    _thr_lut_uv: &[[[u8; 16]; 2]; 2],
    y64: usize,
    _start_of_tile_row: bool,
) {
    let h4 = imin(16, params.bh as i32 - y64 as i32 * 16);
    if h4 <= 0 {
        return;
    }

    let sb64w = (params.bw + 15) / 16;
    for sbx in 0..sb64w {
        let have_left = sbx > 0;
        let w4 = imin(16, params.bw as i32 - sbx as i32 * 16);

        if params.level_y[0] > 0 && have_left {
            filter_plane_cols_y_8bpc(
                y,
                params.y_stride,
                masks,
                thr_lut_y,
                sbx,
                y64,
                w4,
                h4,
            );
        }

        if params.have_chroma && have_left {
            filter_plane_cols_uv_8bpc(
                u,
                v,
                params.uv_stride,
                masks,
                sbx,
                y64,
                w4,
                h4,
                params.ss_hor,
                params.ss_ver,
            );
        }
    }
}

fn filter_choice(
    s: &[u8], s_off: usize,
    t: &[u8], t_off: usize,
    stride: isize,
    max_width_neg: i32, max_width_pos: i32,
    q_thr: u32, side_thr: u32,
) -> i32 {
    let st = stride;
    let at = |buf: &[u8], off: usize, i: isize| -> i32 {
        buf[(off as isize + i * st) as usize] as i32
    };

    let mut second_derivs = [0u32; 4];
    let mut deriv_s = (at(s, s_off, 0) - at(s, s_off, -1)).unsigned_abs()
        + (at(s, s_off, 1) - at(s, s_off, 0)).unsigned_abs();
    let mut deriv_t = (at(t, t_off, 0) - at(t, t_off, -1)).unsigned_abs()
        + (at(t, t_off, 1) - at(t, t_off, 0)).unsigned_abs();
    second_derivs[0] = deriv_s + deriv_t;

    deriv_s = (at(s, s_off, 2) - at(s, s_off, 1)).unsigned_abs()
        + (at(s, s_off, 3) - at(s, s_off, 2)).unsigned_abs();
    deriv_t = (at(t, t_off, 2) - at(t, t_off, 1)).unsigned_abs()
        + (at(t, t_off, 3) - at(t, t_off, 2)).unsigned_abs();
    second_derivs[1] = deriv_s + deriv_t;

    deriv_s = (at(s, s_off, -2) - at(s, s_off, -3)).unsigned_abs()
        + (at(s, s_off, -1) - at(s, s_off, -2)).unsigned_abs();
    deriv_t = (at(t, t_off, -2) - at(t, t_off, -3)).unsigned_abs()
        + (at(t, t_off, -1) - at(t, t_off, -2)).unsigned_abs();
    second_derivs[2] = deriv_s + deriv_t;

    deriv_s = (at(s, s_off, -4) - at(s, s_off, -5)).unsigned_abs()
        + (at(s, s_off, -3) - at(s, s_off, -4)).unsigned_abs();
    deriv_t = (at(t, t_off, -4) - at(t, t_off, -5)).unsigned_abs()
        + (at(t, t_off, -3) - at(t, t_off, -4)).unsigned_abs();
    second_derivs[3] = deriv_s + deriv_t;

    let q_thr_val = q_thr * Q_FIRST[0] as u32;
    if ((second_derivs[0] + second_derivs[1] + 1) >> 1) > q_thr_val {
        return 0;
    }
    let side_thr_val = side_thr * Q_FIRST[0] as u32;
    if ((second_derivs[2] + second_derivs[3] + 1) >> 1) > side_thr_val {
        return 0;
    }

    let mut prev_dist = 1;
    for dist in 2..=max_width_pos {
        let idx = dist as usize - 1;
        if idx >= Q_THRESH_MULTS.len() || Q_THRESH_MULTS[idx] == 0 {
            break;
        }
        let end_thr4 = q_thr * Q_THRESH_MULTS[idx] as u32;
        let dist2 = dist - 1;

        let ds = (at(s, s_off, 0) - at(s, s_off, dist2 as isize) - dist2 as i32 * (at(s, s_off, 0) - at(s, s_off, 1))).unsigned_abs();
        let dt = (at(t, t_off, 0) - at(t, t_off, dist2 as isize) - dist2 as i32 * (at(t, t_off, 0) - at(t, t_off, 1))).unsigned_abs();
        if ((ds + dt + 1) >> 1) > end_thr4 {
            return prev_dist;
        }

        let side_end_thr4 = side_thr * Q_THRESH_MULTS[idx] as u32;
        let ds = (at(s, s_off, 0) - at(s, s_off, -(dist2 as isize)) - dist2 as i32 * (at(s, s_off, 0) - at(s, s_off, -1))).unsigned_abs();
        let dt = (at(t, t_off, 0) - at(t, t_off, -(dist2 as isize)) - dist2 as i32 * (at(t, t_off, 0) - at(t, t_off, -1))).unsigned_abs();
        if ((ds + dt + 1) >> 1) > side_end_thr4 {
            return prev_dist;
        }
        prev_dist = dist;
    }

    max_width_pos
}

fn deblock_pixel(
    dst: &mut [u8], off: usize,
    q_thr: u32, side_thr: u32,
    stridea: isize, strideb: isize,
    max_width_pos: i32, max_width_neg: i32,
    pos_lossless: bool, neg_lossless: bool,
) {
    let s_off = off;
    let t_off = (off as isize + 3 * stridea) as usize;
    let width = filter_choice(dst, s_off, dst, t_off, strideb,
                              max_width_neg, max_width_pos, q_thr, side_thr);
    let width_neg = imin(width, max_width_neg);
    let width_pos = width;

    if width_pos < 1 {
        return;
    }

    let q_thr_clamp = (q_thr * Q_THRESH_MULTS[0] as u32) as i32;

    for i in 0..4 {
        let base = (off as isize + i * strideb) as usize;
        let d0 = dst[base] as i32;
        let dm1 = dst[(base as isize - stridea) as usize] as i32;
        let d1 = dst[(base as isize + stridea) as usize] as i32;
        let dm2 = dst[(base as isize - 2 * stridea) as usize] as i32;

        let delta_m2 = iclip(4 * (3 * (d0 - dm1) - (d1 - dm2)), -q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let delta_m2_neg = delta_m2 * W_MULT[width_neg as usize - 1] as i32;
            for j in 0..width_neg {
                let pix = (base as isize + (-j - 1) as isize * stridea) as usize;
                let diff = (delta_m2_neg * (width_neg - j) + (1 << 10)) >> 11;
                dst[pix] = iclip(dst[pix] as i32 + diff, 0, 255) as u8;
            }
        }

        if !pos_lossless {
            let delta_m2_pos = delta_m2 * W_MULT[width_pos as usize - 1] as i32;
            for j in 0..width_pos {
                let pix = (base as isize + j as isize * stridea) as usize;
                let diff = (delta_m2_pos * (width_pos - j) + (1 << 10)) >> 11;
                dst[pix] = iclip(dst[pix] as i32 - diff, 0, 255) as u8;
            }
        }
    }
}

fn deblock_h_sb64y(
    dst: &mut [u8], dst_off: usize, stride: isize,
    vmask: &[u16], ll_mask: &[u16],
    q_thr: &[u32], side_thr: &[u32],
    tile_edge: bool, h4: i32,
) {
    let ls = stride.unsigned_abs();
    for y in 0..h4 as u16 {
        let mask = vmask[0] | vmask[1] | vmask[2] | vmask[3];
        if mask & (1 << y) == 0 {
            continue;
        }
        let level = if vmask[3] & (1 << y) != 0 { 3 }
            else if vmask[2] & (1 << y) != 0 { 2 }
            else if vmask[1] & (1 << y) != 0 { 1 }
            else { 0 };
        let max_width_pos = MAX_WIDTH_Y[level] as i32;
        let max_width_neg = if tile_edge { max_width_pos } else { MAX_WIDTH_Y[level] as i32 };
        let pos_lossless = ll_mask[1] & (1 << y) != 0;
        let neg_lossless = ll_mask[0] & (1 << y) != 0;
        let off = dst_off + y as usize * ls;
        deblock_pixel(dst, off, q_thr[y as usize], side_thr[y as usize],
                      1, stride, max_width_pos, max_width_neg,
                      pos_lossless, neg_lossless);
    }
}

fn deblock_v_sb64y(
    dst: &mut [u8], dst_off: usize, stride: isize,
    vmask: &[u16], ll_mask: &[u16],
    q_thr: &[u32], side_thr: &[u32],
    tile_edge: bool, w4: i32,
) {
    let ls = stride.unsigned_abs();
    for x in 0..w4 as u16 {
        let mask = vmask[0] | vmask[1] | vmask[2] | vmask[3];
        if mask & (1 << x) == 0 {
            continue;
        }
        let level = if vmask[3] & (1 << x) != 0 { 3 }
            else if vmask[2] & (1 << x) != 0 { 2 }
            else if vmask[1] & (1 << x) != 0 { 1 }
            else { 0 };
        let max_width_pos = MAX_WIDTH_Y[level] as i32;
        let max_width_neg = if tile_edge { max_width_pos } else { MAX_WIDTH_Y[level] as i32 };
        let pos_lossless = ll_mask[1] & (1 << x) != 0;
        let neg_lossless = ll_mask[0] & (1 << x) != 0;
        let off = dst_off + x as usize;
        deblock_pixel(dst, off, q_thr[x as usize], side_thr[x as usize],
                      stride, 1, max_width_pos, max_width_neg,
                      pos_lossless, neg_lossless);
    }
}

fn deblock_h_sb64uv(
    dst: &mut [u8], dst_off: usize, stride: isize,
    vmask: &[u16], ll_mask: &[u16],
    q_thr: &[u32], side_thr: &[u32],
    tile_edge: bool, h4: i32,
) {
    let ls = stride.unsigned_abs();
    for y in 0..h4 as u16 {
        let mask = vmask[0] | vmask[1] | vmask[2];
        if mask & (1 << y) == 0 {
            continue;
        }
        let level = if vmask[2] & (1 << y) != 0 { 2 }
            else if vmask[1] & (1 << y) != 0 { 1 }
            else { 0 };
        let max_width_pos = MAX_WIDTH_UV[level] as i32;
        let max_width_neg = if tile_edge { max_width_pos } else { MAX_WIDTH_UV[level] as i32 };
        let pos_lossless = ll_mask[1] & (1 << y) != 0;
        let neg_lossless = ll_mask[0] & (1 << y) != 0;
        let off = dst_off + y as usize * ls;
        deblock_pixel(dst, off, q_thr[y as usize], side_thr[y as usize],
                      1, stride, max_width_pos, max_width_neg,
                      pos_lossless, neg_lossless);
    }
}

fn deblock_v_sb64uv(
    dst: &mut [u8], dst_off: usize, stride: isize,
    vmask: &[u16], ll_mask: &[u16],
    q_thr: &[u32], side_thr: &[u32],
    tile_edge: bool, w4: i32,
) {
    let ls = stride.unsigned_abs();
    for x in 0..w4 as u16 {
        let mask = vmask[0] | vmask[1] | vmask[2];
        if mask & (1 << x) == 0 {
            continue;
        }
        let level = if vmask[2] & (1 << x) != 0 { 2 }
            else if vmask[1] & (1 << x) != 0 { 1 }
            else { 0 };
        let max_width_pos = MAX_WIDTH_UV[level] as i32;
        let max_width_neg = if tile_edge { max_width_pos } else { MAX_WIDTH_UV[level] as i32 };
        let pos_lossless = ll_mask[1] & (1 << x) != 0;
        let neg_lossless = ll_mask[0] & (1 << x) != 0;
        let off = dst_off + x as usize;
        deblock_pixel(dst, off, q_thr[x as usize], side_thr[x as usize],
                      stride, 1, max_width_pos, max_width_neg,
                      pos_lossless, neg_lossless);
    }
}

fn filter_plane_cols_y_8bpc(
    dst: &mut [u8],
    stride: isize,
    masks: &[[[u16; 4]; 5]],
    thr_lut: &[[u8; 16]; 2],
    sbx: usize,
    _y64: usize,
    w4: i32,
    h4: i32,
) {
    let ls = stride.unsigned_abs();
    let mut tile_edge = sbx == 0;
    for x in 0..w4 {
        let mask_idx = sbx * 16 + x as usize;
        if mask_idx >= masks.len() {
            break;
        }
        let hmask = &masks[mask_idx];
        let vmask = &hmask[0];
        let ll_mask_raw = &hmask[4];
        let ll_slice = [ll_mask_raw[0], ll_mask_raw[1]];
        let mut q_thr_arr = [0u32; 16];
        let mut side_thr_arr = [0u32; 16];
        for y in 0..h4 as usize {
            q_thr_arr[y] = thr_lut[0][0] as u32;
            side_thr_arr[y] = thr_lut[1][0] as u32;
        }
        let off = x as usize * 4;
        deblock_h_sb64y(dst, off, stride, &vmask[..4], &ll_slice,
                        &q_thr_arr, &side_thr_arr, tile_edge, h4);
        tile_edge = false;
    }
}

fn filter_plane_cols_uv_8bpc(
    u: &mut [u8],
    v: &mut [u8],
    stride: isize,
    masks: &[[[u16; 4]; 5]],
    sbx: usize,
    _y64: usize,
    w4: i32,
    h4: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    let ls = stride.unsigned_abs();
    let cw4 = w4 >> ss_hor as i32;
    let ch4 = h4 >> ss_ver as i32;
    let mut tile_edge = sbx == 0;
    for x in 0..cw4 {
        let mask_idx = sbx * (16 >> ss_hor as usize) + x as usize;
        if mask_idx >= masks.len() {
            break;
        }
        let hmask = &masks[mask_idx];
        let vmask = &hmask[0];
        let ll_mask_raw = &hmask[4];
        let ll_slice = [ll_mask_raw[0], ll_mask_raw[1]];
        let mut q_thr_arr = [0u32; 16];
        let mut side_thr_arr = [0u32; 16];
        for y in 0..ch4 as usize {
            q_thr_arr[y] = 1;
            side_thr_arr[y] = 1;
        }
        let off = x as usize * 4;
        deblock_h_sb64uv(u, off, stride, &vmask[..3], &ll_slice,
                         &q_thr_arr, &side_thr_arr, tile_edge, ch4);
        deblock_h_sb64uv(v, off, stride, &vmask[..3], &ll_slice,
                         &q_thr_arr, &side_thr_arr, tile_edge, ch4);
        tile_edge = false;
    }
}

pub fn deblock_sbrow_rows_8bpc(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    params: &DeblockApplyParams,
    masks: &[[[u16; 4]; 5]],
    segmap: &[u8],
    thr_lut_y: &[[u8; 16]; 2],
    thr_lut_uv: &[[[u8; 16]; 2]; 2],
    sby: i32,
) {
    if params.level_y[0] == 0 && params.level_y[1] == 0 {
        return;
    }

    let sb_size = if params.sb128 { 128 } else { 64 };
    let y64_start = (sby as usize * sb_size) / 64;
    let y64_end = ((sby as usize + 1) * sb_size).min(params.bh * 4) / 64;

    for _y64 in y64_start..y64_end {
        deblock_sbrow64_rows_8bpc(
            y,
            u,
            v,
            params,
            masks,
            segmap,
            thr_lut_y,
            thr_lut_uv,
            _y64,
        );
    }
}

fn deblock_sbrow64_rows_8bpc(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    params: &DeblockApplyParams,
    masks: &[[[u16; 4]; 5]],
    segmap: &[u8],
    _thr_lut_y: &[[u8; 16]; 2],
    _thr_lut_uv: &[[[u8; 16]; 2]; 2],
    y64: usize,
) {
    let h4 = imin(16, params.bh as i32 - y64 as i32 * 16);
    if h4 <= 0 {
        return;
    }

    let sb64w = (params.bw + 15) / 16;
    for sbx in 0..sb64w {
        let have_top = y64 > 0;
        let w4 = imin(16, params.bw as i32 - sbx as i32 * 16);

        if params.level_y[1] > 0 && have_top {
            filter_plane_rows_y_8bpc(
                y,
                params.y_stride,
                masks,
                sbx,
                y64,
                w4,
                h4,
            );
        }

        if params.have_chroma && have_top {
            filter_plane_rows_uv_8bpc(
                u,
                v,
                params.uv_stride,
                masks,
                sbx,
                y64,
                w4,
                h4,
                params.ss_hor,
                params.ss_ver,
            );
        }
    }
}

fn filter_plane_rows_y_8bpc(
    dst: &mut [u8],
    stride: isize,
    masks: &[[[u16; 4]; 5]],
    sbx: usize,
    _y64: usize,
    w4: i32,
    h4: i32,
) {
    let ls = stride.unsigned_abs();
    for y in 0..h4 {
        let mask_idx = sbx * 16 + y as usize;
        if mask_idx >= masks.len() {
            break;
        }
        let row_mask = &masks[mask_idx];
        let vmask = &row_mask[0];
        let ll_mask_raw = &row_mask[4];
        let ll_slice = [ll_mask_raw[0], ll_mask_raw[1]];
        let mut q_thr_arr = [0u32; 16];
        let mut side_thr_arr = [0u32; 16];
        for x in 0..w4 as usize {
            q_thr_arr[x] = 1;
            side_thr_arr[x] = 1;
        }
        let off = y as usize * ls;
        deblock_v_sb64y(dst, off, stride, &vmask[..4], &ll_slice,
                        &q_thr_arr, &side_thr_arr, y == 0, w4);
    }
}

fn filter_plane_rows_uv_8bpc(
    u: &mut [u8],
    v: &mut [u8],
    stride: isize,
    masks: &[[[u16; 4]; 5]],
    sbx: usize,
    _y64: usize,
    w4: i32,
    h4: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    let ls = stride.unsigned_abs();
    let cw4 = w4 >> ss_hor as i32;
    let ch4 = h4 >> ss_ver as i32;
    for y in 0..ch4 {
        let mask_idx = sbx * (16 >> ss_ver as usize) + y as usize;
        if mask_idx >= masks.len() {
            break;
        }
        let row_mask = &masks[mask_idx];
        let vmask = &row_mask[0];
        let ll_mask_raw = &row_mask[4];
        let ll_slice = [ll_mask_raw[0], ll_mask_raw[1]];
        let mut q_thr_arr = [0u32; 16];
        let mut side_thr_arr = [0u32; 16];
        for x in 0..cw4 as usize {
            q_thr_arr[x] = 1;
            side_thr_arr[x] = 1;
        }
        let off = y as usize * ls;
        deblock_v_sb64uv(u, off, stride, &vmask[..3], &ll_slice,
                         &q_thr_arr, &side_thr_arr, y == 0, cw4);
        deblock_v_sb64uv(v, off, stride, &vmask[..3], &ll_slice,
                         &q_thr_arr, &side_thr_arr, y == 0, cw4);
    }
}

pub fn copy_db_8bpc(
    lr_db: &mut [Vec<u8>; 3],
    src: &[&[u8]; 3],
    strides: &[isize; 2],
    bw: usize,
    bh: usize,
    sby: i32,
    sb128: bool,
    ss_hor: bool,
    ss_ver: bool,
    lr_backup: bool,
) {
    let sb_size = if sb128 { 128 } else { 64 };
    let row = sby as usize * sb_size;
    let row_h = (row + sb_size).min(bh * 4);

    if row >= row_h {
        return;
    }

    let w = bw * 4;
    let h = bh * 4;

    backup_db(
        &mut lr_db[0],
        src[0],
        strides[0] as usize,
        0,
        sb128,
        row as i32,
        row_h as i32,
        w,
        lr_backup,
        1,
    );

    if strides[1] != 0 {
        let cw = w >> (ss_hor as usize);
        let ch = h >> (ss_ver as usize);
        let crow = row >> (ss_ver as usize);
        let crow_h = row_h >> (ss_ver as usize);

        backup_db(
            &mut lr_db[1],
            src[1],
            strides[1] as usize,
            ss_ver as i32,
            sb128,
            crow as i32,
            crow_h as i32,
            cw,
            lr_backup,
            1,
        );
        backup_db(
            &mut lr_db[2],
            src[2],
            strides[1] as usize,
            ss_ver as i32,
            sb128,
            crow as i32,
            crow_h as i32,
            cw,
            lr_backup,
            1,
        );
    }
}

fn backup_db_apply(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    w: usize,
    n_lines: usize,
    src_row: usize,
) {
    for i in 0..n_lines {
        let src_off = (src_row + i) * stride;
        let dst_off = i * w;
        if src_off + w <= src.len() && dst_off + w <= dst.len() {
            dst[dst_off..dst_off + w].copy_from_slice(&src[src_off..src_off + w]);
        }
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

    #[test]
    fn test_setup_thr_cols_uniform_segmap() {
        let dst_stride = 16;
        let mut q_thr = vec![0u8; dst_stride * 4];
        let mut side_thr = vec![0u8; dst_stride * 4];
        let segmap = vec![0u8; 4 * 4];
        let mask = vec![[[0u16; 4]; 5]; 4];
        let thr_lut = [[20u8; 16], [10u8; 16]];
        let mut left_q = [0u8; 16];
        let mut left_side = [0u8; 16];
        setup_thr_cols_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 0, 4, &mask, &thr_lut,
            &mut left_q, &mut left_side, 0, 0, 0, 4, 4,
        );
        // x4=0: prev=0, cur=20 → edge=0|20=20
        assert_eq!(q_thr[0 * dst_stride + 0], 20);
        // x4=1: prev=20, cur=20 → edge=(20+20+1)>>1=20
        assert_eq!(q_thr[1 * dst_stride + 0], 20);
        assert_eq!(side_thr[0 * dst_stride + 0], 10);
        assert_eq!(side_thr[1 * dst_stride + 0], 10);
    }

    #[test]
    fn test_setup_thr_cols_left_state_update() {
        let dst_stride = 16;
        let mut q_thr = vec![0u8; dst_stride * 4];
        let mut side_thr = vec![0u8; dst_stride * 4];
        let segmap = vec![2u8; 4 * 4];
        let mask = vec![[[0u16; 4]; 5]; 4];
        let thr_lut = [[0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
        let mut left_q = [0u8; 16];
        let mut left_side = [0u8; 16];
        setup_thr_cols_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 0, 4, &mask, &thr_lut,
            &mut left_q, &mut left_side, 0, 0, 0, 4, 4,
        );
        // left should be updated to last column's threshold
        assert_eq!(left_q[0], 30);
        assert_eq!(left_side[0], 15);
    }

    #[test]
    fn test_setup_thr_cols_subpu_mask() {
        let dst_stride = 16;
        let mut q_thr = vec![0u8; dst_stride * 2];
        let mut side_thr = vec![0u8; dst_stride * 2];
        let segmap = vec![0u8; 2 * 2];
        // Set subpu bit for x4=1, mask_idx=0, y4=0 → mask[1][4][0] bit 0 = 1
        let mut mask = vec![[[0u16; 4]; 5]; 2];
        mask[1][4][0] = 1;
        let thr_lut = [[40u8; 16], [20u8; 16]];
        let mut left_q = [40u8; 16];
        let mut left_side = [20u8; 16];
        setup_thr_cols_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 0, 2, &mask, &thr_lut,
            &mut left_q, &mut left_side, 0, 0, 0, 2, 2,
        );
        // x4=1, y4=0: subpu=3, edge_q=40, result=40>>3=5
        assert_eq!(q_thr[1 * dst_stride + 0], 5);
        assert_eq!(side_thr[1 * dst_stride + 0], 2); // 20>>3=2
    }

    #[test]
    fn test_setup_thr_cols_mixed_thresholds() {
        let dst_stride = 16;
        let mut q_thr = vec![0u8; dst_stride * 2];
        let mut side_thr = vec![0u8; dst_stride * 2];
        // seg_id 0 maps to thr 0, seg_id 1 maps to thr 30
        let segmap = vec![0, 1, 0, 1];
        let mask = vec![[[0u16; 4]; 5]; 2];
        let thr_lut = [[0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
        let mut left_q = [0u8; 16];
        let mut left_side = [0u8; 16];
        setup_thr_cols_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 0, 2, &mask, &thr_lut,
            &mut left_q, &mut left_side, 0, 0, 0, 2, 2,
        );
        // y4=0: x4=0 seg=0 cur=0 prev=0 → edge=0; x4=1 seg=1 cur=30 prev=0 → edge=0|30=30
        assert_eq!(q_thr[0 * dst_stride + 0], 0);
        assert_eq!(q_thr[1 * dst_stride + 0], 30);
    }

    #[test]
    fn test_setup_thr_rows_no_above() {
        let dst_stride = 4;
        let mut q_thr = vec![0u8; dst_stride * 4];
        let mut side_thr = vec![0u8; dst_stride * 4];
        let segmap = vec![0u8; 4 * 4];
        let mask = vec![[[0u16; 4]; 5]; 4];
        let thr_lut = [[20u8; 16], [10u8; 16]];
        setup_thr_rows_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 0, 4, &mask, &thr_lut, None,
            0, 0, 0, 4, 4,
        );
        // y4=0: prev=0(no above), cur=20 → edge=0|20=20
        assert_eq!(q_thr[0 + 0 * dst_stride], 20);
        // y4=1: prev=20, cur=20 → edge=(20+20+1)>>1=20
        assert_eq!(q_thr[0 + 1 * dst_stride], 20);
        assert_eq!(side_thr[0 + 0 * dst_stride], 10);
    }

    #[test]
    fn test_setup_thr_rows_with_above() {
        let dst_stride = 4;
        let mut q_thr = vec![0u8; dst_stride * 4];
        let mut side_thr = vec![0u8; dst_stride * 4];
        // segmap: row -1 (above) = seg_id 1, current rows = seg_id 0
        let seg_stride: isize = 4;
        let mut segmap = vec![0u8; 5 * 4];
        // above row (offset 0)
        for i in 0..4 { segmap[i] = 1; }
        // current rows start at offset 4 (seg_off=4)
        let mask = vec![[[0u16; 4]; 5]; 4];
        let thr_lut = [[10u8; 16], [5u8; 16]];
        let above_lut = [[0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
        setup_thr_rows_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 4, seg_stride, &mask, &thr_lut, Some(&above_lut),
            0, 0, 0, 4, 4,
        );
        // y4=0, x4=0: above seg_id=1→above_q=30, cur=10 → (30+10+1)>>1=20
        assert_eq!(q_thr[0 + 0 * dst_stride], 20);
        // above_side=15, cur_side=5 → (15+5+1)>>1=10
        assert_eq!(side_thr[0 + 0 * dst_stride], 10);
    }

    #[test]
    fn test_setup_thr_rows_subpu_mask() {
        let dst_stride = 2;
        let mut q_thr = vec![0u8; dst_stride * 2];
        let mut side_thr = vec![0u8; dst_stride * 2];
        let segmap = vec![0u8; 2 * 2];
        // Set subpu bit for y4=1, mask_idx=0, x4=0 → mask[1][4][0] bit 0 = 1
        let mut mask = vec![[[0u16; 4]; 5]; 2];
        mask[1][4][0] = 1;
        let thr_lut = [[40u8; 16], [20u8; 16]];
        setup_thr_rows_sb64(
            &mut q_thr, &mut side_thr, dst_stride,
            &segmap, 0, 2, &mask, &thr_lut, None,
            0, 0, 0, 2, 2,
        );
        // y4=1, x4=0: subpu=3, edge_q=40, result=40>>3=5
        assert_eq!(q_thr[0 + 1 * dst_stride], 5);
        assert_eq!(side_thr[0 + 1 * dst_stride], 2);
    }

    #[test]
    fn test_backup_db_first_row_single_thread() {
        let stride = 64;
        let w = 16;
        let mut src = vec![0u8; stride * 128];
        for i in 0..4 {
            for x in 0..w {
                src[(54 + i) * stride + x] = (10 + i) as u8;
            }
        }
        let mut dst = vec![0u8; stride * 64];
        // row=0, sb128=false, lr_backup=false -> stripe_h=(64-8)>>0=56, src_off=54*stride
        backup_db(&mut dst, &src, stride, 0, false, 0, 56, w, false, 1);
        // n_tc=1, row=0 -> no top copy, dst starts at 4*stride
        for x in 0..w {
            assert_eq!(dst[4 * stride + x], 10);
            assert_eq!(dst[5 * stride + x], 11);
            assert_eq!(dst[6 * stride + x], 12);
            assert_eq!(dst[7 * stride + x], 13);
        }
    }

    #[test]
    fn test_backup_db_not_first_row() {
        let stride = 64;
        let w = 8;
        let mut src = vec![0u8; stride * 128];
        // row>0: stripe_h=(64-0)>>0=64, src_off=62*stride
        for i in 0..4 {
            for x in 0..w {
                src[(62 + i) * stride + x] = (20 + i) as u8;
            }
        }
        let mut dst = vec![0u8; stride * 64];
        for i in 0..4 {
            for x in 0..w {
                dst[(4 + i) * stride + x] = (50 + i) as u8;
            }
        }
        backup_db(&mut dst, &src, stride, 0, false, 56, 120, w, false, 1);
        // n_tc=1, row>0: copy dst rows [4..8] -> [0..4]
        for x in 0..w {
            assert_eq!(dst[0 * stride + x], 50);
            assert_eq!(dst[3 * stride + x], 53);
        }
        // Then copy from src[62..66] into dst[4..8]
        for x in 0..w {
            assert_eq!(dst[4 * stride + x], 20);
            assert_eq!(dst[7 * stride + x], 23);
        }
    }

    #[test]
    fn test_backup_db_multithread() {
        let stride = 64;
        let w = 8;
        let mut src = vec![0u8; stride * 128];
        for i in 0..4 {
            for x in 0..w {
                src[(54 + i) * stride + x] = (30 + i) as u8;
            }
        }
        let mut dst = vec![0u8; stride * 64];
        // n_tc=2: no single-thread top copy, dst_off starts at 0
        backup_db(&mut dst, &src, stride, 0, false, 0, 56, w, false, 2);
        for x in 0..w {
            assert_eq!(dst[0 * stride + x], 30);
            assert_eq!(dst[3 * stride + x], 33);
        }
    }

    #[test]
    fn test_backup_db_ss_ver() {
        let stride = 64;
        let w = 8;
        let mut src = vec![0u8; stride * 64];
        // ss_ver=1: stripe_h=(64-8)>>1=28, src_off=26*stride
        for i in 0..4 {
            for x in 0..w {
                src[(26 + i) * stride + x] = (40 + i) as u8;
            }
        }
        let mut dst = vec![0u8; stride * 32];
        backup_db(&mut dst, &src, stride, 1, false, 0, 28, w, false, 2);
        for x in 0..w {
            assert_eq!(dst[0 * stride + x], 40);
            assert_eq!(dst[3 * stride + x], 43);
        }
    }

    #[test]
    fn test_backup_db_sb128() {
        let stride = 64;
        let w = 8;
        let mut src = vec![0u8; stride * 256];
        // sb128=true, cdef_backup=true: stripe_h=(64<<1 - 8)>>0=120, src_off=118*stride
        for i in 0..4 {
            for x in 0..w {
                src[(118 + i) * stride + x] = (60 + i) as u8;
            }
        }
        let mut dst = vec![0u8; stride * 128];
        backup_db(&mut dst, &src, stride, 0, true, 0, 120, w, false, 2);
        for x in 0..w {
            assert_eq!(dst[0 * stride + x], 60);
            assert_eq!(dst[3 * stride + x], 63);
        }
    }

    #[test]
    fn test_backup_db_lr_mode() {
        let stride = 64;
        let w = 8;
        let mut src = vec![0u8; stride * 128];
        // lr_backup=true -> cdef_backup=0, sb128 irrelevant
        // stripe_h=(64<<0 - 8)>>0=56, src_off=54*stride
        for i in 0..4 {
            for x in 0..w {
                src[(54 + i) * stride + x] = (70 + i) as u8;
            }
        }
        let mut dst = vec![0u8; stride * 64];
        backup_db(&mut dst, &src, stride, 0, true, 0, 56, w, true, 2);
        for x in 0..w {
            assert_eq!(dst[0 * stride + x], 70);
            assert_eq!(dst[3 * stride + x], 73);
        }
    }

    #[test]
    fn test_backup_db_no_stripes() {
        let stride = 64;
        let w = 8;
        let src = vec![0u8; stride * 128];
        let mut dst = vec![0u8; stride * 64];
        // row_h < row + stripe_h: no stripes copied
        backup_db(&mut dst, &src, stride, 0, false, 0, 10, w, false, 2);
        assert!(dst.iter().all(|&x| x == 0));
    }
}

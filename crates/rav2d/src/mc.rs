use crate::intops::{iclip, imax, imin};
use crate::tables::{EXT_WARP_FILTER, MC_SUBPEL_FILTERS, MC_WARP_FILTER};

pub const INTERMEDIATE_BITS_8BPC: i32 = 4;
pub const PREP_BIAS_8BPC: i32 = 0;

pub fn put_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        dst[y * dst_stride..y * dst_stride + w].copy_from_slice(&src[y * src_stride..y * src_stride + w]);
    }
}

pub fn prep_8bpc(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = ((src[y * src_stride + x] as i32) << INTERMEDIATE_BITS_8BPC) as i16;
        }
    }
}

pub fn avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    let sh = INTERMEDIATE_BITS_8BPC + 1;
    let rnd = 1 << INTERMEDIATE_BITS_8BPC;
    for y in 0..h {
        for x in 0..w {
            let ti = y * w + x;
            dst[y * dst_stride + x] =
                iclip((tmp1[ti] as i32 + tmp2[ti] as i32 + rnd) >> sh, 0, 255) as u8;
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
    let sh = INTERMEDIATE_BITS_8BPC + 4;
    let rnd = 8 << INTERMEDIATE_BITS_8BPC;
    for y in 0..h {
        for x in 0..w {
            let ti = y * w + x;
            dst[y * dst_stride + x] = iclip(
                (tmp1[ti] as i32 * weight + tmp2[ti] as i32 * (16 - weight) + rnd) >> sh,
                0,
                255,
            ) as u8;
        }
    }
}

pub fn mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    let sh = INTERMEDIATE_BITS_8BPC + 6;
    let rnd = 32 << INTERMEDIATE_BITS_8BPC;
    for y in 0..h {
        for x in 0..w {
            let ti = y * w + x;
            let m = mask[ti] as i32;
            dst[y * dst_stride + x] = iclip(
                (tmp1[ti] as i32 * m + tmp2[ti] as i32 * (64 - m) + rnd) >> sh,
                0,
                255,
            ) as u8;
        }
    }
}

pub fn blend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        for x in 0..w {
            let di = y * dst_stride + x;
            let ti = y * w + x;
            let m = mask[ti] as i32;
            dst[di] = ((dst[di] as i32 * (64 - m) + tmp[ti] as i32 * m + 32) >> 6) as u8;
        }
    }
}

pub fn morph_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    alpha: i32,
    beta: i32,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let di = y * dst_stride + x;
            dst[di] = iclip((alpha * dst[di] as i32 + beta) >> 8, 0, 255) as u8;
        }
    }
}

pub fn emu_edge_8bpc(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: isize,
    y: isize,
    dst: &mut [u8],
    dst_stride: usize,
    r: &[u8],
    ref_stride: usize,
) {
    let ref_y = iclip(y as i32, 0, ih as i32 - 1) as usize;
    let ref_x = iclip(x as i32, 0, iw as i32 - 1) as usize;
    let ref_off = ref_y * ref_stride + ref_x;

    let left_ext = iclip(-x as i32, 0, bw as i32 - 1) as usize;
    let right_ext = iclip((x + bw as isize - iw as isize) as i32, 0, bw as i32 - 1) as usize;
    let top_ext = iclip(-y as i32, 0, bh as i32 - 1) as usize;
    let bottom_ext = iclip((y + bh as isize - ih as isize) as i32, 0, bh as i32 - 1) as usize;

    let center_w = bw - left_ext - right_ext;
    let center_h = bh - top_ext - bottom_ext;

    let mut roff = ref_off;
    for cy in 0..center_h {
        let blk_y = top_ext + cy;
        let blk_off = blk_y * dst_stride;
        dst[blk_off + left_ext..blk_off + left_ext + center_w]
            .copy_from_slice(&r[roff..roff + center_w]);
        if left_ext > 0 {
            let fill = dst[blk_off + left_ext];
            dst[blk_off..blk_off + left_ext].fill(fill);
        }
        if right_ext > 0 {
            let fill = dst[blk_off + left_ext + center_w - 1];
            dst[blk_off + left_ext + center_w..blk_off + bw].fill(fill);
        }
        roff += ref_stride;
    }

    let first_row_off = top_ext * dst_stride;
    for ty in 0..top_ext {
        dst.copy_within(first_row_off..first_row_off + bw, ty * dst_stride);
    }

    if bottom_ext > 0 {
        let last_row_start = (top_ext + center_h - 1) * dst_stride;
        for by in 0..bottom_ext {
            let dst_start = (top_ext + center_h + by) * dst_stride;
            dst.copy_within(last_row_start..last_row_start + bw, dst_start);
        }
    }
}

pub fn sad_nxn_8bpc(
    p0: &[u8],
    p0_stride: usize,
    p1: &[u8],
    p1_stride: usize,
    w: usize,
    h: usize,
) -> i32 {
    let mut sad = 0i32;
    let mut o0 = 0;
    let mut o1 = 0;
    let mut y = 0;
    while y < h {
        for x in 0..w {
            sad += (p0[o0 + x] as i32 - p1[o1 + x] as i32).abs();
        }
        o0 += p0_stride * 2;
        o1 += p1_stride * 2;
        y += 2;
    }
    sad
}

fn get_h_filter(mx: i32, filter_type: i32, w: usize) -> Option<[i8; 8]> {
    if mx == 0 { return None; }
    let f = if filter_type == -1 {
        EXT_WARP_FILTER[(mx - 1) as usize]
    } else if w > 4 {
        MC_SUBPEL_FILTERS[filter_type as usize][(mx - 1) as usize]
    } else {
        MC_SUBPEL_FILTERS[(3 + (filter_type & 1)) as usize][(mx - 1) as usize]
    };
    Some(f)
}

fn get_v_filter(my: i32, filter_type: i32, h: usize) -> Option<[i8; 8]> {
    if my == 0 { return None; }
    let f = if filter_type == -1 {
        EXT_WARP_FILTER[(my - 1) as usize]
    } else if h > 4 {
        MC_SUBPEL_FILTERS[filter_type as usize][(my - 1) as usize]
    } else {
        MC_SUBPEL_FILTERS[(3 + (filter_type & 1)) as usize][(my - 1) as usize]
    };
    Some(f)
}

#[inline(always)]
fn filter_8tap_u8(src: &[u8], center: usize, f: &[i8; 8], stride: isize) -> i32 {
    let c = center as isize;
    f[0] as i32 * src[(c - 3 * stride) as usize] as i32
        + f[1] as i32 * src[(c - 2 * stride) as usize] as i32
        + f[2] as i32 * src[(c - stride) as usize] as i32
        + f[3] as i32 * src[center] as i32
        + f[4] as i32 * src[(c + stride) as usize] as i32
        + f[5] as i32 * src[(c + 2 * stride) as usize] as i32
        + f[6] as i32 * src[(c + 3 * stride) as usize] as i32
        + f[7] as i32 * src[(c + 4 * stride) as usize] as i32
}

#[inline(always)]
fn filter_8tap_i16(mid: &[i16], center: usize, f: &[i8; 8], stride: isize) -> i32 {
    let c = center as isize;
    f[0] as i32 * mid[(c - 3 * stride) as usize] as i32
        + f[1] as i32 * mid[(c - 2 * stride) as usize] as i32
        + f[2] as i32 * mid[(c - stride) as usize] as i32
        + f[3] as i32 * mid[center] as i32
        + f[4] as i32 * mid[(c + stride) as usize] as i32
        + f[5] as i32 * mid[(c + 2 * stride) as usize] as i32
        + f[6] as i32 * mid[(c + 3 * stride) as usize] as i32
        + f[7] as i32 * mid[(c + 4 * stride) as usize] as i32
}

/// 8-tap subpixel MC filter. `src_off` is the origin in `src` — must have
/// 3 rows above and 4 rows below, and 3 cols left and 4 cols right of padding.
pub fn put_8tap_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    let bits = 6 + (filter_type < 0) as i32;
    let intermediate_bits = INTERMEDIATE_BITS_8BPC;
    let intermediate_rnd = ((1 << bits) >> 1) + ((1 << (bits - intermediate_bits)) >> 1);
    let fh = get_h_filter(mx, filter_type, w);
    let fv = get_v_filter(my, filter_type, h);
    let ss = src_stride as isize;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = vec![0i16; 64 * tmp_h];
            for y in 0..tmp_h {
                for x in 0..w {
                    let si = (src_off as isize + (y as isize - 3) * src_stride as isize + x as isize) as usize;
                    mid[y * 64 + x] = ((filter_8tap_u8(src, si, &fh, 1)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits)) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = (y + 3) * 64 + x;
                    dst[y * dst_stride + x] = iclip(
                        (filter_8tap_i16(&mid, mi, &fv, 64)
                            + ((1 << (bits + intermediate_bits)) >> 1))
                            >> (bits + intermediate_bits),
                        0, 255,
                    ) as u8;
                }
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    dst[y * dst_stride + x] = iclip(
                        (filter_8tap_u8(src, si, &fh, 1) + intermediate_rnd) >> bits,
                        0, 255,
                    ) as u8;
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    dst[y * dst_stride + x] = iclip(
                        (filter_8tap_u8(src, si, &fv, ss)
                            + ((1 << bits) >> 1)) >> bits,
                        0, 255,
                    ) as u8;
                }
            }
        }
        (None, None) => {
            put_8bpc(dst, dst_stride, &src[src_off..], src_stride, w, h);
        }
    }
}

pub fn prep_8tap_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    let bits = 6 + (filter_type < 0) as i32;
    let intermediate_bits = INTERMEDIATE_BITS_8BPC;
    let fh = get_h_filter(mx, filter_type, w);
    let fv = get_v_filter(my, filter_type, h);
    let ss = src_stride as isize;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = vec![0i16; 64 * tmp_h];
            for y in 0..tmp_h {
                for x in 0..w {
                    let si = (src_off as isize + (y as isize - 3) * src_stride as isize + x as isize) as usize;
                    mid[y * 64 + x] = ((filter_8tap_u8(src, si, &fh, 1)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits)) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = (y + 3) * 64 + x;
                    tmp[y * tmp_stride + x] = ((filter_8tap_i16(&mid, mi, &fv, 64)
                        + ((1 << bits) >> 1)) >> bits) as i16
                        - PREP_BIAS_8BPC as i16;
                }
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    tmp[y * tmp_stride + x] = ((filter_8tap_u8(src, si, &fh, 1)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits)) as i16
                        - PREP_BIAS_8BPC as i16;
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    tmp[y * tmp_stride + x] = ((filter_8tap_u8(src, si, &fv, ss)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits)) as i16
                        - PREP_BIAS_8BPC as i16;
                }
            }
        }
        (None, None) => {
            prep_8bpc(tmp, &src[src_off..], src_stride, w, h);
        }
    }
}

pub fn w_mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    let sh = INTERMEDIATE_BITS_8BPC + 6;
    let rnd = 32 << INTERMEDIATE_BITS_8BPC;
    let mask_sh = 8; // bitdepth(8) + intermediate_bits(4) - 4
    let mask_rnd = 1i32 << (mask_sh - 5);

    let mut t1off = 0usize;
    let mut t2off = 0usize;
    let mut doff = 0usize;
    let mut moff = 0usize;

    for row in 0..h {
        let mut x = 0usize;
        while x < w {
            let m = imin(38 + (((tmp1[t1off + x] as i32 - tmp2[t2off + x] as i32).abs() + mask_rnd) >> mask_sh), 64);
            dst[doff + x] = iclip(
                (tmp1[t1off + x] as i32 * m + tmp2[t2off + x] as i32 * (64 - m) + rnd) >> sh,
                0, 255,
            ) as u8;

            if ss_hor {
                x += 1;
                let n = imin(38 + (((tmp1[t1off + x] as i32 - tmp2[t2off + x] as i32).abs() + mask_rnd) >> mask_sh), 64);
                dst[doff + x] = iclip(
                    (tmp1[t1off + x] as i32 * n + tmp2[t2off + x] as i32 * (64 - n) + rnd) >> sh,
                    0, 255,
                ) as u8;

                if row & 1 != 0 && ss_ver {
                    mask[moff + (x >> 1)] = ((m + n + mask[moff + (x >> 1)] as i32 + 2 - sign) >> 2) as u8;
                } else if ss_ver {
                    mask[moff + (x >> 1)] = (m + n) as u8;
                } else {
                    mask[moff + (x >> 1)] = ((m + n + 1 - sign) >> 1) as u8;
                }
            } else {
                mask[moff + x] = m as u8;
            }
            x += 1;
        }

        t1off += w;
        t2off += w;
        doff += dst_stride;
        if !ss_ver || (row & 1 != 0) {
            moff += mask_stride;
        }
    }
}

#[inline(always)]
fn bilin(a: i32, b: i32, mxy: i32) -> i32 {
    16 * a + mxy * (b - a)
}

#[inline(always)]
fn bilin_rnd(a: i32, b: i32, mxy: i32, sh: i32) -> i32 {
    (bilin(a, b, mxy) + ((1 << sh) >> 1)) >> sh
}

pub fn put_bilin_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    if mx != 0 {
        if my != 0 {
            let mut mid = vec![0i16; 64 * (h + 1)];
            for y in 0..h + 1 {
                for x in 0..w {
                    let si = y * src_stride + x;
                    mid[y * 64 + x] = bilin_rnd(
                        src[si] as i32, src[si + 1] as i32, mx, 0,
                    ) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = y * 64 + x;
                    dst[y * dst_stride + x] = iclip(
                        bilin_rnd(mid[mi] as i32, mid[mi + 64] as i32, my, 8),
                        0, 255,
                    ) as u8;
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let si = y * src_stride + x;
                    let px = bilin_rnd(src[si] as i32, src[si + 1] as i32, mx, 0);
                    dst[y * dst_stride + x] = iclip((px + 8) >> 4, 0, 255) as u8;
                }
            }
        }
    } else if my != 0 {
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x;
                dst[y * dst_stride + x] = iclip(
                    bilin_rnd(src[si] as i32, src[si + src_stride] as i32, my, 4),
                    0, 255,
                ) as u8;
            }
        }
    } else {
        put_8bpc(dst, dst_stride, src, src_stride, w, h);
    }
}

pub fn prep_bilin_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    if mx != 0 {
        if my != 0 {
            let mut mid = vec![0i16; 64 * (h + 1)];
            for y in 0..h + 1 {
                for x in 0..w {
                    let si = y * src_stride + x;
                    mid[y * 64 + x] = bilin_rnd(
                        src[si] as i32, src[si + 1] as i32, mx, 0,
                    ) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = y * 64 + x;
                    tmp[y * tmp_stride + x] = bilin_rnd(
                        mid[mi] as i32, mid[mi + 64] as i32, my, 4,
                    ) as i16;
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let si = y * src_stride + x;
                    tmp[y * tmp_stride + x] = bilin_rnd(
                        src[si] as i32, src[si + 1] as i32, mx, 0,
                    ) as i16;
                }
            }
        }
    } else if my != 0 {
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x;
                tmp[y * tmp_stride + x] = bilin_rnd(
                    src[si] as i32, src[si + src_stride] as i32, my, 0,
                ) as i16;
            }
        }
    } else {
        prep_8bpc(tmp, src, src_stride, w, h);
    }
}

pub fn sad8x8_8bpc(
    p0: &[u8],
    p0_stride: usize,
    p1: &[u8],
    p1_stride: usize,
) -> u32 {
    let mut sad = 0u32;
    for y in 0..8 {
        for x in 0..8 {
            sad += (p0[y * p0_stride + x] as i32 - p1[y * p1_stride + x] as i32).unsigned_abs();
        }
    }
    sad
}

pub fn sad_refine_mv_8bpc(
    p0: &[u8],
    p0_stride: usize,
    p1: &[u8],
    p1_stride: usize,
    w: usize,
    h: usize,
    is_implicit: bool,
) -> (i32, i32) {
    let sadw = w + 4;
    let sadh = h + 4;
    let sad_thr = (sadw * sadh * 2) as i32;
    let mut best_sad = i32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    if is_implicit {
        best_sad = sad_nxn_8bpc(
            &p0[2 * p0_stride + 2..], p0_stride,
            &p1[2 * p1_stride + 2..], p1_stride,
            sadw, sadh,
        );
        best_sad = (best_sad * 7 + 7) >> 3;
        if best_sad < sad_thr {
            return (best_dx, best_dy);
        }
    }

    for y_off in -2i32..=2 {
        for x_off in -2i32..=2 {
            if x_off == 0 && y_off == 0 { continue; }
            let sad = sad_nxn_8bpc(
                &p0[((2 + y_off) as usize) * p0_stride + (2 + x_off) as usize..], p0_stride,
                &p1[((2 - y_off) as usize) * p1_stride + (2 - x_off) as usize..], p1_stride,
                sadw, sadh,
            );
            if sad >= best_sad { continue; }
            best_sad = sad;
            best_dx = x_off;
            best_dy = y_off;
        }
    }
    (best_dx, best_dy)
}

#[derive(Clone, Copy, Default)]
pub struct OpflRegressionData {
    pub su2: i32,
    pub suv: i32,
    pub sv2: i32,
    pub suw: i32,
    pub svw: i32,
}

pub fn opfl_derive_mv_8bpc(
    out: &mut [OpflRegressionData],
    p0: &[u8],
    p0_stride: usize,
    p1: &[u8],
    p1_stride: usize,
    w: usize,
    h: usize,
    bs: usize,
    d: [i8; 2],
) {
    let mut tmp0 = [0i16; 64 * 16];
    let mut tmp1 = [0i16; 64 * 16];

    for y in 0..h {
        for x in 0..w {
            let p0p = p0[y * p0_stride + x] as i32;
            let p1p = p1[y * p1_stride + x] as i32;
            tmp0[y * 64 + x] = (d[0] as i32 * p0p + d[1] as i32 * p1p) as i16;
            tmp1[y * 64 + x] = (p0p - p1p) as i16;
        }
    }

    let mut gx0 = [0i16; 64 * 16];
    let mut gy0 = [0i16; 64 * 16];

    let mut bx = 0usize;
    while bx < w {
        let x_end = imin(bx as i32 + 16, w as i32) as usize;
        let min_x = bx & !15;
        let max_x = x_end - 1;
        let max_y = h - 1;
        for y in 0..h {
            for x in bx..x_end {
                let pa = tmp0[y * 64 + imax(min_x as i32, x as i32 - 2) as usize] as i32;
                let pb = tmp0[y * 64 + imax(min_x as i32, x as i32 - 1) as usize] as i32;
                let pc = tmp0[y * 64 + imin(max_x as i32, x as i32 + 1) as usize] as i32;
                let pd = tmp0[y * 64 + imin(max_x as i32, x as i32 + 2) as usize] as i32;
                let e1 = (x + 1 > max_x || x < 1 + min_x) as i32;
                let x0 = ((pc - pb) * 42 + (pd - pa) * -5) * (1 + e1);
                gx0[y * 64 + x] = ((x0 + 63 + (x0 > 0) as i32) >> 7) as i16;

                let qa = tmp0[imax(0, y as i32 - 2) as usize * 64 + x] as i32;
                let qb = tmp0[imax(0, y as i32 - 1) as usize * 64 + x] as i32;
                let qc = tmp0[imin(max_y as i32, y as i32 + 1) as usize * 64 + x] as i32;
                let qd = tmp0[imin(max_y as i32, y as i32 + 2) as usize * 64 + x] as i32;
                let e2 = (y + 1 > max_y || y < 1) as i32;
                let y0 = ((qc - qb) * 42 + (qd - qa) * -5) * (1 + e2);
                gy0[y * 64 + x] = ((y0 + 63 + (y0 > 0) as i32) >> 7) as i16;
            }
        }
        bx += 16;
    }

    let mut oi = 0;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let mut su2 = (bs * bs) as i32;
            let mut suv = 0i32;
            let mut sv2 = (bs * bs) as i32;
            let mut suw = 0i32;
            let mut svw = 0i32;
            for py in y..y + bs {
                for px in x..x + bs {
                    let u = gx0[py * 64 + px] as i32;
                    let v = gy0[py * 64 + px] as i32;
                    let ww = tmp1[py * 64 + px] as i32;
                    su2 += u * u;
                    suv += u * v;
                    sv2 += v * v;
                    suw += u * ww;
                    svw += v * ww;
                }
            }
            out[oi] = OpflRegressionData { su2, suv, sv2, suw, svw };
            oi += 1;
            x += bs;
        }
        y += bs;
    }
}

fn filter_warp_rnd(src: &[i16], x: usize, f: &[i8; 8], stride: usize, sh: i32) -> i16 {
    let v = f[0] as i32 * src[x.wrapping_sub(3 * stride)] as i32
        + f[1] as i32 * src[x.wrapping_sub(2 * stride)] as i32
        + f[2] as i32 * src[x.wrapping_sub(stride)] as i32
        + f[3] as i32 * src[x] as i32
        + f[4] as i32 * src[x + stride] as i32
        + f[5] as i32 * src[x + 2 * stride] as i32
        + f[6] as i32 * src[x + 3 * stride] as i32
        + f[7] as i32 * src[x + 4 * stride] as i32
        + ((1 << sh) >> 1);
    (v >> sh) as i16
}

fn filter_warp_rnd_px(src: &[u8], x: usize, f: &[i8; 8], stride: usize, sh: i32) -> i16 {
    let v = f[0] as i32 * src[x.wrapping_sub(3 * stride)] as i32
        + f[1] as i32 * src[x.wrapping_sub(2 * stride)] as i32
        + f[2] as i32 * src[x.wrapping_sub(stride)] as i32
        + f[3] as i32 * src[x] as i32
        + f[4] as i32 * src[x + stride] as i32
        + f[5] as i32 * src[x + 2 * stride] as i32
        + f[6] as i32 * src[x + 3 * stride] as i32
        + f[7] as i32 * src[x + 4 * stride] as i32
        + ((1 << sh) >> 1);
    (v >> sh) as i16
}

pub fn warp_affine_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let mut mid = [0i16; 15 * 8];

    let mut soff = src_off.wrapping_sub(3 * src_stride);
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let fi = (192 + ((tmx + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            mid[y * 8 + x] = filter_warp_rnd_px(src, soff + x, f, 1, 3);
            tmx += abcd[0] as i32;
        }
        soff += src_stride;
        mx += abcd[1] as i32;
    }

    for y in 0..8 {
        let mid_base = (3 + y) * 8;
        let mut tmy = my;
        for x in 0..8 {
            let fi = (192 + ((tmy + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            let v = filter_warp_rnd(&mid, mid_base + x, f, 8, 11);
            dst[y * dst_stride + x] = iclip(v as i32, 0, 255) as u8;
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}

pub fn warp_affine_8x8t_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let mut mid = [0i16; 15 * 8];

    let mut soff = src_off.wrapping_sub(3 * src_stride);
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let fi = (192 + ((tmx + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            mid[y * 8 + x] = filter_warp_rnd_px(src, soff + x, f, 1, 3);
            tmx += abcd[0] as i32;
        }
        soff += src_stride;
        mx += abcd[1] as i32;
    }

    for y in 0..8 {
        let mid_base = (3 + y) * 8;
        let mut tmy = my;
        for x in 0..8 {
            let fi = (192 + ((tmy + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            tmp[y * tmp_stride + x] = filter_warp_rnd(&mid, mid_base + x, f, 8, 7) - PREP_BIAS_8BPC as i16;
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}

fn filter_8tap_ring(rows: &[[i16; 64]; 8], order: &[usize; 8], x: usize, f: &[i8; 8]) -> i32 {
    let mut sum = 0i32;
    for i in 0..8 {
        sum += f[i] as i32 * rows[order[i]][x] as i32;
    }
    sum
}

pub fn put_8tap_scaled_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: isize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
    filter_type: i32,
) {
    let intermediate_bits: i32 = 4;
    let intermediate_rnd: i32 = 1 << (intermediate_bits - 1);
    let mut mid = [[0i16; 64]; 8];
    let mut order = [0usize; 8];
    for i in 0..8 { order[i] = i; }
    let mut in_y: i32 = -8;
    let sp = (src_off as isize - src_stride * 3) as usize;

    let mut src_p = sp;
    let mut dst_p = 0usize;

    for _y in 0..h {
        let src_y = my >> 10;
        let fv = get_v_filter((my & 0x3ff) >> 6, filter_type, h);

        while in_y < src_y {
            let old = order[0];
            for i in 0..7 { order[i] = order[i + 1]; }
            order[7] = old;

            let mut imx = mx;
            let mut ioff = 0usize;
            let row = order[7];
            for x in 0..w {
                let fh = get_h_filter(imx >> 6, filter_type, w);
                mid[row][x] = if let Some(ref f) = fh {
                    let c = src_p + ioff;
                    (filter_8tap_u8(src, c, f, 1) >> (6 - intermediate_bits)) as i16
                } else {
                    ((src[src_p + ioff] as i32) << intermediate_bits) as i16
                };
                imx += dx;
                ioff += (imx >> 10) as usize;
                imx &= 0x3ff;
            }

            src_p = (src_p as isize + src_stride) as usize;
            in_y += 1;
        }

        for x in 0..w {
            dst[dst_p + x] = if let Some(ref f) = fv {
                let sum = filter_8tap_ring(&mid, &order, x, f);
                iclip((sum + (1 << (5 + intermediate_bits))) >> (6 + intermediate_bits), 0, 255) as u8
            } else {
                iclip((mid[order[3]][x] as i32 + intermediate_rnd) >> intermediate_bits, 0, 255) as u8
            };
        }

        my += dy;
        dst_p += dst_stride;
    }
}

pub fn prep_8tap_scaled_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: isize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
    filter_type: i32,
) {
    let intermediate_bits: i32 = 4;
    let mut mid = [[0i16; 64]; 8];
    let mut order = [0usize; 8];
    for i in 0..8 { order[i] = i; }
    let mut in_y: i32 = -8;
    let sp = (src_off as isize - src_stride * 3) as usize;

    let mut src_p = sp;
    let mut tmp_p = 0usize;

    for _y in 0..h {
        let src_y = my >> 10;
        let fv = get_v_filter((my & 0x3ff) >> 6, filter_type, h);

        while in_y < src_y {
            let old = order[0];
            for i in 0..7 { order[i] = order[i + 1]; }
            order[7] = old;

            let mut imx = mx;
            let mut ioff = 0usize;
            let row = order[7];
            for x in 0..w {
                let fh = get_h_filter(imx >> 6, filter_type, w);
                mid[row][x] = if let Some(ref f) = fh {
                    let c = src_p + ioff;
                    (filter_8tap_u8(src, c, f, 1) >> (6 - intermediate_bits)) as i16
                } else {
                    ((src[src_p + ioff] as i32) << intermediate_bits) as i16
                };
                imx += dx;
                ioff += (imx >> 10) as usize;
                imx &= 0x3ff;
            }

            src_p = (src_p as isize + src_stride) as usize;
            in_y += 1;
        }

        for x in 0..w {
            tmp[tmp_p + x] = if let Some(ref f) = fv {
                let sum = filter_8tap_ring(&mid, &order, x, f);
                ((sum + (1 << 5)) >> 6) as i16
            } else {
                mid[order[3]][x]
            };
        }

        my += dy;
        tmp_p += tmp_stride;
    }
}

pub fn put_bilin_scaled_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: isize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
) {
    let intermediate_bits: i32 = 4;
    let mut mid = [[0i16; 64]; 2];
    let mut in_y: i32 = -2;
    let mut src_p = src_off;
    let mut dst_p = 0usize;

    for _row in 0..h {
        let y = my >> 10;
        let mid1_idx = (y & 1) as usize;
        let mid2_idx = mid1_idx ^ 1;
        let dmy = my & 0x3ff;

        while in_y < y {
            let mut imx = mx;
            let mut ioff = 0usize;
            let ri = (in_y & 1) as usize;
            for x in 0..w {
                let frac = imx >> 6;
                mid[ri][x] = bilin_rnd(
                    src[src_p + ioff] as i32,
                    src[src_p + ioff + 1] as i32,
                    frac, 4 - intermediate_bits,
                ) as i16;
                imx += dx;
                ioff += (imx >> 10) as usize;
                imx &= 0x3ff;
            }
            src_p = (src_p as isize + src_stride) as usize;
            in_y += 1;
        }

        for x in 0..w {
            dst[dst_p + x] = iclip(
                bilin_rnd(mid[mid1_idx][x] as i32, mid[mid2_idx][x] as i32,
                          dmy >> 6, 4 + intermediate_bits),
                0, 255,
            ) as u8;
        }

        my += dy;
        dst_p += dst_stride;
    }
}

pub fn prep_bilin_scaled_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: isize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
) {
    let intermediate_bits: i32 = 4;
    let mut mid = [[0i16; 64]; 2];
    let mut in_y: i32 = -2;
    let mut src_p = src_off;
    let mut tmp_p = 0usize;

    for _row in 0..h {
        let y = my >> 10;
        let mid1_idx = (y & 1) as usize;
        let mid2_idx = mid1_idx ^ 1;
        let dmy = my & 0x3ff;

        while in_y < y {
            let mut imx = mx;
            let mut ioff = 0usize;
            let ri = (in_y & 1) as usize;
            for x in 0..w {
                let frac = imx >> 6;
                mid[ri][x] = bilin_rnd(
                    src[src_p + ioff] as i32,
                    src[src_p + ioff + 1] as i32,
                    frac, 4 - intermediate_bits,
                ) as i16;
                imx += dx;
                ioff += (imx >> 10) as usize;
                imx &= 0x3ff;
            }
            src_p = (src_p as isize + src_stride) as usize;
            in_y += 1;
        }

        for x in 0..w {
            tmp[tmp_p + x] = bilin_rnd(
                mid[mid1_idx][x] as i32, mid[mid2_idx][x] as i32,
                dmy >> 6, 4,
            ) as i16;
        }

        my += dy;
        tmp_p += tmp_stride;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad_identical() {
        let p = vec![128u8; 64];
        assert_eq!(sad_nxn_8bpc(&p, 8, &p, 8, 8, 8), 0);
    }

    #[test]
    fn test_sad_all_different() {
        let p0 = vec![0u8; 64];
        let p1 = vec![10u8; 64];
        let sad = sad_nxn_8bpc(&p0, 8, &p1, 8, 8, 8);
        assert_eq!(sad, 8 * 4 * 10);
    }

    #[test]
    fn test_sad_skips_odd_rows() {
        let mut p0 = vec![0u8; 32];
        let p1 = vec![0u8; 32];
        for i in 0..4 {
            p0[i * 8 + 1] = 5;
        }
        let sad = sad_nxn_8bpc(&p0, 8, &p1, 8, 4, 4);
        assert_eq!(sad, 2 * 5);
    }

    #[test]
    fn test_sad_stride() {
        let mut p0 = vec![100u8; 128];
        let p1 = vec![100u8; 128];
        p0[0] = 200;
        let sad = sad_nxn_8bpc(&p0, 16, &p1, 16, 4, 4);
        assert_eq!(sad, 100);
    }

    #[test]
    fn test_put_copy() {
        let src = vec![42u8; 64];
        let mut dst = vec![0u8; 64];
        put_8bpc(&mut dst, 8, &src, 8, 8, 8);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_put_stride() {
        let src = vec![10u8; 32];
        let mut dst = vec![0u8; 32];
        put_8bpc(&mut dst, 8, &src, 8, 4, 4);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 10);
            }
        }
    }

    #[test]
    fn test_prep_shift() {
        let src = vec![128u8; 16];
        let mut tmp = vec![0i16; 16];
        prep_8bpc(&mut tmp, &src, 4, 4, 4);
        for &v in &tmp {
            assert_eq!(v, (128 << INTERMEDIATE_BITS_8BPC) as i16);
        }
    }

    #[test]
    fn test_avg_midpoint() {
        let tmp1: Vec<i16> = vec![(100 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![(200 << 4) as i16; 16];
        let mut dst = vec![0u8; 16];
        avg_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4);
        for &v in &dst {
            assert_eq!(v, 150);
        }
    }

    #[test]
    fn test_w_avg_full_weight() {
        let tmp1: Vec<i16> = vec![(200 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![0i16; 16];
        let mut dst = vec![0u8; 16];
        w_avg_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, 16);
        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_w_avg_zero_weight() {
        let tmp1: Vec<i16> = vec![0i16; 16];
        let tmp2: Vec<i16> = vec![(100 << 4) as i16; 16];
        let mut dst = vec![0u8; 16];
        w_avg_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, 0);
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_mask_uniform() {
        let tmp1: Vec<i16> = vec![(255 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![0i16; 16];
        let mask_buf = vec![64u8; 16];
        let mut dst = vec![0u8; 16];
        mask_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_mask_half() {
        let tmp1: Vec<i16> = vec![(200 << 4) as i16; 16];
        let tmp2: Vec<i16> = vec![(100 << 4) as i16; 16];
        let mask_buf = vec![32u8; 16];
        let mut dst = vec![0u8; 16];
        mask_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 150);
        }
    }

    #[test]
    fn test_blend_zero_mask() {
        let mut dst = vec![200u8; 16];
        let tmp = vec![50u8; 16];
        let mask_buf = vec![0u8; 16];
        blend_8bpc(&mut dst, 4, &tmp, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_blend_full_mask() {
        let mut dst = vec![200u8; 16];
        let tmp = vec![50u8; 16];
        let mask_buf = vec![64u8; 16];
        blend_8bpc(&mut dst, 4, &tmp, 4, 4, &mask_buf);
        for &v in &dst {
            assert_eq!(v, 50);
        }
    }

    #[test]
    fn test_morph_identity() {
        let mut dst = vec![128u8; 16];
        morph_8bpc(&mut dst, 4, 256, 0, 4, 4);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_morph_half() {
        let mut dst = vec![200u8; 16];
        morph_8bpc(&mut dst, 4, 128, 0, 4, 4);
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_morph_clamps() {
        let mut dst = vec![200u8; 16];
        morph_8bpc(&mut dst, 4, 512, 0, 4, 4);
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_emu_edge_inside() {
        let r = vec![42u8; 64];
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, 2, 2, &mut dst, 4, &r, 8);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 42);
            }
        }
    }

    #[test]
    fn test_emu_edge_left_extend() {
        let mut r = vec![0u8; 64];
        for y in 0..8 {
            r[y * 8] = 99;
        }
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, -2, 0, &mut dst, 4, &r, 8);
        for y in 0..4 {
            assert_eq!(dst[y * 4], 99);
            assert_eq!(dst[y * 4 + 1], 99);
        }
    }

    #[test]
    fn test_emu_edge_top_extend() {
        let r = vec![77u8; 64];
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, 0, -2, &mut dst, 4, &r, 8);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 77);
            }
        }
    }

    #[test]
    fn test_emu_edge_corner() {
        let mut r = vec![50u8; 64];
        r[0] = 200;
        let mut dst = vec![0u8; 16];
        emu_edge_8bpc(4, 4, 8, 8, -2, -2, &mut dst, 4, &r, 8);
        assert_eq!(dst[0], 200);
        assert_eq!(dst[1], 200);
    }

    #[test]
    fn test_w_mask_444() {
        let tmp1 = vec![(200i16) << 4; 16];
        let tmp2 = vec![(100i16) << 4; 16];
        let mut dst = vec![0u8; 16];
        let mut mask = vec![0u8; 16];
        w_mask_8bpc(&mut dst, 4, &tmp1, &tmp2, 4, 4, &mut mask, 4, 0, false, false);
        for &v in &dst {
            assert!(v > 0);
        }
        for &m in &mask[..16] {
            assert!(m >= 38 && m <= 64);
        }
    }

    #[test]
    fn test_w_mask_420() {
        let tmp1 = vec![(150i16) << 4; 64];
        let tmp2 = vec![(150i16) << 4; 64];
        let mut dst = vec![0u8; 64];
        let mut mask = vec![0u8; 16];
        w_mask_8bpc(&mut dst, 8, &tmp1, &tmp2, 8, 8, &mut mask, 4, 0, true, true);
        for &v in &dst {
            assert_eq!(v, 150);
        }
        for &m in &mask[..16] {
            assert_eq!(m, 38);
        }
    }

    #[test]
    fn test_put_8tap_no_filter() {
        let src = vec![42u8; 64];
        let mut dst = vec![0u8; 64];
        put_8tap_8bpc(&mut dst, 8, &src, 0, 8, 8, 8, 0, 0, 0);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_put_8tap_h_only() {
        let src = vec![128u8; 22 * 22];
        let mut dst = vec![0u8; 64];
        let so = 3 * 22 + 3;
        put_8tap_8bpc(&mut dst, 8, &src, so, 22, 8, 8, 8, 0, 0);
        for &v in &dst[..64] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_put_8tap_v_only() {
        let src = vec![128u8; 22 * 22];
        let mut dst = vec![0u8; 64];
        let so = 3 * 22 + 3;
        put_8tap_8bpc(&mut dst, 8, &src, so, 22, 8, 8, 0, 8, 0);
        for &v in &dst[..64] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_put_8tap_hv() {
        let src = vec![128u8; 22 * 22];
        let mut dst = vec![0u8; 64];
        let so = 3 * 22 + 3;
        put_8tap_8bpc(&mut dst, 8, &src, so, 22, 8, 8, 8, 8, 0);
        for &v in &dst[..64] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_put_8tap_smooth() {
        let src = vec![128u8; 22 * 22];
        let mut dst = vec![0u8; 64];
        let so = 3 * 22 + 3;
        put_8tap_8bpc(&mut dst, 8, &src, so, 22, 8, 8, 8, 8, 1);
        for &v in &dst[..64] {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_prep_8tap_hv() {
        let src = vec![128u8; 22 * 22];
        let mut tmp = vec![0i16; 64];
        let so = 3 * 22 + 3;
        prep_8tap_8bpc(&mut tmp, 8, &src, so, 22, 8, 8, 8, 8, 0);
        for &v in &tmp {
            assert_eq!(v, 2048);
        }
    }

    #[test]
    fn test_put_bilin_no_interp() {
        let src = vec![42u8; 64];
        let mut dst = vec![0u8; 64];
        put_bilin_8bpc(&mut dst, 8, &src, 8, 8, 8, 0, 0);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_put_bilin_h_only() {
        let mut src = vec![100u8; 80];
        for y in 0..8 { src[y * 10 + 1] = 200; }
        let mut dst = vec![0u8; 64];
        put_bilin_8bpc(&mut dst, 8, &src, 10, 1, 8, 8, 0);
        for y in 0..8 {
            assert!(dst[y * 8] > 100 && dst[y * 8] < 200);
        }
    }

    #[test]
    fn test_put_bilin_v_only() {
        let mut src = vec![100u8; 80];
        for x in 0..8 { src[10 + x] = 200; }
        let mut dst = vec![0u8; 64];
        put_bilin_8bpc(&mut dst, 8, &src, 10, 8, 1, 0, 8);
        assert!(dst[0] > 100 && dst[0] < 200);
    }

    #[test]
    fn test_put_bilin_hv() {
        let src = vec![128u8; 80];
        let mut dst = vec![0u8; 64];
        put_bilin_8bpc(&mut dst, 8, &src, 10, 4, 4, 8, 8);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 128);
            }
        }
    }

    #[test]
    fn test_prep_bilin_no_interp() {
        let src = vec![128u8; 64];
        let mut tmp = vec![0i16; 64];
        prep_bilin_8bpc(&mut tmp, 8, &src, 8, 8, 8, 0, 0);
        for &v in &tmp {
            assert_eq!(v, (128 << INTERMEDIATE_BITS_8BPC) as i16);
        }
    }

    #[test]
    fn test_prep_bilin_h_only() {
        let mut src = vec![100u8; 80];
        for y in 0..8 { src[y * 10 + 1] = 200; }
        let mut tmp = vec![0i16; 64];
        prep_bilin_8bpc(&mut tmp, 8, &src, 10, 1, 8, 8, 0);
        for y in 0..8 {
            let v = tmp[y * 8] as i32;
            assert!(v > 100 * 16 && v < 200 * 16);
        }
    }

    #[test]
    fn test_sad8x8_identical() {
        let p = vec![100u8; 64];
        assert_eq!(sad8x8_8bpc(&p, 8, &p, 8), 0);
    }

    #[test]
    fn test_sad8x8_different() {
        let p0 = vec![0u8; 64];
        let p1 = vec![10u8; 64];
        assert_eq!(sad8x8_8bpc(&p0, 8, &p1, 8), 640);
    }

    #[test]
    fn test_sad_refine_identical() {
        let p = vec![128u8; 20 * 20];
        let (dx, dy) = sad_refine_mv_8bpc(&p, 20, &p, 20, 8, 8, false);
        assert_eq!((dx, dy), (-2, -2));
    }

    #[test]
    fn test_sad_refine_implicit_low_sad() {
        let p = vec![128u8; 20 * 20];
        let (dx, dy) = sad_refine_mv_8bpc(&p, 20, &p, 20, 8, 8, true);
        assert_eq!((dx, dy), (0, 0));
    }

    #[test]
    fn test_sad_refine_finds_offset() {
        let mut p0 = vec![100u8; 20 * 20];
        let mut p1 = vec![100u8; 20 * 20];
        for y in 0..12 {
            for x in 0..12 {
                p0[(y + 3) * 20 + (x + 3)] = 200;
                p1[(y + 1) * 20 + (x + 1)] = 200;
            }
        }
        let (dx, dy) = sad_refine_mv_8bpc(&p0, 20, &p1, 20, 8, 8, false);
        assert!(dx != 0 || dy != 0);
    }

    #[test]
    fn test_warp_affine_identity() {
        let mut src = vec![0u8; 22 * 22];
        for y in 0..22 {
            for x in 0..22 {
                src[y * 22 + x] = 128;
            }
        }
        let mut dst = [0u8; 64];
        let abcd = [0i16, 0, 0, 0];
        let src_off = 3 * 22 + 3;
        warp_affine_8x8_8bpc(&mut dst, 8, &src, 22, src_off, &abcd, 0, 0);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_warp_affine_no_panic() {
        let src = vec![100u8; 22 * 22];
        let mut dst = [0u8; 64];
        let abcd = [64i16, 0, 0, 64];
        warp_affine_8x8_8bpc(&mut dst, 8, &src, 22, 3 * 22 + 3, &abcd, 0, 0);
        assert!(dst.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_warp_affine_8x8t_identity() {
        let src = vec![128u8; 22 * 22];
        let mut tmp = [0i16; 64];
        let abcd = [0i16, 0, 0, 0];
        warp_affine_8x8t_8bpc(&mut tmp, 8, &src, 22, 3 * 22 + 3, &abcd, 0, 0);
        for &v in &tmp {
            assert_eq!(v, 2048);
        }
    }

    #[test]
    fn test_put_8tap_scaled_identity() {
        let stride: isize = 16;
        let src = vec![128u8; 256];
        let src_off = (stride * 4) as usize;
        let mut dst = [0u8; 64];
        put_8tap_scaled_8bpc(
            &mut dst, 8, &src, src_off, stride,
            4, 4, 0, 0, 1024, 1024, 0,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 128);
            }
        }
    }

    #[test]
    fn test_put_8tap_scaled_subpel() {
        let stride: isize = 16;
        let src = vec![100u8; 256];
        let src_off = (stride * 4) as usize;
        let mut dst = [0u8; 64];
        put_8tap_scaled_8bpc(
            &mut dst, 8, &src, src_off, stride,
            4, 4, 8 << 6, 8 << 6, 1024, 1024, 0,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 100);
            }
        }
    }

    #[test]
    fn test_prep_8tap_scaled_identity() {
        let stride: isize = 16;
        let src = vec![128u8; 256];
        let src_off = (stride * 4) as usize;
        let mut tmp = [0i16; 64];
        prep_8tap_scaled_8bpc(
            &mut tmp, 8, &src, src_off, stride,
            4, 4, 0, 0, 1024, 1024, 0,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(tmp[y * 8 + x], (128 << 4) as i16);
            }
        }
    }

    #[test]
    fn test_put_bilin_scaled_identity() {
        let stride: isize = 16;
        let src = vec![128u8; 128];
        let src_off = (stride * 2) as usize;
        let mut dst = [0u8; 64];
        put_bilin_scaled_8bpc(
            &mut dst, 8, &src, src_off, stride,
            4, 4, 0, 0, 1024, 1024,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 128);
            }
        }
    }

    #[test]
    fn test_put_bilin_scaled_uniform_subpel() {
        let stride: isize = 16;
        let src = vec![100u8; 128];
        let src_off = (stride * 2) as usize;
        let mut dst = [0u8; 64];
        put_bilin_scaled_8bpc(
            &mut dst, 8, &src, src_off, stride,
            4, 4, 8 << 6, 8 << 6, 1024, 1024,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 8 + x], 100);
            }
        }
    }

    #[test]
    fn test_prep_bilin_scaled_identity() {
        let stride: isize = 16;
        let src = vec![128u8; 128];
        let src_off = (stride * 2) as usize;
        let mut tmp = [0i16; 64];
        prep_bilin_scaled_8bpc(
            &mut tmp, 8, &src, src_off, stride,
            4, 4, 0, 0, 1024, 1024,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(tmp[y * 8 + x], (128 << 4) as i16);
            }
        }
    }

    #[test]
    fn test_opfl_derive_mv_uniform() {
        let p0 = vec![128u8; 64 * 16];
        let p1 = vec![128u8; 64 * 16];
        let mut out = [OpflRegressionData::default(); 4];
        opfl_derive_mv_8bpc(&mut out, &p0, 64, &p1, 64, 16, 16, 8, [1, -1]);
        for r in &out {
            assert_eq!(r.suw, 0);
            assert_eq!(r.svw, 0);
        }
    }

    #[test]
    fn test_opfl_derive_mv_output_count() {
        let p0 = vec![100u8; 64 * 16];
        let p1 = vec![100u8; 64 * 16];
        let mut out = [OpflRegressionData::default(); 8];
        opfl_derive_mv_8bpc(&mut out, &p0, 64, &p1, 64, 32, 16, 8, [1, -1]);
        for r in &out[..4] {
            assert_eq!(r.su2, 64);
        }
    }

    #[test]
    fn test_opfl_derive_mv_nonzero_diff() {
        let p0 = vec![100u8; 64 * 8];
        let mut p1 = vec![100u8; 64 * 8];
        for x in 4..8 { p1[2 * 64 + x] = 200; }
        let mut out = [OpflRegressionData::default(); 1];
        opfl_derive_mv_8bpc(&mut out, &p0, 64, &p1, 64, 8, 8, 8, [1, -1]);
        assert_ne!(out[0].suw, 0);
    }

    #[test]
    fn test_opfl_derive_mv_bs4() {
        let p0 = vec![128u8; 64 * 8];
        let p1 = vec![128u8; 64 * 8];
        let mut out = [OpflRegressionData::default(); 4];
        opfl_derive_mv_8bpc(&mut out, &p0, 64, &p1, 64, 8, 8, 4, [1, -1]);
        for r in &out {
            assert_eq!(r.su2, 16);
            assert_eq!(r.suv, 0);
        }
    }
}

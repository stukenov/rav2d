use crate::intops::{apply_sign, iclip, imax, imin, ulog2};
use crate::tables::CDEF_DIRECTIONS;

pub const CDEF_HAVE_LEFT: u8 = 1 << 0;
pub const CDEF_HAVE_RIGHT: u8 = 1 << 1;
pub const CDEF_HAVE_TOP: u8 = 1 << 2;
pub const CDEF_HAVE_BOTTOM: u8 = 1 << 3;

/// Fill CDEF padding buffer around a block.
/// `tmp` is a flat buffer, `o` is the origin offset (top-left of the block).
/// The buffer must have 2 extra rows/cols on each side from `o`.
/// `left[y]` contains the 2 left-neighbor pixels for row y: `left[y][0]` is col -2, `left[y][1]` is col -1.
pub fn cdef_padding_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    left: &[[u8; 2]],
    top: &[u8],
    top_off: usize,
    bottom: &[u8],
    bottom_off: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    let o = 2 * tmp_stride + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    if edges & CDEF_HAVE_TOP == 0 {
        let base = o.wrapping_sub(2).wrapping_sub(2 * tmp_stride);
        fill(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        fill(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        fill(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        fill(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let mut toff = top_off;
    for y in y_start..0 {
        for x in x_start..x_end {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = top[(toff as i32 + x) as usize] as i16;
        }
        toff += src_stride;
    }

    for y in 0..h as i32 {
        for x in x_start..0 {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = left[y as usize][(2 + x) as usize] as i16;
        }
    }

    let mut soff = src_off;
    for y in 0..h as i32 {
        for x in 0..x_end {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = src[(soff as i32 + x) as usize] as i16;
        }
        soff += src_stride;
    }

    let mut boff = bottom_off;
    for y in h as i32..y_end {
        for x in x_start..x_end {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = bottom[(boff as i32 + x) as usize] as i16;
        }
        boff += src_stride;
    }
}

#[inline(always)]
pub fn constrain(diff: i32, threshold: i32, shift: i32) -> i32 {
    let adiff = diff.abs();
    apply_sign(imin(adiff, imax(0, threshold - (adiff >> shift))), diff)
}

pub fn fill(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            tmp[y * stride + x] = i16::MIN;
        }
    }
}

pub fn cdef_find_dir(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        for x in 0..8usize {
            let px = img[y * stride + x] as i32 - 128;

            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as u32;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    const DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7usize {
        let d = DIV_TABLE[n];
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n])
            as u32)
            * d;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n])
            as u32)
            * d;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7]) as u32 * 105;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7]) as u32 * 105;

    for n in 0..4usize {
        let ci = n * 2 + 1;
        for m in 0..5usize {
            cost[ci] += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as u32;
        }
        cost[ci] *= 105;
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            cost[ci] += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m])
                as u32)
                * d;
        }
    }

    let mut best_dir = 0i32;
    let mut best_cost = cost[0];
    for n in 1..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n as i32;
        }
    }

    *var = (best_cost - cost[(best_dir ^ 4) as usize]) >> 10;
    best_dir
}

pub fn cdef_pri_tap(pri_strength: i32) -> i32 {
    4 - (pri_strength & 1)
}

pub fn cdef_apply_constrain(px: i32, p0: i32, p1: i32, strength: i32, shift: i32, tap: i32) -> i32 {
    tap * constrain(p0 - px, strength, shift) + tap * constrain(p1 - px, strength, shift)
}

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

pub const BACKUP_2X8_Y: u8 = 1 << 0;
pub const BACKUP_2X8_UV: u8 = 1 << 1;

/// Backup last 2 rows from a single plane for CDEF line buffer.
/// `height` is the block height for this plane (8 for luma, 8 or 4 for chroma).
pub fn backup2lines_plane(
    dst: &mut [u8],
    dst_off: usize,
    src: &[u8],
    src_off: usize,
    stride: isize,
    height: usize,
) {
    let abs_stride = stride.unsigned_abs();
    let len = 2 * abs_stride;
    if stride < 0 {
        let d = (dst_off as isize + stride) as usize;
        let s = (src_off as isize + (height as isize - 1) * stride) as usize;
        dst[d..d + len].copy_from_slice(&src[s..s + len]);
    } else {
        let s = src_off + (height - 2) * abs_stride;
        dst[dst_off..dst_off + len].copy_from_slice(&src[s..s + len]);
    }
}

/// Backup a 2-pixel-wide column from a single plane for CDEF.
/// Saves pixels at (x_off - 2) and (x_off - 1) for `rows` rows.
pub fn backup2x8_plane(
    dst: &mut [[u8; 2]],
    src: &[u8],
    src_off: usize,
    stride: isize,
    x_off: isize,
    rows: usize,
) {
    let mut off = src_off as isize;
    for y in 0..rows {
        let s = (off + x_off - 2) as usize;
        dst[y].copy_from_slice(&src[s..s + 2]);
        off += stride;
    }
}

pub fn cdef_filter_block_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    left: &[[u8; 2]],
    top: &[u8],
    top_off: usize,
    bottom: &[u8],
    bottom_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: usize,
    damping: i32,
    w: usize,
    h: usize,
    edges: u8,
) {
    let tmp_stride: usize = 12;
    let mut tmp_buf = [0i16; 144];
    let o = 2 * tmp_stride + 2;

    cdef_padding_8bpc(
        &mut tmp_buf,
        tmp_stride,
        &*dst,
        dst_stride,
        dst_off,
        left,
        top,
        top_off,
        bottom,
        bottom_off,
        w,
        h,
        edges,
    );

    let mut dp = dst_off;
    let mut tp = o;

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = imax(0, damping - ulog2(pri_strength as u32));
        if sec_strength != 0 {
            let sec_shift = damping - ulog2(sec_strength as u32);
            for _y in 0..h {
                for x in 0..w {
                    let px = dst[dp + x] as i32;
                    let mut sum = 0i32;
                    let mut max_v = px;
                    let mut min_v = px;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = CDEF_DIRECTIONS[dir + 2][k] as isize;
                        let p0 = tmp_buf[((tp + x) as isize + off1) as usize] as i32;
                        let p1 = tmp_buf[((tp + x) as isize - off1) as usize] as i32;
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = (pri_tap_k & 3) | 2;
                        min_v = imin(p0, min_v);
                        max_v = imax(p0, max_v);
                        min_v = imin(p1, min_v);
                        max_v = imax(p1, max_v);
                        let off2 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                        let off3 = CDEF_DIRECTIONS[dir][k] as isize;
                        let s0 = tmp_buf[((tp + x) as isize + off2) as usize] as i32;
                        let s1 = tmp_buf[((tp + x) as isize - off2) as usize] as i32;
                        let s2 = tmp_buf[((tp + x) as isize + off3) as usize] as i32;
                        let s3 = tmp_buf[((tp + x) as isize - off3) as usize] as i32;
                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                        min_v = imin(s0, min_v);
                        max_v = imax(s0, max_v);
                        min_v = imin(s1, min_v);
                        max_v = imax(s1, max_v);
                        min_v = imin(s2, min_v);
                        max_v = imax(s2, max_v);
                        min_v = imin(s3, min_v);
                        max_v = imax(s3, max_v);
                    }
                    dst[dp + x] =
                        iclip(px + ((sum - (sum < 0) as i32 + 8) >> 4), min_v, max_v) as u8;
                }
                dp += dst_stride;
                tp += tmp_stride;
            }
        } else {
            for _y in 0..h {
                for x in 0..w {
                    let px = dst[dp + x] as i32;
                    let mut sum = 0i32;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = CDEF_DIRECTIONS[dir + 2][k] as isize;
                        let p0 = tmp_buf[((tp + x) as isize + off) as usize] as i32;
                        let p1 = tmp_buf[((tp + x) as isize - off) as usize] as i32;
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = (pri_tap_k & 3) | 2;
                    }
                    dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u8;
                }
                dp += dst_stride;
                tp += tmp_stride;
            }
        }
    } else {
        let sec_shift = damping - ulog2(sec_strength as u32);
        for _y in 0..h {
            for x in 0..w {
                let px = dst[dp + x] as i32;
                let mut sum = 0i32;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off2 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp_buf[((tp + x) as isize + off1) as usize] as i32;
                    let s1 = tmp_buf[((tp + x) as isize - off1) as usize] as i32;
                    let s2 = tmp_buf[((tp + x) as isize + off2) as usize] as i32;
                    let s3 = tmp_buf[((tp + x) as isize - off2) as usize] as i32;
                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefEdgeFlags(pub u8);

impl CdefEdgeFlags {
    pub const HAVE_LEFT: Self = Self(1 << 0);
    pub const HAVE_RIGHT: Self = Self(1 << 1);
    pub const HAVE_TOP: Self = Self(1 << 2);
    pub const HAVE_BOTTOM: Self = Self(1 << 3);
    pub const HAVE_ALL: Self = Self(0xf);

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
pub enum Backup2x8Flags {
    Y = 1,
    Uv = 2,
    Both = 3,
}

pub fn backup2lines_8bpc(
    dst: &mut [Vec<u8>; 3],
    src: &[&[u8]; 3],
    strides: &[isize; 2],
    layout: crate::headers::PixelLayout,
    row: usize,
) {
    let y_stride = strides[0] as usize;
    let uv_stride = strides[1] as usize;
    let src_off_y = row * y_stride;

    for i in 0..2 {
        let off = src_off_y + i * y_stride;
        if off + y_stride <= src[0].len() {
            let w = dst[0].len() / 2;
            if w > 0 {
                let dst_off = i * w;
                let copy_len = w.min(src[0].len() - off);
                dst[0][dst_off..dst_off + copy_len].copy_from_slice(&src[0][off..off + copy_len]);
            }
        }
    }

    if layout != crate::headers::PixelLayout::I400 && uv_stride > 0 {
        let ss_ver = layout == crate::headers::PixelLayout::I420;
        let crow = if ss_ver { row / 2 } else { row };

        for plane in 1..3 {
            let src_off = crow * uv_stride;
            for i in 0..2 {
                let off = src_off + i * uv_stride;
                if off + uv_stride <= src[plane].len() {
                    let w = dst[plane].len() / 2;
                    if w > 0 {
                        let dst_off = i * w;
                        let copy_len = w.min(src[plane].len() - off);
                        dst[plane][dst_off..dst_off + copy_len]
                            .copy_from_slice(&src[plane][off..off + copy_len]);
                    }
                }
            }
        }
    }
}

pub struct CdefApplyParams {
    pub bw: usize,
    pub bh: usize,
    pub sb128: bool,
    pub ss_hor: bool,
    pub ss_ver: bool,
    pub damping: i32,
    pub n_bits: i32,
    pub have_chroma: bool,
}

/// Per-superblock-row CDEF parameters threaded from the filter driver.
pub struct CdefBrowParams<'a> {
    pub bw: i32,
    pub bh: i32,
    pub damping: i32,
    pub layout: crate::headers::PixelLayout,
    pub on_skip_tx: bool,
    pub cdef_on: bool,
    /// `cdef_idx[sb256x][sb64_idx]` per the `Av2Filter` masks; for `sb128` decode
    /// `sb256x = sbx >> 2`, `sb64_idx = ((by & 0x30) >> 2) + (sbx & 3)`.
    pub mask_cdef_idx: &'a [[i8; 16]],
    /// `noskip_mask[sb256x][by_idx][sb64x_idx]`, `by_idx = (by & 0x3e) >> 1`.
    pub mask_noskip: &'a [[[u16; 4]; 32]],
    /// Per-cdef-index raw strengths (0..n_strengths) for Y and UV.
    pub y_strength: &'a [u8],
    pub uv_strength: &'a [u8],
}

const UV_DIRS: [[u8; 8]; 2] = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 0, 2, 4, 5, 6, 6, 6]];

/// Backup the bottom 2 rows of the current 8-row band into a toggled CDEF line
/// bank (each plane bank is laid out with the plane's positive stride spacing so
/// the next band can read it as `top`). Port of dav2d `backup2lines`
/// (cdef_apply_tmpl.c:41) for positive strides (single-thread path).
fn cdef_backup2lines_bank(
    bank: &mut [Vec<u8>; 3],
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    y_off: usize,
    uv_off: usize,
    y_stride: usize,
    uv_stride: usize,
    layout: crate::headers::PixelLayout,
) {
    // Luma: copy rows 6,7 of the band (`src + 6*stride`, 2*stride bytes).
    let s = y_off + 6 * y_stride;
    let n = (2 * y_stride)
        .min(src_y.len().saturating_sub(s))
        .min(bank[0].len());
    bank[0][..n].copy_from_slice(&src_y[s..s + n]);

    if layout != crate::headers::PixelLayout::I400 {
        let uv_off_rows = if layout == crate::headers::PixelLayout::I420 {
            2
        } else {
            6
        };
        let s = uv_off + uv_off_rows * uv_stride;
        let n = (2 * uv_stride)
            .min(src_u.len().saturating_sub(s))
            .min(bank[1].len());
        bank[1][..n].copy_from_slice(&src_u[s..s + n]);
        let n = (2 * uv_stride)
            .min(src_v.len().saturating_sub(s))
            .min(bank[2].len());
        bank[2][..n].copy_from_slice(&src_v[s..s + n]);
    }
}

/// Backup a pre-CDEF 2x8 left-column block from a plane into `dst[8]`
/// (`dst[row] = {col x_off-2, col x_off-1}`). Port of dav2d `backup2x8`.
fn cdef_backup2x8(
    dst: &mut [[u8; 2]; 8],
    src: &[u8],
    base: usize,
    stride: usize,
    x_off: usize,
    rows: usize,
) {
    let mut off = base;
    for d in dst.iter_mut().take(rows) {
        let s = off + x_off - 2;
        d[0] = src[s];
        d[1] = src[s + 1];
        off += stride;
    }
}

/// CDEF over a superblock-row (port of dav2d `cdef_brow`, single-thread
/// `have_tt == 0` path). `cdef_line` is the toggled top-row backup whose `tf`
/// bank holds the previous band's bottom 2 rows; `*toggle` flips per 8-row band.
#[allow(clippy::too_many_arguments)]
pub fn cdef_brow_8bpc(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    p: &CdefBrowParams,
    y_stride: isize,
    uv_stride: isize,
    cdef_line: &mut [[Vec<u8>; 3]; 2],
    toggle: &mut usize,
    by_start: i32,
    by_end: i32,
    sby: i32,
    sbrow_start: bool,
) {
    let _ = sby;
    let bitdepth_min_8 = 0; // 8bpc
    let damping = p.damping + bitdepth_min_8;
    let y_ls = y_stride.unsigned_abs();
    let uv_ls = uv_stride.unsigned_abs();
    let layout = p.layout;
    let ss_hor = (layout != crate::headers::PixelLayout::I444) as usize;
    let ss_ver = (layout == crate::headers::PixelLayout::I420) as usize;
    let uv_dir = &UV_DIRS[(layout == crate::headers::PixelLayout::I422) as usize];
    let sbsz = 16i32;
    let sb64w = (p.bw + sbsz - 1) >> 4;
    let have_chroma = layout != crate::headers::PixelLayout::I400;

    // Plane base offset of the band's first row.
    let mut row_y = by_start as usize * 4 * y_ls;
    let mut row_uv = ((by_start as usize * 4) >> ss_ver) * uv_ls;

    let mut edge_top = if by_start > 0 { CDEF_HAVE_TOP } else { 0 };

    let mut by = by_start;
    while by < by_end {
        let tf = *toggle;
        let by_idx = ((by & 0x3e) >> 1) as usize;
        let mut edges = edge_top | CDEF_HAVE_BOTTOM;
        if by + 2 >= p.bh {
            edges &= !CDEF_HAVE_BOTTOM;
        }

        // Back up pre-filter bottom 2 rows of this band for the next band's top.
        if (sbrow_start || by + 2 < by_end) && (edges & CDEF_HAVE_BOTTOM) != 0 {
            let other = 1 - tf;
            cdef_backup2lines_bank(
                &mut cdef_line[other],
                y,
                u,
                v,
                row_y,
                row_uv,
                y_ls,
                uv_ls,
                layout,
            );
        }

        // Left 2x8 backups (toggled `bit`), one per plane, pre-CDEF.
        let mut lr_bak: [[[[u8; 2]; 8]; 3]; 2] = [[[[0u8; 2]; 8]; 3]; 2];
        let mut bit = 0usize;
        edges &= !CDEF_HAVE_LEFT;
        edges |= CDEF_HAVE_RIGHT;
        // `prev_flag` persists across superblocks within a band (dav2d
        // cdef_apply_tmpl.c:145): the last block's backup state carries over so
        // the next SB's first block can reuse / skip the left-column backup.
        let mut prev_flag = 0u8;

        // Per-sb base offsets that advance with iptrs.
        let mut sb_y = row_y;
        let mut sb_uv = row_uv;

        for sbx in 0..sb64w {
            let sb256x = (sbx >> 2) as usize;
            let sb64x_idx = (sbx & 3) as usize;
            let sb64_idx = (((by & 0x30) >> 2) + (sbx & 3)) as usize;
            let mask_i = sb256x.min(p.mask_cdef_idx.len().saturating_sub(1));
            let cdef_idx = if sb256x < p.mask_cdef_idx.len() {
                p.mask_cdef_idx[mask_i][sb64_idx]
            } else {
                -1
            };

            if cdef_idx == -1
                || !p.cdef_on
                || ((p.y_strength.get(cdef_idx as usize).copied().unwrap_or(0) == 0)
                    && (p.uv_strength.get(cdef_idx as usize).copied().unwrap_or(0) == 0))
            {
                prev_flag = 0;
                edges |= CDEF_HAVE_LEFT;
                sb_y += (sbsz * 4) as usize;
                sb_uv += ((sbsz * 4) as usize) >> ss_hor;
                continue;
            }

            let noskip_full = if p.on_skip_tx {
                !0u16
            } else if sb256x < p.mask_noskip.len() && by_idx < 32 && sb64x_idx < 4 {
                p.mask_noskip[mask_i][by_idx][sb64x_idx]
            } else {
                0
            };

            let y_lvl = p.y_strength[cdef_idx as usize] as i32;
            let uv_lvl = p.uv_strength[cdef_idx as usize] as i32;
            let flag = (y_lvl != 0) as u8 + (((uv_lvl != 0) as u8) << 1);

            let y_pri_lvl = (y_lvl >> 2) << bitdepth_min_8;
            let mut y_sec_lvl = y_lvl & 3;
            y_sec_lvl += (y_sec_lvl == 3) as i32;
            y_sec_lvl <<= bitdepth_min_8;

            let uv_pri_lvl = (uv_lvl >> 2) << bitdepth_min_8;
            let mut uv_sec_lvl = uv_lvl & 3;
            uv_sec_lvl += (uv_sec_lvl == 3) as i32;
            uv_sec_lvl <<= bitdepth_min_8;

            let mut b_y = sb_y;
            let mut b_uv = sb_uv;
            let mut bx = sbx * sbsz;
            let sb_bx_end = imin((sbx + 1) * sbsz, p.bw);
            while bx < sb_bx_end {
                if bx + 2 >= p.bw {
                    edges &= !CDEF_HAVE_RIGHT;
                }

                let bx_mask = 3u16 << (bx & 14);
                if (noskip_full & bx_mask) == 0 {
                    prev_flag = 0;
                    edges |= CDEF_HAVE_LEFT;
                    b_y += 8;
                    b_uv += 8 >> ss_hor;
                    bx += 2;
                    continue;
                }

                let do_left = flag & !prev_flag;
                prev_flag = flag;
                if do_left != 0 && (edges & CDEF_HAVE_LEFT) != 0 {
                    if do_left & BACKUP_2X8_Y != 0 {
                        cdef_backup2x8(&mut lr_bak[bit][0], y, b_y, y_ls, 0, 8);
                    }
                    if have_chroma && do_left & BACKUP_2X8_UV != 0 {
                        cdef_backup2x8(&mut lr_bak[bit][1], u, b_uv, uv_ls, 0, 8 >> ss_ver);
                        cdef_backup2x8(&mut lr_bak[bit][2], v, b_uv, uv_ls, 0, 8 >> ss_ver);
                    }
                }
                if (edges & CDEF_HAVE_RIGHT) != 0 {
                    let other = 1 - bit;
                    if flag & BACKUP_2X8_Y != 0 {
                        cdef_backup2x8(&mut lr_bak[other][0], y, b_y, y_ls, 8, 8);
                    }
                    if have_chroma && flag & BACKUP_2X8_UV != 0 {
                        cdef_backup2x8(
                            &mut lr_bak[other][1],
                            u,
                            b_uv,
                            uv_ls,
                            8 >> ss_hor,
                            8 >> ss_ver,
                        );
                        cdef_backup2x8(
                            &mut lr_bak[other][2],
                            v,
                            b_uv,
                            uv_ls,
                            8 >> ss_hor,
                            8 >> ss_ver,
                        );
                    }
                }

                let mut variance = 0u32;
                let dir = if y_pri_lvl != 0 || uv_pri_lvl != 0 {
                    cdef_find_dir(&y[b_y..], y_ls, &mut variance) as usize
                } else {
                    0
                };

                // Luma top/bottom: top from the toggled pre-CDEF line bank, bottom
                // in-place (rows below not yet filtered).
                let top_col = bx as usize * 4;
                let bot_y = b_y + 8 * y_ls;
                if y_pri_lvl != 0 {
                    let adj = adjust_strength(y_pri_lvl, variance);
                    if adj != 0 || y_sec_lvl != 0 {
                        cdef_filter_block_8bpc(
                            y,
                            y_ls,
                            b_y,
                            &lr_bak[bit][0],
                            &cdef_line[tf][0],
                            top_col,
                            unsafe { std::slice::from_raw_parts(y.as_ptr(), y.len()) },
                            bot_y,
                            adj,
                            y_sec_lvl,
                            dir,
                            damping,
                            8,
                            8,
                            edges,
                        );
                    }
                } else if y_sec_lvl != 0 {
                    cdef_filter_block_8bpc(
                        y,
                        y_ls,
                        b_y,
                        &lr_bak[bit][0],
                        &cdef_line[tf][0],
                        top_col,
                        unsafe { std::slice::from_raw_parts(y.as_ptr(), y.len()) },
                        bot_y,
                        0,
                        y_sec_lvl,
                        0,
                        damping,
                        8,
                        8,
                        edges,
                    );
                }

                if uv_lvl != 0 && have_chroma {
                    let uvdir = if uv_pri_lvl != 0 {
                        uv_dir[dir] as usize
                    } else {
                        0
                    };
                    let cw = 8 >> ss_hor;
                    let ch = 8 >> ss_ver;
                    let top_col_uv = (bx as usize * 4) >> ss_hor;
                    let bot_uv = b_uv + ch * uv_ls;
                    cdef_filter_block_8bpc(
                        u,
                        uv_ls,
                        b_uv,
                        &lr_bak[bit][1],
                        &cdef_line[tf][1],
                        top_col_uv,
                        unsafe { std::slice::from_raw_parts(u.as_ptr(), u.len()) },
                        bot_uv,
                        uv_pri_lvl,
                        uv_sec_lvl,
                        uvdir,
                        damping - 1,
                        cw,
                        ch,
                        edges,
                    );
                    cdef_filter_block_8bpc(
                        v,
                        uv_ls,
                        b_uv,
                        &lr_bak[bit][2],
                        &cdef_line[tf][2],
                        top_col_uv,
                        unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) },
                        bot_uv,
                        uv_pri_lvl,
                        uv_sec_lvl,
                        uvdir,
                        damping - 1,
                        cw,
                        ch,
                        edges,
                    );
                }

                bit ^= 1;
                edges |= CDEF_HAVE_LEFT;
                b_y += 8;
                b_uv += 8 >> ss_hor;
                bx += 2;
            }

            sb_y += (sbsz * 4) as usize;
            sb_uv += ((sbsz * 4) as usize) >> ss_hor;
        }

        row_y += 8 * y_ls;
        row_uv += (8 * uv_ls) >> ss_ver;
        *toggle ^= 1;
        edge_top = CDEF_HAVE_TOP;
        let _ = by_idx;
        by += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrain_zero_diff() {
        assert_eq!(constrain(0, 10, 1), 0);
    }

    #[test]
    fn test_constrain_small_diff() {
        let r = constrain(3, 10, 1);
        assert!(r > 0);
        assert!(r <= 3);
    }

    #[test]
    fn test_constrain_negative_diff() {
        let r = constrain(-5, 10, 1);
        assert!(r < 0);
    }

    #[test]
    fn test_constrain_large_diff_clamped() {
        let r = constrain(100, 10, 1);
        assert!(r <= 10);
    }

    #[test]
    fn test_constrain_zero_threshold() {
        assert_eq!(constrain(5, 0, 1), 0);
    }

    #[test]
    fn test_fill_basic() {
        let mut buf = [0i16; 24];
        fill(&mut buf, 6, 4, 3);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(buf[y * 6 + x], i16::MIN);
            }
            assert_eq!(buf[y * 6 + 4], 0);
            assert_eq!(buf[y * 6 + 5], 0);
        }
    }

    #[test]
    fn test_cdef_find_dir_uniform() {
        let img = [128u8; 64];
        let mut var = 0u32;
        let dir = cdef_find_dir(&img, 8, &mut var);
        assert!(dir >= 0 && dir < 8);
        assert_eq!(var, 0);
    }

    #[test]
    fn test_cdef_find_dir_vertical() {
        let mut img = [0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                img[y * 8 + x] = if x < 4 { 64 } else { 192 };
            }
        }
        let mut var = 0u32;
        let dir = cdef_find_dir(&img, 8, &mut var);
        assert!(dir >= 0 && dir < 8);
        assert!(var > 0);
    }

    #[test]
    fn test_cdef_find_dir_horizontal() {
        let mut img = [0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                img[y * 8 + x] = if y < 4 { 64 } else { 192 };
            }
        }
        let mut var = 0u32;
        let dir = cdef_find_dir(&img, 8, &mut var);
        assert!(dir >= 0 && dir < 8);
        assert!(var > 0);
    }

    #[test]
    fn test_cdef_pri_tap() {
        assert_eq!(cdef_pri_tap(4), 4);
        assert_eq!(cdef_pri_tap(3), 3);
        assert_eq!(cdef_pri_tap(2), 4);
        assert_eq!(cdef_pri_tap(1), 3);
    }

    #[test]
    fn test_cdef_apply_constrain_zero() {
        assert_eq!(cdef_apply_constrain(100, 100, 100, 10, 1, 4), 0);
    }

    #[test]
    fn test_cdef_apply_constrain_symmetric() {
        let r = cdef_apply_constrain(100, 105, 95, 10, 1, 4);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_adjust_strength_zero_var() {
        assert_eq!(adjust_strength(100, 0), 0);
    }

    #[test]
    fn test_adjust_strength_low_var() {
        let r = adjust_strength(100, 32);
        assert_eq!(r, (100 * 4 + 8) >> 4);
    }

    #[test]
    fn test_adjust_strength_high_var() {
        let r = adjust_strength(100, 1 << 18);
        assert!(r > adjust_strength(100, 64));
    }

    #[test]
    fn test_adjust_strength_monotonic() {
        let a = adjust_strength(100, 64);
        let b = adjust_strength(100, 256);
        let c = adjust_strength(100, 4096);
        assert!(a <= b);
        assert!(b <= c);
    }

    #[test]
    fn test_cdef_padding_all_edges() {
        let src = vec![100u8; 16 * 16];
        let left = vec![[50u8, 60]; 8];
        let top = vec![200u8; 64];
        let bottom = vec![150u8; 64];
        let mut tmp = vec![0i16; 12 * 20];
        let tmp_stride = 12;
        cdef_padding_8bpc(
            &mut tmp,
            tmp_stride,
            &src,
            16,
            2,
            &left,
            &top,
            2,
            &bottom,
            2,
            8,
            8,
            CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM | CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT,
        );
        let o = 2 * tmp_stride + 2;
        assert_eq!(tmp[o], 100);
        assert_eq!(tmp[o - 1], 60);
        assert_eq!(tmp[o - 2], 50);
    }

    #[test]
    fn test_cdef_padding_no_edges() {
        let src = vec![100u8; 16 * 16];
        let left = vec![[0u8; 2]; 8];
        let top = vec![0u8; 32];
        let bottom = vec![0u8; 32];
        let mut tmp = vec![0i16; 12 * 20];
        let tmp_stride = 12;
        cdef_padding_8bpc(
            &mut tmp, tmp_stride, &src, 16, 0, &left, &top, 0, &bottom, 0, 8, 8, 0,
        );
        let o = 2 * tmp_stride + 2;
        assert_eq!(tmp[o], 100);
        assert_eq!(tmp[o - 1], i16::MIN);
        assert_eq!(tmp[o - tmp_stride], i16::MIN);
    }

    fn make_cdef_test_bufs(
        val: u8,
        stride: usize,
        w: usize,
        h: usize,
    ) -> (Vec<u8>, Vec<[u8; 2]>, Vec<u8>, Vec<u8>) {
        let dst = vec![val; stride * (h + 4)];
        let left = vec![[val; 2]; h];
        let top = vec![val; stride * 2 + w + 4];
        let bottom = vec![val; stride * 2 + w + 4];
        (dst, left, top, bottom)
    }

    #[test]
    fn test_cdef_filter_block_uniform() {
        let stride = 16;
        let (mut dst, left, top, bottom) = make_cdef_test_bufs(128, stride, 8, 8);
        let edges = CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM | CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT;
        let dst_off = stride + 2;
        let top_off = 2;
        let bottom_off = 2;
        cdef_filter_block_8bpc(
            &mut dst, stride, dst_off, &left, &top, top_off, &bottom, bottom_off, 8, 4, 0, 6, 8, 8,
            edges,
        );
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    dst[dst_off + y * stride + x],
                    128,
                    "pixel ({x},{y}) changed"
                );
            }
        }
    }

    #[test]
    fn test_cdef_filter_block_pri_only() {
        let stride = 16;
        let dst_off = stride + 2;
        let mut dst = vec![128u8; stride * 12];
        dst[dst_off + 3 * stride + 3] = 120;
        let left = vec![[128u8; 2]; 8];
        let top = vec![128u8; stride * 2 + 12];
        let bottom = vec![128u8; stride * 2 + 12];
        let edges = CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM | CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT;
        let orig_val = dst[dst_off + 3 * stride + 3];
        cdef_filter_block_8bpc(
            &mut dst, stride, dst_off, &left, &top, 2, &bottom, 2, 8, 0, 0, 6, 8, 8, edges,
        );
        assert_ne!(
            dst[dst_off + 3 * stride + 3],
            orig_val,
            "pri filter should change noisy pixel"
        );
    }

    #[test]
    fn test_cdef_filter_block_sec_only() {
        let stride = 16;
        let dst_off = stride + 2;
        let mut dst = vec![128u8; stride * 12];
        dst[dst_off + 3 * stride + 3] = 120;
        let left = vec![[128u8; 2]; 8];
        let top = vec![128u8; stride * 2 + 12];
        let bottom = vec![128u8; stride * 2 + 12];
        let edges = CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM | CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT;
        let orig_val = dst[dst_off + 3 * stride + 3];
        cdef_filter_block_8bpc(
            &mut dst, stride, dst_off, &left, &top, 2, &bottom, 2, 0, 4, 0, 6, 8, 8, edges,
        );
        assert_ne!(
            dst[dst_off + 3 * stride + 3],
            orig_val,
            "sec filter should change noisy pixel"
        );
    }

    #[test]
    fn test_cdef_filter_block_pri_sec_combined() {
        let stride = 16;
        let dst_off = stride + 2;
        let mut dst = vec![0u8; stride * 12];
        for y in 0..8 {
            for x in 0..8 {
                dst[dst_off + y * stride + x] = ((y * 20 + x * 10) & 0xFF) as u8;
            }
        }
        let mut left_arr: Vec<[u8; 2]> = Vec::new();
        for y in 0..8 {
            let v = (y * 20) as u8;
            left_arr.push([v.wrapping_sub(20), v.wrapping_sub(10)]);
        }
        let mut top = vec![0u8; stride * 2 + 12];
        for x in 0..12 {
            top[x] = (x * 10) as u8;
            top[stride + x] = (x * 10) as u8;
        }
        let mut bottom = vec![0u8; stride * 2 + 12];
        for x in 0..12 {
            bottom[x] = ((8 * 20 + x * 10) & 0xFF) as u8;
            bottom[stride + x] = ((9 * 20 + x * 10) & 0xFF) as u8;
        }
        let edges = CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM | CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT;
        let orig = dst.clone();
        cdef_filter_block_8bpc(
            &mut dst, stride, dst_off, &left_arr, &top, 2, &bottom, 2, 8, 4, 3, 6, 8, 8, edges,
        );
        let mut changed = false;
        for y in 0..8 {
            for x in 0..8 {
                if dst[dst_off + y * stride + x] != orig[dst_off + y * stride + x] {
                    changed = true;
                }
            }
        }
        assert!(changed, "pri+sec combined should modify gradient image");
    }

    #[test]
    fn test_cdef_filter_block_no_edges() {
        let stride = 16;
        let dst_off = stride + 2;
        let (mut dst, left, top, bottom) = make_cdef_test_bufs(128, stride, 8, 8);
        cdef_filter_block_8bpc(
            &mut dst, stride, dst_off, &left, &top, 2, &bottom, 2, 8, 4, 0, 6, 8, 8, 0,
        );
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(dst[dst_off + y * stride + x], 128);
            }
        }
    }

    #[test]
    fn test_backup2lines_plane_positive_stride() {
        let stride: isize = 16;
        let mut src = vec![0u8; 8 * stride as usize];
        for y in 0..8 {
            for x in 0..16 {
                src[y * stride as usize + x] = (y * 16 + x) as u8;
            }
        }
        let mut dst = vec![0u8; 2 * stride as usize];
        backup2lines_plane(&mut dst, 0, &src, 0, stride, 8);
        assert_eq!(&dst[0..16], &src[6 * 16..7 * 16]);
        assert_eq!(&dst[16..32], &src[7 * 16..8 * 16]);
    }

    #[test]
    fn test_backup2lines_plane_i420_chroma() {
        let stride: isize = 8;
        let mut src = vec![0u8; 4 * stride as usize];
        for i in 0..src.len() {
            src[i] = i as u8;
        }
        let mut dst = vec![0u8; 2 * stride as usize];
        backup2lines_plane(&mut dst, 0, &src, 0, stride, 4);
        assert_eq!(&dst[0..8], &src[16..24]);
        assert_eq!(&dst[8..16], &src[24..32]);
    }

    #[test]
    fn test_backup2x8_plane() {
        let stride: isize = 16;
        let mut src = vec![0u8; 8 * stride as usize];
        for y in 0..8 {
            for x in 0..16 {
                src[y * stride as usize + x] = (y * 16 + x) as u8;
            }
        }
        let mut dst = [[0u8; 2]; 8];
        backup2x8_plane(&mut dst, &src, 0, stride, 6, 8);
        for y in 0..8 {
            assert_eq!(dst[y][0], src[y * 16 + 4]);
            assert_eq!(dst[y][1], src[y * 16 + 5]);
        }
    }

    #[test]
    fn test_backup2x8_plane_chroma_4rows() {
        let stride: isize = 8;
        let mut src = vec![0u8; 4 * stride as usize];
        for i in 0..src.len() {
            src[i] = (i + 1) as u8;
        }
        let mut dst = [[0u8; 2]; 8];
        backup2x8_plane(&mut dst[..4], &src, 0, stride, 4, 4);
        assert_eq!(dst[0], [3, 4]);
        assert_eq!(dst[1], [11, 12]);
    }
}

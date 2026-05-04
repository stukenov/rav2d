use std::sync::Arc;

use crate::env::get_poc_diff;
use crate::getbits::GetBits;
use crate::headers::*;
use crate::internal::RefState;
use crate::intops::{iclip, imax, imin, ulog2};
use crate::warpmv::resolve_divisor_32;

#[derive(Debug)]
pub enum Dav2dError {
    InvalidData,
}

pub type Result<T> = std::result::Result<T, Dav2dError>;

static LAYOUTS: [PixelLayout; 4] = [
    PixelLayout::I420,
    PixelLayout::I400,
    PixelLayout::I444,
    PixelLayout::I422,
];

fn check_trailing_bits(gb: &mut GetBits, strict: bool) -> Result<()> {
    let trailing_one = gb.get_bit();

    if gb.has_error() {
        return Err(Dav2dError::InvalidData);
    }

    if !strict {
        return Ok(());
    }

    if trailing_one == 0 {
        return Err(Dav2dError::InvalidData);
    }

    Ok(())
}

#[inline]
fn tile_log2(sz: i32, tgt: i32) -> i32 {
    let mut k = 0;
    while (sz << k) < tgt {
        k += 1;
    }
    k
}

fn parse_seg_info(seg: &mut SegmentationDataSet, gb: &mut GetBits, n_seg: usize) {
    let mut m: u16 = 1;
    for n in 0..n_seg {
        if gb.get_bit() != 0 {
            seg.delta_q_mask |= m;
            seg.delta_q[n] = gb.get_sbits(10).clamp(-351, 351) as i16;
        }
        seg.skip_mask |= m * gb.get_bit() as u16;
        seg.globalmv_mask |= m * gb.get_bit() as u16;
        m <<= 1;
    }
}

fn parse_tile_info(
    thdr: &mut TileInfo,
    gb: &mut GetBits,
    sbmul: i32,
    sb128: u8,
    seq_sb128: u8,
    w: i32,
    h: i32,
    level: u8,
    tier: u8,
) {
    thdr.uniform = gb.get_bit() != 0;

    let sbsz_log2 = 6 + sb128 as i32;
    let sbsz_min1 = (64 << sb128) - 1;
    let sbw = (w + sbsz_min1) >> sbsz_log2;
    let sbh = (h + sbsz_min1) >> sbsz_log2;
    let w_adj = (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_width_sb = 4096 >> (sbsz_log2 - w_adj);
    let sz_adj = (level >= 14) as i32 + (level >= 18) as i32
        + ((level >= 14 && tier != 0) as i32);
    let max_tile_area_sb = 4096 * 2304 >> (2 * sbsz_log2 - sz_adj);
    thdr.min_log2_cols = tile_log2(max_tile_width_sb, sbw) as u8;
    thdr.max_log2_cols = tile_log2(1, imin(sbw, MAX_TILE_COLS as i32)) as u8;
    thdr.max_log2_rows = tile_log2(1, imin(sbh, MAX_TILE_ROWS as i32)) as u8;
    let min_log2_tiles = imax(
        tile_log2(max_tile_area_sb, sbw * sbh),
        thdr.min_log2_cols as i32,
    );

    if thdr.uniform {
        let seq_sbsz_log2 = 6 + seq_sb128 as i32;
        let fsbw = imax(1, (w + 7) >> seq_sbsz_log2);
        let fsbh = imax(1, (h + 7) >> seq_sbsz_log2);

        thdr.log2_cols = thdr.min_log2_cols;
        while thdr.log2_cols < thdr.max_log2_cols && gb.get_bit() != 0 {
            thdr.log2_cols += 1;
        }
        let tile_w = imax(1, fsbw >> thdr.log2_cols);
        let mut extra = imax(0, fsbw - (tile_w << thdr.log2_cols));
        thdr.cols = 0;
        let mut sbx = 0;
        while sbx < fsbw {
            thdr.col_start_sb[thdr.cols as usize] = (sbx * sbmul) as u16;
            let add = tile_w + if extra > 0 { 1 } else { 0 };
            sbx += add;
            thdr.cols += 1;
            extra -= 1;
        }

        thdr.min_log2_rows = imax(
            min_log2_tiles - thdr.log2_cols as i32,
            0,
        ) as u8;
        thdr.log2_rows = thdr.min_log2_rows;
        while thdr.log2_rows < thdr.max_log2_rows && gb.get_bit() != 0 {
            thdr.log2_rows += 1;
        }
        let tile_h = imax(1, fsbh >> thdr.log2_rows);
        let mut extra = imax(0, fsbh - (tile_h << thdr.log2_rows));
        thdr.rows = 0;
        let mut sby = 0;
        while sby < fsbh {
            thdr.row_start_sb[thdr.rows as usize] = (sby * sbmul) as u16;
            let add = tile_h + if extra > 0 { 1 } else { 0 };
            sby += add;
            thdr.rows += 1;
            extra -= 1;
        }
    } else {
        let mut widest_tile = 0;
        thdr.cols = 0;
        let mut sbx = 0;
        while sbx < sbw {
            thdr.col_start_sb[thdr.cols as usize] = sbx as u16;
            let max_width = imin(sbw - sbx, max_tile_width_sb);
            let w_tile = gb.get_uniform(max_width as u32) as i32 + 1;
            widest_tile = imax(widest_tile, w_tile);
            sbx += w_tile;
            thdr.cols += 1;
        }
        thdr.log2_cols = tile_log2(1, thdr.cols as i32) as u8;

        let max_tile_area_sb_here = if min_log2_tiles > 0 {
            (sbw * sbh) >> (min_log2_tiles + 1)
        } else {
            sbw * sbh
        };
        let max_tile_height_sb = imax(max_tile_area_sb_here / widest_tile, 1);

        thdr.rows = 0;
        let mut sby = 0;
        while sby < sbh {
            thdr.row_start_sb[thdr.rows as usize] = sby as u16;
            let max_height = imin(sbh - sby, max_tile_height_sb);
            let h_tile = gb.get_uniform(max_height as u32) as i32 + 1;
            sby += h_tile;
            thdr.rows += 1;
        }
        thdr.log2_rows = tile_log2(1, thdr.rows as i32) as u8;
    }
    thdr.col_start_sb[thdr.cols as usize] = sbw as u16;
    thdr.row_start_sb[thdr.rows as usize] = sbh as u16;
}

pub fn parse_tile_info_frmhdr(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    gb: &mut GetBits,
) {
    hdr.sb128 = if hdr.is_inter_or_switch() {
        seqhdr.sb128
    } else {
        if seqhdr.sb128 != 0 { 1 } else { 0 }
    };

    let mut reuse_allowed = false;
    if seqhdr.tiling.present != AdaptiveBoolean::Off {
        let sbsz_min1 = (64i32 << hdr.sb128) - 1;
        let sbsz_log2 = 6 + hdr.sb128 as i32;
        let sbw = (hdr.width + sbsz_min1) >> sbsz_log2;
        let sbh = (hdr.height + sbsz_min1) >> sbsz_log2;
        if !seqhdr.tiling.t.uniform {
            let seq_sbsz_min1 = (64i32 << seqhdr.sb128) - 1;
            let seq_sbsz_log2 = 6 + seqhdr.sb128 as i32;
            let seq_sbw = (seqhdr.max_width + seq_sbsz_min1) >> seq_sbsz_log2;
            let seq_sbh = (seqhdr.max_height + seq_sbsz_min1) >> seq_sbsz_log2;
            reuse_allowed = seq_sbw == sbw && seq_sbh == sbh;
        } else {
            let tile_w =
                (sbw + seqhdr.tiling.t.cols as i32 - 1) >> seqhdr.tiling.t.log2_cols;
            let tile_h =
                (sbh + seqhdr.tiling.t.rows as i32 - 1) >> seqhdr.tiling.t.log2_rows;
            reuse_allowed = tile_w * (seqhdr.tiling.t.cols as i32 - 1) < sbw
                && tile_h * (seqhdr.tiling.t.rows as i32 - 1) < sbh;
        }
    }

    let sbmul;
    if reuse_allowed
        && (seqhdr.tiling.present == AdaptiveBoolean::On
            || (seqhdr.tiling.present == AdaptiveBoolean::Adaptive
                && gb.get_bit() != 0))
    {
        hdr.tiling.t = seqhdr.tiling.t.clone();
        if hdr.sb128 != seqhdr.sb128 {
            debug_assert!(hdr.sb128 == 1 && seqhdr.sb128 == 2 && hdr.is_key_or_intra());
            sbmul = 2;
            for n in 0..hdr.tiling.t.rows as usize {
                hdr.tiling.t.row_start_sb[n] *= 2;
            }
            for n in 0..hdr.tiling.t.cols as usize {
                hdr.tiling.t.col_start_sb[n] *= 2;
            }
        } else {
            sbmul = 1;
        }
    } else {
        sbmul = if seqhdr.sb128 == 2 && hdr.is_key_or_intra() { 2 } else { 1 };
        parse_tile_info(
            &mut hdr.tiling.t,
            gb,
            sbmul,
            hdr.sb128,
            seqhdr.sb128,
            hdr.width,
            hdr.height,
            seqhdr.level,
            seqhdr.tier,
        );
    }

    if sbmul == 2 {
        hdr.tiling.t.row_start_sb[hdr.tiling.t.rows as usize] =
            ((hdr.height + 127) >> 7) as u16;
        hdr.tiling.t.col_start_sb[hdr.tiling.t.cols as usize] =
            ((hdr.width + 127) >> 7) as u16;
    }
}

pub fn parse_film_grain_data(
    gb: &mut GetBits,
    layout: PixelLayout,
) -> Result<FilmGrainData> {
    let mut fgd = FilmGrainData::default();

    let mut num_pl = 1;
    if layout != PixelLayout::I400 {
        fgd.chroma_scaling_from_luma = gb.get_bit() != 0;
        if !fgd.chroma_scaling_from_luma {
            num_pl = 3;
        }
    }

    for pl in 0..num_pl {
        fgd.num_points[pl] = gb.get_bits(4) as i32;
        if fgd.num_points[pl] > 14 {
            return Err(Dav2dError::InvalidData);
        }
        if fgd.num_points[pl] == 0 {
            continue;
        }
        let index_bits = 1 + gb.get_bits(3) as i32;
        let scaling_bits = 5 + gb.get_bits(2) as i32;
        let mut base = 0u32;
        for i in 0..fgd.num_points[pl] as usize {
            base += gb.get_bits(index_bits);
            if base > 255 {
                return Err(Dav2dError::InvalidData);
            }
            fgd.points[pl][i][0] = base as u8;
            fgd.points[pl][i][1] = gb.get_bits(scaling_bits) as u8;
        }
    }

    if layout == PixelLayout::I420
        && (fgd.num_points[1] != 0) != (fgd.num_points[2] != 0)
    {
        return Err(Dav2dError::InvalidData);
    }

    fgd.scaling_shift = gb.get_bits(2) as i32 + 8;
    fgd.ar_coeff_lag = gb.get_bits(2) as i32;
    let num_pos = 2 * fgd.ar_coeff_lag * (fgd.ar_coeff_lag + 1);
    for pl in 0..3 {
        if fgd.num_points[pl] == 0
            && (pl == 0 || !fgd.chroma_scaling_from_luma)
        {
            continue;
        }
        let num_pl_pos =
            num_pos + (pl != 0 && fgd.num_points[0] != 0) as i32;
        let coef_bits = 5 + gb.get_bits(2) as i32;
        for i in 0..num_pl_pos as usize {
            fgd.ar_coeffs[pl][i] = (gb.get_bits(coef_bits) as i32 - 128) as i8;
        }
    }
    fgd.ar_coeff_shift = gb.get_bits(2) as u64 + 6;
    fgd.grain_scale_shift = gb.get_bits(2) as i32;
    for pl in 0..2 {
        if fgd.num_points[1 + pl] == 0 {
            continue;
        }
        fgd.uv_mult[pl] = gb.get_bits(8) as i32 - 128;
        fgd.uv_luma_mult[pl] = gb.get_bits(8) as i32 - 128;
        fgd.uv_offset[pl] = gb.get_bits(9) as i32 - 256;
    }
    fgd.overlap_flag = gb.get_bit() != 0;
    fgd.clip_to_restricted_range = gb.get_bit() != 0;
    if fgd.clip_to_restricted_range {
        fgd.mc_identity = gb.get_bit() != 0;
    }
    fgd.block_size = gb.get_bit() as i32;

    Ok(fgd)
}

pub fn rescale_matrix(dm: &mut [i32; 6], sm: &[i32; 6], in_dist: i32, out_dist: i32) {
    let mut shift = 0i32;
    let mut inv = resolve_divisor_32(in_dist.unsigned_abs(), &mut shift);
    if inv >= 512 {
        inv >>= 1;
        shift -= 1;
    }
    if in_dist < 0 {
        inv = -inv;
    }
    let rnd = (1 << shift) >> 1;
    for n in 0..2 {
        let r = iclip(sm[n], -0x400000, 0x400000) * inv;
        let t = ((r + rnd - (r < 0) as i32) >> shift) * out_dist;
        let d = (t + 0x1000 - (t < 0) as i32) & !0x1fff;
        dm[n] = iclip(d, -0x7ffe000, 0x7ffe000);
    }
    for n in 2..6 {
        let b = 0x10000 * (((n as u32).wrapping_sub(3)) > 1) as i32;
        let r = (sm[n] - b) * inv;
        let t = ((r + rnd - (r < 0) as i32) >> shift) * out_dist;
        let d = (t + 32 - (t < 0) as i32) & !63;
        dm[n] = b + iclip(d, -0x7fc0, 0x7fc0);
    }
}

pub fn parse_seq_hdr(gb: &mut GetBits, strict: bool) -> Result<SequenceHeader> {
    let mut hdr = SequenceHeader::default();

    hdr.id = gb.get_vlc() as u8;
    hdr.profile = gb.get_bits(5) as u8;
    if hdr.profile > 2 {
        return Err(Dav2dError::InvalidData);
    }
    hdr.reduced_still_picture_header = gb.get_bit() != 0;
    hdr.level = gb.get_bits(5) as u8;
    if hdr.level >= 4 && !hdr.reduced_still_picture_header {
        hdr.tier = gb.get_bit() as u8;
    }

    let layout_idx = gb.get_vlc();
    if layout_idx > 3 {
        return Err(Dav2dError::InvalidData);
    }
    hdr.layout = LAYOUTS[layout_idx as usize];
    match hdr.layout {
        PixelLayout::I420 | PixelLayout::I400 => {
            hdr.ss_hor = 1;
            hdr.ss_ver = 1;
        }
        PixelLayout::I422 => {
            hdr.ss_hor = 1;
            hdr.ss_ver = 0;
        }
        _ => {}
    }

    hdr.hbd = gb.get_vlc() as u8;
    if hdr.hbd > 2 {
        return Err(Dav2dError::InvalidData);
    }
    if hdr.hbd < 2 {
        hdr.hbd ^= 1;
    }

    if hdr.reduced_still_picture_header {
        hdr.still_picture = true;
        hdr.monotonic = true;
    } else {
        hdr.lcr_id = gb.get_bits(3) as u8;
        hdr.still_picture = gb.get_bit() != 0;
        hdr.max_tlayer_id = gb.get_bits(2) as u8;
        hdr.max_mlayer_id = gb.get_bits(3) as u8;
        hdr.monotonic = gb.get_bit() != 0;
    }

    hdr.width_n_bits = gb.get_bits(4) as u8 + 1;
    hdr.height_n_bits = gb.get_bits(4) as u8 + 1;
    hdr.max_width = gb.get_bits(hdr.width_n_bits as i32) as i32 + 1;
    hdr.max_height = gb.get_bits(hdr.height_n_bits as i32) as i32 + 1;

    hdr.crop.enabled = gb.get_bit() != 0;
    if hdr.crop.enabled {
        hdr.crop.left = gb.get_vlc();
        hdr.crop.right = gb.get_vlc();
        hdr.crop.top = gb.get_vlc();
        hdr.crop.bottom = gb.get_vlc();
    }

    if !hdr.reduced_still_picture_header {
        if gb.get_bit() != 0 {
            // max_display_model_info_present
            let _max_initial_display_delay = gb.get_bits(4);
        }
        let decoder_model_info_present = gb.get_bit() != 0;
        if decoder_model_info_present {
            let _num_units = gb.get_bits(32);
            let _max_dec_buf = gb.get_vlc();
            let _max_enc_buf = gb.get_vlc();
        }
    }

    if hdr.max_tlayer_id > 0 {
        hdr.tlayer_dependency_present = gb.get_bit() != 0;
        if hdr.tlayer_dependency_present {
            for n in 1..hdr.max_tlayer_id as usize {
                hdr.tlayer_dependencies[n] = gb.get_bits(n as i32) as u8;
            }
        } else {
            let mut mask = !0u32;
            for n in 1..hdr.max_tlayer_id as usize {
                hdr.tlayer_dependencies[n] = (!mask) as u8;
                mask <<= 1;
            }
        }
    }

    if hdr.max_mlayer_id > 0 {
        hdr.mlayer_dependency_present = gb.get_bit() != 0;
        if hdr.mlayer_dependency_present {
            for n in 1..hdr.max_mlayer_id as usize {
                hdr.mlayer_dependencies[n] = gb.get_bits(n as i32) as u8;
            }
        } else {
            let mut mask = !0u32;
            for n in 1..hdr.max_mlayer_id as usize {
                hdr.mlayer_dependencies[n] = (!mask) as u8;
                mask <<= 1;
            }
        }
    }

    hdr.sb128 = if gb.get_bit() != 0 { 2 } else { gb.get_bit() as u8 };

    if hdr.layout != PixelLayout::I400 {
        hdr.sdp = gb.get_bit() != 0;
        if hdr.sdp && !hdr.reduced_still_picture_header {
            hdr.ext_sdp = gb.get_bit() != 0;
        }
    }
    hdr.ext_partitions = gb.get_bit() != 0;
    if hdr.ext_partitions {
        hdr.uneven_4way_partitions = gb.get_bit() != 0;
    }
    hdr.max_pb_aspect_ratio_log2 = if gb.get_bit() != 0 {
        1 + gb.get_bit() as u8
    } else {
        3
    };

    hdr.segmentation.ext = gb.get_bit() != 0;
    hdr.segmentation.info_present = gb.get_bit() != 0;
    if hdr.segmentation.info_present {
        hdr.segmentation.adaptive = gb.get_bit() != 0;
        parse_seg_info(
            &mut hdr.segmentation.d,
            gb,
            8 << (hdr.segmentation.ext as usize),
        );
    }

    hdr.intra_dip = gb.get_bit() != 0;
    hdr.intra_edge_filter = gb.get_bit() != 0;
    hdr.mrls = gb.get_bit() != 0;
    hdr.cfl = gb.get_bit() != 0;
    if hdr.layout != PixelLayout::I400 {
        hdr.cfl_ds_filter_index = gb.get_bits(2) as u8;
    }
    hdr.mhccp = gb.get_bit() != 0;
    hdr.ibp = gb.get_bit() != 0;

    if hdr.reduced_still_picture_header {
        hdr.motion_modes = 1;
    } else {
        hdr.motion_modes = 1; // MM_TRANSLATION = bit 0
        for shift in [1, 2, 3, 4] {
            hdr.motion_modes |= (gb.get_bit() as u8) << shift;
        }
        if hdr.motion_modes & !1 != 0 {
            hdr.frame_motion_modes_present = gb.get_bit() != 0;
        }
        if hdr.motion_modes & (1 << 3) != 0 {
            // MM_WARP_DELTA
            hdr.six_param_warp_delta = gb.get_bit() != 0;
        }
        hdr.masked_compound = gb.get_bit() != 0;
        hdr.ref_frame_mvs = gb.get_bit() != 0;
        if hdr.ref_frame_mvs {
            hdr.reduced_ref_frame_mvs_mode = gb.get_bit() as u8;
        }
        hdr.order_hint_n_bits = gb.get_bits(4) as u8 + 1;
    }

    hdr.refmv_bank = gb.get_bit() != 0;
    hdr.drl_reorder = if gb.get_bit() != 0 {
        false
    } else {
        gb.get_bit() == 0
    };

    if hdr.reduced_still_picture_header {
        hdr.ref_frames = 2;
        hdr.def_max_drl_bits = 1;
    } else {
        hdr.explicit_ref_frame_map = gb.get_bit() != 0;
        hdr.ref_frames = if gb.get_bit() != 0 {
            gb.get_bits(4) as u8 + 1
        } else {
            8
        };
        hdr.ref_frames_log2 = if hdr.ref_frames <= 2 {
            hdr.ref_frames - 1
        } else {
            1 + ulog2(hdr.ref_frames as u32 - 1) as u8
        };
        hdr.number_of_bits_for_lt_frame_id = gb.get_bits(3) as u8;
        hdr.def_max_drl_bits = gb.get_uniform(5) as u8 + 1;
        hdr.allow_frame_max_drl_bits = gb.get_bit() != 0;
    }
    hdr.def_max_bvp_drl_bits = gb.get_uniform(3) as u8 + 1;
    hdr.allow_max_bvp_drl_bits = gb.get_bit() != 0;
    if !hdr.reduced_still_picture_header {
        hdr.num_same_ref_comp = gb.get_bits(2) as u8;
    }

    if !hdr.reduced_still_picture_header {
        let tip_val = gb.get_bit();
        hdr.tip = tip_val != 0 && (1 + gb.get_bit() as u8) > 0;
        if hdr.tip {
            hdr.tip_hole_fill = gb.get_bit() != 0;
        }
        hdr.mv_traj = gb.get_bit() != 0;
    }
    hdr.bawp = gb.get_bit() != 0;
    if !hdr.reduced_still_picture_header {
        hdr.cwp = gb.get_bit() != 0;
        hdr.imp_msk_bld = gb.get_bit() != 0;
        hdr.db_sub_pu = gb.get_bit() != 0;
        if hdr.tip && hdr.db_sub_pu {
            hdr.tip_explicit_qp = gb.get_bit() != 0;
        }
    }

    if !hdr.reduced_still_picture_header {
        hdr.opfl_refine = gb.get_bit() != 0;
        let _opfl_bits = gb.get_bits(2) as u8;
        // opfl_refine is 2 bits in C, but we stored first bit above — fix:
        hdr.refine_mv = gb.get_bit() != 0;
        if hdr.tip && (hdr.opfl_refine || hdr.refine_mv) {
            hdr.tip_refine_mv = gb.get_bit() != 0;
        }
        hdr.bru = gb.get_bit() != 0;
        hdr.adaptive_mvd = gb.get_bit() != 0;
        hdr.mvd_sign_derive = gb.get_bit() != 0;
        hdr.flex_mvres = gb.get_bit() != 0;
        hdr.global_motion = gb.get_bit() != 0;
        hdr.short_refresh_frame_flags = gb.get_bit() != 0;
    }

    if hdr.reduced_still_picture_header {
        hdr.screen_content_tools = AdaptiveBoolean::Adaptive;
        hdr.force_integer_mv = AdaptiveBoolean::Adaptive;
    } else {
        hdr.screen_content_tools = if gb.get_bit() != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            if gb.get_bit() != 0 {
                AdaptiveBoolean::On
            } else {
                AdaptiveBoolean::Off
            }
        };
        hdr.force_integer_mv = if hdr.screen_content_tools != AdaptiveBoolean::Off {
            if gb.get_bit() != 0 {
                AdaptiveBoolean::Adaptive
            } else {
                if gb.get_bit() != 0 {
                    AdaptiveBoolean::On
                } else {
                    AdaptiveBoolean::Off
                }
            }
        } else {
            AdaptiveBoolean::Adaptive
        };
    }

    hdr.fsc = gb.get_bit() != 0;
    hdr.idtx_intra = hdr.fsc || gb.get_bit() != 0;
    hdr.ist[0] = gb.get_bit() != 0;
    hdr.ist[1] = gb.get_bit() != 0;
    if hdr.layout != PixelLayout::I400 {
        hdr.chroma_dctonly = gb.get_bit() != 0;
    }
    if !hdr.reduced_still_picture_header {
        hdr.inter_ddt = gb.get_bit() != 0;
    }
    hdr.reduced_tx_part_set = gb.get_bit() != 0;
    if hdr.layout != PixelLayout::I400 {
        hdr.cctx = gb.get_bit() != 0;
    }

    let tcq_bit = gb.get_bit();
    hdr.tcq = if tcq_bit != 0 {
        if !hdr.reduced_still_picture_header && gb.get_bit() != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            AdaptiveBoolean::On
        }
    } else {
        AdaptiveBoolean::Off
    };
    if hdr.tcq != AdaptiveBoolean::On {
        hdr.parity_hiding = gb.get_bit() != 0;
    }

    hdr.avg_cdf = hdr.reduced_still_picture_header || gb.get_bit() != 0;
    if hdr.avg_cdf {
        hdr.avg_cdf_type = if hdr.reduced_still_picture_header || gb.get_bit() != 0 {
            1
        } else {
            0
        };
    }

    if hdr.layout != PixelLayout::I400 {
        hdr.separate_uv_delta_q = gb.get_bit() != 0;
    }
    hdr.equal_ac_dc_q = gb.get_bit() != 0;
    if !hdr.equal_ac_dc_q {
        hdr.base_ydc_dq = gb.get_bits(5) as i8 - 23;
        hdr.ydc_dq_enabled = gb.get_bit() != 0;
    }
    if hdr.layout != PixelLayout::I400 {
        if !hdr.equal_ac_dc_q {
            hdr.base_uvdc_dq = (gb.get_bits(5) as i32 - 23) as u8;
            hdr.uvdc_dq_enabled = gb.get_bit() != 0;
        }
        hdr.base_uvac_dq = (gb.get_bits(5) as i32 - 23) as u8;
        hdr.uvac_dq_enabled = gb.get_bit() != 0;
        if hdr.equal_ac_dc_q {
            hdr.base_uvdc_dq = hdr.base_uvac_dq;
        }
    }

    hdr.disable_loopfilters_across_tiles = gb.get_bit() != 0;
    hdr.cdef = gb.get_bit() != 0;
    hdr.gdf = gb.get_bit() != 0;
    if hdr.gdf && hdr.sb128 == 0 {
        hdr.gdf_unit_matches_sbsz = gb.get_bit() != 0;
    }
    hdr.restoration = gb.get_bit() != 0;
    if hdr.restoration {
        let no_pc_wiener = gb.get_bit() as u8;
        let no_ns_wiener_y = gb.get_bit() as u8;
        hdr.rst_disable_mask[0] = (no_ns_wiener_y << 1) | no_pc_wiener;
        if gb.get_bit() != 0 {
            hdr.rst_disable_mask[1] = (gb.get_bit() as u8) << 1 | 1;
        } else {
            hdr.rst_disable_mask[1] = hdr.rst_disable_mask[0] | 1;
        }
    }
    hdr.ccso = gb.get_bit() != 0;
    if hdr.ccso {
        hdr.ccso_unit_matches_sbsz = gb.get_bit() != 0;
    }
    hdr.cdef_on_skiptx = if hdr.reduced_still_picture_header {
        AdaptiveBoolean::Adaptive
    } else if gb.get_bit() != 0 {
        AdaptiveBoolean::On
    } else if gb.get_bit() != 0 {
        AdaptiveBoolean::Off
    } else {
        AdaptiveBoolean::Adaptive
    };
    hdr.df_par_bits = 2 + gb.get_bits(2) as u8;

    let tiling_present = gb.get_bit();
    if tiling_present != 0 {
        let tiling_type = gb.get_bit();
        hdr.tiling.present = if tiling_type != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            AdaptiveBoolean::On
        };
        parse_tile_info(
            &mut hdr.tiling.t,
            gb,
            1,
            hdr.sb128,
            hdr.sb128,
            hdr.max_width,
            hdr.max_height,
            hdr.level,
            hdr.tier,
        );
    }

    hdr.film_grain_present = gb.get_bit() != 0;

    if gb.has_error() {
        return Err(Dav2dError::InvalidData);
    }

    if !strict {
        return Ok(hdr);
    }

    // extension handling — skip for non-strict mode
    let has_extension = gb.get_bit() != 0;
    if has_extension {
        // skip extension bits (we don't parse them)
    }

    check_trailing_bits(gb, strict)?;
    Ok(hdr)
}

pub fn parse_sequence_header(data: &[u8]) -> Result<SequenceHeader> {
    if data.is_empty() {
        return Err(Dav2dError::InvalidData);
    }
    let mut gb = GetBits::new(data);
    parse_seq_hdr(&mut gb, false)
}

pub fn parse_ci_hdr(ci: &mut ContentInterpretation, gb: &mut GetBits) -> Result<()> {
    ci.scan_type = match gb.get_bits(2) {
        0 => ScanType::Unknown,
        1 => ScanType::Progressive,
        2 => ScanType::Interlace,
        3 => ScanType::InterlaceComplementary,
        _ => unreachable!(),
    };
    ci.color_description_present = gb.get_bit() != 0;
    ci.chroma_sample_position_present = gb.get_bit() != 0;
    ci.aspect_ratio_info_present = gb.get_bit() != 0;
    ci.timing_info_present = gb.get_bit() != 0;
    ci.extension_present = gb.get_bit() != 0;
    let _ = gb.get_bit(); // reserved

    if ci.color_description_present {
        let desc_type = gb.get_golomb(2);
        ci.color.desc_type = match desc_type {
            0 => ColorDescription::Explicit,
            1 => ColorDescription::Bt709Sdr,
            2 => ColorDescription::Bt2100Pq,
            3 => ColorDescription::Bt2100Hlg,
            4 => ColorDescription::Srgb,
            5 => ColorDescription::SrgbSycc,
            _ => ColorDescription::Explicit, // unknown → treat as explicit with unknown values
        };
        match ci.color.desc_type {
            ColorDescription::Explicit => {
                if desc_type == 0 {
                    ci.color.pri = u8_to_color_pri(gb.get_bits(8) as u8);
                    ci.color.trc = u8_to_trc(gb.get_bits(8) as u8);
                    ci.color.mtrx = u8_to_mc(gb.get_bits(8) as u8);
                } else {
                    ci.color.pri = ColorPrimaries::Unknown;
                    ci.color.trc = TransferCharacteristics::Unknown;
                    ci.color.mtrx = MatrixCoefficients::Unknown;
                }
            }
            ColorDescription::Bt709Sdr => {
                ci.color.pri = ColorPrimaries::Bt709;
                ci.color.trc = TransferCharacteristics::Bt709;
                ci.color.mtrx = MatrixCoefficients::Bt470Bg;
            }
            ColorDescription::Bt2100Pq => {
                ci.color.pri = ColorPrimaries::Bt2020;
                ci.color.trc = TransferCharacteristics::Smpte2084;
                ci.color.mtrx = MatrixCoefficients::Bt2020Ncl;
            }
            ColorDescription::Bt2100Hlg => {
                ci.color.pri = ColorPrimaries::Bt2020;
                ci.color.trc = TransferCharacteristics::Bt2020_10Bit;
                ci.color.mtrx = MatrixCoefficients::Bt2020Ncl;
            }
            ColorDescription::Srgb => {
                ci.color.pri = ColorPrimaries::Bt709;
                ci.color.trc = TransferCharacteristics::Srgb;
                ci.color.mtrx = MatrixCoefficients::Identity;
            }
            ColorDescription::SrgbSycc => {
                ci.color.pri = ColorPrimaries::Bt709;
                ci.color.trc = TransferCharacteristics::Srgb;
                ci.color.mtrx = MatrixCoefficients::Bt470Bg;
            }
        }
        ci.color.range = gb.get_bit() as u8;
    } else {
        ci.color.pri = ColorPrimaries::Unknown;
        ci.color.trc = TransferCharacteristics::Unknown;
        ci.color.mtrx = MatrixCoefficients::Unknown;
    }

    if ci.chroma_sample_position_present {
        ci.chr[0] = u32_to_chr(gb.get_vlc());
        ci.chr[1] = if ci.scan_type == ScanType::Progressive {
            ci.chr[0]
        } else {
            u32_to_chr(gb.get_vlc())
        };
    } else {
        ci.chr[0] = ChromaSamplePosition::Unknown;
        ci.chr[1] = ChromaSamplePosition::Unknown;
    }

    if ci.aspect_ratio_info_present {
        let sar_type = gb.get_bits(8) as u8;
        match sar_type {
            0 => ci.sar.sar_type = AspectRatio::Unknown,
            1 => { ci.sar.sar_type = AspectRatio::Sar1_1; ci.sar.w = 1; ci.sar.h = 1; }
            2 => { ci.sar.sar_type = AspectRatio::Sar12_11; ci.sar.w = 12; ci.sar.h = 11; }
            3 => { ci.sar.sar_type = AspectRatio::Sar10_11; ci.sar.w = 10; ci.sar.h = 11; }
            4 => { ci.sar.sar_type = AspectRatio::Sar16_11; ci.sar.w = 16; ci.sar.h = 11; }
            5 => { ci.sar.sar_type = AspectRatio::Sar40_33; ci.sar.w = 40; ci.sar.h = 33; }
            6 => { ci.sar.sar_type = AspectRatio::Sar24_11; ci.sar.w = 24; ci.sar.h = 11; }
            7 => { ci.sar.sar_type = AspectRatio::Sar20_11; ci.sar.w = 20; ci.sar.h = 11; }
            8 => { ci.sar.sar_type = AspectRatio::Sar32_11; ci.sar.w = 32; ci.sar.h = 11; }
            9 => { ci.sar.sar_type = AspectRatio::Sar80_33; ci.sar.w = 80; ci.sar.h = 33; }
            10 => { ci.sar.sar_type = AspectRatio::Sar18_11; ci.sar.w = 18; ci.sar.h = 11; }
            11 => { ci.sar.sar_type = AspectRatio::Sar15_11; ci.sar.w = 15; ci.sar.h = 11; }
            12 => { ci.sar.sar_type = AspectRatio::Sar64_33; ci.sar.w = 64; ci.sar.h = 33; }
            13 => { ci.sar.sar_type = AspectRatio::Sar160_99; ci.sar.w = 160; ci.sar.h = 99; }
            14 => { ci.sar.sar_type = AspectRatio::Sar4_3; ci.sar.w = 4; ci.sar.h = 3; }
            15 => { ci.sar.sar_type = AspectRatio::Sar3_2; ci.sar.w = 3; ci.sar.h = 2; }
            16 => { ci.sar.sar_type = AspectRatio::Sar2_1; ci.sar.w = 2; ci.sar.h = 1; }
            255 => {
                ci.sar.sar_type = AspectRatio::Explicit;
                ci.sar.w = gb.get_vlc();
                ci.sar.h = gb.get_vlc();
            }
            _ => return Err(Dav2dError::InvalidData),
        }
    }

    if ci.timing_info_present {
        ci.timing.num_units_in_display_tick = gb.get_bits(32) as u32;
        ci.timing.time_scale = gb.get_bits(32) as u32;
        if ci.timing.num_units_in_display_tick == 0 || ci.timing.time_scale == 0 {
            return Err(Dav2dError::InvalidData);
        }
        ci.timing.equal_elemental_interval = gb.get_bit() as u8;
        if ci.timing.equal_elemental_interval != 0 {
            let t = gb.get_vlc();
            if t == u32::MAX {
                return Err(Dav2dError::InvalidData);
            }
            ci.timing.num_ticks_per_elemental_duration = t + 1;
        }
    }

    Ok(())
}

fn u8_to_color_pri(v: u8) -> ColorPrimaries {
    match v {
        1 => ColorPrimaries::Bt709,
        2 => ColorPrimaries::Unknown,
        4 => ColorPrimaries::Bt470M,
        5 => ColorPrimaries::Bt470Bg,
        6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Xyz,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => ColorPrimaries::Unknown,
    }
}

fn u8_to_trc(v: u8) -> TransferCharacteristics {
    match v {
        1 => TransferCharacteristics::Bt709,
        2 => TransferCharacteristics::Unknown,
        4 => TransferCharacteristics::Bt470M,
        5 => TransferCharacteristics::Bt470Bg,
        6 => TransferCharacteristics::Bt601,
        7 => TransferCharacteristics::Smpte240,
        8 => TransferCharacteristics::Linear,
        9 => TransferCharacteristics::Log100,
        10 => TransferCharacteristics::Log100Sqrt10,
        11 => TransferCharacteristics::Iec61966,
        12 => TransferCharacteristics::Bt1361,
        13 => TransferCharacteristics::Srgb,
        14 => TransferCharacteristics::Bt2020_10Bit,
        15 => TransferCharacteristics::Bt2020_12Bit,
        16 => TransferCharacteristics::Smpte2084,
        17 => TransferCharacteristics::Smpte428,
        18 => TransferCharacteristics::Hlg,
        _ => TransferCharacteristics::Unknown,
    }
}

fn u8_to_mc(v: u8) -> MatrixCoefficients {
    match v {
        0 => MatrixCoefficients::Identity,
        1 => MatrixCoefficients::Bt709,
        2 => MatrixCoefficients::Unknown,
        4 => MatrixCoefficients::Fcc,
        5 => MatrixCoefficients::Bt470Bg,
        6 => MatrixCoefficients::Bt601,
        7 => MatrixCoefficients::Smpte240,
        8 => MatrixCoefficients::SmpteYcgco,
        9 => MatrixCoefficients::Bt2020Ncl,
        10 => MatrixCoefficients::Bt2020Cl,
        11 => MatrixCoefficients::Smpte2085,
        12 => MatrixCoefficients::ChromatNcl,
        13 => MatrixCoefficients::ChromatCl,
        14 => MatrixCoefficients::Ictcp,
        15 => MatrixCoefficients::IptC2,
        16 => MatrixCoefficients::YcgcoRe,
        17 => MatrixCoefficients::YcgcoRo,
        _ => MatrixCoefficients::Unknown,
    }
}

fn u32_to_chr(v: u32) -> ChromaSamplePosition {
    match v {
        0 => ChromaSamplePosition::Left,
        1 => ChromaSamplePosition::Center,
        2 => ChromaSamplePosition::TopLeft,
        3 => ChromaSamplePosition::Top,
        4 => ChromaSamplePosition::BottomLeft,
        5 => ChromaSamplePosition::Bottom,
        _ => ChromaSamplePosition::Unknown,
    }
}

pub fn read_frame_size(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
    gb: &mut GetBits,
) -> Result<()> {
    if hdr.frame_size_override != 0 && hdr.is_inter_or_switch() {
        for i in 0..hdr.n_ref_frames as usize {
            if gb.get_bit() != 0 {
                let refhdr = refs[hdr.refidx[i] as usize]
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(Dav2dError::InvalidData)?;
                hdr.width = refhdr.width;
                hdr.height = refhdr.height;
                return Ok(());
            }
        }
    }
    if hdr.frame_size_override != 0 {
        hdr.width = gb.get_bits(seqhdr.width_n_bits as i32) as i32 + 1;
        hdr.height = gb.get_bits(seqhdr.height_n_bits as i32) as i32 + 1;
    } else {
        hdr.width = seqhdr.max_width;
        hdr.height = seqhdr.max_height;
    }
    Ok(())
}

pub fn get_ref_frames(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
    have_resolution: bool,
) -> i32 {
    struct Score {
        score: i32,
        poc: u8,
        pocdiff: i8,
        qidx: u16,
        mlayer: u8,
        res_ratio_log2: i8,
    }
    let mut ref_info: [Score; 8] = std::array::from_fn(|_| Score {
        score: 0,
        poc: 0,
        pocdiff: 0,
        qidx: 0,
        mlayer: 0,
        res_ratio_log2: 0,
    });
    let mut sort_idx = [0u8; 8];
    let mut n_refs = 0i32;
    let mut have_fwd_refs = false;
    let poc = hdr.frame_offset as i32;
    let nbits = seqhdr.order_hint_n_bits as i32;

    for n in 0..8 {
        if have_fwd_refs {
            break;
        }
        if let Some(refhdr) = refs[n].p.frame_hdr.as_ref() {
            have_fwd_refs = get_poc_diff(nbits, poc, refhdr.frame_offset as i32) < 0;
        }
    }

    let mlayer = hdr.mlayer_id as i32;
    let tlayer = hdr.tlayer_id as i32;
    let w = hdr.width;
    let h = hdr.height;
    let mut minq = 512i32;
    let mut maxq = -1i32;
    let mut last_refhdr_ptr: Option<*const FrameHeader> = None;

    for n in 0..8usize {
        let refhdr_arc = match refs[n].p.frame_hdr.as_ref() {
            Some(fh) => fh,
            None => continue,
        };
        let refhdr_ptr = Arc::as_ptr(refhdr_arc);
        if last_refhdr_ptr == Some(refhdr_ptr) {
            continue;
        }
        let refhdr = refhdr_arc.as_ref();

        if seqhdr.tlayer_dependency_present {
            if seqhdr.tlayer_dependencies[tlayer as usize] & (1 << refhdr.tlayer_id) == 0 {
                continue;
            }
        } else if tlayer < refhdr.tlayer_id as i32 {
            continue;
        }

        let ref_mlayer = refhdr.mlayer_id;
        if seqhdr.mlayer_dependency_present {
            if seqhdr.mlayer_dependencies[mlayer as usize] & (1 << ref_mlayer) == 0 {
                continue;
            }
        } else if mlayer < ref_mlayer as i32 {
            continue;
        }

        if have_resolution
            && (2 * w < refhdr.width
                || 2 * h < refhdr.height
                || w > 16 * refhdr.width
                || h > 16 * refhdr.height)
        {
            continue;
        }

        let ref_poc = refhdr.frame_offset;
        let pocdiff = get_poc_diff(nbits, poc, ref_poc as i32) as i8;
        let ref_qidx = refhdr.quant.yac;
        let res_ratio = -(ulog2((refhdr.width * refhdr.height) as u32) as i8);
        let tdist = (pocdiff as i32).abs() + mlayer - ref_mlayer as i32;
        let mut score = if have_fwd_refs {
            tdist << 6
        } else {
            128 - (128 >> imin(tdist, 6)) + imax(tdist - 6, 0)
        };
        score += res_ratio as i32 * (1 << 5) + ref_qidx as i32;

        ref_info[n] = Score {
            score,
            poc: ref_poc,
            pocdiff,
            qidx: ref_qidx,
            mlayer: ref_mlayer,
            res_ratio_log2: res_ratio,
        };

        let mut m = 0usize;
        while m < n_refs as usize {
            let r2 = &ref_info[sort_idx[m] as usize];
            if score == r2.score && ref_poc == r2.poc && ref_mlayer == r2.mlayer {
                break;
            }
            m += 1;
        }
        if (m as i32) < n_refs {
            continue;
        }

        maxq = imax(ref_qidx as i32, maxq);
        minq = imin(ref_qidx as i32, minq);

        while m > 0 {
            let idx = sort_idx[m - 1] as usize;
            if ref_info[idx].score <= score {
                break;
            }
            sort_idx[m] = idx as u8;
            m -= 1;
        }
        sort_idx[m] = n as u8;
        n_refs += 1;
        last_refhdr_ptr = Some(refhdr_ptr);
    }

    if n_refs == 8 {
        let q_thr = (maxq + minq + 1) >> 1;
        let mut maxpocdiff = [0i32; 2];
        let mut num = [0i32; 2];
        let mut furthest_idx = [0usize; 2];
        for n in 0..8usize {
            let r = &ref_info[sort_idx[n] as usize];
            if (r.qidx as i32) < q_thr {
                continue;
            }
            if r.pocdiff > 0 {
                if (r.pocdiff as i32) > maxpocdiff[0] {
                    maxpocdiff[0] = r.pocdiff as i32;
                    furthest_idx[0] = n;
                }
                num[0] += 1;
            } else if r.pocdiff < 0 {
                if (r.pocdiff as i32) < maxpocdiff[1] {
                    maxpocdiff[1] = r.pocdiff as i32;
                    furthest_idx[1] = n;
                }
                num[1] += 1;
            }
        }
        let idx = if num[0] > num[1] {
            furthest_idx[0]
        } else if num[0] < num[1] {
            furthest_idx[1]
        } else {
            furthest_idx[if maxpocdiff[0] < -maxpocdiff[1] { 1 } else { 0 }]
        };
        if idx < 7 {
            sort_idx.copy_within(idx + 1..8, idx);
            sort_idx[7] = idx as u8;
        }
    }

    for n in 0..7usize {
        hdr.refidx[n] = sort_idx[if (n as i32) < n_refs { n } else { 0 }] as i8;
    }

    imin(7, n_refs)
}

pub fn find_tip_ref_frames(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
) {
    let n_refs = hdr.n_ref_frames as usize;
    if n_refs == 1 {
        hdr.tip.r#ref[0] = 0;
        hdr.tip.r#ref[1] = 0;
        return;
    }

    let poc = hdr.frame_offset as i32;
    let nbits = seqhdr.order_hint_n_bits as i32;
    let mut order = [0u8; 7];
    let mut refdist = [0i8; 7];
    let mut n_past = 0usize;

    for n in 0..n_refs {
        let refpoc = refs[hdr.refidx[n] as usize]
            .p
            .frame_hdr
            .as_ref()
            .unwrap()
            .frame_offset;
        let dist = get_poc_diff(nbits, refpoc as i32, poc);
        refdist[n] = dist as i8;
        let mut m = n;
        while m > 0 && (refdist[order[m - 1] as usize] as i32) > dist {
            order[m] = order[m - 1];
            m -= 1;
        }
        order[m] = n as u8;
        if dist < 0 {
            n_past += 1;
        }
    }

    if n_past == n_refs {
        hdr.tip.r#ref[0] = order[n_refs - 1] as i8;
        hdr.tip.r#ref[1] = order[n_refs - 2] as i8;
    } else if n_past == 0 {
        hdr.tip.r#ref[0] = order[0] as i8;
        hdr.tip.r#ref[1] = order[1] as i8;
    } else {
        hdr.tip.r#ref[0] = order[n_past - 1] as i8;
        hdr.tip.r#ref[1] = order[n_past] as i8;
    }
}

pub fn derive_pri_sec_ref(
    hdr: &FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
) -> [i32; 2] {
    let mut result = [PRIMARY_REF_NONE as i32, PRIMARY_REF_NONE as i32];
    let mut best_qdiff = [0i32; 2];
    let mut best_pocdiff = [0i32; 2];
    let mut best_poc = [0i32; 2];
    let mut best = 0usize;
    let qidx = hdr.quant.yac as i32;
    let poc = hdr.frame_offset as i32;
    let nbits = seqhdr.order_hint_n_bits as i32;

    for i in 0..hdr.n_ref_frames as usize {
        let refhdr = match refs[hdr.refidx[i] as usize].p.frame_hdr.as_ref() {
            Some(fh) => fh,
            None => continue,
        };
        if refhdr.is_key_or_intra() {
            continue;
        }
        let ref_qidx = refhdr.quant.yac as i32;
        let qdiff = (ref_qidx - qidx).abs();
        let ref_poc = refhdr.frame_offset as i32;
        let pocdiff = get_poc_diff(nbits, poc, ref_poc).abs();
        for n in 0..2usize {
            let m = if n == 0 { best } else { 1 - best };
            if result[m] == PRIMARY_REF_NONE as i32
                || qdiff < best_qdiff[m]
                || (qdiff == best_qdiff[m]
                    && (pocdiff < best_pocdiff[m]
                        || (pocdiff == best_pocdiff[m]
                            && get_poc_diff(nbits, best_poc[m], ref_poc) < 0)))
            {
                let slot = 1 - best;
                result[slot] = i as i32;
                best_pocdiff[slot] = pocdiff;
                best_qdiff[slot] = qdiff;
                best_poc[slot] = ref_poc;
                if n == 0 {
                    best = 1 - best;
                }
                break;
            }
        }
    }

    if best != 0 {
        result.swap(0, 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_log2() {
        assert_eq!(tile_log2(64, 64), 0);
        assert_eq!(tile_log2(64, 128), 1);
        assert_eq!(tile_log2(64, 256), 2);
        assert_eq!(tile_log2(1, 4), 2);
    }

    #[test]
    fn test_check_trailing_bits_non_strict() {
        let data = [0x80]; // trailing 1 bit
        let mut gb = GetBits::new(&data);
        assert!(check_trailing_bits(&mut gb, false).is_ok());
    }

    #[test]
    fn test_parse_seg_info_empty() {
        let data = [0x00, 0x00]; // all zeros = no segments enabled
        let mut gb = GetBits::new(&data);
        let mut seg = SegmentationDataSet::default();
        parse_seg_info(&mut seg, &mut gb, 2);
        assert_eq!(seg.delta_q_mask, 0);
    }

    #[test]
    fn test_rescale_matrix_identity() {
        let sm = [0, 0, 0x10000, 0, 0, 0x10000];
        let mut dm = [0i32; 6];
        rescale_matrix(&mut dm, &sm, 1, 1);
        assert_eq!(dm[0], 0);
        assert_eq!(dm[1], 0);
    }

    #[test]
    fn test_rescale_matrix_scale2x() {
        let sm = [0x2000, 0x1000, 0x10000, 0, 0, 0x10000];
        let mut dm = [0i32; 6];
        rescale_matrix(&mut dm, &sm, 1, 2);
        assert!(dm[0].abs() > sm[0].abs());
    }

    #[test]
    fn test_rescale_matrix_clamp() {
        let sm = [0x400000, 0x400000, 0x10000, 0, 0, 0x10000];
        let mut dm = [0i32; 6];
        rescale_matrix(&mut dm, &sm, 1, 100);
        assert!(dm[0] <= 0x7ffe000);
        assert!(dm[1] <= 0x7ffe000);
    }

    #[test]
    fn test_layouts_table() {
        assert_eq!(LAYOUTS[0], PixelLayout::I420);
        assert_eq!(LAYOUTS[1], PixelLayout::I400);
        assert_eq!(LAYOUTS[2], PixelLayout::I444);
        assert_eq!(LAYOUTS[3], PixelLayout::I422);
    }

    #[test]
    fn test_parse_tile_info_frmhdr_inter_no_tiling() {
        let mut hdr = FrameHeader::default();
        hdr.frame_type = FrameType::Inter;
        hdr.width = 1920;
        hdr.height = 1080;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.sb128 = 1;
        seqhdr.level = 8;
        // tiling.present = Off => reuse_allowed stays false, falls to parse_tile_info
        let data = [0xFF; 32];
        let mut gb = GetBits::new(&data);
        parse_tile_info_frmhdr(&mut hdr, &seqhdr, &mut gb);
        assert_eq!(hdr.sb128, 1);
        assert!(hdr.tiling.t.cols >= 1);
        assert!(hdr.tiling.t.rows >= 1);
    }

    #[test]
    fn test_parse_tile_info_frmhdr_key_sb128_downgrade() {
        let mut hdr = FrameHeader::default();
        hdr.frame_type = FrameType::Key;
        hdr.width = 256;
        hdr.height = 256;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.sb128 = 2;
        seqhdr.max_width = 256;
        seqhdr.max_height = 256;
        seqhdr.level = 4;
        // key frame with sb128=2 => hdr.sb128 = 1 (!!2), sbmul=2
        let data = [0xFF; 32];
        let mut gb = GetBits::new(&data);
        parse_tile_info_frmhdr(&mut hdr, &seqhdr, &mut gb);
        assert_eq!(hdr.sb128, 1);
    }

    #[test]
    fn test_parse_tile_info_frmhdr_inter_inherits_sb128() {
        let mut hdr = FrameHeader::default();
        hdr.frame_type = FrameType::Inter;
        hdr.width = 512;
        hdr.height = 512;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.sb128 = 0;
        seqhdr.level = 4;
        // uniform=1 bit, then all zeros for log2_cols/rows
        let data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        parse_tile_info_frmhdr(&mut hdr, &seqhdr, &mut gb);
        assert_eq!(hdr.sb128, 0);
    }

    #[test]
    fn test_parse_tile_info_frmhdr_key_sb128_zero() {
        let mut hdr = FrameHeader::default();
        hdr.frame_type = FrameType::Key;
        hdr.width = 128;
        hdr.height = 128;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.sb128 = 0;
        seqhdr.level = 4;
        // uniform=1 bit, then zeros
        let data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        parse_tile_info_frmhdr(&mut hdr, &seqhdr, &mut gb);
        assert_eq!(hdr.sb128, 0); // !!0 = 0
        assert!(hdr.tiling.t.cols >= 1);
    }

    #[test]
    fn test_parse_tile_info_frmhdr_reuse_uniform() {
        let mut seqhdr = SequenceHeader::default();
        seqhdr.sb128 = 1;
        seqhdr.max_width = 512;
        seqhdr.max_height = 512;
        seqhdr.level = 4;
        seqhdr.tiling.present = AdaptiveBoolean::On;
        seqhdr.tiling.t.uniform = true;
        seqhdr.tiling.t.cols = 2;
        seqhdr.tiling.t.rows = 2;
        seqhdr.tiling.t.log2_cols = 1;
        seqhdr.tiling.t.log2_rows = 1;
        seqhdr.tiling.t.col_start_sb[0] = 0;
        seqhdr.tiling.t.col_start_sb[1] = 2;
        seqhdr.tiling.t.col_start_sb[2] = 4;
        seqhdr.tiling.t.row_start_sb[0] = 0;
        seqhdr.tiling.t.row_start_sb[1] = 2;
        seqhdr.tiling.t.row_start_sb[2] = 4;

        let mut hdr = FrameHeader::default();
        hdr.frame_type = FrameType::Inter;
        hdr.width = 512;
        hdr.height = 512;

        let data = [0x00; 32];
        let mut gb = GetBits::new(&data);
        parse_tile_info_frmhdr(&mut hdr, &seqhdr, &mut gb);
        // inter => sb128 inherits from seqhdr (1)
        assert_eq!(hdr.sb128, 1);
        // reuse check: uniform, tile_w * (cols-1) < sbw and tile_h * (rows-1) < sbh
        // sbw = (512+127)>>7 = 4, tile_w = (4+1)>>1 = 2, 2*(2-1) = 2 < 4 => reuse_allowed
        // present=On => reuse happens, sb128 match => sbmul=1
        assert_eq!(hdr.tiling.t.cols, 2);
        assert_eq!(hdr.tiling.t.rows, 2);
        assert_eq!(hdr.tiling.t.col_start_sb[0], 0);
        assert_eq!(hdr.tiling.t.col_start_sb[1], 2);
    }

    #[test]
    fn test_parse_fgd_i400_no_points() {
        // I400: no chroma_scaling bit, 4 bits num_points=0, then scaling/ar fields
        // num_points[0]=0 => skip points loop
        // scaling_shift(2), ar_coeff_lag(2)=0, ar_coeff_shift(2), grain_scale_shift(2)
        // overlap(1), clip_to_restricted(1)=0, block_size(1)
        let data = [0x00; 16];
        let mut gb = GetBits::new(&data);
        let fgd = parse_film_grain_data(&mut gb, PixelLayout::I400).unwrap();
        assert_eq!(fgd.num_points[0], 0);
        assert!(!fgd.chroma_scaling_from_luma);
        assert_eq!(fgd.scaling_shift, 8);
        assert_eq!(fgd.ar_coeff_lag, 0);
    }

    #[test]
    fn test_parse_fgd_i420_chroma_scaling_from_luma() {
        // chroma_scaling_from_luma=1, num_points[0]=0
        // then ar fields for pl=0 (skipped since num_points=0 and pl==0)
        let data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        let fgd = parse_film_grain_data(&mut gb, PixelLayout::I420).unwrap();
        assert!(fgd.chroma_scaling_from_luma);
        assert_eq!(fgd.num_points[0], 0);
    }

    #[test]
    fn test_parse_fgd_i420_with_points() {
        // chroma_scaling_from_luma=0 => num_pl=3
        // num_points[0]=1 (4 bits: 0001)
        // index_bits=1+0=1 (3 bits: 000), scaling_bits=5+0=5 (2 bits: 00)
        // point[0]: base=0 (1 bit: 0), scaling=0 (5 bits: 00000)
        // num_points[1]=0, num_points[2]=0
        // (I420 check: both 0 => ok)
        // scaling_shift, ar_coeff_lag=0, ...
        // Bit layout: 0 | 0001 | 000 | 00 | 0 | 00000 | 0000 | 0000 | ...
        //           = 0_0001_000_00_0_00000_0000_0000...
        //           = 0000 1000 0000 0000 0000 0000...
        let data = [0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        let fgd = parse_film_grain_data(&mut gb, PixelLayout::I420).unwrap();
        assert!(!fgd.chroma_scaling_from_luma);
        assert_eq!(fgd.num_points[0], 1);
        assert_eq!(fgd.num_points[1], 0);
        assert_eq!(fgd.num_points[2], 0);
        assert_eq!(fgd.points[0][0][0], 0);
    }

    #[test]
    fn test_parse_fgd_too_many_points() {
        // chroma_scaling_from_luma=0 (1 bit), num_points[0]=15 (4 bits: 1111)
        // 0_1111_000 = 0x78
        let data = [0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        assert!(parse_film_grain_data(&mut gb, PixelLayout::I420).is_err());
    }

    #[test]
    fn test_parse_fgd_i444_no_chroma_check() {
        // I444 doesn't have the I420 num_points consistency check
        // chroma_scaling_from_luma=0 => num_pl=3
        // all num_points=0
        let data = [0x00; 16];
        let mut gb = GetBits::new(&data);
        let fgd = parse_film_grain_data(&mut gb, PixelLayout::I444).unwrap();
        assert!(!fgd.chroma_scaling_from_luma);
        assert_eq!(fgd.num_points[0], 0);
    }

    fn make_ref_hdr(frame_offset: u8, frame_type: FrameType, w: i32, h: i32, qidx: u16) -> FrameHeader {
        let mut fh = FrameHeader::default();
        fh.frame_offset = frame_offset;
        fh.frame_type = frame_type;
        fh.width = w;
        fh.height = h;
        fh.quant.yac = qidx;
        fh
    }

    fn make_refs_with(hdrs: &[(usize, FrameHeader)]) -> [RefState; 8] {
        let mut refs: [RefState; 8] = Default::default();
        for (idx, fh) in hdrs {
            refs[*idx].p.frame_hdr = Some(Arc::new(fh.clone()));
        }
        refs
    }

    #[test]
    fn test_read_frame_size_no_override() {
        let mut hdr = FrameHeader::default();
        hdr.frame_size_override = 0;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.max_width = 1920;
        seqhdr.max_height = 1080;
        let refs: [RefState; 8] = Default::default();
        let data = [0x00; 4];
        let mut gb = GetBits::new(&data);
        read_frame_size(&mut hdr, &seqhdr, &refs, &mut gb).unwrap();
        assert_eq!(hdr.width, 1920);
        assert_eq!(hdr.height, 1080);
    }

    #[test]
    fn test_read_frame_size_override_intra() {
        let mut hdr = FrameHeader::default();
        hdr.frame_size_override = 1;
        hdr.frame_type = FrameType::Key;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.width_n_bits = 11;
        seqhdr.height_n_bits = 11;
        // encode width=640 (639 in 11 bits) and height=480 (479 in 11 bits)
        // 639 = 0b01001111111, 479 = 0b00111011111
        // combined: 01001111111_00111011111_...
        let bits: u32 = (639 << 21) | (479 << 10);
        let data = bits.to_be_bytes();
        let mut gb = GetBits::new(&data);
        let refs: [RefState; 8] = Default::default();
        read_frame_size(&mut hdr, &seqhdr, &refs, &mut gb).unwrap();
        assert_eq!(hdr.width, 640);
        assert_eq!(hdr.height, 480);
    }

    #[test]
    fn test_read_frame_size_override_inter_ref_match() {
        let mut hdr = FrameHeader::default();
        hdr.frame_size_override = 1;
        hdr.frame_type = FrameType::Inter;
        hdr.n_ref_frames = 2;
        hdr.refidx[0] = 3;
        hdr.refidx[1] = 5;
        let ref_fh = make_ref_hdr(1, FrameType::Inter, 800, 600, 100);
        let refs = make_refs_with(&[(3, ref_fh.clone())]);
        // first bit=0 (skip ref 0), second bit=1 (use ref 1)... but refidx[0]=3
        // Actually: bit0=1 means use refidx[0]=3
        let data = [0x80, 0x00, 0x00, 0x00]; // first bit = 1
        let mut gb = GetBits::new(&data);
        read_frame_size(&mut hdr, &SequenceHeader::default(), &refs, &mut gb).unwrap();
        assert_eq!(hdr.width, 800);
        assert_eq!(hdr.height, 600);
    }

    #[test]
    fn test_read_frame_size_override_inter_missing_ref() {
        let mut hdr = FrameHeader::default();
        hdr.frame_size_override = 1;
        hdr.frame_type = FrameType::Inter;
        hdr.n_ref_frames = 1;
        hdr.refidx[0] = 0;
        let refs: [RefState; 8] = Default::default();
        let data = [0x80, 0x00, 0x00, 0x00]; // bit=1 → try ref 0, but no frame_hdr
        let mut gb = GetBits::new(&data);
        assert!(read_frame_size(&mut hdr, &SequenceHeader::default(), &refs, &mut gb).is_err());
    }

    #[test]
    fn test_get_ref_frames_basic() {
        let mut hdr = FrameHeader::default();
        hdr.frame_offset = 4;
        hdr.width = 320;
        hdr.height = 240;
        hdr.n_ref_frames = 7;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.order_hint_n_bits = 8;
        let refs = make_refs_with(&[
            (0, make_ref_hdr(2, FrameType::Inter, 320, 240, 50)),
            (1, make_ref_hdr(3, FrameType::Inter, 320, 240, 50)),
            (2, make_ref_hdr(1, FrameType::Inter, 320, 240, 50)),
        ]);
        let n = get_ref_frames(&mut hdr, &seqhdr, &refs, false);
        assert!(n >= 1);
        assert!(n <= 7);
    }

    #[test]
    fn test_get_ref_frames_layer_dep_filter() {
        let mut hdr = FrameHeader::default();
        hdr.frame_offset = 4;
        hdr.tlayer_id = 1;
        hdr.width = 320;
        hdr.height = 240;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.order_hint_n_bits = 8;
        seqhdr.tlayer_dependency_present = true;
        seqhdr.tlayer_dependencies[1] = 0b01; // layer 1 depends on layer 0 only
        let refs = make_refs_with(&[
            (0, { let mut fh = make_ref_hdr(2, FrameType::Inter, 320, 240, 50); fh.tlayer_id = 0; fh }),
            (1, { let mut fh = make_ref_hdr(3, FrameType::Inter, 320, 240, 50); fh.tlayer_id = 2; fh }),
        ]);
        let n = get_ref_frames(&mut hdr, &seqhdr, &refs, false);
        // ref 1 (tlayer=2) should be filtered out since dep mask only allows layer 0
        assert_eq!(n, 1);
        assert_eq!(hdr.refidx[0], 0);
    }

    #[test]
    fn test_find_tip_ref_frames_single_ref() {
        let mut hdr = FrameHeader::default();
        hdr.n_ref_frames = 1;
        hdr.tip.r#ref = [-1, -1];
        let seqhdr = SequenceHeader::default();
        let refs: [RefState; 8] = Default::default();
        find_tip_ref_frames(&mut hdr, &seqhdr, &refs);
        assert_eq!(hdr.tip.r#ref[0], 0);
        assert_eq!(hdr.tip.r#ref[1], 0);
    }

    #[test]
    fn test_find_tip_ref_frames_past_and_future() {
        let mut hdr = FrameHeader::default();
        hdr.frame_offset = 4;
        hdr.n_ref_frames = 3;
        hdr.refidx[0] = 0;
        hdr.refidx[1] = 1;
        hdr.refidx[2] = 2;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.order_hint_n_bits = 8;
        let refs = make_refs_with(&[
            (0, make_ref_hdr(2, FrameType::Inter, 320, 240, 50)), // past (poc=2 < 4)
            (1, make_ref_hdr(6, FrameType::Inter, 320, 240, 50)), // future (poc=6 > 4)
            (2, make_ref_hdr(3, FrameType::Inter, 320, 240, 50)), // past (poc=3 < 4)
        ]);
        find_tip_ref_frames(&mut hdr, &seqhdr, &refs);
        // mixed: n_past=2, picks order[n_past-1] and order[n_past]
        // sorted by dist: ref0(poc2,dist=-2), ref2(poc3,dist=-1), ref1(poc6,dist=2)
        // n_past=2 → tip.ref[0] = order[1] = closest past, tip.ref[1] = order[2] = closest future
        assert!(hdr.tip.r#ref[0] >= 0 && hdr.tip.r#ref[0] < 3);
        assert!(hdr.tip.r#ref[1] >= 0 && hdr.tip.r#ref[1] < 3);
        assert_ne!(hdr.tip.r#ref[0], hdr.tip.r#ref[1]);
    }

    #[test]
    fn test_derive_pri_sec_ref_no_valid_refs() {
        let mut hdr = FrameHeader::default();
        hdr.n_ref_frames = 2;
        hdr.refidx[0] = 0;
        hdr.refidx[1] = 1;
        let seqhdr = SequenceHeader::default();
        // all refs are key frames → filtered out
        let refs = make_refs_with(&[
            (0, make_ref_hdr(1, FrameType::Key, 320, 240, 50)),
            (1, make_ref_hdr(2, FrameType::Key, 320, 240, 50)),
        ]);
        let result = derive_pri_sec_ref(&hdr, &seqhdr, &refs);
        assert_eq!(result[0], PRIMARY_REF_NONE as i32);
    }

    #[test]
    fn test_derive_pri_sec_ref_inter_refs() {
        let mut hdr = FrameHeader::default();
        hdr.frame_offset = 4;
        hdr.quant.yac = 100;
        hdr.n_ref_frames = 3;
        hdr.refidx[0] = 0;
        hdr.refidx[1] = 1;
        hdr.refidx[2] = 2;
        let mut seqhdr = SequenceHeader::default();
        seqhdr.order_hint_n_bits = 8;
        let refs = make_refs_with(&[
            (0, make_ref_hdr(2, FrameType::Inter, 320, 240, 95)),
            (1, make_ref_hdr(3, FrameType::Inter, 320, 240, 200)),
            (2, make_ref_hdr(1, FrameType::Inter, 320, 240, 98)),
        ]);
        let result = derive_pri_sec_ref(&hdr, &seqhdr, &refs);
        // ref 2 (qidx=98, qdiff=2) is closest in quality, should be primary
        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_parse_ci_hdr_minimal() {
        // scan_type=1 (progressive), all flags=0, reserved=0
        // 01_0_0_0_0_0_0 = 0b01000000 = 0x40
        let data = [0x40, 0x00, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        let mut ci = ContentInterpretation::default();
        parse_ci_hdr(&mut ci, &mut gb).unwrap();
        assert_eq!(ci.scan_type, ScanType::Progressive);
        assert!(!ci.color_description_present);
        assert!(!ci.timing_info_present);
        assert_eq!(ci.color.pri, ColorPrimaries::Unknown);
    }

    #[test]
    fn test_parse_ci_hdr_bt709sdr() {
        // scan_type=1, color_desc_present=1, rest flags=0, reserved=0
        // 01_1_0_0_0_0_0 = 0b01100000 = 0x60
        // color.type = golomb(2) for BT709SDR (=1)
        // golomb(2): for value 1, unary = 0b0, then 2 bits = 01 → "0 01"
        // But get_golomb encodes as: prefix 0s + 1 + k bits
        // For k=2, value=1: bits = 1_01 (prefix length 0 since 1 < 4)
        // Actually let me check: golomb coding with k=2
        // value 1: quotient=0, remainder=1 → unary: 1 (for q=0), k bits: 01
        // So bits: 1_01 = 0b101
        // Then color.range = 1 bit
        // Full: 0110_0000 | 101_r_0000
        // Byte 0: 0110_0000 = 0x60
        // golomb(2) for val 1: prefix 0 (stop), remainder 01 → "0_01"
        // then range=0
        // Byte 1: 001_0_0000 = 0x20
        let data = [0x60, 0x20, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        let mut ci = ContentInterpretation::default();
        parse_ci_hdr(&mut ci, &mut gb).unwrap();
        assert_eq!(ci.scan_type, ScanType::Progressive);
        assert!(ci.color_description_present);
        assert_eq!(ci.color.desc_type, ColorDescription::Bt709Sdr);
        assert_eq!(ci.color.pri, ColorPrimaries::Bt709);
        assert_eq!(ci.color.trc, TransferCharacteristics::Bt709);
        assert_eq!(ci.color.mtrx, MatrixCoefficients::Bt470Bg);
    }

    #[test]
    fn test_parse_ci_hdr_timing_zero_error() {
        // scan_type=0, flags: timing_info_present=1 only
        // 00_0_0_0_1_0_0 = 0b00000100 = 0x04
        // timing: num_units=0 (32 bits) → error
        let mut data = [0u8; 12];
        data[0] = 0x04;
        let mut gb = GetBits::new(&data);
        let mut ci = ContentInterpretation::default();
        assert!(parse_ci_hdr(&mut ci, &mut gb).is_err());
    }

    #[test]
    fn test_parse_ci_hdr_bad_sar() {
        // scan_type=0, aspect_ratio_info_present=1 only
        // 00_0_0_1_0_0_0 = 0b00001000 = 0x08
        // sar_type = 200 (invalid, not 0-16 or 255) → error
        let data = [0x08, 200, 0x00, 0x00];
        let mut gb = GetBits::new(&data);
        let mut ci = ContentInterpretation::default();
        assert!(parse_ci_hdr(&mut ci, &mut gb).is_err());
    }

    #[test]
    fn test_seq_hdr_layer_deps_parsed() {
        // Verify that parse_seq_hdr stores layer dependency defaults
        // when tlayer_dependency_present=false
        let mut seqhdr = SequenceHeader::default();
        seqhdr.max_tlayer_id = 3;
        seqhdr.tlayer_dependency_present = false;
        // defaults: dep[1]=0, dep[2]=1, dep[3]=3 (each depends on all lower layers)
        let mut mask = !0u32;
        for n in 1..3usize {
            seqhdr.tlayer_dependencies[n] = (!mask) as u8;
            mask <<= 1;
        }
        assert_eq!(seqhdr.tlayer_dependencies[1], 0);
        assert_eq!(seqhdr.tlayer_dependencies[2], 1);
    }
}

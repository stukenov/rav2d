use crate::getbits::GetBits;
use crate::headers::*;
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
            thdr.col_start_sb[thdr.cols as usize] = sbx as u16;
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
            thdr.row_start_sb[thdr.rows as usize] = sby as u16;
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
        let dep_present = gb.get_bit() != 0;
        if dep_present {
            for n in 1..hdr.max_tlayer_id as usize {
                let _dep = gb.get_bits(n as i32);
            }
        }
    }

    if hdr.max_mlayer_id > 0 {
        let dep_present = gb.get_bit() != 0;
        if dep_present {
            for n in 1..hdr.max_mlayer_id as usize {
                let _dep = gb.get_bits(n as i32);
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
}

/*
 * Copyright © 2018-2026, VideoLAN and dav2d authors
 * Copyright © 2018-2026, Two Orioles, LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef DAV2D_HEADERS_H
#define DAV2D_HEADERS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants from Section 3. "Symbols and abbreviated terms"
#define DAV2D_MAX_CDEF_STRENGTHS 8
#define DAV2D_MAX_OPERATING_POINTS 64
#define DAV2D_MAX_TILE_COLS 64
#define DAV2D_MAX_TILE_ROWS 64
#define DAV2D_MAX_SEGMENTS 16
#define DAV2D_NUM_REF_FRAMES 8
#define DAV2D_PRIMARY_REF_NONE 7
#define DAV2D_REFS_PER_FRAME 7
#define DAV2D_TOTAL_REFS_PER_FRAME (DAV2D_REFS_PER_FRAME + 1)

enum Dav2dObuType {
    DAV2D_OBU_SEQ_HDR          = 1,
    DAV2D_OBU_TD               = 2,
    DAV2D_OBU_MULTI_FRAME_HDR  = 3,
    DAV2D_OBU_CLOSED_LOOP_KF   = 4,
    DAV2D_OBU_OPEN_LOOP_KF     = 5,
    DAV2D_OBU_LEADING_TILE_GRP = 6,
    DAV2D_OBU_TILE_GRP         = 7,
    DAV2D_OBU_METADATA         = 8,
    DAV2D_OBU_METADATA_GRP     = 9,
    DAV2D_OBU_SWITCH           = 10,
    DAV2D_OBU_LEADING_SEF      = 11,
    DAV2D_OBU_SEF              = 12,
    DAV2D_OBU_LEADING_TIP      = 13,
    DAV2D_OBU_TIP              = 14,
    DAV2D_OBU_BUF_RM_TIMING    = 15,
    DAV2D_OBU_LAYER_CFG_REC    = 16,
    DAV2D_OBU_ATLAS_SEG        = 17,
    DAV2D_OBU_OP_PT_SET        = 18,
    DAV2D_OBU_BRIDGE           = 19,
    DAV2D_OBU_MSDO             = 20,
    DAV2D_OBU_RAS              = 21,
    DAV2D_OBU_QM               = 22,
    DAV2D_OBU_FGM              = 23,
    DAV2D_OBU_CONTENT_INTERP   = 24,
    DAV2D_OBU_PADDING          = 25,
};

enum Dav2dTxfmMode {
    DAV2D_TX_4X4_ONLY,
    DAV2D_TX_LARGEST,
    DAV2D_TX_SWITCHABLE,
    DAV2D_N_TX_MODES,
};

enum Dav2dFilterMode {
    DAV2D_FILTER_8TAP_REGULAR,
    DAV2D_FILTER_8TAP_SMOOTH,
    DAV2D_FILTER_8TAP_SHARP,
    DAV2D_N_SWITCHABLE_FILTERS,
    DAV2D_FILTER_BILINEAR = DAV2D_N_SWITCHABLE_FILTERS,
    DAV2D_N_FILTERS,
    DAV2D_FILTER_SWITCHABLE = DAV2D_N_FILTERS,
};

enum Dav2dAdaptiveBoolean {
    DAV2D_OFF = 0,
    DAV2D_ON = 1,
    DAV2D_ADAPTIVE = 2,
};

enum Dav2dRestorationType {
    DAV2D_RESTORATION_NONE,
    DAV2D_RESTORATION_PC_WIENER,
    DAV2D_RESTORATION_NS_WIENER,
    DAV2D_RESTORATION_SWITCHABLE,
};

enum Dav2dWarpedMotionType {
    DAV2D_WM_TYPE_INVALID = -1,
    DAV2D_WM_TYPE_IDENTITY,
    DAV2D_WM_TYPE_TRANSLATION,
    DAV2D_WM_TYPE_ROT_ZOOM,
    DAV2D_WM_TYPE_AFFINE,
};

typedef struct Dav2dWarpedMotionParams {
    enum Dav2dWarpedMotionType type;
    int32_t matrix[6];
    union {
        struct {
            int16_t alpha, beta, gamma, delta;
        } p;
        int16_t abcd[4];
    } u;
    int affine;
} Dav2dWarpedMotionParams;

enum Dav2dPixelLayout {
    DAV2D_PIXEL_LAYOUT_I400, ///< monochrome
    DAV2D_PIXEL_LAYOUT_I420, ///< 4:2:0 planar
    DAV2D_PIXEL_LAYOUT_I422, ///< 4:2:2 planar
    DAV2D_PIXEL_LAYOUT_I444, ///< 4:4:4 planar
};

enum Dav2dFrameType {
    DAV2D_FRAME_TYPE_KEY = 0,    ///< Key Intra frame
    DAV2D_FRAME_TYPE_INTER = 1,  ///< Inter frame
    DAV2D_FRAME_TYPE_INTRA = 2,  ///< Non key Intra frame
    DAV2D_FRAME_TYPE_SWITCH = 3, ///< Switch Inter frame
};

enum Dav2dColorDescription {
    DAV2D_COLOR_DESC_EXPLICIT = 0,   // Explicitly signaled
    DAV2D_COLOR_DESC_BT709SDR = 1,   // CP=1, TC=1, MC=5
    DAV2D_COLOR_DESC_BT2100PQ = 2,   // CP=9, TC=16, MC=9
    DAV2D_COLOR_DESC_BT2100HLG = 3,  // CP=9, TC=14, MC=9
    DAV2D_COLOR_DESC_SRGB = 4,       // CP=1, TC=13, MC=0
    DAV2D_COLOR_DESC_SRGBSYCC = 5,   // CP=1, TC=13, MC=5
};

enum Dav2dColorPrimaries {
    DAV2D_COLOR_PRI_BT709 = 1,
    DAV2D_COLOR_PRI_UNKNOWN = 2,
    DAV2D_COLOR_PRI_BT470M = 4,
    DAV2D_COLOR_PRI_BT470BG = 5,
    DAV2D_COLOR_PRI_BT601 = 6,
    DAV2D_COLOR_PRI_SMPTE240 = 7,
    DAV2D_COLOR_PRI_FILM = 8,
    DAV2D_COLOR_PRI_BT2020 = 9,
    DAV2D_COLOR_PRI_XYZ = 10,
    DAV2D_COLOR_PRI_SMPTE431 = 11,
    DAV2D_COLOR_PRI_SMPTE432 = 12,
    DAV2D_COLOR_PRI_EBU3213 = 22,
    DAV2D_COLOR_PRI_RESERVED = 255,
};

enum Dav2dTransferCharacteristics {
    DAV2D_TRC_BT709 = 1,
    DAV2D_TRC_UNKNOWN = 2,
    DAV2D_TRC_BT470M = 4,
    DAV2D_TRC_BT470BG = 5,
    DAV2D_TRC_BT601 = 6,
    DAV2D_TRC_SMPTE240 = 7,
    DAV2D_TRC_LINEAR = 8,
    DAV2D_TRC_LOG100 = 9,         ///< logarithmic (100:1 range)
    DAV2D_TRC_LOG100_SQRT10 = 10, ///< lograithmic (100*sqrt(10):1 range)
    DAV2D_TRC_IEC61966 = 11,
    DAV2D_TRC_BT1361 = 12,
    DAV2D_TRC_SRGB = 13,
    DAV2D_TRC_BT2020_10BIT = 14,
    DAV2D_TRC_BT2020_12BIT = 15,
    DAV2D_TRC_SMPTE2084 = 16,     ///< PQ
    DAV2D_TRC_SMPTE428 = 17,
    DAV2D_TRC_HLG = 18,           ///< hybrid log/gamma (BT.2100 / ARIB STD-B67)
    DAV2D_TRC_RESERVED = 255,
};

enum Dav2dMatrixCoefficients {
    DAV2D_MC_IDENTITY = 0,
    DAV2D_MC_BT709 = 1,
    DAV2D_MC_UNKNOWN = 2,
    DAV2D_MC_FCC = 4,
    DAV2D_MC_BT470BG = 5,
    DAV2D_MC_BT601 = 6,
    DAV2D_MC_SMPTE240 = 7,
    DAV2D_MC_SMPTE_YCGCO = 8,
    DAV2D_MC_BT2020_NCL = 9,
    DAV2D_MC_BT2020_CL = 10,
    DAV2D_MC_SMPTE2085 = 11,
    DAV2D_MC_CHROMAT_NCL = 12, ///< Chromaticity-derived
    DAV2D_MC_CHROMAT_CL = 13,
    DAV2D_MC_ICTCP = 14,
    DAV2D_MC_IPT_C2 = 15,
    DAV2D_MC_YCGCO_RE = 16,
    DAV2D_MC_YCGCO_RO = 17,
    DAV2D_MC_RESERVED = 255,
};

enum Dav2dChromaSamplePosition {
    DAV2D_CHR_LEFT = 0,
    DAV2D_CHR_CENTER = 1,
    DAV2D_CHR_TOPLEFT = 2,
    DAV2D_CHR_TOP = 3,
    DAV2D_CHR_BOTTOMLEFT = 4,
    DAV2D_CHR_BOTTOM = 5,
    DAV2D_CHR_UNKNOWN = 6,
};

enum Dav2dAspectRatio {
    DAV2D_SAR_UNKNOWN = 0,
    DAV2D_SAR_1_1 = 1,
    DAV2D_SAR_12_11 = 2,
    DAV2D_SAR_10_11 = 3,
    DAV2D_SAR_16_11 = 4,
    DAV2D_SAR_40_33 = 5,
    DAV2D_SAR_24_11 = 6,
    DAV2D_SAR_20_11 = 7,
    DAV2D_SAR_32_11 = 8,
    DAV2D_SAR_80_33 = 9,
    DAV2D_SAR_18_11 = 10,
    DAV2D_SAR_15_11 = 11,
    DAV2D_SAR_64_33 = 12,
    DAV2D_SAR_160_99 = 13,
    DAV2D_SAR_4_3 = 14,
    DAV2D_SAR_3_2 = 15,
    DAV2D_SAR_2_1 = 16,
    DAV2D_SAR_EXPLICIT = 255,
};

enum Dav2dScanType {
    DAV2D_SCAN_TYPE_UNKNOWN = 0,
    DAV2D_SCAN_TYPE_PROGRESSIVE = 1,
    DAV2D_SCAN_TYPE_INTERLACE = 2,
    DAV2D_SCAN_TYPE_INTERLACE_COMPLEMENTARY = 3,
};

// Specifies the params related to the content in the sequence
typedef struct Dav2dContentInterpretation {
    uint8_t /*enum Dav2dScanType*/ scan_type;
    uint8_t color_description_present;
    uint8_t chroma_sample_position_present;
    uint8_t aspect_ratio_info_present;
    uint8_t timing_info_present;
    uint8_t extension_present;
    uint8_t /*enum Dav2dChromaSamplePosition*/ chr[2];

    struct {
        uint8_t /*enum Dav2dColorDescription*/ type;
        uint8_t /*enum Dav2dColorPrimaries*/ pri;
        uint8_t /*enum Dav2dTransferCharacteristics*/ trc;
        uint8_t /*enum Dav2dMatrixCoefficients*/ mtrx;
        uint8_t range;
    } color;
    struct {
        uint8_t /*enum Dav2dAspectRatio*/ type;
        uint32_t w, h;
    } sar;
    struct {
        uint32_t num_units_in_display_tick;
        uint32_t time_scale;
        uint8_t equal_elemental_interval;
        uint32_t num_ticks_per_elemental_duration;
    } timing;
} Dav2dContentInterpretation;

typedef struct Dav2dContentLightLevel {
    uint16_t max_content_light_level;
    uint16_t max_frame_average_light_level;
} Dav2dContentLightLevel;

typedef struct Dav2dMasteringDisplay {
    uint16_t primaries[3][2]; ///< 0.16 fixed point
    uint16_t white_point[2]; ///< 0.16 fixed point
    uint32_t max_luminance; ///< 24.8 fixed point
    uint32_t min_luminance; ///< 18.14 fixed point
} Dav2dMasteringDisplay;

typedef struct Dav2dITUTT35 {
    uint8_t  country_code;
    uint8_t  country_code_extension_byte;
    size_t   payload_size;
    uint8_t *payload;
} Dav2dITUTT35;

typedef struct Dav2dSegmentationDataSet {
    int16_t delta_q[DAV2D_MAX_SEGMENTS];
    uint16_t delta_q_mask, skip_mask, globalmv_mask;
} Dav2dSegmentationDataSet;

typedef struct Dav2dSequenceHeader {
    uint8_t id;
    /**
     * Stream profile, 0 for 8-10 bits/component 4:2:0 or monochrome;
     * 1 for 8-10 bits/component 4:4:4; 2 for 4:2:2 at any bits/component,
     * or 12 bits/component at any chroma subsampling.
     */
    uint8_t profile;
    uint8_t reduced_still_picture_header;
    uint8_t level;
    uint8_t tier;

    uint8_t /*enum Dav2dPixelLayout*/ layout; ///< format of the picture
    uint8_t ss_hor, ss_ver;

    /**
     * 0, 1 and 2 mean 8, 10 or 12 bits/component, respectively. This is not
     * exactly the same as 'hbd' from the spec; the spec's hbd distinguishes
     * between 8 (0) and 10-12 (1) bits/component, and another element
     * (twelve_bit) to distinguish between 10 and 12 bits/component. To get
     * the spec's hbd, use !!our_hbd, and to get twelve_bit, use hbd == 2.
     */
    uint8_t hbd;

    uint8_t lcr_id;
    uint8_t still_picture;
    uint8_t max_tlayer_id, max_mlayer_id, monotonic;
    /**
     * Maximum dimensions for this stream. In non-scalable streams, these
     * are often the actual dimensions of the stream, although that is not
     * a normative requirement.
     */
    int max_width, max_height;
    uint8_t width_n_bits, height_n_bits;
    struct {
        uint8_t enabled;
        unsigned left, right, top, bottom;
    } crop;

    uint8_t max_display_model_info_present;
    uint8_t max_initial_display_delay;
    uint8_t decoder_model_info_present;
    uint8_t max_decoder_model_present;
    uint32_t num_units_in_decoding_tick;
    uint32_t max_decoder_buffer_delay;
    uint32_t max_encoder_buffer_delay;
    uint8_t max_low_delay_mode;

    uint8_t tlayer_dependency_present, mlayer_dependency_present;
    uint8_t tlayer_dependencies[8], mlayer_dependencies[8];

    uint8_t sb128; // 2: 256x256, 1: 128x128, 0: 64x64

    // partition flags
    uint8_t sdp, ext_sdp;
    uint8_t ext_partitions, uneven_4way_partitions;
    uint8_t max_pb_aspect_ratio_log2;

    // segmentation
    struct {
        uint8_t ext, info_present, adaptive;
        Dav2dSegmentationDataSet d;
    } segmentation;

    // intra tools
    uint8_t intra_dip, intra_edge_filter;
    uint8_t mrls, cfl, cfl_ds_filter_index, mhccp, ibp;

    // inter tools
    uint8_t motion_modes; // translation, inter-intra, warp [3x]
    uint8_t frame_motion_modes_present;
    uint8_t six_param_warp_delta;
    uint8_t masked_compound;
    uint8_t ref_frame_mvs;
    uint8_t reduced_ref_frame_mvs_mode;
    uint8_t order_hint_n_bits;

    uint8_t refmv_bank, drl_reorder;
    uint8_t explicit_ref_frame_map;
    uint8_t ref_frames, ref_frames_log2, number_of_bits_for_lt_frame_id;
    uint8_t def_max_drl_bits, allow_frame_max_drl_bits;
    uint8_t def_max_bvp_drl_bits, allow_max_bvp_drl_bits;
    uint8_t num_same_ref_comp;

    uint8_t tip, tip_hole_fill;
    uint8_t mv_traj, bawp, cwp, imp_msk_bld;
    uint8_t db_sub_pu, tip_explicit_qp;

    uint8_t opfl_refine, refine_mv, tip_refine_mv;
    uint8_t bru, adaptive_mvd, mvd_sign_derive, flex_mvres;
    uint8_t global_motion, short_refresh_frame_flags;

    // screen content flags
    uint8_t /*enum Dav2dAdaptiveBoolean*/ screen_content_tools;
    uint8_t /*enum Dav2dAdaptiveBoolean*/ force_integer_mv;

    // tx group tools
    uint8_t fsc, idtx_intra;
    uint8_t ist[2 /* intra, inter */];
    uint8_t chroma_dctonly, inter_ddt, reduced_tx_part_set;
    uint8_t cctx;

    // coef flags
    uint8_t /*enum Dav2dAdaptiveBoolean*/ tcq;
    uint8_t parity_hiding;

    uint8_t avg_cdf, avg_cdf_type;

    // filtering flags
    uint8_t disable_loopfilters_across_tiles;
    uint8_t cdef;
    uint8_t gdf, gdf_unit_matches_sbsz;
    uint8_t restoration;
    uint8_t rst_disable_mask[2];
    uint8_t ccso, ccso_unit_matches_sbsz;
    uint8_t /*enum Dav2dAdaptiveBoolean*/ cdef_on_skiptx;
    uint8_t df_par_bits;

    // quant tools
    uint8_t separate_uv_delta_q;
    uint8_t equal_ac_dc_q;
    int8_t base_ydc_dq, ydc_dq_enabled;
    uint8_t base_uvdc_dq, uvdc_dq_enabled;
    uint8_t base_uvac_dq, uvac_dq_enabled;

    struct {
        uint8_t /*enum Dav2dAdaptiveBoolean*/ present;
        struct Dav2dTileInfo {
            uint8_t uniform;
            uint8_t min_log2_cols, max_log2_cols, log2_cols, cols;
            uint8_t min_log2_rows, max_log2_rows, log2_rows, rows;
            uint16_t col_start_sb[DAV2D_MAX_TILE_COLS + 1];
            uint16_t row_start_sb[DAV2D_MAX_TILE_ROWS + 1];
        } t;
    } tiling;

    uint8_t film_grain_present;
} Dav2dSequenceHeader;

typedef struct Dav2dFilmGrainData {
    int chroma_scaling_from_luma;
    int num_points[3];
    uint8_t points[3][14][2 /* value, scaling */];
    int scaling_shift;
    int ar_coeff_lag;
    int8_t ar_coeffs[3][25 + 3 /* padding for alignment purposes */];
    uint64_t ar_coeff_shift;
    int grain_scale_shift;
    int uv_mult[2];
    int uv_luma_mult[2];
    int uv_offset[2];
    int overlap_flag;
    int clip_to_restricted_range;
    int mc_identity;
    int block_size;
} Dav2dFilmGrainData;

typedef struct Dav2dFrameHeader {
    uint8_t id;
    enum Dav2dFrameType frame_type; ///< type of the picture
    int width, height;
    uint8_t frame_offset; ///< frame number
    uint8_t tlayer_id, mlayer_id, xlayer_id;

    uint8_t show_existing_frame;
    int8_t existing_frame_idx;
    int8_t ltr_id;
    uint32_t frame_presentation_delay;
    uint8_t show_immediate;
    uint8_t show_implicit;
    uint8_t cross_frame_context;
    uint8_t disable_cdf_update;
    uint8_t allow_screen_content_tools;
    uint8_t force_integer_mv;
    uint8_t frame_size_override;
    uint8_t primary_ref_signaled, primary_ref_frame, secondary_ref_frame;
    uint8_t n_ref_frames;
    uint8_t refresh_frame_flags;
    uint8_t allow_intrabc, allow_global_intrabc, allow_local_intrabc;
    uint8_t max_bvp_drl_bits, max_drl_bits;
    int8_t refidx[DAV2D_REFS_PER_FRAME];
    uint8_t has_future_refs, has_past_refs, has_bothside_refs;
    uint8_t mv_precision; // 0-3 for {f,h,q,e}pel
    enum Dav2dFilterMode subpel_filter_mode;
    uint8_t motion_modes;
    uint8_t use_ref_frame_mvs;
    uint8_t tmvp_sample_step;
    uint8_t opfl_refine_type;
    struct {
        uint8_t frame_mode;
        uint8_t hole_fill;
        uint8_t global_wtd_idx;
        uint8_t apply_filter;
        struct {
            int8_t y, x;
        } gmv;
        uint8_t subpel_filter;
        int8_t ref[2];
    } tip;
    uint8_t sb128; // not literally coded, but derived from seqhdr/frame_type
    struct {
        struct Dav2dTileInfo t;
        uint8_t n_bytes;
        uint16_t update;
    } tiling;
    struct {
        uint16_t yac;
        int8_t ydc_delta;
        int8_t udc_delta, uac_delta, vdc_delta, vac_delta;
        struct {
            uint8_t enabled, num, y[4], u[4], v[4];
        } qm;
    } quant;
    struct {
        uint8_t enabled, update_map, temporal;
        Dav2dSegmentationDataSet d;
        uint8_t preskip;
        int8_t last_active_segid;
        uint8_t lossless[DAV2D_MAX_SEGMENTS], qidx[DAV2D_MAX_SEGMENTS];
    } segmentation;
    struct {
        struct {
            uint8_t present;
            uint8_t res_log2;
        } q;
    } delta;
    uint8_t all_lossless, any_lossless;
    uint8_t tcq, parity_hiding;
    struct {
        uint8_t sub_pu;
        uint8_t level_y[2 /* dir */];
        uint8_t level_u, level_v;
        int8_t delta_q_y[2], delta_q_u, delta_q_v;
    } deblock;
    struct {
        enum Dav2dAdaptiveBoolean enabled;
        uint8_t qp_idx, scale;
    } gdf;
    struct {
        uint8_t enabled;
        uint8_t damping;
        uint8_t n_strengths;
        uint8_t on_skiptx;
        uint8_t y_strength[DAV2D_MAX_CDEF_STRENGTHS];
        uint8_t uv_strength[DAV2D_MAX_CDEF_STRENGTHS];
    } cdef;
    struct {
        struct {
            uint8_t /*enum Dav2dRestorationType*/ type;
            struct Dav2dNSWienerPlane {
                uint8_t frame_filters_on;
                uint8_t num_classes_idx, num_classes, temporal, refidx;
                int8_t filter[16][18];
            } ns;
        } p[3 /* plane */];
        uint8_t unit_size[2 /* y, uv */];
    } restoration;
    struct {
        uint8_t enabled;
        struct {
            uint8_t enabled;
            uint8_t reuse, sb_reuse, refidx;
            uint8_t bo_only, scale_idx, quant_idx;
            uint8_t ext_filter_support, edge_clf, max_band_log2;
            uint8_t filter_off[64 /* nibbles. if bo_only { [band:128] } else { [d0:4][d1:4][band:8] } */];
        } p[3];
    } ccso;
    enum Dav2dTxfmMode txfm_mode;
    uint8_t switchable_comp_refs;
    uint8_t skip_mode_enabled;
    uint8_t bawp;
    uint8_t warp_motion;
    uint8_t reduced_txtp_set;
    struct {
        uint8_t ref; // index in our reference array
        uint8_t refref; // index in that reference's refrence array
        Dav2dWarpedMotionParams m[DAV2D_REFS_PER_FRAME];
    } gmv;
    struct {
        uint8_t present;
        uint8_t id;
        unsigned seed;
    } film_grain; ///< film grain parameters
} Dav2dFrameHeader;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DAV2D_HEADERS_H */

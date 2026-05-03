pub const MAX_CDEF_STRENGTHS: usize = 8;
pub const MAX_OPERATING_POINTS: usize = 64;
pub const MAX_TILE_COLS: usize = 64;
pub const MAX_TILE_ROWS: usize = 64;
pub const MAX_SEGMENTS: usize = 16;
pub const NUM_REF_FRAMES: usize = 8;
pub const PRIMARY_REF_NONE: u8 = 7;
pub const REFS_PER_FRAME: usize = 7;
pub const TOTAL_REFS_PER_FRAME: usize = REFS_PER_FRAME + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObuType {
    SeqHdr = 1,
    Td = 2,
    MultiFrameHdr = 3,
    ClosedLoopKf = 4,
    OpenLoopKf = 5,
    LeadingTileGrp = 6,
    TileGrp = 7,
    Metadata = 8,
    MetadataGrp = 9,
    Switch = 10,
    LeadingSef = 11,
    Sef = 12,
    LeadingTip = 13,
    Tip = 14,
    BufRmTiming = 15,
    LayerCfgRec = 16,
    AtlasSeg = 17,
    OpPtSet = 18,
    Bridge = 19,
    Msdo = 20,
    Ras = 21,
    Qm = 22,
    Fgm = 23,
    ContentInterp = 24,
    Padding = 25,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxfmMode {
    #[default]
    Only4x4 = 0,
    Largest = 1,
    Switchable = 2,
}
pub const N_TX_MODES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FilterMode {
    #[default]
    Regular8Tap = 0,
    Smooth8Tap = 1,
    Sharp8Tap = 2,
    Bilinear = 3,
    Switchable = 4,
}
pub const N_SWITCHABLE_FILTERS: usize = 3;
pub const N_FILTERS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum AdaptiveBoolean {
    #[default]
    Off = 0,
    On = 1,
    Adaptive = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RestorationType {
    #[default]
    None = 0,
    PcWiener = 1,
    NsWiener = 2,
    Switchable = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum WarpedMotionType {
    Invalid = -1,
    Identity = 0,
    Translation = 1,
    RotZoom = 2,
    Affine = 3,
}

impl Default for WarpedMotionType {
    fn default() -> Self {
        Self::Identity
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PixelLayout {
    #[default]
    I400 = 0,
    I420 = 1,
    I422 = 2,
    I444 = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    #[default]
    Key = 0,
    Inter = 1,
    Intra = 2,
    Switch = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorDescription {
    #[default]
    Explicit = 0,
    Bt709Sdr = 1,
    Bt2100Pq = 2,
    Bt2100Hlg = 3,
    Srgb = 4,
    SrgbSycc = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorPrimaries {
    Bt709 = 1,
    #[default]
    Unknown = 2,
    Bt470M = 4,
    Bt470Bg = 5,
    Bt601 = 6,
    Smpte240 = 7,
    Film = 8,
    Bt2020 = 9,
    Xyz = 10,
    Smpte431 = 11,
    Smpte432 = 12,
    Ebu3213 = 22,
    Reserved = 255,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TransferCharacteristics {
    Bt709 = 1,
    #[default]
    Unknown = 2,
    Bt470M = 4,
    Bt470Bg = 5,
    Bt601 = 6,
    Smpte240 = 7,
    Linear = 8,
    Log100 = 9,
    Log100Sqrt10 = 10,
    Iec61966 = 11,
    Bt1361 = 12,
    Srgb = 13,
    Bt2020_10Bit = 14,
    Bt2020_12Bit = 15,
    Smpte2084 = 16,
    Smpte428 = 17,
    Hlg = 18,
    Reserved = 255,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,
    Bt709 = 1,
    #[default]
    Unknown = 2,
    Fcc = 4,
    Bt470Bg = 5,
    Bt601 = 6,
    Smpte240 = 7,
    SmpteYcgco = 8,
    Bt2020Ncl = 9,
    Bt2020Cl = 10,
    Smpte2085 = 11,
    ChromatNcl = 12,
    ChromatCl = 13,
    Ictcp = 14,
    IptC2 = 15,
    YcgcoRe = 16,
    YcgcoRo = 17,
    Reserved = 255,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ChromaSamplePosition {
    Left = 0,
    Center = 1,
    TopLeft = 2,
    Top = 3,
    BottomLeft = 4,
    Bottom = 5,
    #[default]
    Unknown = 6,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WarpedMotionParams {
    pub wm_type: WarpedMotionType,
    pub matrix: [i32; 6],
    pub abcd: [i16; 4],
    pub affine: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SegmentationDataSet {
    pub delta_q: [i16; MAX_SEGMENTS],
    pub delta_q_mask: u16,
    pub skip_mask: u16,
    pub globalmv_mask: u16,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ContentLightLevel {
    pub max_content_light_level: u16,
    pub max_frame_average_light_level: u16,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MasteringDisplay {
    pub primaries: [[u16; 2]; 3],
    pub white_point: [u16; 2],
    pub max_luminance: u32,
    pub min_luminance: u32,
}

#[derive(Debug, Clone)]
pub struct TileInfo {
    pub uniform: bool,
    pub min_log2_cols: u8,
    pub max_log2_cols: u8,
    pub log2_cols: u8,
    pub cols: u8,
    pub min_log2_rows: u8,
    pub max_log2_rows: u8,
    pub log2_rows: u8,
    pub rows: u8,
    pub col_start_sb: Box<[u16; MAX_TILE_COLS + 1]>,
    pub row_start_sb: Box<[u16; MAX_TILE_ROWS + 1]>,
}

impl Default for TileInfo {
    fn default() -> Self {
        Self {
            uniform: false,
            min_log2_cols: 0,
            max_log2_cols: 0,
            log2_cols: 0,
            cols: 0,
            min_log2_rows: 0,
            max_log2_rows: 0,
            log2_rows: 0,
            rows: 0,
            col_start_sb: Box::new([0u16; MAX_TILE_COLS + 1]),
            row_start_sb: Box::new([0u16; MAX_TILE_ROWS + 1]),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SequenceHeader {
    pub id: u8,
    pub profile: u8,
    pub reduced_still_picture_header: bool,
    pub level: u8,
    pub tier: u8,
    pub layout: PixelLayout,
    pub ss_hor: u8,
    pub ss_ver: u8,
    pub hbd: u8,
    pub lcr_id: u8,
    pub still_picture: bool,
    pub max_tlayer_id: u8,
    pub max_mlayer_id: u8,
    pub monotonic: bool,
    pub max_width: i32,
    pub max_height: i32,
    pub width_n_bits: u8,
    pub height_n_bits: u8,

    pub crop: CropInfo,
    pub sb128: u8,

    pub sdp: bool,
    pub ext_sdp: bool,
    pub ext_partitions: bool,
    pub uneven_4way_partitions: bool,
    pub max_pb_aspect_ratio_log2: u8,

    pub segmentation: SeqSegmentation,

    pub intra_dip: bool,
    pub intra_edge_filter: bool,
    pub mrls: bool,
    pub cfl: bool,
    pub cfl_ds_filter_index: u8,
    pub mhccp: bool,
    pub ibp: bool,

    pub motion_modes: u8,
    pub frame_motion_modes_present: bool,
    pub six_param_warp_delta: bool,
    pub masked_compound: bool,
    pub ref_frame_mvs: bool,
    pub reduced_ref_frame_mvs_mode: u8,
    pub order_hint_n_bits: u8,

    pub refmv_bank: bool,
    pub drl_reorder: bool,
    pub explicit_ref_frame_map: bool,
    pub ref_frames: u8,
    pub ref_frames_log2: u8,
    pub number_of_bits_for_lt_frame_id: u8,
    pub def_max_drl_bits: u8,
    pub allow_frame_max_drl_bits: bool,
    pub def_max_bvp_drl_bits: u8,
    pub allow_max_bvp_drl_bits: bool,
    pub num_same_ref_comp: u8,

    pub tip: bool,
    pub tip_hole_fill: bool,
    pub mv_traj: bool,
    pub bawp: bool,
    pub cwp: bool,
    pub imp_msk_bld: bool,

    pub db_sub_pu: bool,
    pub tip_explicit_qp: bool,

    pub opfl_refine: bool,
    pub refine_mv: bool,
    pub tip_refine_mv: bool,
    pub bru: bool,
    pub adaptive_mvd: bool,
    pub mvd_sign_derive: bool,
    pub flex_mvres: bool,
    pub global_motion: bool,
    pub short_refresh_frame_flags: bool,

    pub screen_content_tools: AdaptiveBoolean,
    pub force_integer_mv: AdaptiveBoolean,

    pub fsc: bool,
    pub idtx_intra: bool,
    pub ist: [bool; 2],
    pub chroma_dctonly: bool,
    pub inter_ddt: bool,
    pub reduced_tx_part_set: bool,
    pub cctx: bool,

    pub tcq: AdaptiveBoolean,
    pub parity_hiding: bool,

    pub avg_cdf: bool,
    pub avg_cdf_type: u8,

    pub disable_loopfilters_across_tiles: bool,
    pub cdef: bool,
    pub gdf: bool,
    pub gdf_unit_matches_sbsz: bool,
    pub restoration: bool,
    pub rst_disable_mask: [u8; 2],
    pub ccso: bool,
    pub ccso_unit_matches_sbsz: bool,
    pub cdef_on_skiptx: AdaptiveBoolean,
    pub df_par_bits: u8,

    pub separate_uv_delta_q: bool,
    pub equal_ac_dc_q: bool,
    pub base_ydc_dq: i8,
    pub ydc_dq_enabled: bool,
    pub base_uvdc_dq: u8,
    pub uvdc_dq_enabled: bool,
    pub base_uvac_dq: u8,
    pub uvac_dq_enabled: bool,

    pub tiling: SeqTiling,
    pub film_grain_present: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CropInfo {
    pub enabled: bool,
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

#[derive(Debug, Clone, Default)]
pub struct SeqSegmentation {
    pub ext: bool,
    pub info_present: bool,
    pub adaptive: bool,
    pub d: SegmentationDataSet,
}

#[derive(Debug, Clone, Default)]
pub struct SeqTiling {
    pub present: AdaptiveBoolean,
    pub t: TileInfo,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FilmGrainData {
    pub chroma_scaling_from_luma: bool,
    pub num_points: [i32; 3],
    pub points: [[[u8; 2]; 14]; 3],
    pub scaling_shift: i32,
    pub ar_coeff_lag: i32,
    pub ar_coeffs: [[i8; 28]; 3],
    pub ar_coeff_shift: u64,
    pub grain_scale_shift: i32,
    pub uv_mult: [i32; 2],
    pub uv_luma_mult: [i32; 2],
    pub uv_offset: [i32; 2],
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
    pub mc_identity: bool,
    pub block_size: i32,
}

#[derive(Debug, Clone, Default)]
pub struct FhTip {
    pub frame_mode: u8,
    pub hole_fill: u8,
    pub global_wtd_idx: u8,
    pub apply_filter: u8,
    pub gmv_y: i8,
    pub gmv_x: i8,
    pub subpel_filter: u8,
    pub r#ref: [i8; 2],
}

#[derive(Debug, Clone, Default)]
pub struct FhTiling {
    pub t: TileInfo,
    pub n_bytes: u8,
    pub update: u16,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhQuantQm {
    pub enabled: u8,
    pub num: u8,
    pub y: [u8; 4],
    pub u: [u8; 4],
    pub v: [u8; 4],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhQuant {
    pub yac: u16,
    pub ydc_delta: i8,
    pub udc_delta: i8,
    pub uac_delta: i8,
    pub vdc_delta: i8,
    pub vac_delta: i8,
    pub qm: FhQuantQm,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhSegmentation {
    pub enabled: u8,
    pub update_map: u8,
    pub temporal: u8,
    pub d: SegmentationDataSet,
    pub preskip: u8,
    pub last_active_segid: i8,
    pub lossless: [u8; MAX_SEGMENTS],
    pub qidx: [u8; MAX_SEGMENTS],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhDeltaQ {
    pub present: u8,
    pub res_log2: u8,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhDelta {
    pub q: FhDeltaQ,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhDeblock {
    pub sub_pu: u8,
    pub level_y: [u8; 2],
    pub level_u: u8,
    pub level_v: u8,
    pub delta_q_y: [i8; 2],
    pub delta_q_u: i8,
    pub delta_q_v: i8,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhGdf {
    pub enabled: AdaptiveBoolean,
    pub qp_idx: u8,
    pub scale: u8,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhCdef {
    pub enabled: u8,
    pub damping: u8,
    pub n_strengths: u8,
    pub on_skiptx: u8,
    pub y_strength: [u8; MAX_CDEF_STRENGTHS],
    pub uv_strength: [u8; MAX_CDEF_STRENGTHS],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NSWienerPlane {
    pub frame_filters_on: u8,
    pub num_classes_idx: u8,
    pub num_classes: u8,
    pub temporal: u8,
    pub refidx: u8,
    pub filter: [[i8; 18]; 16],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhRestorationPlane {
    pub restoration_type: u8,
    pub ns: NSWienerPlane,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhRestoration {
    pub p: [FhRestorationPlane; 3],
    pub unit_size: [u8; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct FhCcsoPlane {
    pub enabled: u8,
    pub reuse: u8,
    pub sb_reuse: u8,
    pub refidx: u8,
    pub bo_only: u8,
    pub scale_idx: u8,
    pub quant_idx: u8,
    pub ext_filter_support: u8,
    pub edge_clf: u8,
    pub max_band_log2: u8,
    pub filter_off: [u8; 64],
}

impl Default for FhCcsoPlane {
    fn default() -> Self {
        Self {
            enabled: 0,
            reuse: 0,
            sb_reuse: 0,
            refidx: 0,
            bo_only: 0,
            scale_idx: 0,
            quant_idx: 0,
            ext_filter_support: 0,
            edge_clf: 0,
            max_band_log2: 0,
            filter_off: [0; 64],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhCcso {
    pub enabled: u8,
    pub p: [FhCcsoPlane; 3],
}

#[derive(Debug, Clone, Default)]
pub struct FhGmv {
    pub r#ref: u8,
    pub refref: u8,
    pub m: [WarpedMotionParams; REFS_PER_FRAME],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FhFilmGrain {
    pub present: u8,
    pub id: u8,
    pub seed: u32,
}

#[derive(Debug, Clone, Default)]
pub struct FrameHeader {
    pub id: u8,
    pub frame_type: FrameType,
    pub width: i32,
    pub height: i32,
    pub frame_offset: u8,
    pub tlayer_id: u8,
    pub mlayer_id: u8,
    pub xlayer_id: u8,
    pub show_existing_frame: u8,
    pub existing_frame_idx: i8,
    pub ltr_id: i8,
    pub frame_presentation_delay: u32,
    pub show_immediate: u8,
    pub show_implicit: u8,
    pub cross_frame_context: u8,
    pub disable_cdf_update: u8,
    pub allow_screen_content_tools: u8,
    pub force_integer_mv: u8,
    pub frame_size_override: u8,
    pub primary_ref_signaled: u8,
    pub primary_ref_frame: u8,
    pub secondary_ref_frame: u8,
    pub n_ref_frames: u8,
    pub refresh_frame_flags: u8,
    pub allow_intrabc: u8,
    pub allow_global_intrabc: u8,
    pub allow_local_intrabc: u8,
    pub max_bvp_drl_bits: u8,
    pub max_drl_bits: u8,
    pub refidx: [i8; REFS_PER_FRAME],
    pub has_future_refs: u8,
    pub has_past_refs: u8,
    pub has_bothside_refs: u8,
    pub mv_precision: u8,
    pub subpel_filter_mode: FilterMode,
    pub motion_modes: u8,
    pub use_ref_frame_mvs: u8,
    pub tmvp_sample_step: u8,
    pub opfl_refine_type: u8,
    pub tip: FhTip,
    pub sb128: u8,
    pub tiling: FhTiling,
    pub quant: FhQuant,
    pub segmentation: FhSegmentation,
    pub delta: FhDelta,
    pub all_lossless: u8,
    pub any_lossless: u8,
    pub tcq: u8,
    pub parity_hiding: u8,
    pub deblock: FhDeblock,
    pub gdf: FhGdf,
    pub cdef: FhCdef,
    pub restoration: FhRestoration,
    pub ccso: FhCcso,
    pub txfm_mode: TxfmMode,
    pub switchable_comp_refs: u8,
    pub skip_mode_enabled: u8,
    pub bawp: u8,
    pub warp_motion: u8,
    pub reduced_txtp_set: u8,
    pub gmv: FhGmv,
    pub film_grain: FhFilmGrain,
}

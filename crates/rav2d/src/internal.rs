use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU32};

use crate::cdf::CdfContext;
use crate::dsp::{DSPContext, PalDSPContext, RefmvsDSPContext};
use crate::env::{BlockContext, SBEdgeCtx};
use crate::headers::{
    ContentInterpretation, ContentLightLevel, FilmGrainData, FrameHeader, MAX_SEGMENTS,
    MasteringDisplay, SequenceHeader, WarpedMotionParams,
};
use crate::levels::{Av2Block, BlockSize, N_RECT_TX_SIZES, RefPair};
use crate::lf_mask::{Av2Filter, Av2Restoration, Av2RestorationUnit};
use crate::refmvs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskType {
    Init = 0,
    InitCdf = 1,
    TileEntropy = 2,
    EntropyProgress = 3,
    TileMvResolution = 4,
    TileReconstruction = 5,
    DeblockCols = 6,
    DeblockRows = 7,
    Cdef = 8,
    LoopRestoration = 9,
    ReconstructionProgress = 10,
    FgPrep = 11,
    FgApply = 12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pass {
    Entropy = 1,
    MvRes = 2,
    Recon = 4,
}

pub const PASS_ALL: u8 = Pass::Entropy as u8 | Pass::MvRes as u8 | Pass::Recon as u8;

#[derive(Clone, Copy, Default)]
pub struct CodedBlockInfo {
    pub eob: [i16; 3],
    pub txtp: [u16; 3],
}

#[derive(Clone, Copy, Default)]
pub struct ScalableMotionParams {
    pub scale: i32,
    pub step: i32,
}

#[derive(Default)]
pub struct NsWienerBank {
    pub bank_size: [u8; 16],
    pub bank_idx: [u8; 16],
    pub filter: [[[i8; 32]; 16]; 4],
}

pub struct TileState {
    pub cdf: CdfContext,
    pub msac_buf: Vec<u8>,

    pub tiling: TileBounds,

    pub progress: [AtomicI32; 3],
    pub frame_thread: [TileStateFrameThread; 2],

    pub lowest_pixel: Vec<[[i32; 2]; 7]>,

    pub dqmem: [[[u32; 2]; 3]; MAX_SEGMENTS],
    pub last_qidx: i32,

    pub lr_ref: [Vec<Av2RestorationUnit>; 3],

    pub ns_wiener_bank: [NsWienerBank; 3],

    pub tile_start_off: u32,
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            cdf: Default::default(),
            msac_buf: Vec::new(),
            tiling: Default::default(),
            progress: [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)],
            frame_thread: Default::default(),
            lowest_pixel: Vec::new(),
            dqmem: [[[0; 2]; 3]; MAX_SEGMENTS],
            last_qidx: 0,
            lr_ref: Default::default(),
            ns_wiener_bank: Default::default(),
            tile_start_off: 0,
        }
    }
}

#[derive(Clone, Default)]
pub struct TileBounds {
    pub col_start: i32,
    pub col_end: i32,
    pub row_start: i32,
    pub row_end: i32,
    pub col: i32,
    pub row: i32,
}

#[derive(Default)]
pub struct TileStateFrameThread {
    pub pal_idx: Vec<u8>,
    pub cbi: Vec<CodedBlockInfo>,
    pub cf: Vec<i32>,
    pub partition: [Vec<u8>; 2],
}

pub struct FrameThread {
    pub next_tile_row: [i32; 2],
    pub entropy_progress: AtomicI32,
    pub deblock_progress: AtomicI32,
    pub b: Vec<Av2Block>,
    pub cbi: Vec<CodedBlockInfo>,
    pub pal_idx: Vec<u8>,
    pub cf: Vec<i32>,
    pub partition: Vec<u8>,
    pub prog_sz: i32,
    pub tile_start_off: Vec<u32>,
    pub scheduled: i32,
}

impl Default for FrameThread {
    fn default() -> Self {
        Self {
            next_tile_row: [0; 2],
            entropy_progress: AtomicI32::new(0),
            deblock_progress: AtomicI32::new(0),
            b: Vec::new(),
            cbi: Vec::new(),
            pal_idx: Vec::new(),
            cf: Vec::new(),
            partition: Vec::new(),
            prog_sz: 0,
            tile_start_off: Vec::new(),
            scheduled: 0,
        }
    }
}

#[derive(Default)]
pub struct LoopFilterState {
    pub mask: Vec<Av2Filter>,
    pub lr_mask: Vec<Av2Restoration>,
    pub segmap_uv: Vec<u8>,
    pub uv_segmap_stride: isize,
    pub cdef_buf_plane_sz: [i32; 2],
    pub cdef_buf_sbh: i32,
    pub lr_buf_plane_sz: [i32; 4],
    pub re_sz: i32,
    pub base_q: i32,
    pub gdf_ref_dst_idx: i32,
    pub start_of_tile_row: Vec<u8>,
    pub restore_planes: i32,
    pub wiener_idx: usize,
    pub ns_subclass_class_idx: Option<usize>,
    pub lr_db_line: [Vec<u8>; 3],
    pub lr_cdef_line: [Vec<u8>; 3],
    pub p: [Vec<u8>; 3],
    pub ns_subclass_lut: Vec<u8>,
    pub pc_subclass_lut: Vec<u8>,
    pub pc_filters: Vec<[i16; 13]>,
}

#[derive(Default)]
pub struct FrameContext {
    pub seq_hdr: Arc<SequenceHeader>,
    pub frame_hdr: Arc<FrameHeader>,

    pub cur: ThreadPicture,
    pub refp: [ThreadPicture; 7],

    pub mvs: Vec<refmvs::TemporalBlock>,
    pub ref_mvs: [Option<Vec<refmvs::TemporalBlock>>; 7],

    pub cur_segmap: Vec<u8>,
    pub prev_segmap: Option<Vec<u8>>,

    pub cur_ccsomap: Vec<u8>,
    pub prev_ccsomap: [Option<Vec<u8>>; 3],

    pub refpoc: [u8; 7],
    pub refrefpoc: [[u8; 7]; 7],
    pub refcnt: [u8; 7],
    pub refdir: [u8; 8],
    pub refdir_intra: i8,
    pub furthest_future_refidx: i8,
    pub absrefdist: [u8; 7],
    pub refdist: [i8; 7],
    pub skip_mode_refs: RefPair,
    pub gmv_warp_allowed: [u8; 7],
    pub use_pri_sec_cdf: i32,

    pub tile: Vec<TileGroup>,
    pub n_tile_data: i32,

    pub svc: [[ScalableMotionParams; 2]; 7],

    pub ts: Vec<TileState>,
    pub n_ts: i32,
    pub dsp: Arc<[DSPContext; 3]>,

    pub b4_stride: isize,
    pub bw: i32,
    pub bh: i32,
    pub sb256w: i32,
    pub sb256h: i32,
    pub sbh: i32,
    pub sb_shift: i32,
    pub sb_step: i32,
    pub ss_ver: i32,
    pub ss_hor: i32,

    pub dq: [[[u32; 2]; 3]; MAX_SEGMENTS],
    pub qm: [[Option<Vec<u8>>; 3]; N_RECT_TX_SIZES],

    pub a: Vec<BlockContext>,
    pub a_sz: i32,
    pub rf: refmvs::Frame,
    pub bitdepth_max: i32,
    pub root_bs: BlockSize,

    pub frame_thread: FrameThread,
    pub lf: LoopFilterState,

    /// In-loop filter flag word (DAV2D_INLOOPFILTER_* bits) threaded from the
    /// decoder's `Settings.inloop_filters`. Defaults to 0 (filters off) so any
    /// path constructing a `FrameContext` directly keeps pre-filter behaviour;
    /// `submit_frame` sets it from the configured filters.
    pub inloop_filters: u32,

    /// Output picture buffer that reconstruction writes into (the pixel planes
    /// that `cur` only describes as metadata).
    pub cur_pic: crate::picture::Picture,
}

#[derive(Clone, Default)]
pub struct ThreadPicture {
    pub visible: bool,
    pub showable: bool,
    pub frame_hdr: Option<Arc<FrameHeader>>,
    pub progress: [Arc<AtomicU32>; 2],
}

pub struct TileGroup {
    pub data: Vec<u8>,
    pub start: i32,
    pub end: i32,
}

pub struct TaskContext {
    pub bx: i32,
    pub by: i32,
    pub cbx: i32,
    pub cby: i32,
    pub sdp_cfl_disallowed: i32,
    pub intra_region: i32,
    pub l: BlockContext,
    pub a_sb_cache: SBEdgeCtx,
    pub is_coded: [[u64; 64]; 2],

    pub pb: PbContext,
    pub rt: refmvs::Tile,

    pub chroma_txtp: [[u16; 2]; 256],
    pub chroma_eob: [[i16; 2]; 256],
    pub cf_uv: Vec<i32>,

    pub luma_intra_dir_mode_map: [u8; 256],
    pub luma_fsc_map: [u8; 256],

    pub txtp_map: [u8; 256],
    pub warpmv: [WarpedMotionParams; 2],
    pub lf_mask: Vec<Av2Filter>,
    pub top_pre_cdef_toggle: i32,
    pub u_has_cf: u8,

    pub pass: u8,
}

#[derive(Clone, Default)]
pub struct PbContext {
    pub col_start: i32,
    pub row_start: i32,
    pub bawp: [BawpParams; 3],
}

#[derive(Clone, Copy, Default)]
pub struct BawpParams {
    pub alpha: i32,
    pub beta: i32,
}

pub struct CdfThreadContext {
    pub cdf: CdfContext,
    pub progress: AtomicI32,
}

pub struct DecoderContext {
    pub seq_hdr: Option<Arc<SequenceHeader>>,
    pub frame_hdr: Option<Arc<FrameHeader>>,

    pub tile: Vec<TileGroup>,
    pub n_tile_data: i32,
    pub n_tiles: i32,

    pub refs: [RefState; 8],
    pub cdf: Vec<CdfThreadContext>,

    pub dsp: Arc<[DSPContext; 3]>,
    pub pal_dsp: PalDSPContext,
    pub refmvs_dsp: RefmvsDSPContext,

    pub content_light: Option<ContentLightLevel>,
    pub mastering_display: Option<MasteringDisplay>,
    pub ci: Option<ContentInterpretation>,
    pub fgm: [Option<FilmGrainData>; 8],

    pub apply_grain: bool,
    pub operating_point: i32,
    pub operating_point_idc: u32,
    pub all_layers: bool,
    pub max_spatial_id: i32,
    pub frame_size_limit: u32,
    pub strict_std_compliance: bool,
    pub output_invisible_frames: bool,
    pub n_passes: i32,

    /// In-loop filter flag word (DAV2D_INLOOPFILTER_* bits) from the configured
    /// `Settings.inloop_filters`. Threaded onto each `FrameContext` in
    /// `submit_frame` so the per-superblock-row filter pass can gate each stage.
    pub inloop_filters: u32,

    /// Bring-up gate: run the single-threaded frame decode from `parse_obus`.
    /// Off by default while reconstruction and the entropy-path bugs are being
    /// worked through; the orchestration runs end-to-end when enabled.
    pub run_decode: bool,

    /// The most recently reconstructed picture, handed off to the decoder's
    /// output queue by `gen_picture`. (Minimal output path; visibility/POC
    /// reordering lands with full output queueing.)
    pub frame_out: Option<crate::picture::Picture>,
}

#[derive(Default)]
pub struct RefState {
    pub p: ThreadPicture,
    pub segmap: Option<Vec<u8>>,
    pub refmvs: Option<Vec<refmvs::TemporalBlock>>,
    pub ccsomap: Option<Vec<u8>>,
    pub refpoc: [u8; 7],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_values() {
        assert_eq!(TaskType::Init as u8, 0);
        assert_eq!(TaskType::FgApply as u8, 12);
    }

    #[test]
    fn test_pass_flags() {
        assert_eq!(PASS_ALL, 7);
        assert_eq!(
            Pass::Entropy as u8 | Pass::MvRes as u8 | Pass::Recon as u8,
            7
        );
    }

    #[test]
    fn test_coded_block_info_default() {
        let cbi = CodedBlockInfo::default();
        assert_eq!(cbi.eob, [0; 3]);
        assert_eq!(cbi.txtp, [0; 3]);
    }

    #[test]
    fn test_tile_bounds_default() {
        let tb = TileBounds::default();
        assert_eq!(tb.col_start, 0);
        assert_eq!(tb.row, 0);
    }

    #[test]
    fn test_thread_picture_default() {
        let tp = ThreadPicture::default();
        assert!(!tp.visible);
        assert!(!tp.showable);
    }

    #[test]
    fn test_ref_state_default() {
        let rs = RefState::default();
        assert!(rs.segmap.is_none());
        assert_eq!(rs.refpoc, [0; 7]);
    }

    #[test]
    fn test_bawp_params_default() {
        let bp = BawpParams::default();
        assert_eq!(bp.alpha, 0);
        assert_eq!(bp.beta, 0);
    }

    #[test]
    fn test_ns_wiener_bank_default() {
        let nwb = NsWienerBank::default();
        assert_eq!(nwb.bank_size[0], 0);
        assert_eq!(nwb.filter[0][0][0], 0);
    }
}

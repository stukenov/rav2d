pub const TIP_FRAME: usize = 7;
pub const INVALID_MV: i32 = 0x200000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObuMetaType {
    HdrCll = 1,
    HdrMdcv = 2,
    Scalability = 3,
    ItutT35 = 4,
    Timecode = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxfmSize {
    #[default]
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
    Tx64x64 = 4,
}
pub const N_TX_SIZES: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RectTxfmSize {
    Rtx4x8 = 5,
    Rtx8x4 = 6,
    Rtx8x16 = 7,
    Rtx16x8 = 8,
    Rtx16x32 = 9,
    Rtx32x16 = 10,
    Rtx32x64 = 11,
    Rtx64x32 = 12,
    Rtx4x16 = 13,
    Rtx16x4 = 14,
    Rtx8x32 = 15,
    Rtx32x8 = 16,
    Rtx16x64 = 17,
    Rtx64x16 = 18,
    Rtx4x32 = 19,
    Rtx32x4 = 20,
    Rtx8x64 = 21,
    Rtx64x8 = 22,
    Rtx4x64 = 23,
    Rtx64x4 = 24,
}
pub const N_RECT_TX_SIZES: usize = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Tx1dType {
    Dct = 0,
    Identity = 1,
    Adst = 2,
    FlipAdst = 3,
    Ddt = 4,
    FlipDdt = 5,
    Wht = 6,
}
pub const N_TX_1D_TYPES: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxClass {
    Class2D = 0,
    Class2DInv = 1,
    ClassH = 2,
    ClassV = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum IntraPredMode {
    #[default]
    DcPred = 0,
    VertPred = 1,
    HorPred = 2,
    DiagDownLeftPred = 3,
    DiagDownRightPred = 4,
    VertRightPred = 5,
    HorDownPred = 6,
    HorUpPred = 7,
    VertLeftPred = 8,
    SmoothPred = 9,
    SmoothVPred = 10,
    SmoothHPred = 11,
    PaethPred = 12,
}
pub const N_INTRA_PRED_MODES: usize = 13;
pub const CFL_PRED: u8 = 13;
pub const N_UV_INTRA_PRED_MODES: usize = 14;
pub const LEFT_DC_PRED: u8 = 3;
pub const TOP_DC_PRED: u8 = 14;
pub const DC_128_PRED: u8 = 15;
pub const Z1_PRED: u8 = 16;
pub const Z2_PRED: u8 = 17;
pub const Z3_PRED: u8 = 18;
pub const DIP_PRED: u8 = 13;

pub const ANGLE_SMOOTH_LEFT_EDGE_FLAG: i32 = 1 << 9;
pub const ANGLE_SMOOTH_TOP_EDGE_FLAG: i32 = 1 << 10;
pub const ANGLE_USE_EDGE_FILTER_FLAG: i32 = 1 << 11;
pub const ANGLE_IBP_FLAG: i32 = 1 << 12;
pub const ANGLE_MRL_IDX_SHIFT: i32 = 13;
pub const ANGLE_MRL_IDX_MASK: i32 = 3 << 13;
pub const ANGLE_MULTI_MRL_FLAG: i32 = 1 << 15;
pub const ANGLE_HAS_LEFT_FLAG: i32 = 1 << 16;
pub const ANGLE_HAS_TOP_FLAG: i32 = 1 << 17;
pub const ANGLE_DIP_FLAG: i32 = 1 << 18;
pub const ANGLE_IS_LUMA: i32 = 1 << 19;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterIntraPredMode {
    DcPred = 0,
    VertPred = 1,
    HorPred = 2,
    SmoothPred = 3,
}
pub const N_INTER_INTRA_PRED_MODES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum BlockPartition {
    Invalid = -1,
    None = 0,
    H = 1,
    V = 2,
    H3 = 3,
    V3 = 4,
    H4A = 5,
    H4B = 6,
    V4A = 7,
    V4B = 8,
    Split = 9,
}
pub const N_PARTITIONS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxPartition {
    None = 0,
    Split = 1,
    H = 2,
    V = 3,
    H4 = 4,
    V4 = 5,
    H5 = 6,
    V5 = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum BlockSize {
    Invalid = -1,
    Bs256x256 = 0,
    Bs256x128 = 1,
    Bs128x256 = 2,
    Bs128x128 = 3,
    Bs128x64 = 4,
    Bs64x128 = 5,
    Bs64x64 = 6,
    Bs64x32 = 7,
    Bs64x16 = 8,
    Bs64x8 = 9,
    Bs64x4 = 10,
    Bs32x64 = 11,
    Bs32x32 = 12,
    Bs32x16 = 13,
    Bs32x8 = 14,
    Bs32x4 = 15,
    Bs16x64 = 16,
    Bs16x32 = 17,
    Bs16x16 = 18,
    Bs16x8 = 19,
    Bs16x4 = 20,
    Bs8x64 = 21,
    Bs8x32 = 22,
    Bs8x16 = 23,
    Bs8x8 = 24,
    Bs8x4 = 25,
    Bs4x64 = 26,
    Bs4x32 = 27,
    Bs4x16 = 28,
    Bs4x8 = 29,
    Bs4x4 = 30,
}
pub const N_BS_SIZES: usize = 31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterPredMode {
    NearMv = 13,
    GlobalMv = 14,
    NewMv = 15,
    WarpMv = 16,
    WarpNewMv = 17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompInterPredMode {
    NearMvNearMv = 18,
    NearMvNewMv = 19,
    NewMvNearMv = 20,
    GlobalMvGlobalMv = 21,
    NewMvNewMv = 22,
    JointNewMv = 23,
    OpflNearMvNearMv = 24,
    OpflNearMvNewMv = 25,
    OpflNewMvNearMv = 26,
    OpflNewMvNewMv = 27,
    OpflJointNewMv = 28,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CompInterType {
    #[default]
    None = 0,
    Avg = 1,
    Wedge = 2,
    Seg = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MotionMode {
    #[default]
    Translation = 0,
    InterIntra = 1,
    WarpCausal = 2,
    WarpDelta = 3,
    WarpExtend = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CflType {
    #[default]
    Explicit = 0,
    Implicit = 1,
    Mhccp = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CflMhDir {
    #[default]
    Center = 0,
    Top = 1,
    Left = 2,
    All = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union Mv {
    pub c: MvXY,
    pub n: u64,
}

impl Default for Mv {
    fn default() -> Self {
        Self { n: 0 }
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct MvXY {
    pub y: i32,
    pub x: i32,
}

impl std::fmt::Debug for Mv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = unsafe { self.c };
        write!(f, "Mv({}, {})", c.y, c.x)
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(2))]
pub union RefPair {
    pub r: [i8; 2],
    pub pair: i16,
}

impl Default for RefPair {
    fn default() -> Self {
        Self { pair: 0 }
    }
}

impl std::fmt::Debug for RefPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = unsafe { self.r };
        write!(f, "RefPair({}, {})", r[0], r[1])
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct IsSm {
    pub a: i32,
    pub l: i32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union CflAlphaOrMhDir {
    pub cfl_alpha: [i8; 2],
    pub cfl_mh_dir: u8,
}

impl Default for CflAlphaOrMhDir {
    fn default() -> Self {
        Self { cfl_alpha: [0; 2] }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Av2BlockIntra {
    pub intrabc_mv: Mv,
    pub dpcm: [u8; 2],
    pub y_mode: u8,
    pub mrl_index: u8,
    pub multi_mrl: u8,
    pub dip: u8,
    pub morph_pred: u8,
    pub is_refmv: u8,
    pub is_qpel: u8,
    pub uv_mode: u8,
    pub pal_sz: u8,
    pub y_angle: i8,
    pub uv_angle: i8,
    pub cfl_type: i8,
    pub cfl: CflAlphaOrMhDir,
    pub is_sm: [IsSm; 2],
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Av2BlockInter {
    pub mv: [Mv; 2],
    pub wedge_idx: i8,
    pub wedge_sign: i8,
    pub mask_sign: u8,
    pub interintra_mode: u8,
    pub matrix: [i8; 4],
    pub drl_idx: [u8; 2],
    pub warp_ref_idx: u8,
    pub warpmv_with_mvd: u8,
    pub comp_type: u8,
    pub inter_mode: u8,
    pub motion_mode: u8,
    pub warp_ii: u8,
    pub cwp_idx: i8,
    pub mv_prec: i8,
    pub amvd: i8,
    pub bawp: [u8; 2],
    pub filter: u8,
    pub refine_mv: u8,
    pub mtxbak: [i32; 6],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union Av2BlockData {
    pub intra: Av2BlockIntra,
    pub inter: Av2BlockInter,
}

impl Default for Av2BlockData {
    fn default() -> Self {
        Self {
            inter: Av2BlockInter::default(),
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Av2Block {
    pub bs: i8,
    pub cbs: i8,
    pub is_intra: u8,
    pub intrabc: u8,
    pub seg_id: u8,
    pub skip_mode: u8,
    pub skip_txfm: u8,
    pub tx_part: u8,
    pub fsc: u8,
    pub tx_size_ll: u8,
    pub ref_pair: RefPair,
    pub data: Av2BlockData,
}

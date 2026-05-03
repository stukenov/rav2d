use crate::levels::{Mv, RefPair};

#[derive(Clone, Copy, Default)]
#[repr(C, align(2))]
pub struct TrajMap {
    pub y: i8,
    pub x: i8,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct SnglMvBlock {
    pub mv: Mv,
    pub r#ref: u8,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub union QMv {
    pub c: [i8; 2],
    pub n: u16,
}

impl Default for QMv {
    fn default() -> Self {
        Self { n: 0 }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct TemporalBlock {
    pub mv: TemporalBlockMv,
    pub r#ref: RefPair,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union TemporalBlockMv {
    pub mv: [QMv; 2],
    pub n: u32,
}

impl Default for TemporalBlockMv {
    fn default() -> Self {
        Self { n: 0 }
    }
}

impl Default for TemporalBlock {
    fn default() -> Self {
        Self {
            mv: TemporalBlockMv::default(),
            r#ref: RefPair::default(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Block {
    pub mv: [Mv; 2],
    pub r#ref: RefPair,
    pub bs: u8,
    pub mf: i8,
    pub ox4: u8,
    pub oy4: u8,
    pub subpel_filter: u8,
    pub warp_type: i8,
    pub lmv: [Mv; 2],
    pub m: [i32; 6],
}

impl Default for Block {
    fn default() -> Self {
        Self {
            mv: [Mv::default(); 2],
            r#ref: RefPair::default(),
            bs: 0,
            mf: 0,
            ox4: 0,
            oy4: 0,
            subpel_filter: 0,
            warp_type: 0,
            lmv: [Mv::default(); 2],
            m: [0; 6],
        }
    }
}

pub struct MfmvRef {
    pub r#ref: u8,
    pub tgt: i8,
    pub dir: u8,
}

pub struct Frame {
    pub iw4: i32,
    pub ih4: i32,
    pub iw8: i32,
    pub ih8: i32,
    pub sbsz: i32,
    pub mfmv_sbsz8: i32,
    pub mfmv_edge: i32,
    pub mfmv_k_shift: i32,
    pub use_ref_frame_mvs: i32,
    pub tip: FrameTip,
    pub ref_sign: [u8; 7],
    pub pocdiff: [i8; 7],
    pub ref_flip: u64,
    pub abspocdiff: [u8; 7],
    pub mfmv_mask: u8,
    pub mfmv: [MfmvRef; 4],
    pub mfmv_ref2cur: [i8; 4],
    pub mfmv_ref2ref: [[i8; 7]; 4],
    pub mfmv_ref2idx: [[i8; 7]; 4],
    pub mfmv_ref2sf: [[[i32; 2]; 7]; 4],
    pub n_mfmvs: i32,
    pub n_blocks: i32,
    pub rp: Vec<TemporalBlock>,
    pub rp_stride: isize,
    pub rp_proj: Vec<SnglMvBlock>,
    pub have_threading: bool,
    pub have_frame_threading: bool,
}

pub struct FrameTip {
    pub sf: [i32; 2],
    pub r#ref: RefPair,
    pub delta: i8,
}

pub struct TileRange {
    pub start: i32,
    pub end: i32,
}

pub struct MvBank {
    pub mv: [[[Mv; 2]; 4]; 9],
    pub cwp_idx: [[i8; 4]; 3],
    pub r#ref: [RefPair; 4],
    pub size: [u8; 9],
    pub idx: [u8; 9],
    pub hits: [u8; 2],
    pub avail: u8,
}

pub struct WarpBank {
    pub mat: [[[i32; 6]; 4]; 7],
    pub warp_type: [[i8; 4]; 7],
    pub hits: u8,
    pub size: [u8; 7],
    pub idx: [u8; 7],
}

pub struct Tile {
    pub rp_proj: Vec<SnglMvBlock>,
    pub ra: Vec<Block>,
    pub ra_tl: Block,
    pub r: Vec<Block>,
    pub tile_col: TileRange,
    pub tile_row: TileRange,
    pub bank: MvBank,
    pub warp: WarpBank,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_block_default() {
        let tb = TemporalBlock::default();
        assert_eq!(unsafe { tb.mv.n }, 0);
    }

    #[test]
    fn test_block_default() {
        let b = Block::default();
        assert_eq!(b.bs, 0);
        assert_eq!(b.mf, 0);
    }

    #[test]
    fn test_traj_map_size() {
        assert_eq!(std::mem::size_of::<TrajMap>(), 2);
    }
}

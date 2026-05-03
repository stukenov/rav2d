use crate::intops::{apply_sign, iclip, imax, ulog2};
use crate::levels::{Mv, MvXY, RefPair};

pub const INVALID_TRAJ: u16 = 0x8080;

static DIV_MULT: [u16; 32] = [
       0, 16384, 8192, 5461, 4096, 3276, 2730, 2340,
    2048,  1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024,   963,  910,  862,  819,  780,  744,  712,
     682,   655,  630,  606,  585,  564,  546,  528,
];

pub fn mv_projection(mv: Mv, num: i32, den: i32, min: i32, max: i32) -> Mv {
    debug_assert!(den > 0 && den < 32);
    debug_assert!(num > -32 && num < 32);
    let (y, x) = unsafe { (mv.c.y, mv.c.x) };
    let frac = num as i64 * DIV_MULT[den as usize] as i64;
    let py = y as i64 * frac;
    let px = x as i64 * frac;
    Mv {
        c: MvXY {
            y: iclip(((py + 8192 + (py >> 63)) >> 14) as i32, min, max),
            x: iclip(((px + 8192 + (px >> 63)) >> 14) as i32, min, max),
        },
    }
}

pub fn scale_mv(mv: Mv, sf: i32) -> Mv {
    let (y_in, x_in) = unsafe { (mv.c.y, mv.c.x) };
    let y = y_in as i64 * sf as i64;
    let x = x_in as i64 * sf as i64;
    Mv {
        c: MvXY {
            y: iclip(
                ((y + 0x2000 - (y < 0) as i64) >> 14) as i32,
                -0xffff,
                0xffff,
            ),
            x: iclip(
                ((x + 0x2000 - (x < 0) as i64) >> 14) as i32,
                -0xffff,
                0xffff,
            ),
        },
    }
}

pub fn quantize_mv_comp(absv: u32) -> u32 {
    debug_assert!(absv < 2048);
    if absv == 0 {
        return 0;
    }
    let nbits = iclip(ulog2(absv) - 4, 0, 6) as u32;
    let has_bits = (nbits != 0) as u32;
    let res = (absv - (16 * has_bits << nbits)) >> nbits;
    res + (nbits + has_bits) * 16
}

pub fn quantize_mv(mv: Mv) -> QMv {
    let (y, x) = unsafe { (mv.c.y, mv.c.x) };
    let absy = y.unsigned_abs();
    let absx = x.unsigned_abs();
    if imax(absx as i32, absy as i32) >= 2048 {
        return QMv { n: INVALID_TRAJ };
    }
    QMv {
        c: [
            apply_sign(quantize_mv_comp(absy) as i32, y) as i8,
            apply_sign(quantize_mv_comp(absx) as i32, x) as i8,
        ],
    }
}

pub fn dequantize_mv_comp(v: i32) -> i32 {
    let absv = v.unsigned_abs();
    debug_assert!(absv < 0x80);
    let nbits = (absv >> 4).wrapping_sub(if absv >= 16 { 1 } else { 0 });
    let has_bits = (nbits != 0) as u32;
    let mut res = (absv - (nbits + has_bits) * 16) << nbits;
    res += 16 * has_bits << nbits;
    if v < 0 { -(res as i32) } else { res as i32 }
}

pub fn dequantize_mv(mv: QMv) -> Mv {
    if unsafe { mv.n } == INVALID_TRAJ {
        return Mv {
            c: MvXY {
                y: crate::levels::INVALID_MV,
                x: 0,
            },
        };
    }
    let c = unsafe { mv.c };
    Mv {
        c: MvXY {
            y: dequantize_mv_comp(c[0] as i32),
            x: dequantize_mv_comp(c[1] as i32),
        },
    }
}

pub fn get_warpmv_proj(
    warp_type: i8,
    m: &[i32; 6],
    x: i32,
    y: i32,
    minx: i32,
    maxx: i32,
    miny: i32,
    maxy: i32,
) -> Mv {
    if warp_type <= 0 {
        return Mv { n: 0 };
    }
    let xc = (m[2] - (1 << 16)) * x + m[3] * y + m[0];
    let yc = (m[5] - (1 << 16)) * y + m[4] * x + m[1];
    let ry = iclip((yc + 0x1000 - (yc < 0) as i32) >> 13, -0xffff, 0xffff);
    let rx = iclip((xc + 0x1000 - (xc < 0) as i32) >> 13, -0xffff, 0xffff);
    Mv {
        c: MvXY {
            y: iclip(ry, miny, maxy),
            x: iclip(rx, minx, maxx),
        },
    }
}

pub fn abs_closest_ref(ref2ref: &[i8; 7], cur2ref: &[i8; 7], dir: bool) -> u32 {
    let mut b = 0xffu32;
    for n in 0..7 {
        let a = (ref2ref[n] as i32).unsigned_abs();
        if ((cur2ref[n] > 0 && ref2ref[n] > 0 && dir)
            || (cur2ref[n] < 0 && ref2ref[n] < 0 && !dir))
            && a < b
        {
            b = a;
        }
    }
    b
}

pub fn topo_insert(
    mut cnt: i32,
    idx: usize,
    order: &mut [i8],
    rev_order: &mut [i8],
    cnv: &[[i8; 7]],
    refcnt: &[u8],
) -> i32 {
    if rev_order[idx] != -1 {
        return cnt;
    }
    rev_order[idx] = 0;
    if refcnt[idx] != 0 {
        for n in 0..7 {
            let r_idx = cnv[idx][n];
            if r_idx == -1 {
                continue;
            }
            cnt = topo_insert(cnt, r_idx as usize, order, rev_order, cnv, refcnt);
        }
    }
    order[cnt as usize] = idx as i8;
    rev_order[idx] = cnt as i8;
    cnt + 1
}

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

    #[test]
    fn test_mv_projection_identity() {
        let mv = Mv { c: MvXY { y: 100, x: 200 } };
        let p = mv_projection(mv, 1, 1, -0xffff, 0xffff);
        let (y, x) = unsafe { (p.c.y, p.c.x) };
        assert_eq!(y, 100);
        assert_eq!(x, 200);
    }

    #[test]
    fn test_mv_projection_scale() {
        let mv = Mv { c: MvXY { y: 64, x: -32 } };
        let p = mv_projection(mv, 2, 1, -0xffff, 0xffff);
        let (y, x) = unsafe { (p.c.y, p.c.x) };
        assert_eq!(y, 128);
        assert_eq!(x, -64);
    }

    #[test]
    fn test_mv_projection_clamp() {
        let mv = Mv { c: MvXY { y: 10000, x: 10000 } };
        let p = mv_projection(mv, 16, 1, -1000, 1000);
        let (y, x) = unsafe { (p.c.y, p.c.x) };
        assert_eq!(y, 1000);
        assert_eq!(x, 1000);
    }

    #[test]
    fn test_scale_mv_identity() {
        let mv = Mv { c: MvXY { y: 100, x: -50 } };
        let s = scale_mv(mv, 1 << 14);
        let (y, x) = unsafe { (s.c.y, s.c.x) };
        assert_eq!(y, 100);
        assert_eq!(x, -50);
    }

    #[test]
    fn test_quantize_mv_comp_zero() {
        assert_eq!(quantize_mv_comp(0), 0);
    }

    #[test]
    fn test_quantize_mv_comp_small() {
        assert_eq!(quantize_mv_comp(1), 1);
        assert_eq!(quantize_mv_comp(15), 15);
        assert_eq!(quantize_mv_comp(16), 16);
        assert_eq!(quantize_mv_comp(31), 31);
        assert_eq!(quantize_mv_comp(32), 32);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        for v in [0, 1, 5, 15, 16, 32, 64, 128, 256, 512, 1024] {
            let q = quantize_mv_comp(v);
            let d = dequantize_mv_comp(q as i32) as u32;
            assert!(
                d <= v && v - d < (1 << ((v as f64).log2() as u32).saturating_sub(3)),
                "v={v} q={q} d={d}"
            );
        }
    }

    #[test]
    fn test_quantize_mv_large_returns_invalid() {
        let mv = Mv { c: MvXY { y: 3000, x: 0 } };
        let q = quantize_mv(mv);
        assert_eq!(unsafe { q.n }, INVALID_TRAJ);
    }

    #[test]
    fn test_dequantize_mv_invalid() {
        let q = QMv { n: INVALID_TRAJ };
        let mv = dequantize_mv(q);
        assert_eq!(unsafe { mv.c.y }, crate::levels::INVALID_MV);
    }

    #[test]
    fn test_dequantize_mv_comp_basic() {
        assert_eq!(dequantize_mv_comp(0), 0);
        assert_eq!(dequantize_mv_comp(1), 1);
        assert_eq!(dequantize_mv_comp(-1), -1);
        assert_eq!(dequantize_mv_comp(15), 15);
    }

    #[test]
    fn test_get_warpmv_proj_disabled() {
        let m = [0i32; 6];
        let mv = get_warpmv_proj(0, &m, 10, 20, -100, 100, -100, 100);
        assert_eq!(unsafe { mv.n }, 0);
    }

    #[test]
    fn test_get_warpmv_proj_identity() {
        let m = [0, 0, 1 << 16, 0, 0, 1 << 16];
        let mv = get_warpmv_proj(1, &m, 100, 200, -0xffff, 0xffff, -0xffff, 0xffff);
        let (y, x) = unsafe { (mv.c.y, mv.c.x) };
        assert_eq!(y, 0);
        assert_eq!(x, 0);
    }

    #[test]
    fn test_get_warpmv_proj_clamp() {
        let m = [1_000_000, 1_000_000, 1 << 16, 0, 0, 1 << 16];
        let mv = get_warpmv_proj(1, &m, 0, 0, -100, 100, -100, 100);
        let (y, x) = unsafe { (mv.c.y, mv.c.x) };
        assert_eq!(y, 100);
        assert_eq!(x, 100);
    }

    #[test]
    fn test_abs_closest_ref_basic() {
        let ref2ref = [1, -2, 3, 0, 0, 0, 0i8];
        let cur2ref = [1, -1, 1, 0, 0, 0, 0i8];
        assert_eq!(abs_closest_ref(&ref2ref, &cur2ref, true), 1);
    }

    #[test]
    fn test_abs_closest_ref_no_match() {
        let ref2ref = [1, 2, 3, 4, 5, 6, 7i8];
        let cur2ref = [-1, -2, -3, -4, -5, -6, -7i8];
        assert_eq!(abs_closest_ref(&ref2ref, &cur2ref, true), 0xff);
    }

    #[test]
    fn test_topo_insert_basic() {
        let mut order = [-1i8; 7];
        let mut rev_order = [-1i8; 7];
        let cnv = [[-1i8; 7]; 7];
        let refcnt = [0u8; 7];
        let cnt = topo_insert(0, 3, &mut order, &mut rev_order, &cnv, &refcnt);
        assert_eq!(cnt, 1);
        assert_eq!(order[0], 3);
        assert_eq!(rev_order[3], 0);
    }

    #[test]
    fn test_topo_insert_chain() {
        let mut order = [-1i8; 7];
        let mut rev_order = [-1i8; 7];
        let mut cnv = [[-1i8; 7]; 7];
        cnv[0][0] = 1;
        cnv[1][0] = 2;
        let refcnt = [1u8; 7];
        let cnt = topo_insert(0, 0, &mut order, &mut rev_order, &cnv, &refcnt);
        assert_eq!(cnt, 3);
        assert_eq!(order[0], 2);
        assert_eq!(order[1], 1);
        assert_eq!(order[2], 0);
    }
}

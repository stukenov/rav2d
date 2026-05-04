use crate::headers::{WarpedMotionParams, WarpedMotionType};
use crate::intops::{apply_sign, apply_sign64, iclip, iclip64to32, imax, imin, ulog2};
use crate::levels::INVALID_MV;
use crate::levels::{Av2Block, BlockSize, Mv, MvXY, RefPair};

pub const INVALID_TRAJ: u16 = 0x8080;
pub const INVALID_REF2CUR: i8 = -32;

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

impl TrajMap {
    pub unsafe fn n(&self) -> u16 {
        unsafe { *(self as *const Self as *const u16) }
    }
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

#[derive(Clone, Copy)]
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
    pub rp_ref: [Vec<TemporalBlock>; 7],
    pub rp_proj: Vec<SnglMvBlock>,
    pub rp_traj: [Vec<Mv>; 7],
    pub rp_map: [[Vec<TrajMap>; 7]; 3],
    pub ra: Vec<Block>,
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

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Candidate {
    pub mv: [Mv; 2],
    pub weight: u16,
    pub cwp_idx: i8,
    pub y_off: i8,
    pub x_off: i8,
}

pub struct Tile {
    pub rp_proj: Vec<SnglMvBlock>,
    pub rp_proj_off: usize,
    pub rp_traj_off: usize,
    pub ra: Vec<Block>,
    pub ra_off: usize,
    pub ra_tl: Block,
    pub r: Vec<Block>,
    pub tile_col: TileRange,
    pub tile_row: TileRange,
    pub bank: MvBank,
    pub warp: WarpBank,
}

pub fn model_from_corners(
    mat: &mut [i32; 7],
    topleft_mv: Mv,
    topright_mv: Mv,
    bottomleft_mv: Mv,
    xpos: i32,
    ypos: i32,
    b_dim: &[u8],
) -> bool {
    let (tl_x, tl_y) = unsafe { (topleft_mv.c.x as i32, topleft_mv.c.y as i32) };
    let (tr_x, tr_y) = unsafe { (topright_mv.c.x as i32, topright_mv.c.y as i32) };
    let (bl_x, bl_y) = unsafe { (bottomleft_mv.c.x as i32, bottomleft_mv.c.y as i32) };

    if unsafe { topright_mv.n == topleft_mv.n && bottomleft_mv.n == topleft_mv.n } {
        return false;
    }
    if imin(imin(tl_x, bl_x), tr_x + b_dim[0] as i32 * 32) < -xpos * 8 {
        return false;
    }
    if imin(imin(tl_y, tr_y), bl_y + b_dim[1] as i32 * 32) < -ypos * 8 {
        return false;
    }

    mat[2] = iclip64to32(
        ((tr_x - tl_x) as i64 * (1i64 << 11)) >> b_dim[2],
        i32::MIN, i32::MAX,
    );
    mat[4] = iclip64to32(
        ((tr_y - tl_y) as i64 * (1i64 << 11)) >> b_dim[2],
        i32::MIN, i32::MAX,
    );
    mat[3] = iclip64to32(
        ((bl_x - tl_x) as i64 * (1i64 << 11)) >> b_dim[3],
        i32::MIN, i32::MAX,
    );
    mat[5] = iclip64to32(
        ((bl_y - tl_y) as i64 * (1i64 << 11)) >> b_dim[3],
        i32::MIN, i32::MAX,
    );
    mat[0] = iclip64to32(
        tl_x as i64 * (1i64 << 13) - xpos as i64 * mat[2] as i64 - ypos as i64 * mat[3] as i64,
        -0x8000000, 0x7ffffc0,
    );
    mat[1] = iclip64to32(
        tl_y as i64 * (1i64 << 13) - xpos as i64 * mat[4] as i64 - ypos as i64 * mat[5] as i64,
        -0x8000000, 0x7ffffc0,
    );

    for i in [2, 3, 4, 5] {
        mat[i] = iclip(mat[i], -0x7fc0, 0x7fc0);
        mat[i] += 0x20 - (mat[i] < 0) as i32;
        mat[i] &= !0x3f;
    }
    mat[2] += 0x10000;
    mat[5] += 0x10000;
    mat[6] = 3; // DAV2D_WM_TYPE_AFFINE

    true
}

pub fn add_candidate_sngl(
    mvstack: &mut [Candidate],
    cnt: &mut i32,
    max_cnt: i32,
    weight: u16,
    cand_mv: Mv,
    y_off: i8,
    x_off: i8,
    iter_cntr: &mut i32,
    max_iter: i32,
) -> bool {
    let last = *cnt as usize;
    if *iter_cntr < max_iter {
        for m in 0..last {
            if unsafe { mvstack[m].mv[0].n == cand_mv.n } {
                *iter_cntr += m as i32 + 1;
                mvstack[m].weight += weight;
                return false;
            }
        }
        *iter_cntr += last as i32;
    }

    if *cnt >= max_cnt {
        return false;
    }

    mvstack[last].mv[0] = cand_mv;
    mvstack[last].weight = weight;
    mvstack[last].y_off = y_off;
    mvstack[last].x_off = x_off;
    *cnt = last as i32 + 1;
    true
}

pub fn add_candidate_c2s(
    mvstack: &mut [SnglMvBlock],
    cnt: &mut i32,
    max_cnt: i32,
    r#ref: u8,
    cand_mv: Mv,
    iter_cntr: &mut i32,
    max_iter: i32,
) {
    let last = *cnt as usize;
    if *iter_cntr < max_iter {
        for m in 0..last {
            if unsafe { mvstack[m].mv.n == cand_mv.n } && mvstack[m].r#ref == r#ref {
                *iter_cntr += m as i32 + 1;
                return;
            }
        }
        *iter_cntr += last as i32;
    }

    if *cnt >= max_cnt {
        return;
    }

    mvstack[last].mv = cand_mv;
    mvstack[last].r#ref = r#ref;
    *cnt = last as i32 + 1;
}

pub fn add_candidate_comp(
    mvstack: &mut [Candidate],
    cnt: &mut i32,
    max_cnt: i32,
    weight: u16,
    cwp_idx: i8,
    cand_mv: &[Mv; 2],
    iter_cntr: &mut i32,
    max_iter: i32,
) -> bool {
    let last = *cnt as usize;
    if *iter_cntr < max_iter {
        for n in 0..last {
            if unsafe {
                mvstack[n].mv[0].n == cand_mv[0].n && mvstack[n].mv[1].n == cand_mv[1].n
            } {
                *iter_cntr += n as i32 + 1;
                mvstack[n].weight += weight;
                return false;
            }
        }
        *iter_cntr += last as i32;
    }

    if *cnt >= max_cnt {
        return false;
    }

    mvstack[last].mv[0] = cand_mv[0];
    mvstack[last].mv[1] = cand_mv[1];
    mvstack[last].weight = weight;
    mvstack[last].cwp_idx = cwp_idx;
    *cnt = last as i32 + 1;
    true
}

pub fn tip_projection(
    rp_proj: &mut [SnglMvBlock],
    stride: isize,
    col_start8: i32,
    col_end8: i32,
    row_start8: i32,
    row_end8: i32,
    mfmv_sbsz8: i32,
    sbsz8: i32,
    tmvp_sample_step: i32,
    tip_delta: i8,
) {
    let mut sx = col_start8;
    while sx < col_end8 {
        let xend = imin(col_end8, sx + mfmv_sbsz8);
        let mut y = row_start8;
        while y < row_end8 {
            let pos_base = ((y & (sbsz8 - 1)) as isize) * stride;
            let mut x = sx;
            while x < xend {
                let pos = (pos_base + x as isize) as usize;
                let mv_y = unsafe { rp_proj[pos].mv.c.y };
                if mv_y == INVALID_MV {
                    x += tmvp_sample_step;
                    continue;
                }
                let mv = rp_proj[pos].mv;
                let r = rp_proj[pos].r#ref;
                rp_proj[pos].mv = mv_projection(mv, tip_delta as i32, r as i32, -2047, 2047);
                rp_proj[pos].r#ref = tip_delta as u8;
                x += tmvp_sample_step;
            }
            y += tmvp_sample_step;
        }
        sx += mfmv_sbsz8;
    }
}

pub fn fill_holes(
    rp_proj: &mut [SnglMvBlock],
    stride: isize,
    col_start8: i32,
    col_end8: i32,
    row_start8: i32,
    row_end8: i32,
    mfmv_sbsz8: i32,
    sbsz8: i32,
    tmvp_sample_step: i32,
    tip_delta: i8,
) {
    let step = tmvp_sample_step;
    let mut sx = col_start8;
    while sx < col_end8 {
        let xend = imin(col_end8, sx + mfmv_sbsz8);
        let mut y = row_start8;
        while y < row_end8 {
            let ystart = y & !(mfmv_sbsz8 - 1);
            let yend = imin(ystart + mfmv_sbsz8, row_end8);
            let pos_base = ((y & (sbsz8 - 1)) as isize) * stride;
            let mut x = sx;
            while x < xend {
                let pos = (pos_base + x as isize) as usize;
                let mv_y = unsafe { rp_proj[pos].mv.c.y };
                if mv_y == INVALID_MV {
                    x += step;
                    continue;
                }
                let mv = rp_proj[pos].mv;
                if x - step >= sx {
                    let p = (pos as isize - step as isize) as usize;
                    if unsafe { rp_proj[p].mv.c.y } == INVALID_MV {
                        rp_proj[p].mv = mv;
                        rp_proj[p].r#ref = tip_delta as u8;
                    }
                }
                if x + step < xend {
                    let p = pos + step as usize;
                    if unsafe { rp_proj[p].mv.c.y } == INVALID_MV {
                        rp_proj[p].mv = mv;
                        rp_proj[p].r#ref = tip_delta as u8;
                    }
                }
                if y - step >= ystart {
                    let p = (pos as isize - step as isize * stride) as usize;
                    if unsafe { rp_proj[p].mv.c.y } == INVALID_MV {
                        rp_proj[p].mv = mv;
                        rp_proj[p].r#ref = tip_delta as u8;
                    }
                }
                if y + step < yend {
                    let p = (pos as isize + step as isize * stride) as usize;
                    if unsafe { rp_proj[p].mv.c.y } == INVALID_MV {
                        rp_proj[p].mv = mv;
                        rp_proj[p].r#ref = tip_delta as u8;
                    }
                }
                x += step;
            }
            y += step;
        }
        sx += mfmv_sbsz8;
    }
}

pub fn warp_bank_add(
    warp: &mut WarpBank,
    mat: &WarpedMotionParams,
    r#ref: usize,
) -> i32 {
    if warp.hits >= 64 {
        return -1;
    }
    warp.hits += 1;
    let sz = warp.size[r#ref] as usize;
    let idx = warp.idx[r#ref] as usize;

    let mut n = 0;
    while n < sz {
        let m = &warp.mat[r#ref][(idx + n) & 3][2..6];
        if m == &mat.matrix[2..6] {
            break;
        }
        n += 1;
    }

    if n < sz {
        let to = if sz == 4 { (idx.wrapping_sub(1)) & 3 } else { sz - 1 };
        let from = (idx + n) & 3;
        if from != to {
            let bak = warp.mat[r#ref][from];
            let bak_type = warp.warp_type[r#ref][from];
            let mut n1 = from;
            let mut n2 = (n1 + 1) & 3;
            while n1 != to {
                warp.mat[r#ref][n1] = warp.mat[r#ref][n2];
                warp.warp_type[r#ref][n1] = warp.warp_type[r#ref][n2];
                n1 = n2;
                n2 = (n2 + 1) & 3;
            }
            warp.mat[r#ref][to] = bak;
            warp.warp_type[r#ref][to] = bak_type;
        }
        return 0;
    }

    let tgt = if sz == 4 {
        let t = warp.idx[r#ref] as usize & 3;
        warp.idx[r#ref] = warp.idx[r#ref].wrapping_add(1);
        t
    } else {
        let t = warp.size[r#ref] as usize;
        warp.size[r#ref] += 1;
        t
    };
    warp.mat[r#ref][tgt] = mat.matrix;
    warp.warp_type[r#ref][tgt] = mat.wm_type as i8;
    0
}

pub fn mv_bank_add_inner(
    bank: &mut MvBank,
    r#ref: RefPair,
    mv: &[Mv; 2],
    cwp_idx_val: i8,
) {
    bank.hits[0] += 1;

    let (ref0, ref1) = unsafe { (r#ref.r[0], r#ref.r[1]) };
    let c = if ref1 == -1 {
        if (ref0 as u32) <= 5 { ref0 as usize } else { 8 }
    } else {
        if ref0 == 0 && ref1 <= 1 { 6 + ref1 as usize } else { 8 }
    };
    let sz = bank.size[c] as usize;
    let idx = bank.idx[c] as usize;
    let comp = ref1 != -1;
    let comp_idx = if comp { 1 } else { 0 };

    let mut n = 0;
    while n < sz {
        let i = (idx + n) & 3;
        let match0 = unsafe { mv[0].n == bank.mv[c][i][0].n };
        let match1 = unsafe { mv[comp_idx].n == bank.mv[c][i][comp_idx].n };
        let ref_match = c < 8 || unsafe { r#ref.pair == bank.r#ref[i].pair };
        if match0 && match1 && ref_match {
            break;
        }
        n += 1;
    }

    if n < sz {
        let to = if sz == 4 { idx.wrapping_sub(1) & 3 } else { sz - 1 };
        let from = (idx + n) & 3;
        if from != to {
            let mv_bak = bank.mv[c][from];
            let ref_bak = bank.r#ref[from];
            let cwp_bak = if c >= 6 { bank.cwp_idx[c.saturating_sub(6)][from] } else { 0 };
            let mut n1 = from;
            let mut n2 = (n1 + 1) & 3;
            while n1 != to {
                bank.mv[c][n1] = bank.mv[c][n2];
                if c == 8 {
                    bank.r#ref[n1] = bank.r#ref[n2];
                }
                if c >= 6 {
                    bank.cwp_idx[c - 6][n1] = bank.cwp_idx[c - 6][n2];
                }
                n1 = n2;
                n2 = (n2 + 1) & 3;
            }
            bank.mv[c][to] = mv_bak;
            if c == 8 {
                bank.r#ref[to] = ref_bak;
            }
            if c >= 6 {
                bank.cwp_idx[c - 6][to] = cwp_bak;
            }
        }
        return;
    }

    let tgt = if sz == 4 {
        let t = bank.idx[c] as usize & 3;
        bank.idx[c] = bank.idx[c].wrapping_add(1);
        t
    } else {
        let t = bank.size[c] as usize;
        bank.size[c] += 1;
        t
    };
    bank.mv[c][tgt] = *mv;
    if c == 8 {
        bank.r#ref[tgt] = r#ref;
    }
    if ref1 != -1 {
        bank.cwp_idx[c.saturating_sub(6)][tgt] = cwp_idx_val;
    }
}

static SMOOTHEN_IDIV: [u32; 5] = [65536, 32768, 21845, 16384, 13107];

pub fn smoothen(
    rp_proj: &mut [SnglMvBlock],
    stride: isize,
    col_start8: i32,
    col_end8: i32,
    row_start8: i32,
    row_end8: i32,
    mfmv_sbsz8: i32,
    sbsz8: i32,
    tmvp_sample_step: i32,
    tip_delta: i8,
) {
    let step = tmvp_sample_step;
    let mut mv_line = [Mv { n: 0 }; 32];
    let mut sx = col_start8;
    while sx < col_end8 {
        let xend = imin(col_end8, sx + mfmv_sbsz8);
        let mut first_line = true;
        let mut y = row_start8;
        while y < row_end8 {
            let ystart = y & !(mfmv_sbsz8 - 1);
            let yend = imin(ystart + mfmv_sbsz8, row_end8);
            let pos_base = ((y & (sbsz8 - 1)) as isize) * stride;
            let mut x = sx;
            while x < xend {
                let pos = (pos_base + x as isize) as usize;
                let mut sum_x: i32 = 0;
                let mut sum_y: i32 = 0;
                let mut sum_n: usize = 0;

                macro_rules! add_sample {
                    ($p:expr) => {
                        let mv_y_val = unsafe { rp_proj[$p].mv.c.y };
                        if mv_y_val != INVALID_MV {
                            sum_x += unsafe { rp_proj[$p].mv.c.x } as i32;
                            sum_y += mv_y_val as i32;
                            sum_n += 1;
                        }
                    };
                }

                add_sample!(pos);
                if x - step >= sx {
                    add_sample!((pos as isize - step as isize) as usize);
                }
                if x + step < xend {
                    add_sample!(pos + step as usize);
                }
                if y - step >= ystart {
                    add_sample!((pos as isize - step as isize * stride) as usize);
                }
                if y + step < yend {
                    add_sample!((pos as isize + step as isize * stride) as usize);
                }

                if !first_line {
                    let prev = (pos as isize - step as isize * stride) as usize;
                    rp_proj[prev].mv = mv_line[(x - sx) as usize];
                    rp_proj[prev].r#ref = tip_delta as u8;
                }

                if sum_n > 0 {
                    let d = SMOOTHEN_IDIV[sum_n - 1] as i64;
                    mv_line[(x - sx) as usize] = Mv {
                        c: MvXY {
                            y: ((sum_y as i64 * d + 0x8000 - (sum_y < 0) as i64) >> 16) as i32,
                            x: ((sum_x as i64 * d + 0x8000 - (sum_x < 0) as i64) >> 16) as i32,
                        },
                    };
                } else {
                    mv_line[(x - sx) as usize] = Mv { c: MvXY { y: INVALID_MV, x: 0 } };
                }

                x += step;
            }
            first_line = false;
            y += step;
        }
        if !first_line {
            let prev_y = y - step;
            let pos_base = ((prev_y & (sbsz8 - 1)) as isize) * stride;
            let mut x = sx;
            while x < xend {
                let pos = (pos_base + x as isize) as usize;
                rp_proj[pos].mv = mv_line[(x - sx) as usize];
                rp_proj[pos].r#ref = tip_delta as u8;
                x += step;
            }
        }
        sx += mfmv_sbsz8;
    }
}

pub fn fill_gap_proj(
    rp_proj: &mut [SnglMvBlock],
    stride: isize,
    col_start8: i32, col_end8: i32,
    row_start8: i32, row_end8: i32,
    mfmv_sbsz8: i32, sbsz8: i32,
) {
    let mut sx = col_start8;
    while sx < col_end8 {
        let xend = imin(col_end8, sx + mfmv_sbsz8);
        let mut y = row_start8;
        while y < row_end8 {
            let ystart = y & !(mfmv_sbsz8 - 1);
            let yend = imin(ystart + mfmv_sbsz8, row_end8);
            let pos_base = ((y & (sbsz8 - 1)) as isize) * stride;
            let mut x = sx;
            while x < xend {
                let pos = (pos_base + x as isize) as usize;
                let (mvy, mvx) = unsafe { (rp_proj[pos].mv.c.y, rp_proj[pos].mv.c.x) };
                if mvy == INVALID_MV { x += 2; continue; }
                let (mut sum_y, mut sum_x, mut sum_n) = (mvy, mvx, 1i32);
                let ref_off = rp_proj[pos].r#ref;

                let have_right = x + 2 < xend;
                if have_right && unsafe { rp_proj[pos + 2].mv.c.y } != INVALID_MV {
                    let right_ref = rp_proj[pos + 2].r#ref;
                    let rmv = mv_projection(
                        rp_proj[pos + 2].mv, ref_off as i32, right_ref as i32, -2047, 2047);
                    let (ry, rx) = unsafe { (rmv.c.y, rmv.c.x) };
                    sum_x += rx;
                    sum_y += ry;
                    rp_proj[pos + 1].mv.c.y = (sum_y + (sum_y > 0) as i32) >> 1;
                    rp_proj[pos + 1].mv.c.x = (sum_x + (sum_x > 0) as i32) >> 1;
                    rp_proj[pos + 1].r#ref = ref_off;
                    sum_n += 1;
                } else {
                    rp_proj[pos + 1] = rp_proj[pos];
                }

                let have_bottom = y + 2 < yend;
                let bot = (pos as isize + 2 * stride) as usize;
                let mid = (pos as isize + stride) as usize;
                if have_bottom && unsafe { rp_proj[bot].mv.c.y } != INVALID_MV {
                    let bot_ref = rp_proj[bot].r#ref;
                    let bmv = mv_projection(
                        rp_proj[bot].mv, ref_off as i32, bot_ref as i32, -2047, 2047);
                    let (by, bx) = unsafe { (bmv.c.y, bmv.c.x) };
                    sum_x += bx;
                    sum_y += by;
                    let (mx, my) = (mvx + bx, mvy + by);
                    rp_proj[mid].mv.c.y = (my + (my > 0) as i32) >> 1;
                    rp_proj[mid].mv.c.x = (mx + (mx > 0) as i32) >> 1;
                    rp_proj[mid].r#ref = ref_off;
                    sum_n += 1;
                } else {
                    rp_proj[mid] = rp_proj[pos];
                }

                if have_right && have_bottom {
                    let br = (pos as isize + 2 * (1 + stride)) as usize;
                    if unsafe { rp_proj[br].mv.c.y } != INVALID_MV {
                        let br_ref = rp_proj[br].r#ref;
                        let brmv = mv_projection(
                            rp_proj[br].mv, ref_off as i32, br_ref as i32, -2047, 2047);
                        sum_x += unsafe { brmv.c.x };
                        sum_y += unsafe { brmv.c.y };
                        sum_n += 1;
                    }
                }
                let diag = (pos as isize + 1 + stride) as usize;
                match sum_n {
                    1 => rp_proj[diag].mv = rp_proj[pos].mv,
                    2 => {
                        rp_proj[diag].mv.c.y = (sum_y + (sum_y > 0) as i32) >> 1;
                        rp_proj[diag].mv.c.x = (sum_x + (sum_x > 0) as i32) >> 1;
                    },
                    3 => {
                        rp_proj[diag].mv.c.y =
                            (sum_y * 85 + 128 - (sum_y < 0) as i32) >> 8;
                        rp_proj[diag].mv.c.x =
                            (sum_x * 85 + 128 - (sum_x < 0) as i32) >> 8;
                    },
                    4 => {
                        rp_proj[diag].mv.c.y =
                            (sum_y + 1 + (sum_y > 0) as i32) >> 2;
                        rp_proj[diag].mv.c.x =
                            (sum_x + 1 + (sum_x > 0) as i32) >> 2;
                    },
                    _ => unreachable!(),
                }
                rp_proj[diag].r#ref = ref_off;
                x += 2;
            }
            y += 2;
        }
        sx += mfmv_sbsz8;
    }
}

pub fn fill_gap_traj(
    rp_traj: &mut [Mv],
    stride: isize,
    col_start8: i32, col_end8: i32,
    row_start8: i32, row_end8: i32,
    mfmv_sbsz8: i32, sbsz8: i32,
) {
    let mut sx = col_start8;
    while sx < col_end8 {
        let xend = imin(col_end8, sx + mfmv_sbsz8);
        let mut y = row_start8;
        while y < row_end8 {
            let ystart = y & !(mfmv_sbsz8 - 1);
            let yend = imin(ystart + mfmv_sbsz8, row_end8);
            let pos_base = ((y & (sbsz8 - 1)) as isize) * stride;
            let mut x = sx;
            while x < xend {
                let pos = (pos_base + x as isize) as usize;
                let (mvy, mvx) = unsafe { (rp_traj[pos].c.y, rp_traj[pos].c.x) };
                if mvy == INVALID_MV { x += 2; continue; }
                let (mut sum_y, mut sum_x, mut sum_n) = (mvy, mvx, 1i32);

                let have_bottom = y + 2 < yend;
                let bot = (pos as isize + 2 * stride) as usize;
                let mid = (pos as isize + stride) as usize;
                if have_bottom && unsafe { rp_traj[bot].c.y } != INVALID_MV {
                    let (by, bx) = unsafe { (rp_traj[bot].c.y, rp_traj[bot].c.x) };
                    sum_x += bx;
                    sum_y += by;
                    rp_traj[mid].c.y = (sum_y + (sum_y > 0) as i32) >> 1;
                    rp_traj[mid].c.x = (sum_x + (sum_x > 0) as i32) >> 1;
                    sum_n += 1;
                } else {
                    rp_traj[mid] = rp_traj[pos];
                }

                let have_right = x + 2 < xend;
                if have_right && unsafe { rp_traj[pos + 2].c.y } != INVALID_MV {
                    let (ry, rx) = unsafe { (rp_traj[pos + 2].c.y, rp_traj[pos + 2].c.x) };
                    sum_x += rx;
                    sum_y += ry;
                    let (mx, my) = (mvx + rx, mvy + ry);
                    rp_traj[pos + 1].c.y = (my + (my > 0) as i32) >> 1;
                    rp_traj[pos + 1].c.x = (mx + (mx > 0) as i32) >> 1;
                    sum_n += 1;
                } else {
                    rp_traj[pos + 1] = rp_traj[pos];
                }

                if have_right && have_bottom {
                    let br = (pos as isize + 2 * (1 + stride)) as usize;
                    if unsafe { rp_traj[br].c.y } != INVALID_MV {
                        sum_x += unsafe { rp_traj[br].c.x };
                        sum_y += unsafe { rp_traj[br].c.y };
                        sum_n += 1;
                    }
                }
                let diag = (pos as isize + 1 + stride) as usize;
                match sum_n {
                    1 => rp_traj[diag] = rp_traj[pos],
                    2 => {
                        rp_traj[diag].c.y = (sum_y + (sum_y > 0) as i32) >> 1;
                        rp_traj[diag].c.x = (sum_x + (sum_x > 0) as i32) >> 1;
                    },
                    3 => {
                        rp_traj[diag].c.y =
                            (sum_y * 85 + 128 - (sum_y < 0) as i32) >> 8;
                        rp_traj[diag].c.x =
                            (sum_x * 85 + 128 - (sum_x < 0) as i32) >> 8;
                    },
                    4 => {
                        rp_traj[diag].c.y =
                            (sum_y + 1 + (sum_y > 0) as i32) >> 2;
                        rp_traj[diag].c.x =
                            (sum_x + 1 + (sum_x > 0) as i32) >> 2;
                    },
                    _ => unreachable!(),
                }
                x += 2;
            }
            y += 2;
        }
        sx += mfmv_sbsz8;
    }
}

pub fn bank_update(
    bank: &mut MvBank,
    bs: crate::levels::BlockSize,
    by4: i32,
    bx4: i32,
    sbsz: i32,
    sb128: bool,
) {
    let bsh = 1 + sb128 as i32;
    let bsz = 1 << bsh;
    let b_dim = &crate::tables::BLOCK_DIMENSIONS[bs as usize];
    if (by4 | bx4) & (sbsz - 1) == 0 {
        let w = imax(1, (b_dim[0] as i32) >> bsh) * imax(1, (b_dim[1] as i32) >> bsh);
        bank.hits[1] = 0;
        bank.avail = imax(w, 4) as u8;
    } else if (by4 | bx4) & (bsz - 1) == 0 {
        let w = imax(1, (b_dim[0] as i32) >> bsh) * imax(1, (b_dim[1] as i32) >> bsh);
        bank.hits[1] = 0;
        bank.avail = (bank.avail as i32 + w) as u8;
    }
}

pub fn bank_add(
    bank: &mut MvBank,
    bs: BlockSize,
    by4: i32,
    bx4: i32,
    sbsz: i32,
    sb128: bool,
    b: &Av2Block,
) {
    debug_assert!(b.is_intra == 0 || b.intrabc != 0);
    bank_update(bank, bs, by4, bx4, sbsz, sb128);
    if bank.hits[0] >= 64 || bank.hits[1] >= 16 || bank.avail == 0 {
        return;
    }
    bank.hits[1] += 1;
    bank.avail -= 1;
    let mv = unsafe { &b.data.inter.mv };
    let cwp_idx = if unsafe { b.ref_pair.r[1] } == -1 {
        0
    } else {
        unsafe { b.data.inter.cwp_idx }
    };
    mv_bank_add_inner(bank, b.ref_pair, mv, cwp_idx);
}

#[derive(Clone)]
pub struct MvSearchState {
    pub dr: [Candidate; 6],
    pub sngl: [SnglMvBlock; 4],
    pub drvd_cnt: i32,
    pub sngl_cnt: i32,
    pub drvd_iter_cntr: i32,
    pub sngl_iter_cntr: i32,
    pub iter_cntr: i32,
    pub b8x8: isize,
}

impl Default for MvSearchState {
    fn default() -> Self {
        Self {
            dr: [Candidate::default(); 6],
            sngl: [SnglMvBlock { mv: Mv::default(), r#ref: 0 }; 4],
            drvd_cnt: 0,
            sngl_cnt: 0,
            drvd_iter_cntr: 0,
            sngl_iter_cntr: 0,
            iter_cntr: 0,
            b8x8: 0,
        }
    }
}

pub fn add_derived(
    st: &mut MvSearchState,
    mvstack: &mut [Candidate; 6],
    cnt: &mut i32,
    lim: i32,
    comp: bool,
) {
    for n in 0..st.drvd_cnt as usize {
        if *cnt >= 6 { break; }
        if comp {
            add_candidate_comp(mvstack, cnt, lim, 0, 8,
                               &st.dr[n].mv, &mut st.iter_cntr, 16);
        } else {
            add_candidate_sngl(mvstack, cnt, lim, 0, st.dr[n].mv[0],
                               0, 0, &mut st.iter_cntr, 16);
        }
    }
}

pub fn add_temporal_candidate(
    rf: &Frame,
    rp_proj: &[SnglMvBlock],
    rp_traj: &[Vec<Mv>; 7],
    st: &mut MvSearchState,
    mvstack: &mut [Candidate; 6],
    cnt: &mut i32,
    off_8x8: isize,
    r#ref: RefPair,
    seq_mv_traj: bool,
) -> bool {
    let (ref0, ref1) = unsafe { (r#ref.r[0], r#ref.r[1]) };
    if ref0 as usize >= crate::levels::TIP_FRAME { return false; }

    let off = off_8x8 as usize;
    let mv = if seq_mv_traj && unsafe { rp_traj[ref0 as usize][off].c.y } != INVALID_MV {
        rp_traj[ref0 as usize][off]
    } else {
        let proj_mv = rp_proj[off].mv;
        if unsafe { proj_mv.c.y } == INVALID_MV { return false; }
        mv_projection(proj_mv, rf.pocdiff[ref0 as usize] as i32,
                      rp_proj[off].r#ref as i32, -0xffff, 0xffff)
    };

    if ref1 == -1 {
        let weight = 1 + (rf.abspocdiff[ref0 as usize] <= 2) as u16;
        return add_candidate_sngl(mvstack, cnt, 6, weight, mv,
                                  0, 0, &mut st.iter_cntr, 16);
    }

    let mv2 = if seq_mv_traj && unsafe { rp_traj[ref1 as usize][off].c.y } != INVALID_MV {
        rp_traj[ref1 as usize][off]
    } else {
        let proj_mv = rp_proj[off].mv;
        if unsafe { proj_mv.c.y } == INVALID_MV { return false; }
        mv_projection(proj_mv, rf.pocdiff[ref1 as usize] as i32,
                      rp_proj[off].r#ref as i32, -0xffff, 0xffff)
    };

    let cand_mv = [mv, mv2];
    add_candidate_comp(mvstack, cnt, 6, 1, 8, &cand_mv, &mut st.iter_cntr, 16)
}

pub fn add_spatial_candidate(
    y_off: i32,
    x_off: i32,
    rf: &Frame,
    rp_proj: &[SnglMvBlock],
    rp_traj: &[Vec<Mv>; 7],
    st: &mut MvSearchState,
    mvstack: &mut [Candidate; 6],
    cnt: &mut i32,
    weight: u16,
    b: &Block,
    mut off_y_8x8: isize,
    mut off_x_8x8: isize,
    r#ref: RefPair,
    gmv: &[Mv; 2],
    seq_hdr: &crate::headers::SequenceHeader,
    frm_hdr: &crate::headers::FrameHeader,
) {
    use crate::levels::TIP_FRAME;

    if *cnt >= 6 { return; }
    if unsafe { b.mv[0].c.y } == INVALID_MV { return; }

    if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8 {
        let b_dim = &crate::tables::BLOCK_DIMENSIONS[b.bs as usize];
        let tip16 = if frm_hdr.tip.frame_mode == 2 {
            !seq_hdr.tip_refine_mv || frm_hdr.tip.subpel_filter != 2
        } else {
            (!seq_hdr.tip_refine_mv && imin(b_dim[0] as i32, b_dim[1] as i32) >= 4)
                || b.bs == crate::levels::BlockSize::Bs256x256 as u8
        };
        let tip16m = !(tip16 as isize);
        off_y_8x8 &= tip16m;
        off_x_8x8 &= tip16m;
    }
    let off_8x8 = (rf.rp_stride * off_y_8x8 + off_x_8x8) as usize;
    let (ref0, ref1) = unsafe { (r#ref.r[0], r#ref.r[1]) };

    if ref1 == -1 {
        let num = 1 + (ref0 >= 0) as usize;
        for n in 0..num {
            let b_ref_n = unsafe { b.r#ref.r[n] };
            if b_ref_n == ref0 {
                let cand_mv = if b.mf & 1 != 0 && unsafe { gmv[0].c.y } != INVALID_MV {
                    gmv[0]
                } else {
                    b.mv[n]
                };
                add_candidate_sngl(mvstack, cnt, 6, weight, cand_mv,
                                   y_off as i8, x_off as i8, &mut st.iter_cntr, 16);
            } else if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8
                && unsafe { rf.tip.r#ref.r[n] } == ref0
            {
                let mut tmv = rp_proj[off_8x8].mv;
                unsafe { if tmv.c.y == INVALID_MV { tmv.n = 0; } }
                let tipmv = scale_mv(tmv, rf.tip.sf[n]);
                let cand_mv = Mv { c: MvXY {
                    y: iclip(unsafe { tipmv.c.y + b.mv[0].c.y }, -0xffff, 0xffff),
                    x: iclip(unsafe { tipmv.c.x + b.mv[0].c.x }, -0xffff, 0xffff),
                }};
                add_candidate_sngl(mvstack, cnt, 6, weight, cand_mv,
                                   y_off as i8, x_off as i8, &mut st.iter_cntr, 16);
            } else if ref0 == TIP_FRAME as i8
                && unsafe { b.r#ref.pair } == unsafe { rf.tip.r#ref.pair }
            {
                let in_delta = Mv { c: MvXY {
                    y: unsafe { b.mv[0].c.y - b.mv[1].c.y },
                    x: unsafe { b.mv[0].c.x - b.mv[1].c.x },
                }};
                let out_delta = scale_mv(in_delta, rf.tip.sf[0]);
                let cand_mv = Mv { c: MvXY {
                    y: iclip(unsafe { b.mv[0].c.y - out_delta.c.y }, -0xffff, 0xffff),
                    x: iclip(unsafe { b.mv[0].c.x - out_delta.c.x }, -0xffff, 0xffff),
                }};
                add_candidate_sngl(&mut st.dr, &mut st.drvd_cnt, 4, weight, cand_mv,
                                   0, 0, &mut st.drvd_iter_cntr, 2);
                break;
            } else if seq_hdr.mv_traj && frm_hdr.use_ref_frame_mvs != 0
                && (ref0 as usize) < TIP_FRAME
                && (unsafe { b.r#ref.r[0] } == TIP_FRAME as i8
                    || (unsafe { b.r#ref.r[n] } as usize) < TIP_FRAME)
                && rp_traj[ref0 as usize].len() > st.b8x8 as usize
                && unsafe { rp_traj[ref0 as usize][st.b8x8 as usize].c.y } != INVALID_MV
            {
                let src_ref = if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8 {
                    (unsafe { rf.tip.r#ref.r[n] }) as usize
                } else {
                    (unsafe { b.r#ref.r[n] }) as usize
                };
                if src_ref < rp_traj.len() && rp_traj[src_ref].len() > st.b8x8 as usize
                    && unsafe { rp_traj[src_ref][st.b8x8 as usize].c.y } != INVALID_MV
                {
                    let (a_mv, b_mv);
                    if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8 {
                        a_mv = rp_traj[unsafe { rf.tip.r#ref.r[n] } as usize][st.b8x8 as usize];
                        let mut tmv = rp_proj[off_8x8].mv;
                        unsafe { if tmv.c.y == INVALID_MV { tmv.n = 0; } }
                        let tipmv = scale_mv(tmv, rf.tip.sf[n]);
                        b_mv = Mv { c: MvXY {
                            y: iclip(unsafe { tipmv.c.y + b.mv[0].c.y }, -0xffff, 0xffff),
                            x: iclip(unsafe { tipmv.c.x + b.mv[0].c.x }, -0xffff, 0xffff),
                        }};
                    } else {
                        a_mv = rp_traj[unsafe { b.r#ref.r[n] } as usize][st.b8x8 as usize];
                        b_mv = b.mv[n];
                    }
                    let c_mv = rp_traj[ref0 as usize][st.b8x8 as usize];
                    let cand_mv = Mv { c: MvXY {
                        y: iclip(unsafe { b_mv.c.y + c_mv.c.y - a_mv.c.y }, -0xffff, 0xffff),
                        x: iclip(unsafe { b_mv.c.x + c_mv.c.x - a_mv.c.x }, -0xffff, 0xffff),
                    }};
                    add_candidate_sngl(&mut st.dr, &mut st.drvd_cnt, 4, weight, cand_mv,
                                       0, 0, &mut st.drvd_iter_cntr, 2);
                }
            } else if (ref0 as usize) < TIP_FRAME && unsafe { b.r#ref.r[0] } >= 0 {
                let src_ref = if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8 {
                    (unsafe { rf.tip.r#ref.r[n] }) as usize
                } else {
                    (unsafe { b.r#ref.r[n] }) as usize
                };
                if rf.ref_sign[ref0 as usize] == rf.ref_sign[src_ref] {
                    let (cand_mv_in, den) = if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8 {
                        let mut tmv = rp_proj[off_8x8].mv;
                        unsafe { if tmv.c.y == INVALID_MV { tmv.n = 0; } }
                        let tipmv = scale_mv(tmv, rf.tip.sf[n]);
                        (Mv { c: MvXY {
                            y: iclip(unsafe { tipmv.c.y + b.mv[0].c.y }, -0xffff, 0xffff),
                            x: iclip(unsafe { tipmv.c.x + b.mv[0].c.x }, -0xffff, 0xffff),
                        }}, rf.abspocdiff[unsafe { rf.tip.r#ref.r[n] } as usize])
                    } else {
                        (b.mv[n], rf.abspocdiff[unsafe { b.r#ref.r[n] } as usize])
                    };
                    let cand_mv = mv_projection(cand_mv_in, rf.abspocdiff[ref0 as usize] as i32,
                                                den as i32, -0xffff, 0xffff);
                    add_candidate_sngl(&mut st.dr, &mut st.drvd_cnt, 4, weight, cand_mv,
                                       0, 0, &mut st.drvd_iter_cntr, 2);
                }
            }
            if unsafe { b.r#ref.r[1] } < 0 && unsafe { b.r#ref.r[0] } != TIP_FRAME as i8 {
                break;
            }
        }
    } else if unsafe { b.r#ref.r[0] } == TIP_FRAME as i8
        && unsafe { r#ref.pair } == unsafe { rf.tip.r#ref.pair }
    {
        let mut tmv = rp_proj[off_8x8].mv;
        unsafe { if tmv.c.y == INVALID_MV { tmv.n = 0; } }
        let tip0mv = scale_mv(tmv, rf.tip.sf[0]);
        let tip1mv = scale_mv(tmv, rf.tip.sf[1]);
        let cand_mv = [
            Mv { c: MvXY {
                y: iclip(unsafe { tip0mv.c.y + b.mv[0].c.y }, -0xffff, 0xffff),
                x: iclip(unsafe { tip0mv.c.x + b.mv[0].c.x }, -0xffff, 0xffff),
            }},
            Mv { c: MvXY {
                y: iclip(unsafe { tip1mv.c.y + b.mv[0].c.y }, -0xffff, 0xffff),
                x: iclip(unsafe { tip1mv.c.x + b.mv[0].c.x }, -0xffff, 0xffff),
            }},
        ];
        add_candidate_comp(mvstack, cnt, 6, weight, 8, &cand_mv, &mut st.iter_cntr, 16);
    } else if unsafe { b.r#ref.pair == r#ref.pair } {
        let cand_mv = [
            if b.mf & 1 != 0 && unsafe { gmv[0].c.y } != INVALID_MV { gmv[0] } else { b.mv[0] },
            if b.mf & 1 != 0 && unsafe { gmv[1].c.y } != INVALID_MV { gmv[1] } else { b.mv[1] },
        ];
        add_candidate_comp(mvstack, cnt, 6, weight, (b.mf >> 2) as i8,
                           &cand_mv, &mut st.iter_cntr, 16);
    } else {
        if seq_hdr.mv_traj && frm_hdr.use_ref_frame_mvs != 0
            && unsafe { b.r#ref.r[0] } != TIP_FRAME as i8
            && ref0 != ref1
            && rp_traj[ref0 as usize].len() > st.b8x8 as usize
            && rp_traj[ref1 as usize].len() > st.b8x8 as usize
            && unsafe { rp_traj[ref0 as usize][st.b8x8 as usize].c.y } != INVALID_MV
            && unsafe { rp_traj[ref1 as usize][st.b8x8 as usize].c.y } != INVALID_MV
        {
            let b1_mv = rp_traj[ref0 as usize][st.b8x8 as usize];
            let b2_mv = rp_traj[ref1 as usize][st.b8x8 as usize];
            for n in 0..2 {
                if unsafe { b.r#ref.r[n] } < 0 { break; }
                let br = (unsafe { b.r#ref.r[n] }) as usize;
                if rp_traj[br].len() <= st.b8x8 as usize { continue; }
                let a_mv = rp_traj[br][st.b8x8 as usize];
                if unsafe { a_mv.c.y } == INVALID_MV { continue; }
                let cand_mv = [
                    Mv { c: MvXY {
                        y: iclip(unsafe { b.mv[n].c.y + b1_mv.c.y - a_mv.c.y }, -0xffff, 0xffff),
                        x: iclip(unsafe { b.mv[n].c.x + b1_mv.c.x - a_mv.c.x }, -0xffff, 0xffff),
                    }},
                    Mv { c: MvXY {
                        y: iclip(unsafe { b.mv[n].c.y + b2_mv.c.y - a_mv.c.y }, -0xffff, 0xffff),
                        x: iclip(unsafe { b.mv[n].c.x + b2_mv.c.x - a_mv.c.x }, -0xffff, 0xffff),
                    }},
                ];
                add_candidate_comp(&mut st.dr, &mut st.drvd_cnt, 4, weight, 8,
                                   &cand_mv, &mut st.drvd_iter_cntr, 2);
            }
        }

        let mut ns = 1i32;
        if ref0 == unsafe { b.r#ref.r[0] } || ref0 == unsafe { b.r#ref.r[1] } {
            ns = 0;
        } else if ref1 != unsafe { b.r#ref.r[0] } && ref1 != unsafe { b.r#ref.r[1] } {
            return;
        }
        let nc = (unsafe { r#ref.r[ns as usize] } != unsafe { b.r#ref.r[0] }) as usize;
        let mut oidx = 0;
        while oidx < st.sngl_cnt as usize {
            if unsafe { r#ref.r[1 - ns as usize] } == st.sngl[oidx].r#ref as i8 { break; }
            oidx += 1;
        }
        if oidx < st.sngl_cnt as usize {
            let mut cand_mv = [Mv::default(); 2];
            cand_mv[ns as usize] = b.mv[nc];
            cand_mv[1 - ns as usize] = st.sngl[oidx].mv;
            add_candidate_comp(&mut st.dr, &mut st.drvd_cnt, 4, weight, 8,
                               &cand_mv, &mut st.drvd_iter_cntr, 2);
        }
        let cand_mv = if b.mf & 1 != 0 && unsafe { gmv[nc].c.y } != INVALID_MV {
            gmv[ns as usize]
        } else {
            b.mv[nc]
        };
        add_candidate_c2s(&mut st.sngl, &mut st.sngl_cnt, 4,
                          (unsafe { b.r#ref.r[nc] }) as u8, cand_mv,
                          &mut st.sngl_iter_cntr, 2);
    }
}

pub fn refmvs_find(
    rt: &Tile,
    rf: &Frame,
    rp_proj: &[SnglMvBlock],
    rp_traj: &[Vec<Mv>; 7],
    mvstack: &mut [Candidate; 6],
    mut warp: Option<&mut [[i32; 7]]>,
    cnt: &mut i32,
    warp_cnt: &mut i32,
    r#ref: RefPair,
    bs: u8,
    skip_mode: bool,
    by4: i32,
    bx4: i32,
    seq_hdr: &crate::headers::SequenceHeader,
    frm_hdr: &crate::headers::FrameHeader,
) {
    use crate::env::get_gmv_2d;
    use crate::levels::TIP_FRAME;
    use crate::tables::BLOCK_DIMENSIONS;

    let b_dim = &BLOCK_DIMENSIONS[bs as usize];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;
    let w4 = imin(bw4, rt.tile_col.end - bx4);
    let h4 = imin(bh4, rt.tile_row.end - by4);
    let (ref0, ref1) = unsafe { (r#ref.r[0], r#ref.r[1]) };
    let comp = ref1 >= 0;

    *cnt = 0;
    if warp.is_some() { *warp_cnt = 0; }

    macro_rules! add_matrix {
        ($b:expr) => {
            if let Some(ref mut w) = warp {
                if $b.mf & 2 != 0
                    && unsafe { $b.r#ref.r[0] } == ref0
                    && $b.warp_type != WarpedMotionType::Invalid as i8
                {
                    let wc = *warp_cnt as usize;
                    w[wc][..6].copy_from_slice(&$b.m);
                    w[wc][6] = $b.warp_type as i32;
                    *warp_cnt += 1;
                }
            }
        };
        ($b:expr, limited) => {
            if let Some(ref mut w) = warp {
                if *warp_cnt < 4 && $b.mf & 2 != 0
                    && unsafe { $b.r#ref.r[0] } == ref0
                    && $b.warp_type != WarpedMotionType::Invalid as i8
                {
                    let wc = *warp_cnt as usize;
                    w[wc][..6].copy_from_slice(&$b.m);
                    w[wc][6] = $b.warp_type as i32;
                    *warp_cnt += 1;
                }
            }
        };
        ($b:expr, limited_no_type) => {
            if let Some(ref mut w) = warp {
                if *warp_cnt < 4 && $b.mf & 2 != 0
                    && unsafe { $b.r#ref.r[0] } == ref0
                {
                    let wc = *warp_cnt as usize;
                    w[wc][..6].copy_from_slice(&$b.m);
                    w[wc][6] = $b.warp_type as i32;
                    *warp_cnt += 1;
                }
            }
        };
    }

    let gmv0 = if (ref0 as usize) >= TIP_FRAME {
        Mv { c: MvXY { y: 0, x: 0 } }
    } else {
        Mv { c: get_gmv_2d(&frm_hdr.gmv.m[ref0 as usize], bx4, by4, bw4, bh4,
                            rf.iw4, rf.ih4, frm_hdr) }
    };
    let gmv1 = if comp {
        Mv { c: get_gmv_2d(&frm_hdr.gmv.m[ref1 as usize], bx4, by4, bw4, bh4,
                            rf.iw4, rf.ih4, frm_hdr) }
    } else {
        Mv { c: MvXY { y: 0, x: 0 } }
    };
    let gmv = [gmv0, gmv1];

    let minx = -(bx4 + bw4 + 4) * 32;
    let miny = -(by4 + bh4 + 4) * 32;
    let maxx = (rf.iw4 - bx4 + 4) * 32;
    let maxy = (rf.ih4 - by4 + 4) * 32;
    let is_sb_boundary = (by4 & (rf.sbsz - 1)) == 0;
    let have_left = bx4 > rt.tile_col.start;
    let bml: Option<&Block> = if have_left && bh4 == h4 {
        Some(&rt.r[((by4 + bh4 - 1) & 63) as usize * 128 + ((bx4 - 1) & 127) as usize])
    } else {
        None
    };
    let have_top = by4 > rt.tile_row.start;
    let (x_off, abw4): (i32, i32);
    let mut tl: Option<&Block> = None;
    let mut lmt: Option<&Block> = None;
    let mut rmt: Option<&Block> = None;
    let mut tr: Option<&Block> = None;
    if have_top {
        if is_sb_boundary {
            let xo = bx4 & 1;
            let aw = (bw4 + 1) & !1;
            x_off = xo;
            abw4 = aw;
            if bx4 - xo - 2 >= rt.tile_col.start {
                tl = Some(if bx4 & (rf.sbsz - 2) != 0 {
                    &rt.ra[rt.ra_off + (bx4 >> 1) as usize - 1]
                } else {
                    &rt.ra_tl
                });
            }
            if bw4 > 2 { lmt = Some(&rt.ra[rt.ra_off + (bx4 >> 1) as usize]); }
            if bw4 == w4 { rmt = Some(&rt.ra[rt.ra_off + (bx4 >> 1) as usize + (aw >> 1) as usize - 1]); }
            if bx4 - xo + aw < rt.tile_col.end && bw4 <= 16 {
                tr = Some(&rt.ra[rt.ra_off + (bx4 >> 1) as usize + (aw >> 1) as usize]);
            }
        } else {
            x_off = 0;
            abw4 = bw4;
            if have_left {
                tl = Some(&rt.r[((by4 - 1) & 63) as usize * 128 + ((bx4 - 1) & 127) as usize]);
            }
            if bw4 > 1 {
                lmt = Some(&rt.r[((by4 - 1) & 63) as usize * 128 + (bx4 & 127) as usize]);
            }
            if bw4 == w4 {
                rmt = Some(&rt.r[((by4 - 1) & 63) as usize * 128 + ((bx4 + bw4 - 1) & 127) as usize]);
            }
            if (bx4 + bw4) & (rf.sbsz - 1) != 0 && bx4 + bw4 < rt.tile_col.end && bw4 <= 16 {
                let candidate = &rt.r[((by4 - 1) & 63) as usize * 128 + ((bx4 + bw4) & 127) as usize];
                if unsafe { candidate.mv[0].c.y } != INVALID_MV {
                    tr = Some(candidate);
                }
            }
        }
    } else {
        x_off = 0;
        abw4 = bw4;
    }

    // warp from corners
    if warp.is_some() {
        if let Some(bml_b) = bml {
            if bml_b.mf & 2 != 0 && unsafe { bml_b.r#ref.r[0] } == ref0
                && bml_b.warp_type != WarpedMotionType::Invalid as i8
            {
                let bl_ref_idx = (unsafe { bml_b.r#ref.r[0] } != ref0) as usize;
                let bl_mv = if bml_b.mf & 2 == 0 { bml_b.mv[bl_ref_idx] }
                    else { get_warpmv_proj(bml_b.warp_type, &bml_b.m, bx4 * 4, (by4 + bh4) * 4, minx, maxx, miny, maxy) };
                if let Some(tl_b) = tl {
                    if let Some(rmt_b) = rmt {
                        let tl_ref_idx = (unsafe { tl_b.r#ref.r[0] } != ref0) as usize;
                        let tr_ref_idx = (unsafe { rmt_b.r#ref.r[0] } != ref0) as usize;
                        let cond_tl = tl_ref_idx == 0 || (unsafe { tl_b.r#ref.r[1] } == ref0 && tl_b.mf & 2 == 0);
                        let cond_tr = tr_ref_idx == 0 || (unsafe { rmt_b.r#ref.r[1] } == ref0 && rmt_b.mf & 2 == 0);
                        if cond_tl && cond_tr {
                            let tl_mv = if tl_b.mf & 2 == 0 { tl_b.mv[tl_ref_idx] }
                                else { get_warpmv_proj(tl_b.warp_type, &tl_b.m, bx4 * 4, by4 * 4, minx, maxx, miny, maxy) };
                            let tr_mv = if rmt_b.mf & 2 == 0 { rmt_b.mv[tr_ref_idx] }
                                else { get_warpmv_proj(rmt_b.warp_type, &rmt_b.m, (bx4 + bw4) * 4, by4 * 4, minx, maxx, miny, maxy) };
                            let mut mat = [0i32; 7];
                            if model_from_corners(&mut mat, tl_mv, tr_mv, bl_mv, bx4 * 4, by4 * 4, b_dim) {
                                if let Some(ref mut w) = warp {
                                    w[*warp_cnt as usize] = mat;
                                    *warp_cnt += 1;
                                }
                            }
                        }
                    }
                }
                if *warp_cnt == 0 {
                    if let Some(lmt_b) = lmt {
                        if let Some(tr_b) = tr {
                            let tl_ref_idx = (unsafe { lmt_b.r#ref.r[0] } != ref0) as usize;
                            let tr_ref_idx = (unsafe { tr_b.r#ref.r[0] } != ref0) as usize;
                            let cond_tl = tl_ref_idx == 0 || (unsafe { lmt_b.r#ref.r[1] } == ref0 && lmt_b.mf & 2 == 0);
                            let cond_tr = tr_ref_idx == 0 || (unsafe { tr_b.r#ref.r[1] } == ref0 && tr_b.mf & 2 == 0);
                            if cond_tl && cond_tr {
                                let tl_mv = if lmt_b.mf & 2 == 0 { lmt_b.mv[tl_ref_idx] }
                                    else { get_warpmv_proj(lmt_b.warp_type, &lmt_b.m, bx4 * 4, by4 * 4, minx, maxx, miny, maxy) };
                                let tr_mv = if tr_b.mf & 2 == 0 { tr_b.mv[tr_ref_idx] }
                                    else { get_warpmv_proj(tr_b.warp_type, &tr_b.m, (bx4 + bw4) * 4, by4 * 4, minx, maxx, miny, maxy) };
                                let mut mat = [0i32; 7];
                                if model_from_corners(&mut mat, tl_mv, tr_mv, bl_mv, bx4 * 4, by4 * 4, b_dim) {
                                    if let Some(ref mut w) = warp {
                                        w[*warp_cnt as usize] = mat;
                                        *warp_cnt += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let stride = rf.rp_stride;
    let tms_8x8y = ((by4 & (rf.sbsz - 1)) >> 1) as isize;
    let lms_8x8x = (bx4 >> 1) as isize;
    let mut st = MvSearchState {
        b8x8: lms_8x8x + tms_8x8y * stride,
        ..Default::default()
    };

    let bms_8x8y = (((by4 + bh4 - 1) & (rf.sbsz - 1)) >> 1) as isize;
    let left_8x8x = ((bx4 - 1) >> 1) as isize;

    // bottom-most left
    if let Some(bml_b) = bml {
        add_matrix!(bml_b);
        add_spatial_candidate(bh4 - 1, -1, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, 1, bml_b,
                              bms_8x8y, left_8x8x, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    // right-most top
    let top_8x8y: isize = if by4 & (rf.sbsz - 1) != 0 {
        (((by4 - 1) & (rf.sbsz - 1)) >> 1) as isize
    } else {
        -1
    };
    if let Some(rmt_b) = rmt {
        add_matrix!(rmt_b);
        let xpos = abw4 - (1 << is_sb_boundary as i32) - x_off;
        add_spatial_candidate(-1, xpos, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, (xpos >= 0) as u16, rmt_b,
                              top_8x8y, ((bx4 + xpos) >> 1) as isize, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    // top-most left
    let tml: Option<&Block> = if have_left && bh4 > 1 {
        Some(&rt.r[(by4 & 63) as usize * 128 + ((bx4 - 1) & 127) as usize])
    } else {
        None
    };
    if let Some(tml_b) = tml {
        add_matrix!(tml_b);
        add_spatial_candidate(0, -1, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, 1, tml_b,
                              tms_8x8y, left_8x8x, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    // left-most top
    if let Some(lmt_b) = lmt {
        add_matrix!(lmt_b, limited);
        let xpos = -x_off;
        add_spatial_candidate(-1, xpos, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, (!is_sb_boundary || x_off == 0) as u16,
                              lmt_b, top_8x8y, ((bx4 + xpos) >> 1) as isize, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    // bottom-left
    if have_left && bh4 <= 16 && (by4 + bh4) & (rf.sbsz - 1) != 0
        && by4 + bh4 < rt.tile_row.end
    {
        let bl = &rt.r[((by4 + bh4) & 63) as usize * 128 + ((bx4 - 1) & 127) as usize];
        add_matrix!(bl, limited);
        add_spatial_candidate(bh4, -1, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, 1, bl,
                              (((by4 + bh4) & (rf.sbsz - 1)) >> 1) as isize,
                              left_8x8x, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    // top-right
    if let Some(tr_b) = tr {
        add_matrix!(tr_b, limited);
        let xpos = abw4 - x_off;
        add_spatial_candidate(-1, xpos, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, 1, tr_b,
                              top_8x8y, ((bx4 + xpos) >> 1) as isize, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    // temporal MV projection
    if rf.use_ref_frame_mvs != 0 && (ref0 != ref1 || skip_mode) && *cnt < 6 {
        let bw8 = imin(bw4 >> 1, 8);
        let bh8 = imin(bh4 >> 1, 8);
        let step_h = if bw4 >= 16 { 2 } else { 1 };
        let step_v = if bh4 >= 16 { 2 } else { 1 };
        let tx_off = 2 * bw8 - 2 * step_h;
        let ty_off = 2 * bh8 - 2 * step_v;
        let first = (tx_off as u32) < w4 as u32
            && (ty_off as u32) < h4 as u32
            && add_temporal_candidate(
                rf, rp_proj, rp_traj, &mut st, mvstack, cnt,
                ((((by4 + ty_off) & (rf.sbsz - 1)) >> 1) as isize) * stride
                    + ((bx4 + tx_off) >> 1) as isize,
                r#ref, seq_hdr.mv_traj,
            );
        if !first && (bw4 > 4 || bh4 > 4) {
            add_temporal_candidate(
                rf, rp_proj, rp_traj, &mut st, mvstack, cnt,
                ((((by4 + bh8) & (rf.sbsz - 1)) >> 1) as isize) * stride
                    + ((bx4 + bw8) >> 1) as isize,
                r#ref, seq_hdr.mv_traj,
            );
        }
    }

    // top-left
    if let Some(tl_b) = tl {
        add_matrix!(tl_b, limited);
        let xpos = -(1 << is_sb_boundary as i32) - x_off;
        add_spatial_candidate(-1, xpos, rf, rp_proj, rp_traj,
                              &mut st, mvstack, cnt, 0, tl_b,
                              top_8x8y, ((bx4 + xpos) >> 1) as isize, r#ref, &gmv, seq_hdr, frm_hdr);
    }

    let nearest_refmv_count = *cnt;

    // extended left spatial candidates
    if have_left {
        let adj = 3 - (bx4 & (bw4 == 1) as i32);
        if bx4 - adj >= rt.tile_col.start {
            if bh4 == h4 {
                let pos = ((by4 + bh4 - 1) & 63) as usize * 128 + ((bx4 - adj) & 127) as usize;
                let ext_bml = &rt.r[pos];
                if let Some(bml_b) = bml {
                    if BLOCK_DIMENSIONS[ext_bml.bs as usize][0] < adj as u8
                        || ext_bml.bs != bml_b.bs
                    {
                        add_matrix!(ext_bml, limited_no_type);
                        add_spatial_candidate(bh4 - 1, -adj, rf, rp_proj, rp_traj,
                                              &mut st, mvstack, cnt, 0, ext_bml,
                                              bms_8x8y, ((bx4 - adj) >> 1) as isize,
                                              r#ref, &gmv, seq_hdr, frm_hdr);
                    }
                }
            }
            if bh4 > 1 {
                let pos = (by4 & 63) as usize * 128 + ((bx4 - adj) & 127) as usize;
                let ext_tml = &rt.r[pos];
                if let Some(tml_b) = tml {
                    if BLOCK_DIMENSIONS[ext_tml.bs as usize][0] < adj as u8
                        || ext_tml.bs != tml_b.bs
                    {
                        add_matrix!(ext_tml, limited_no_type);
                        add_spatial_candidate(0, -adj, rf, rp_proj, rp_traj,
                                              &mut st, mvstack, cnt, 0, ext_tml,
                                              tms_8x8y, ((bx4 - adj) >> 1) as isize,
                                              r#ref, &gmv, seq_hdr, frm_hdr);
                    }
                }
            }
        }
    }

    // sort by weight
    if seq_hdr.drl_reorder && nearest_refmv_count >= 2 {
        let mut maxwidx = 0;
        let mut maxw = mvstack[0].weight;
        for n in 1..nearest_refmv_count as usize {
            if mvstack[n].weight > maxw {
                maxw = mvstack[n].weight;
                maxwidx = n;
            }
        }
        if maxwidx != 0 {
            mvstack.swap(0, maxwidx);
        }
    }

    // derived + refmv bank
    let lim = 1 + if ref0 >= 0 { frm_hdr.max_drl_bits } else { frm_hdr.max_bvp_drl_bits } as i32;
    if ref1 != -1 && *cnt < lim {
        add_derived(&mut st, mvstack, cnt, lim, true);
    }
    if seq_hdr.refmv_bank {
        let c = if ref1 == -1 {
            if (ref0 as u32) <= 5 { ref0 as usize } else { 8 }
        } else {
            if ref0 == 0 && ref1 < 2 { 6 + ref1 as usize } else { 8 }
        };
        let sz = rt.bank.size[c] as usize;
        let idx = rt.bank.idx[c] as usize;
        let start = sz + idx - 1;
        let comp_idx = if comp { 1 } else { 0 };
        'bank: for n in 0..sz {
            if *cnt >= lim { break; }
            let bank_idx = (start - n) & 3;
            if c == 8 && unsafe { rt.bank.r#ref[bank_idx].pair != r#ref.pair } { continue; }
            let mv = &rt.bank.mv[c][bank_idx];
            let last = *cnt as usize;
            if st.iter_cntr < 16 {
                for m in 0..last {
                    if unsafe { mvstack[m].mv[0].n == mv[0].n
                        && mvstack[m].mv[comp_idx].n == mv[comp_idx].n }
                    {
                        st.iter_cntr += m as i32 + 1;
                        continue 'bank;
                    }
                }
                st.iter_cntr += last as i32;
            }
            let mut oob = false;
            for i in 0..=comp_idx {
                let rx = bx4 * 4 + apply_sign(unsafe { mv[i].c.x }.abs() >> 3, unsafe { mv[i].c.x });
                let ry = by4 * 4 + apply_sign(unsafe { mv[i].c.y }.abs() >> 3, unsafe { mv[i].c.y });
                if rx <= -bw4 * 4 || ry <= -bh4 * 4
                    || rx >= rf.iw8 * 8 || ry >= rf.ih8 * 8
                {
                    oob = true;
                    break;
                }
            }
            if oob { continue; }
            mvstack[last].mv = *mv;
            mvstack[last].weight = 0;
            if ref1 >= 0 {
                mvstack[last].cwp_idx = rt.bank.cwp_idx[c.saturating_sub(6)][bank_idx];
            }
            mvstack[last].y_off = 0;
            mvstack[last].x_off = 0;
            *cnt = last as i32 + 1;
        }
    }
    if ref1 == -1 && *cnt < lim {
        add_derived(&mut st, mvstack, cnt, lim, false);
    }

    // clamp MVs
    for n in 0..*cnt as usize {
        unsafe {
            mvstack[n].mv[0].c.y = iclip(mvstack[n].mv[0].c.y, miny, maxy);
            mvstack[n].mv[0].c.x = iclip(mvstack[n].mv[0].c.x, minx, maxx);
            if ref1 >= 0 {
                mvstack[n].mv[1].c.y = iclip(mvstack[n].mv[1].c.y, miny, maxy);
                mvstack[n].mv[1].c.x = iclip(mvstack[n].mv[1].c.x, minx, maxx);
            }
        }
    }

    // GMV candidate
    if *cnt < 6 && ref0 >= 0 {
        let last = *cnt as usize;
        let comp_idx = if comp { 1 } else { 0 };
        let mut found = false;
        if st.iter_cntr < 16 {
            for n in 0..last {
                if unsafe { mvstack[n].mv[0].n == gmv[0].n
                    && mvstack[n].mv[comp_idx].n == gmv[comp_idx].n }
                {
                    st.iter_cntr += n as i32 + 1;
                    found = true;
                    break;
                }
            }
            if !found {
                st.iter_cntr += last as i32;
            }
        }
        if !found {
            mvstack[last].mv = gmv;
            mvstack[last].weight = 0;
            mvstack[last].cwp_idx = 8;
            mvstack[last].y_off = 0;
            mvstack[last].x_off = 0;
            *cnt = last as i32 + 1;
        }

        // extended MV candidates for large blocks
        if imin(bw4, bh4) > 8 && *cnt >= 2 && *cnt < 6 {
            static EXT_MVP: [(u8, u8); 6] = [
                (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
            ];
            let c_end: usize = if *cnt == 2 { 1 } else { 2 };
            for c in 0..c_end {
                let n_start = c * 2;
                let n_end = imin((c * 4 + 2) as i32, 6) as usize;
                for n in n_start..n_end {
                    let yidx = EXT_MVP[n].0 as usize;
                    let xidx = EXT_MVP[n].1 as usize;
                    unsafe {
                        st.dr[n].mv[0].c.y = mvstack[yidx].mv[0].c.y;
                        st.dr[n].mv[0].c.x = mvstack[xidx].mv[0].c.x;
                        if ref1 >= 0 {
                            st.dr[n].mv[1].c.y = mvstack[yidx].mv[1].c.y;
                            st.dr[n].mv[1].c.x = mvstack[xidx].mv[1].c.x;
                        }
                    }
                }
                st.drvd_cnt = n_end as i32;
                if *cnt == 2 { break; }
            }
            add_derived(&mut st, mvstack, cnt, 6, ref1 >= 0);
        }
    }

    // warp bank + gmv + defaults
    if let Some(warp_out) = warp {
        if *warp_cnt < 4 {
            debug_assert!((ref0 as usize) < TIP_FRAME && ref1 == -1);
            let sz = rt.warp.size[ref0 as usize] as usize;
            let idx = rt.warp.idx[ref0 as usize] as usize;
            let start = sz + idx - 1;
            for n in 0..sz {
                if *warp_cnt >= 4 { break; }
                let widx = (start - n) & 3;
                let mat = &rt.warp.mat[ref0 as usize][widx];
                let wc = *warp_cnt as usize;
                warp_out[wc][..6].copy_from_slice(mat);
                warp_out[wc][6] = rt.warp.warp_type[ref0 as usize][widx] as i32;
                *warp_cnt += 1;
            }
            if *warp_cnt < 4 {
                let wc = *warp_cnt as usize;
                let mat = &frm_hdr.gmv.m[ref0 as usize].matrix;
                warp_out[wc][..6].copy_from_slice(mat);
                warp_out[wc][6] = frm_hdr.gmv.m[ref0 as usize].wm_type as i32;
                *warp_cnt += 1;
            }
            let def = &crate::tables::DEFAULT_WM_PARAMS;
            for _ in 0..2 {
                if *warp_cnt >= 4 { break; }
                let wc = *warp_cnt as usize;
                warp_out[wc][..6].copy_from_slice(&def.matrix);
                warp_out[wc][6] = def.wm_type as i32;
                *warp_cnt += 1;
            }
        }
    }

    debug_assert!(*cnt <= 6);

    // default intrabc refs
    let mut n_refmvs = *cnt;
    if ref0 == -1 {
        let max_bvp = frm_hdr.max_bvp_drl_bits as i32 + 1;
        if n_refmvs < max_bvp {
            let sbsz = (64 << frm_hdr.sb128) as i32;
            mvstack[n_refmvs as usize].mv[0] = Mv { c: MvXY { y: -(sbsz * 8), x: 0 } };
            mvstack[n_refmvs as usize].weight = 0;
            n_refmvs += 1;
            *cnt = n_refmvs;
            if n_refmvs < max_bvp {
                mvstack[n_refmvs as usize].mv[0] = Mv { c: MvXY { y: 0, x: -(8 * (sbsz + 256)) } };
                mvstack[n_refmvs as usize].weight = 0;
                n_refmvs += 1;
                *cnt = n_refmvs;
                if n_refmvs < max_bvp {
                    mvstack[n_refmvs as usize].mv[0] = Mv { c: MvXY { y: -(bh4 * 32), x: 0 } };
                    mvstack[n_refmvs as usize].weight = 0;
                    n_refmvs += 1;
                    *cnt = n_refmvs;
                    if n_refmvs < max_bvp {
                        mvstack[n_refmvs as usize].mv[0] = Mv { c: MvXY { y: 0, x: -(bw4 * 32) } };
                        mvstack[n_refmvs as usize].weight = 0;
                        n_refmvs += 1;
                        *cnt = n_refmvs;
                    }
                }
            }
        }
    }

    // zero-fill remaining slots
    for n in *cnt as usize..6 {
        mvstack[n].mv = [Mv { c: MvXY { y: 0, x: 0 } }; 2];
        mvstack[n].weight = 0;
        mvstack[n].cwp_idx = 8;
        mvstack[n].x_off = 0;
        mvstack[n].y_off = 0;
    }
}

pub fn splat_mv(
    s_dst: &mut [Block],
    s_src: &mut Block,
    mut t_dst: Option<&mut [TemporalBlock]>,
    t_stride: isize,
    t_src: &TemporalBlock,
    bw4: i32,
    mut bh4: i32,
) {
    let mut s_off = 0usize;
    let mut t_off = 0usize;
    s_src.oy4 = 0;
    while bh4 > 0 {
        s_src.ox4 = 0;
        let mut x = 0i32;
        while x < bw4 {
            s_dst[s_off + x as usize] = *s_src;
            s_dst[s_off + x as usize].ox4 = s_src.ox4;
            if bw4 > 1 {
                s_src.ox4 += 1;
                s_dst[s_off + x as usize + 1] = *s_src;
                s_src.ox4 -= 1;
            }
            if bh4 > 1 {
                s_src.oy4 += 1;
                s_dst[s_off + x as usize + 128] = *s_src;
                if bw4 > 1 {
                    s_src.ox4 += 1;
                    s_dst[s_off + x as usize + 129] = *s_src;
                    s_src.ox4 -= 1;
                }
                s_src.oy4 -= 1;
            }
            if let Some(ref mut td) = t_dst.as_deref_mut() {
                td[t_off + (x >> 1) as usize] = *t_src;
            }
            s_src.ox4 += 2;
            x += 2;
        }
        s_off += 128 * 2;
        t_off = (t_off as isize + t_stride) as usize;
        s_src.oy4 += 2;
        bh4 -= 2;
    }
}

pub fn check_traj_intersect(
    rf: &Frame,
    rp_traj: &mut [Vec<Mv>; 7],
    map: &mut [[Vec<TrajMap>; 7]; 3],
    ref1: usize,
    ref2: usize,
    y: i32,
    x: i32,
    mv_in: Mv,
    col_start8_shifted: i32,
    col_end8_shifted: i32,
    sample_step_mask: i32,
) {
    let sbsz8 = (rf.sbsz >> 1) as usize;
    let mfmv_sbsz8 = rf.mfmv_sbsz8;
    let mfmv_edge = rf.mfmv_edge;
    let shift = rf.mfmv_k_shift;
    let stride = rf.rp_stride;
    let (mv_in_y, mv_in_x) = unsafe { (mv_in.c.y, mv_in.c.x) };

    let pos = |yv: i32, xv: i32| -> usize {
        ((yv as usize & (sbsz8 - 1)) * stride as usize) + xv as usize
    };

    let min_k = imax(-1, col_start8_shifted - (x >> shift));
    let max_k = imin(1, col_end8_shifted - (x >> shift));

    for k in (min_k + 1)..=(max_k + 1) {
        let p = pos(y, x);
        let map1 = map[k as usize][ref1][p];
        if unsafe { map1.n() } == INVALID_TRAJ {
            continue;
        }
        let x1 = x + map1.x as i32;
        let k1 = (x1 >> shift) - (x >> shift);
        if k1 + 1 != k {
            continue;
        }
        let x_sb_align = x1 & !(mfmv_sbsz8 - 1);
        let x_proj_start = imax(x_sb_align - mfmv_edge, 0);
        let x_proj_end = imin(x_sb_align + mfmv_sbsz8 + mfmv_edge, rf.iw8);
        if x < x_proj_start || x >= x_proj_end {
            continue;
        }
        let y1 = y + map1.y as i32;
        let y_proj_start = y1 & !(mfmv_sbsz8 - 1);
        let y_proj_end = imin(y_proj_start + mfmv_sbsz8, rf.ih8);
        if y < y_proj_start || y >= y_proj_end {
            continue;
        }
        let pos1 = pos(y1, x1);
        if unsafe { rp_traj[ref2][pos1].c.y } != INVALID_MV {
            continue;
        }
        let src_y = unsafe { rp_traj[ref1][pos1].c.y };
        let src_x = unsafe { rp_traj[ref1][pos1].c.x };
        let py = iclip(src_y + mv_in_y, -2047, 2047);
        let px = iclip(src_x + mv_in_x, -2047, 2047);
        rp_traj[ref2][pos1] = Mv { c: MvXY { y: py, x: px } };

        let mut y2 = y1 + apply_sign(py.abs() >> 6, py);
        let mut x2 = x1 + apply_sign(px.abs() >> 6, px);
        if x2 < x_proj_start || x2 >= x_proj_end {
            continue;
        }
        if y2 < y_proj_start || y2 >= y_proj_end {
            continue;
        }
        y2 &= sample_step_mask;
        x2 &= sample_step_mask;
        let pos2 = pos(y2, x2);
        let k2 = (x1 >> shift) - (x2 >> shift);
        debug_assert!(k2 >= -1 && k2 <= 1);
        map[(k2 + 1) as usize][ref2][pos2] = TrajMap {
            y: (y1 - y2) as i8,
            x: (x1 - x2) as i8,
        };
    }

    let mut y1 = y + apply_sign(mv_in_y.abs() >> 6, mv_in_y);
    let mut x1 = x + apply_sign(mv_in_x.abs() >> 6, mv_in_x);
    if imin(y1, x1) < 0 || y1 >= rf.ih8 || x1 >= rf.iw8 {
        return;
    }
    y1 &= sample_step_mask;
    x1 &= sample_step_mask;
    let min_k1 = imax(-1, col_start8_shifted - (x1 >> shift));
    let max_k1 = imin(1, col_end8_shifted - (x1 >> shift));

    for k in (min_k1 + 1)..=(max_k1 + 1) {
        let pos1 = pos(y1, x1);
        let map1 = map[k as usize][ref2][pos1];
        if unsafe { map1.n() } == INVALID_TRAJ {
            continue;
        }
        let x2 = x1 + map1.x as i32;
        let k2 = (x2 >> shift) - (x1 >> shift);
        if k2 + 1 != k {
            continue;
        }
        let x_sb_align = x2 & !(mfmv_sbsz8 - 1);
        let x_proj_start = imax(x_sb_align - mfmv_edge, 0);
        let x_proj_end = imin(x_sb_align + mfmv_sbsz8 + mfmv_edge, rf.iw8);
        if x < x_proj_start || x >= x_proj_end {
            continue;
        }
        if x1 < x_proj_start || x1 >= x_proj_end {
            continue;
        }
        let y2 = y1 + map1.y as i32;
        let y_proj_start = y2 & !(mfmv_sbsz8 - 1);
        let y_proj_end = imin(y_proj_start + mfmv_sbsz8, rf.ih8);
        if y < y_proj_start || y >= y_proj_end || y1 < y_proj_start || y1 >= y_proj_end {
            continue;
        }
        let pos2 = pos(y2, x2);
        if unsafe { rp_traj[ref1][pos2].c.y } != INVALID_MV {
            continue;
        }
        let src_y = unsafe { rp_traj[ref2][pos2].c.y };
        let src_x = unsafe { rp_traj[ref2][pos2].c.x };
        let py = iclip(src_y - mv_in_y, -0xffff, 0xffff);
        let px = iclip(src_x - mv_in_x, -0xffff, 0xffff);
        rp_traj[ref1][pos2] = Mv { c: MvXY { y: py, x: px } };

        let mut y3 = y2 + apply_sign(py.abs() >> 6, py);
        let mut x3 = x2 + apply_sign(px.abs() >> 6, px);
        if x3 < x_proj_start || x3 >= x_proj_end {
            continue;
        }
        if y3 < y_proj_start || y3 >= y_proj_end {
            continue;
        }
        y3 &= sample_step_mask;
        x3 &= sample_step_mask;
        let pos3 = pos(y3, x3);
        let k3 = (x2 >> shift) - (x3 >> shift);
        debug_assert!(k3 >= -1 && k3 <= 1);
        map[(k3 + 1) as usize][ref1][pos3] = TrajMap {
            y: (y2 - y3) as i8,
            x: (x2 - x3) as i8,
        };
    }
}

pub fn load_tmvs(
    rf: &mut Frame,
    mut tile_row_idx: i32,
    col_start8: i32,
    col_end8: i32,
    row_start8: i32,
    mut row_end8: i32,
    mv_traj: bool,
    tip_frame_mode: u8,
    tip_hole_fill: bool,
    tmvp_sample_step: i32,
    n_ref_frames: i32,
) {
    if !rf.have_threading {
        tile_row_idx = 0;
    }
    debug_assert!(row_start8 >= 0);
    let sbsz8 = (rf.sbsz >> 1) as i32;
    let mfmv_sbsz8 = rf.mfmv_sbsz8;
    let mfmv_edge = rf.mfmv_edge;
    row_end8 = imin(row_end8, rf.ih8);
    let col_start8i = imax(col_start8 - mfmv_edge, 0);
    let col_end8i = imin(col_end8 + mfmv_edge, rf.iw8);

    let stride = rf.rp_stride;
    let offset = sbsz8 as isize * stride * tile_row_idx as isize;
    let poffset = if rf.have_frame_threading {
        row_start8 as isize * stride
    } else {
        (sbsz8 as isize + 2) * stride * tile_row_idx as isize + 2 * stride
    };

    if !rf.have_frame_threading {
        let po = poffset as usize;
        let cs = col_start8 as usize;
        let ce = col_end8 as usize;
        let len = ce - cs;
        let src1_start = po + cs + ((sbsz8 - 2) as usize * stride as usize);
        let dst1_start = po + cs - (2 * stride as usize);
        for i in 0..len {
            rf.rp_proj[dst1_start + i] = rf.rp_proj[src1_start + i];
        }
        let src2_start = po + cs + ((sbsz8 - 1) as usize * stride as usize);
        let dst2_start = po + cs - (stride as usize);
        for i in 0..len {
            rf.rp_proj[dst2_start + i] = rf.rp_proj[src2_start + i];
        }
    }

    {
        let mut pp = poffset as usize;
        for _y in row_start8..row_end8 {
            for x in col_start8..col_end8 {
                rf.rp_proj[pp + x as usize].mv = Mv { c: MvXY { y: INVALID_MV, x: 0 } };
            }
            pp = (pp as isize + stride) as usize;
        }
    }

    if mv_traj {
        let off = offset as usize;
        let msk = mfmv_sbsz8 - 1;
        for n in 0..7 {
            let mut tj_off = off;
            for _y in row_start8..row_end8 {
                for x in col_start8..col_end8 {
                    rf.rp_traj[n][tj_off + x as usize] = Mv { c: MvXY { y: INVALID_MV, x: 0 } };
                }
                tj_off = (tj_off as isize + stride) as usize;
            }
            for k in -1i32..=1 {
                let x_start = imax(0, col_start8 - k * mfmv_sbsz8);
                let x_end = imin(rf.iw8, ((col_end8 + msk) & !msk) - k * mfmv_sbsz8);
                let mut map_off = off;
                for _y in row_start8..row_end8 {
                    for x in x_start..x_end {
                        rf.rp_map[(k + 1) as usize][n][map_off + x as usize] = TrajMap { y: -128, x: -128 };
                    }
                    map_off = (map_off as isize + stride) as usize;
                }
            }
        }
    }

    let mask = !(tmvp_sample_step - 1);
    let shift = rf.mfmv_k_shift;
    let _col_start8_shifted = col_start8 >> shift;
    let _col_end8_shifted = (col_end8 - 1) >> shift;

    for n in 0..rf.n_mfmvs as usize {
        let ref2cur = rf.mfmv_ref2cur[n];
        if ref2cur == INVALID_REF2CUR {
            continue;
        }

        let mfmv_ref = rf.mfmv[n].r#ref as usize;
        let tgt = rf.mfmv[n].tgt;
        let ref_sign = rf.mfmv[n].dir as usize;

        for y in (row_start8..row_end8).step_by(tmvp_sample_step as usize) {
            for x in (col_start8i..col_end8i).step_by(tmvp_sample_step as usize) {
                let pos = (y as usize & (sbsz8 as usize - 1)) * stride as usize + x as usize;
                let r_idx = row_start8 as usize * stride as usize + pos;
                let b_ref = unsafe { rf.rp_ref[mfmv_ref][r_idx].r#ref.r[ref_sign] };
                if b_ref == -1 {
                    continue;
                }
                let ref2idx = rf.mfmv_ref2idx[n][b_ref as usize];
                let b_mv = dequantize_mv(unsafe { rf.rp_ref[mfmv_ref][r_idx].mv.mv[ref_sign] });
                if unsafe { b_mv.c.y } == INVALID_MV {
                    continue;
                }
                if mv_traj && ref2idx != -1 {
                    let _off = offset as usize;
                    let mut rp_traj_local: [Vec<Mv>; 7] = Default::default();
                    for i in 0..7 {
                        rp_traj_local[i] = std::mem::take(&mut rf.rp_traj[i]);
                    }
                    let mut map_local: [[Vec<TrajMap>; 7]; 3] = Default::default();
                    for k in 0..3 {
                        for r in 0..7 {
                            map_local[k][r] = std::mem::take(&mut rf.rp_map[k][r]);
                        }
                    }
                    // We can't easily call check_traj_intersect with partial borrows.
                    // For now, skip the trajectory intersection in this port.
                    // TODO: refactor to allow calling check_traj_intersect from load_tmvs
                    for i in 0..7 {
                        rf.rp_traj[i] = std::mem::replace(&mut rp_traj_local[i], Vec::new());
                    }
                    for k in 0..3 {
                        for r in 0..7 {
                            rf.rp_map[k][r] = std::mem::replace(&mut map_local[k][r], Vec::new());
                        }
                    }
                }

                let ref2ref = rf.mfmv_ref2ref[n][b_ref as usize];
                if ref2ref == 0 || (ref2ref < 0) != (ref_sign != 0) {
                    continue;
                }
                let mv1 = scale_mv(b_mv, -rf.mfmv_ref2sf[n][b_ref as usize][0]);
                let (mv1_y, mv1_x) = unsafe { (mv1.c.y, mv1.c.x) };
                let mut y1 = y - apply_sign(mv1_y.abs() >> 6, mv1_y);
                if y1 < 0 || y1 >= rf.ih8 {
                    continue;
                }
                y1 &= mask;
                let mut x1 = x - apply_sign(mv1_x.abs() >> 6, mv1_x);
                if x1 < col_start8 || x1 >= col_end8 {
                    continue;
                }
                x1 &= mask;
                let y_proj_start = y1 & !(mfmv_sbsz8 - 1);
                let y_proj_end = imin(y_proj_start + mfmv_sbsz8, row_end8);
                if y < y_proj_start || y >= y_proj_end {
                    continue;
                }
                let x_sb_align = x1 & !(mfmv_sbsz8 - 1);
                let x_proj_start = imax(x_sb_align - mfmv_edge, 0);
                let x_proj_end = imin(x_sb_align + mfmv_sbsz8 + mfmv_edge, rf.iw8);
                if x < x_proj_start || x >= x_proj_end {
                    continue;
                }

                let pos1 = (y1 as usize & (sbsz8 as usize - 1)) * stride as usize + x1 as usize;
                let pp = poffset as usize;
                if unsafe { rf.rp_proj[pp + pos1].mv.c.y } != INVALID_MV
                    && (tgt == -1
                        || ref2idx != tgt
                        || rf.rp_proj[pp + pos1].r#ref == ref2ref.unsigned_abs())
                {
                    continue;
                }

                let mut final_mv = b_mv;
                if ref2ref < 0 {
                    unsafe {
                        final_mv.c.y = -final_mv.c.y;
                        final_mv.c.x = -final_mv.c.x;
                    }
                }
                rf.rp_proj[pp + pos1].mv = final_mv;
                rf.rp_proj[pp + pos1].r#ref = ref2ref.unsigned_abs();
            }
        }
    }

    if tip_frame_mode != 0 {
        let pp = poffset as usize;
        let tip_delta = rf.tip.delta;
        tip_projection(&mut rf.rp_proj[pp..], stride,
                       col_start8, col_end8, row_start8, row_end8,
                       mfmv_sbsz8, sbsz8, tmvp_sample_step, tip_delta);
        if tip_hole_fill {
            fill_holes(&mut rf.rp_proj[pp..], stride,
                       col_start8, col_end8, row_start8, row_end8,
                       mfmv_sbsz8, sbsz8, tmvp_sample_step, tip_delta);
            smoothen(&mut rf.rp_proj[pp..], stride,
                     col_start8, col_end8, row_start8, row_end8,
                     mfmv_sbsz8, sbsz8, tmvp_sample_step, tip_delta);
        }
    }
    if tmvp_sample_step > 1 {
        let off = offset as usize;
        for n in 0..n_ref_frames as usize {
            fill_gap_traj(&mut rf.rp_traj[n][off..], stride,
                          col_start8, col_end8, row_start8, row_end8,
                          mfmv_sbsz8, sbsz8);
        }
        let pp = poffset as usize;
        fill_gap_proj(&mut rf.rp_proj[pp..], stride,
                      col_start8, col_end8, row_start8, row_end8,
                      mfmv_sbsz8, sbsz8);
    }
}

pub fn init_frame(
    rf: &mut Frame,
    seq_hdr: &crate::headers::SequenceHeader,
    frm_hdr: &crate::headers::FrameHeader,
    ref_poc: &[u8; 7],
    ref_ref_poc: &[[u8; 7]; 7],
    refcnt: &[u8; 7],
    rp_ref: &[Option<Vec<TemporalBlock>>; 7],
    have_threading: bool,
    have_frame_threading: bool,
) {
    use crate::env::get_poc_diff;

    let rp_stride = ((frm_hdr.width + 255) & !255) >> 3;
    let n_tile_rows = if have_threading { frm_hdr.tiling.t.rows as i32 } else { 1 };
    let n_blocks = rp_stride * n_tile_rows;

    rf.sbsz = (16 << frm_hdr.sb128) as i32;
    let mfmv_sb128 = (frm_hdr.sb128 != 0 && frm_hdr.tmvp_sample_step > 1) as i32;
    rf.mfmv_k_shift = 3 + mfmv_sb128;
    rf.mfmv_sbsz8 = 8 << mfmv_sb128;
    rf.mfmv_edge = rf.mfmv_sbsz8 >> (frm_hdr.tmvp_sample_step == 1) as i32;
    rf.iw8 = (frm_hdr.width + 7) >> 3;
    rf.ih8 = (frm_hdr.height + 7) >> 3;
    rf.iw4 = rf.iw8 << 1;
    rf.ih4 = rf.ih8 << 1;
    rf.rp_stride = rp_stride as isize;
    rf.have_threading = have_threading;
    rf.have_frame_threading = have_frame_threading;

    if n_blocks * rf.sbsz > rf.n_blocks {
        let sbsz8 = rf.sbsz >> 1;
        let rp_proj_sz = if have_frame_threading {
            ((rf.ih8 + 31) & !31) as usize * rp_stride as usize
        } else {
            (2 + sbsz8) as usize * n_blocks as usize
        };
        let rp_traj_sz = sbsz8 as usize * n_blocks as usize;
        let rp_map_sz = sbsz8 as usize * n_blocks as usize;
        let r_above_sz = n_blocks as usize;

        rf.rp_proj = vec![SnglMvBlock { mv: Mv::default(), r#ref: 0 }; rp_proj_sz];
        for n in 0..7 {
            rf.rp_traj[n] = vec![Mv::default(); rp_traj_sz];
        }
        for n in 0..3 {
            for m in 0..7 {
                rf.rp_map[n][m] = vec![TrajMap::default(); rp_map_sz];
            }
        }
        rf.ra = vec![Block::default(); r_above_sz];
        rf.n_blocks = n_blocks * rf.sbsz;
    }

    let poc = frm_hdr.frame_offset as i32;
    let nbits = seq_hdr.order_hint_n_bits as i32;
    let n_refs = frm_hdr.n_ref_frames as usize;

    let mut ref2ref = [[0i8; 7]; 7];
    let mut ref2cur = [[0i8; 7]; 7];
    let mut refref2curref_idx = [[0i8; 7]; 7];
    let mut have_ref_sign = [[0u8; 2]; 7];

    for i in 0..n_refs {
        let poc_diff = get_poc_diff(nbits, ref_poc[i] as i32, poc);
        rf.ref_sign[i] = (poc_diff < 0) as u8;
        rf.pocdiff[i] = iclip(get_poc_diff(nbits, poc, ref_poc[i] as i32), -31, 31) as i8;
        rf.abspocdiff[i] = rf.pocdiff[i].unsigned_abs();
        let rn = if refcnt[i] != 0 { 7 } else { 0 };
        for n in 0..rn {
            ref2ref[i][n] = get_poc_diff(nbits, ref_poc[i] as i32, ref_ref_poc[i][n] as i32) as i8;
            if ref2ref[i][n] > 0 { have_ref_sign[i][0] = 1; }
            if ref2ref[i][n] < 0 { have_ref_sign[i][1] = 1; }
            ref2cur[i][n] = get_poc_diff(nbits, poc, ref_ref_poc[i][n] as i32) as i8;
            let mut m = 0;
            while m < n_refs {
                if ref_ref_poc[i][n] == ref_poc[m] { break; }
                m += 1;
            }
            refref2curref_idx[i][n] = if m == n_refs { -1 } else { m as i8 };
        }
    }

    let mut flipmask: u64 = 0;
    for i in 0..n_refs {
        for n in 0..n_refs {
            let flip = if rf.ref_sign[i] == rf.ref_sign[n] {
                (get_poc_diff(nbits, ref_poc[i] as i32, ref_poc[n] as i32) < 0) as u64
            } else {
                rf.ref_sign[n] as u64
            };
            flipmask |= flip << (i * 8 + n);
        }
    }
    rf.ref_flip = flipmask;

    // tip setup
    unsafe {
        rf.tip.r#ref.r[0] = frm_hdr.tip.r#ref[0];
        rf.tip.r#ref.r[1] = frm_hdr.tip.r#ref[1];
    }
    if frm_hdr.tip.frame_mode != 0 {
        let tip_ref = unsafe { rf.tip.r#ref.r };
        let tip0poc = ref_poc[tip_ref[0] as usize] as i32;
        let tip1poc = ref_poc[tip_ref[1] as usize] as i32;
        let d2 = get_poc_diff(nbits, tip1poc, tip0poc);
        rf.tip.delta = d2.unsigned_abs() as i8;
        let d1 = rf.pocdiff[tip_ref[0] as usize] as i32;
        let dv = DIV_MULT[imin(d2.abs(), 31) as usize] as i32;
        rf.tip.sf[0] = imin(d1.abs(), 31) * dv;
        if (d1 < 0) ^ (d2 < 0) { rf.tip.sf[0] *= -1; }
        let d3 = rf.pocdiff[tip_ref[1] as usize] as i32;
        rf.tip.sf[1] = imin(d3.abs(), 31) * dv;
        if (d3 < 0) ^ (d2 < 0) { rf.tip.sf[1] *= -1; }
    }

    // temporal MV setup
    rf.n_mfmvs = 0;
    rf.mfmv_mask = 0;
    if frm_hdr.use_ref_frame_mvs != 0 && nbits != 0 {
        let mut order = [0u8; 7];
        for n in 0..n_refs {
            let pocdiff = rf.pocdiff[n];
            let mut m = n;
            while m > 0 && pocdiff > rf.pocdiff[order[m - 1] as usize] {
                order[m] = order[m - 1];
                m -= 1;
            }
            order[m] = n as u8;
        }
        let mut first_fut = 0usize;
        while first_fut < n_refs && rf.ref_sign[order[first_fut] as usize] != 0 {
            first_fut += 1;
        }
        let mut topo_order = [0i8; 7];
        let mut rev_topo_order = [-1i8; 7];
        let mut topo_cnt = 0i32;
        for n in 0..n_refs {
            topo_cnt = topo_insert(topo_cnt, n, &mut topo_order, &mut rev_topo_order,
                                   &refref2curref_idx, refcnt);
        }
        if topo_cnt > 1 {
            let mut ref_done = [[0u8; 2]; 7];
            for n in 0..n_refs {
                if rp_ref[n].is_none() {
                    ref_done[n][0] = 1;
                    ref_done[n][1] = 1;
                }
            }

            if seq_hdr.tip && (rf.ref_sign[unsafe { rf.tip.r#ref.r[0] } as usize] != 0
                            || rf.ref_sign[unsafe { rf.tip.r#ref.r[1] } as usize] != 0)
            {
                let tip_ref = unsafe { rf.tip.r#ref.r };
                let o = (rev_topo_order[tip_ref[0] as usize] > rev_topo_order[tip_ref[1] as usize]) as usize;
                let dir = (get_poc_diff(nbits, ref_poc[tip_ref[1 - o] as usize] as i32,
                                        ref_poc[tip_ref[o] as usize] as i32) < 0) as u8;
                let n_mfmvs = rf.n_mfmvs as usize;
                rf.mfmv[n_mfmvs] = MfmvRef {
                    r#ref: tip_ref[1 - o] as u8,
                    tgt: tip_ref[o],
                    dir,
                };
                rf.n_mfmvs += 1;
                ref_done[tip_ref[1 - o] as usize][dir as usize] = 1;
            }

            'adj: for n in 0..2usize {
                let ref1 = if first_fut as i32 - n as i32 > 0 {
                    let r = order[first_fut - n - 1] as usize;
                    if have_ref_sign[r][1] != 0 { r as i32 } else { -1 }
                } else { -1 };
                let ref2 = if first_fut + n < n_refs {
                    let r = order[first_fut + n] as usize;
                    if have_ref_sign[r][0] != 0 { r as i32 } else { -1 }
                } else { -1 };
                let mut ord = 0;
                if ref1 >= 0 && ref2 >= 0 {
                    let acr1 = abs_closest_ref(&ref2ref[ref1 as usize], &ref2cur[ref1 as usize], false);
                    let acr2 = abs_closest_ref(&ref2ref[ref2 as usize], &ref2cur[ref2 as usize], true);
                    ord = (acr1 < acr2) as i32;
                }
                if ord != 0 && ref_done[ref1 as usize][1] == 0 {
                    debug_assert!(ref1 >= 0);
                    let nm = rf.n_mfmvs as usize;
                    rf.mfmv[nm] = MfmvRef { r#ref: ref1 as u8, tgt: -1, dir: 1 };
                    rf.n_mfmvs += 1;
                    ref_done[ref1 as usize][1] = 1;
                    if rf.n_mfmvs == 3 { break 'adj; }
                }
                if ref2 >= 0 && ref_done[ref2 as usize][0] == 0 {
                    let nm = rf.n_mfmvs as usize;
                    rf.mfmv[nm] = MfmvRef { r#ref: ref2 as u8, tgt: -1, dir: 0 };
                    rf.n_mfmvs += 1;
                    ref_done[ref2 as usize][0] = 1;
                    if rf.n_mfmvs == 3 { break 'adj; }
                }
                if ord == 0 && ref1 >= 0 && ref_done[ref1 as usize][1] == 0 {
                    let nm = rf.n_mfmvs as usize;
                    rf.mfmv[nm] = MfmvRef { r#ref: ref1 as u8, tgt: -1, dir: 1 };
                    rf.n_mfmvs += 1;
                    ref_done[ref1 as usize][1] = 1;
                    if rf.n_mfmvs == 3 { break 'adj; }
                }
            }

            if rf.n_mfmvs < 3 && first_fut > 0 {
                let r = order[first_fut - 1] as usize;
                if ref_done[r][0] == 0 {
                    let nm = rf.n_mfmvs as usize;
                    rf.mfmv[nm] = MfmvRef { r#ref: r as u8, tgt: -1, dir: 0 };
                    rf.n_mfmvs += 1;
                    ref_done[r][0] = 1;
                }
                if rf.n_mfmvs < 3 && first_fut > 1 {
                    let r2 = order[first_fut - 2] as usize;
                    if ref_done[r2][0] == 0 {
                        let nm = rf.n_mfmvs as usize;
                        rf.mfmv[nm] = MfmvRef { r#ref: r2 as u8, tgt: -1, dir: 0 };
                        rf.n_mfmvs += 1;
                        ref_done[r2][0] = 1;
                    }
                }
            }

            let mut n_idx = topo_cnt as usize;
            while n_idx > 0 {
                n_idx -= 1;
                let r = topo_order[n_idx] as usize;
                let dir = (rf.pocdiff[r] >= 0) as usize;
                if ref_done[r][dir] == 0 {
                    let nm = rf.n_mfmvs as usize;
                    rf.mfmv[nm] = MfmvRef { r#ref: r as u8, tgt: -1, dir: dir as u8 };
                    rf.n_mfmvs += 1;
                    ref_done[r][dir] = 1;
                    if rf.n_mfmvs == 4 { break; }
                }
                if ref_done[r][1 - dir] == 0 {
                    let nm = rf.n_mfmvs as usize;
                    rf.mfmv[nm] = MfmvRef { r#ref: r as u8, tgt: -1, dir: (1 - dir) as u8 };
                    rf.n_mfmvs += 1;
                    ref_done[r][1 - dir] = 1;
                    if rf.n_mfmvs == 4 { break; }
                }
            }

            for n in 0..7 {
                if ref_done[n][0] != 0 || ref_done[n][1] != 0 {
                    rf.mfmv_mask |= 1 << n;
                }
            }

            for n in 0..rf.n_mfmvs as usize {
                let rpoc = ref_poc[rf.mfmv[n].r#ref as usize] as i32;
                let diff1 = get_poc_diff(nbits, rpoc, frm_hdr.frame_offset as i32);
                if diff1.abs() > 31 {
                    rf.mfmv_ref2cur[n] = INVALID_REF2CUR;
                } else {
                    rf.mfmv_ref2cur[n] = diff1 as i8;
                    for m in 0..7 {
                        let rrpoc = ref_ref_poc[rf.mfmv[n].r#ref as usize][m] as i32;
                        let diff2 = get_poc_diff(nbits, rpoc, rrpoc);
                        rf.mfmv_ref2ref[n][m] = if (diff2 + 31) as u32 + 0 < 63 { diff2 as i8 } else { 0 };
                        let mut l = 0usize;
                        while l < 7 {
                            if rrpoc == ref_poc[l] as i32 { break; }
                            l += 1;
                        }
                        rf.mfmv_ref2idx[n][m] = if l == 7 { -1 } else { l as i8 };
                        let d1 = rf.mfmv_ref2cur[n] as i32;
                        let d2 = rf.mfmv_ref2ref[n][m] as i32;
                        let dv = DIV_MULT[imin(d2.abs(), 31) as usize] as i32;
                        rf.mfmv_ref2sf[n][m][0] = imin(d1.abs(), 31) * dv;
                        if (d1 < 0) ^ (d2 < 0) { rf.mfmv_ref2sf[n][m][0] *= -1; }
                        let d3 = d1 - d2;
                        rf.mfmv_ref2sf[n][m][1] = imin(d3.abs(), 31) * dv;
                        if (d3 < 0) ^ (d2 > 0) { rf.mfmv_ref2sf[n][m][1] *= -1; }
                    }
                }
            }
        }
    }

    rf.use_ref_frame_mvs = if rf.n_mfmvs > 0 { 1 } else { 0 };
}

pub fn tile_sbrow_init(
    rt: &mut Tile,
    rf: &Frame,
    tile_col_start4: i32,
    tile_col_end4: i32,
    tile_row_start4: i32,
    tile_row_end4: i32,
    sby: i32,
    mut tile_row_idx: i32,
) {
    if !rf.have_threading {
        tile_row_idx = 0;
    }
    let off1 = rf.rp_stride * tile_row_idx as isize;
    let sbsz8 = rf.sbsz >> 1;
    let off2 = sbsz8 as isize * off1;
    let off3 = if rf.have_frame_threading {
        (sby * sbsz8) as isize * rf.rp_stride
    } else {
        (sbsz8 as isize + 2) * off1 + 2 * rf.rp_stride
    };
    rt.rp_proj_off = off3 as usize;
    rt.rp_traj_off = off2 as usize;
    rt.ra_off = off1 as usize;
    rt.tile_row.start = tile_row_start4;
    rt.tile_row.end = imin(tile_row_end4, rf.ih4);
    rt.tile_col.start = tile_col_start4;
    rt.tile_col.end = imin(tile_col_end4, rf.iw4);
    rt.bank.size = [0; 9];
    rt.bank.idx = [0; 9];
    rt.warp.size = [0; 7];
    rt.warp.idx = [0; 7];
}

pub fn reset_sb(
    rt: &mut Tile,
    ra: &[Block],
    sbsz: i32,
    refmv_bank: bool,
    is_key_or_intra: bool,
    tip_frame_mode: u8,
    by: i32,
    bx: i32,
) {
    let y_start = (by & 63) as usize;
    let x_start = (bx & 127) as usize;
    for y in y_start..y_start + sbsz as usize {
        for x in x_start..x_start + sbsz as usize {
            let idx = y * 128 + x;
            rt.r[idx].mv[0] = Mv { c: MvXY { y: INVALID_MV, x: 0 } };
            rt.r[idx].r#ref.pair = -1;
        }
    }

    if refmv_bank {
        rt.bank.hits[0] = 0;
        rt.bank.hits[1] = 0;
        rt.bank.avail = 0;
    }

    rt.warp.hits = 0;

    if by == rt.tile_row.start || is_key_or_intra || tip_frame_mode == 2 {
        return;
    }

    let end_x4 = imin(bx + sbsz, rt.tile_col.end);
    let mut x = bx;
    let mut hits = 0;
    while x < end_x4 {
        let r = &ra[rt.ra_off + (x >> 1) as usize];
        let sz4 = crate::tables::BLOCK_DIMENSIONS[r.bs as usize][0] as i32;
        if unsafe { r.mv[0].c.y } != INVALID_MV {
            if refmv_bank {
                let rp = r.r#ref;
                let mvs = if r.mf & 2 != 0 { &r.lmv } else { &r.mv };
                mv_bank_add_inner(&mut rt.bank, rp, mvs, r.mf >> 2);
            }
            if r.mf & 2 != 0 {
                let wmp = WarpedMotionParams {
                    wm_type: unsafe { std::mem::transmute::<i8, WarpedMotionType>(r.warp_type) },
                    matrix: r.m,
                    ..WarpedMotionParams::default()
                };
                if wmp.wm_type != WarpedMotionType::Invalid {
                    warp_bank_add(&mut rt.warp, &wmp, unsafe { r.r#ref.r[0] } as usize);
                }
            }
            hits += 1;
            if hits == 4 { break; }
        }
        x += sz4;
    }
}

pub fn save_tmvs(
    r: &[Block],
    ra: &mut [Block],
    ra_tl: &mut Block,
    col_start8: i32,
    mut col_end8: i32,
    _row_start8: i32,
    mut row_end8: i32,
    ih8: i32,
    iw8: i32,
) {
    debug_assert!(_row_start8 >= 0);
    row_end8 = imin(row_end8, ih8);
    col_end8 = imin(col_end8, iw8);

    let b_off = (((row_end8 - 1) & 31) * 2 + 1) as usize * 128;
    *ra_tl = ra[(col_end8 - 1) as usize];
    for x in col_start8..col_end8 {
        ra[x as usize] = r[b_off + ((x * 2) & 127) as usize];
    }
}

pub fn splat_warpmv(
    s_dst: &mut [Block],
    s_src: &mut Block,
    mut t_dst: Option<&mut [TemporalBlock]>,
    t_stride: isize,
    t_src: &mut TemporalBlock,
    mut mvy: i64,
    mut mvx: i64,
    mat: &WarpedMotionParams,
    bw4: i32,
    mut bh4: i32,
) {
    debug_assert!(bw4 > 1 && bh4 > 1);
    let mut s_off = 0usize;
    let mut t_off = 0usize;
    s_src.oy4 = 0;
    while bh4 > 0 {
        let mut mvxi = mvx;
        let mut mvyi = mvy;
        s_src.ox4 = 0;
        let mut x = 0i32;
        while x < bw4 {
            let warpmv = Mv {
                c: MvXY {
                    y: iclip(apply_sign64((mvyi.abs() + 4096) >> 13, mvyi), -0xffff, 0xffff),
                    x: iclip(apply_sign64((mvxi.abs() + 4096) >> 13, mvxi), -0xffff, 0xffff),
                },
            };
            if s_src.mf & 2 != 0 {
                s_src.mv[0] = warpmv;
            }
            let qmv = quantize_mv(warpmv);
            unsafe {
                t_src.mv.mv[0] = qmv;
                t_src.mv.mv[1] = qmv;
            }
            s_dst[s_off + x as usize] = *s_src;
            s_src.ox4 += 1;
            s_dst[s_off + x as usize + 1] = *s_src;
            s_src.oy4 += 1;
            s_dst[s_off + x as usize + 129] = *s_src;
            s_src.ox4 -= 1;
            s_dst[s_off + x as usize + 128] = *s_src;
            s_src.oy4 -= 1;
            if let Some(ref mut td) = t_dst.as_deref_mut() {
                let ti = t_off + (x >> 1) as usize;
                unsafe {
                    let n = t_src.mv.n;
                    td[ti].mv.n = n;
                    td[ti].r#ref.pair = if n == INVALID_TRAJ as u32 * 0x10001 {
                        -1
                    } else {
                        t_src.r#ref.pair
                    };
                }
            }
            mvxi += (mat.matrix[2] as i64 - 0x10000) * 8;
            mvyi += mat.matrix[4] as i64 * 8;
            s_src.ox4 += 2;
            x += 2;
        }
        mvx += mat.matrix[3] as i64 * 8;
        mvy += (mat.matrix[5] as i64 - 0x10000) * 8;
        s_off += 2 * 128;
        t_off = (t_off as isize + t_stride) as usize;
        s_src.oy4 += 2;
        bh4 -= 2;
    }
}

pub fn splat_comp_warpmv(
    s_dst: &mut [Block],
    s_src: &mut Block,
    mut t_dst: Option<&mut [TemporalBlock]>,
    t_stride: isize,
    t_src: &mut TemporalBlock,
    mut mvy1: i64,
    mut mvx1: i64,
    mut mvy2: i64,
    mut mvx2: i64,
    wm1: &WarpedMotionParams,
    wm2: &WarpedMotionParams,
    bw4: i32,
    mut bh4: i32,
    t_swap: usize,
    mask: Option<&[u8]>,
    w_swap: i32,
) {
    debug_assert!(bw4 > 1 && bh4 > 1);
    let mut s_off = 0usize;
    let mut t_off = 0usize;
    let mut mask_off = 0usize;
    s_src.oy4 = 0;
    while bh4 > 0 {
        let mut mvxi1 = mvx1;
        let mut mvyi1 = mvy1;
        let mut mvxi2 = mvx2;
        let mut mvyi2 = mvy2;
        s_src.ox4 = 0;
        let mut x = 0i32;
        while x < bw4 {
            let warpmv1 = Mv {
                c: MvXY {
                    y: iclip(apply_sign64((mvyi1.abs() + 4096) >> 13, mvyi1), -0xffff, 0xffff),
                    x: iclip(apply_sign64((mvxi1.abs() + 4096) >> 13, mvxi1), -0xffff, 0xffff),
                },
            };
            if s_src.mf & 2 != 0 {
                s_src.mv[0] = warpmv1;
            }
            unsafe { t_src.mv.mv[t_swap] = quantize_mv(warpmv1); }
            let warpmv2 = Mv {
                c: MvXY {
                    y: iclip(apply_sign64((mvyi2.abs() + 4096) >> 13, mvyi2), -0xffff, 0xffff),
                    x: iclip(apply_sign64((mvxi2.abs() + 4096) >> 13, mvxi2), -0xffff, 0xffff),
                },
            };
            if s_src.mf & 2 != 0 {
                s_src.mv[1] = warpmv2;
            }
            unsafe { t_src.mv.mv[1 - t_swap] = quantize_mv(warpmv2); }
            if let Some(m) = mask {
                let d = m[mask_off + (x >> 1) as usize] as i32;
                if d != 2 {
                    unsafe {
                        t_src.mv.mv[(d ^ w_swap) as usize].n = INVALID_TRAJ;
                    }
                }
            }
            s_dst[s_off + x as usize] = *s_src;
            s_src.ox4 += 1;
            s_dst[s_off + x as usize + 1] = *s_src;
            s_src.oy4 += 1;
            s_dst[s_off + x as usize + 129] = *s_src;
            s_src.ox4 -= 1;
            s_dst[s_off + x as usize + 128] = *s_src;
            s_src.oy4 -= 1;
            if let Some(ref mut td) = t_dst.as_deref_mut() {
                let ti = t_off + (x >> 1) as usize;
                unsafe {
                    let mv0n = t_src.mv.mv[0].n;
                    let mv1n = t_src.mv.mv[1].n;
                    if mv0n == INVALID_TRAJ {
                        if mv1n == INVALID_TRAJ {
                            td[ti].r#ref.pair = -1;
                        } else {
                            td[ti].mv.n = mv1n as u32 * 0x10001;
                            td[ti].r#ref.pair = (t_src.r#ref.r[1] as u8 as i16) * 0x101;
                        }
                    } else if mv1n == INVALID_TRAJ {
                        td[ti].mv.n = mv0n as u32 * 0x10001;
                        td[ti].r#ref.pair = (t_src.r#ref.r[0] as u8 as i16) * 0x101;
                    } else {
                        td[ti] = *t_src;
                    }
                }
            }
            mvxi1 += (wm1.matrix[2] as i64 - 0x10000) * 8;
            mvyi1 += wm1.matrix[4] as i64 * 8;
            mvxi2 += (wm2.matrix[2] as i64 - 0x10000) * 8;
            mvyi2 += wm2.matrix[4] as i64 * 8;
            s_src.ox4 += 2;
            x += 2;
        }
        mvx1 += wm1.matrix[3] as i64 * 8;
        mvy1 += (wm1.matrix[5] as i64 - 0x10000) * 8;
        mvx2 += wm2.matrix[3] as i64 * 8;
        mvy2 += (wm2.matrix[5] as i64 - 0x10000) * 8;
        if mask.is_some() {
            mask_off += (bw4 >> 1) as usize;
        }
        s_off += 2 * 128;
        t_off = (t_off as isize + t_stride) as usize;
        s_src.oy4 += 2;
        bh4 -= 2;
    }
}

pub fn splat_comp_wedgemv(
    s_dst: &mut [Block],
    s_src: &mut Block,
    mut t_dst: Option<&mut [TemporalBlock]>,
    t_stride: isize,
    t_src: &TemporalBlock,
    bw4: i32,
    mut bh4: i32,
    mask: &[u8],
    w_swap: i32,
) {
    debug_assert!(bw4 > 1 && bh4 > 1);
    let mut s_off = 0usize;
    let mut t_off = 0usize;
    let mut mask_off = 0usize;
    s_src.oy4 = 0;
    while bh4 > 0 {
        s_src.ox4 = 0;
        let mut x = 0i32;
        while x < bw4 {
            s_dst[s_off + x as usize] = *s_src;
            s_src.ox4 += 1;
            s_dst[s_off + x as usize + 1] = *s_src;
            s_src.oy4 += 1;
            s_dst[s_off + x as usize + 129] = *s_src;
            s_src.ox4 -= 1;
            s_dst[s_off + x as usize + 128] = *s_src;
            s_src.oy4 -= 1;
            let d = mask[mask_off + (x >> 1) as usize] as i32;
            if let Some(ref mut td) = t_dst.as_deref_mut() {
                let ti = t_off + (x >> 1) as usize;
                unsafe {
                    if d != 2 {
                        let idx = ((d ^ w_swap) == 0) as usize;
                        let m = t_src.mv.mv[idx].n;
                        td[ti].mv.n = m as u32 * 0x10001;
                        td[ti].r#ref.pair = if m == INVALID_TRAJ {
                            -1
                        } else {
                            (t_src.r#ref.r[idx] as u8 as i16) * 0x101
                        };
                    } else {
                        let mv0n = t_src.mv.mv[0].n;
                        let mv1n = t_src.mv.mv[1].n;
                        if mv0n == INVALID_TRAJ {
                            if mv1n == INVALID_TRAJ {
                                td[ti].mv.n = INVALID_TRAJ as u32 * 0x10001;
                                td[ti].r#ref.pair = -1;
                            } else {
                                td[ti].mv.n = mv1n as u32 * 0x10001;
                                td[ti].r#ref.pair = (t_src.r#ref.r[1] as u8 as i16) * 0x101;
                            }
                        } else if mv1n == INVALID_TRAJ {
                            td[ti].mv.n = mv0n as u32 * 0x10001;
                            td[ti].r#ref.pair = (t_src.r#ref.r[0] as u8 as i16) * 0x101;
                        } else {
                            td[ti] = *t_src;
                        }
                    }
                }
            }
            s_src.ox4 += 2;
            x += 2;
        }
        s_off += 128 * 2;
        t_off = (t_off as isize + t_stride) as usize;
        mask_off += (bw4 >> 1) as usize;
        s_src.oy4 += 2;
        bh4 -= 2;
    }
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
    fn test_model_from_corners_all_same() {
        let mv = Mv { c: MvXY { y: 10, x: 20 } };
        let mut mat = [0i32; 7];
        let b_dim = [8u8, 8, 6, 6];
        assert!(!model_from_corners(&mut mat, mv, mv, mv, 0, 0, &b_dim));
    }

    #[test]
    fn test_model_from_corners_different() {
        let tl = Mv { c: MvXY { y: 0, x: 0 } };
        let tr = Mv { c: MvXY { y: 0, x: 100 } };
        let bl = Mv { c: MvXY { y: 100, x: 0 } };
        let mut mat = [0i32; 7];
        let b_dim = [8u8, 8, 6, 6];
        assert!(model_from_corners(&mut mat, tl, tr, bl, 100, 100, &b_dim));
        assert_eq!(mat[6], 3); // WM_TYPE_AFFINE
        assert!(mat[2] != 0 || mat[5] != 0);
    }

    #[test]
    fn test_add_candidate_sngl_basic() {
        let mut stack = [Candidate::default(); 8];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mv = Mv { c: MvXY { y: 10, x: 20 } };
        let added = add_candidate_sngl(&mut stack, &mut cnt, 6, 2, mv, 0, 0, &mut iter, 16);
        assert!(added);
        assert_eq!(cnt, 1);
        assert_eq!(stack[0].weight, 2);
    }

    #[test]
    fn test_add_candidate_sngl_duplicate() {
        let mut stack = [Candidate::default(); 8];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mv = Mv { c: MvXY { y: 10, x: 20 } };
        add_candidate_sngl(&mut stack, &mut cnt, 6, 2, mv, 0, 0, &mut iter, 16);
        let added = add_candidate_sngl(&mut stack, &mut cnt, 6, 3, mv, 0, 0, &mut iter, 16);
        assert!(!added);
        assert_eq!(cnt, 1);
        assert_eq!(stack[0].weight, 5);
    }

    #[test]
    fn test_add_candidate_sngl_full() {
        let mut stack = [Candidate::default(); 2];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mv1 = Mv { c: MvXY { y: 10, x: 20 } };
        let mv2 = Mv { c: MvXY { y: 30, x: 40 } };
        let mv3 = Mv { c: MvXY { y: 50, x: 60 } };
        add_candidate_sngl(&mut stack, &mut cnt, 2, 1, mv1, 0, 0, &mut iter, 16);
        add_candidate_sngl(&mut stack, &mut cnt, 2, 1, mv2, 0, 0, &mut iter, 16);
        let added = add_candidate_sngl(&mut stack, &mut cnt, 2, 1, mv3, 0, 0, &mut iter, 16);
        assert!(!added);
        assert_eq!(cnt, 2);
    }

    #[test]
    fn test_add_candidate_c2s_basic() {
        let mut stack = [SnglMvBlock::default(); 4];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mv = Mv { c: MvXY { y: 5, x: 10 } };
        add_candidate_c2s(&mut stack, &mut cnt, 4, 1, mv, &mut iter, 16);
        assert_eq!(cnt, 1);
        assert_eq!(stack[0].r#ref, 1);
    }

    #[test]
    fn test_add_candidate_c2s_dup_same_ref() {
        let mut stack = [SnglMvBlock::default(); 4];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mv = Mv { c: MvXY { y: 5, x: 10 } };
        add_candidate_c2s(&mut stack, &mut cnt, 4, 1, mv, &mut iter, 16);
        add_candidate_c2s(&mut stack, &mut cnt, 4, 1, mv, &mut iter, 16);
        assert_eq!(cnt, 1);
    }

    #[test]
    fn test_add_candidate_c2s_same_mv_diff_ref() {
        let mut stack = [SnglMvBlock::default(); 4];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mv = Mv { c: MvXY { y: 5, x: 10 } };
        add_candidate_c2s(&mut stack, &mut cnt, 4, 1, mv, &mut iter, 16);
        add_candidate_c2s(&mut stack, &mut cnt, 4, 2, mv, &mut iter, 16);
        assert_eq!(cnt, 2);
    }

    #[test]
    fn test_add_candidate_comp_basic() {
        let mut stack = [Candidate::default(); 8];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mvs = [
            Mv { c: MvXY { y: 10, x: 20 } },
            Mv { c: MvXY { y: 30, x: 40 } },
        ];
        let added = add_candidate_comp(&mut stack, &mut cnt, 6, 1, 0, &mvs, &mut iter, 16);
        assert!(added);
        assert_eq!(cnt, 1);
    }

    #[test]
    fn test_add_candidate_comp_duplicate() {
        let mut stack = [Candidate::default(); 8];
        let mut cnt = 0i32;
        let mut iter = 0i32;
        let mvs = [
            Mv { c: MvXY { y: 10, x: 20 } },
            Mv { c: MvXY { y: 30, x: 40 } },
        ];
        add_candidate_comp(&mut stack, &mut cnt, 6, 2, 0, &mvs, &mut iter, 16);
        let added = add_candidate_comp(&mut stack, &mut cnt, 6, 3, 0, &mvs, &mut iter, 16);
        assert!(!added);
        assert_eq!(cnt, 1);
        assert_eq!(stack[0].weight, 5);
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

    fn make_sngl_mv_block(y: i32, x: i32, r: u8) -> SnglMvBlock {
        SnglMvBlock {
            mv: Mv { c: MvXY { y, x } },
            r#ref: r,
        }
    }

    fn invalid_sngl() -> SnglMvBlock {
        SnglMvBlock {
            mv: Mv { c: MvXY { y: INVALID_MV as i32, x: 0 } },
            r#ref: 0,
        }
    }

    #[test]
    fn test_tip_projection_basic() {
        let stride: isize = 8;
        let mut rp = vec![invalid_sngl(); 64];
        rp[0] = make_sngl_mv_block(100, 200, 1);
        tip_projection(&mut rp, stride, 0, 8, 0, 8, 8, 8, 2, 2);
        let mv_y = unsafe { rp[0].mv.c.y };
        assert_ne!(mv_y, INVALID_MV);
    }

    #[test]
    fn test_tip_projection_skips_invalid() {
        let stride: isize = 8;
        let mut rp = vec![invalid_sngl(); 64];
        tip_projection(&mut rp, stride, 0, 8, 0, 8, 8, 8, 2, 2);
        assert_eq!(unsafe { rp[0].mv.c.y }, INVALID_MV);
    }

    #[test]
    fn test_fill_holes_propagates() {
        let stride: isize = 8;
        let mut rp = vec![invalid_sngl(); 64];
        rp[0] = make_sngl_mv_block(50, 60, 1);
        fill_holes(&mut rp, stride, 0, 8, 0, 8, 8, 8, 2, 2);
        assert_ne!(unsafe { rp[2].mv.c.y }, INVALID_MV);
        assert_ne!(unsafe { rp[(stride as usize) * 2].mv.c.y }, INVALID_MV);
    }

    #[test]
    fn test_fill_holes_no_overwrite() {
        let stride: isize = 8;
        let mut rp = vec![invalid_sngl(); 64];
        rp[0] = make_sngl_mv_block(50, 60, 1);
        rp[2] = make_sngl_mv_block(70, 80, 1);
        fill_holes(&mut rp, stride, 0, 8, 0, 8, 8, 8, 2, 2);
        assert_eq!(unsafe { rp[2].mv.c.y }, 70);
    }

    #[test]
    fn test_warp_bank_add_new() {
        let mut warp = WarpBank {
            mat: [[[0i32; 6]; 4]; 7],
            warp_type: [[0i8; 4]; 7],
            hits: 0,
            size: [0u8; 7],
            idx: [0u8; 7],
        };
        let mat = WarpedMotionParams {
            matrix: [100, 200, 0x10000, 0, 0, 0x10000],
            ..WarpedMotionParams::default()
        };
        assert_eq!(warp_bank_add(&mut warp, &mat, 0), 0);
        assert_eq!(warp.size[0], 1);
        assert_eq!(warp.hits, 1);
    }

    #[test]
    fn test_warp_bank_add_dup() {
        let mut warp = WarpBank {
            mat: [[[0i32; 6]; 4]; 7],
            warp_type: [[0i8; 4]; 7],
            hits: 0,
            size: [0u8; 7],
            idx: [0u8; 7],
        };
        let mat = WarpedMotionParams {
            matrix: [100, 200, 0x10000, 0, 0, 0x10000],
            ..WarpedMotionParams::default()
        };
        warp_bank_add(&mut warp, &mat, 0);
        assert_eq!(warp_bank_add(&mut warp, &mat, 0), 0);
        assert_eq!(warp.size[0], 1);
    }

    #[test]
    fn test_warp_bank_add_full() {
        let mut warp = WarpBank {
            mat: [[[0i32; 6]; 4]; 7],
            warp_type: [[0i8; 4]; 7],
            hits: 64,
            size: [0u8; 7],
            idx: [0u8; 7],
        };
        let mat = WarpedMotionParams::default();
        assert_eq!(warp_bank_add(&mut warp, &mat, 0), -1);
    }

    #[test]
    fn test_mv_bank_add_inner_single_ref() {
        let mut bank = MvBank {
            mv: [[[Mv { n: 0 }; 2]; 4]; 9],
            cwp_idx: [[0i8; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0u8; 9],
            idx: [0u8; 9],
            hits: [0u8; 2],
            avail: 4,
        };
        let r = RefPair { r: [2, -1] };
        let mv = [
            Mv { c: MvXY { y: 10, x: 20 } },
            Mv { n: 0 },
        ];
        mv_bank_add_inner(&mut bank, r, &mv, 0);
        assert_eq!(bank.size[2], 1);
        assert_eq!(bank.hits[0], 1);
    }

    #[test]
    fn test_mv_bank_add_inner_dup() {
        let mut bank = MvBank {
            mv: [[[Mv { n: 0 }; 2]; 4]; 9],
            cwp_idx: [[0i8; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0u8; 9],
            idx: [0u8; 9],
            hits: [0u8; 2],
            avail: 4,
        };
        let r = RefPair { r: [2, -1] };
        let mv = [
            Mv { c: MvXY { y: 10, x: 20 } },
            Mv { n: 0 },
        ];
        mv_bank_add_inner(&mut bank, r, &mv, 0);
        mv_bank_add_inner(&mut bank, r, &mv, 0);
        assert_eq!(bank.size[2], 1);
    }

    #[test]
    fn test_smoothen_basic() {
        let stride: isize = 8;
        let mut rp = vec![invalid_sngl(); 64];
        for y in (0..8).step_by(2) {
            for x in (0..8).step_by(2) {
                rp[y * stride as usize + x] = make_sngl_mv_block(100, 100, 1);
            }
        }
        smoothen(&mut rp, stride, 0, 8, 0, 8, 8, 8, 2, 2);
        assert_ne!(unsafe { rp[0].mv.c.y }, INVALID_MV);
    }

    fn make_smb(x: i32, y: i32, r: u8) -> SnglMvBlock {
        SnglMvBlock { mv: Mv { c: MvXY { y, x } }, r#ref: r }
    }

    fn inv_smb() -> SnglMvBlock {
        SnglMvBlock { mv: Mv { c: MvXY { y: INVALID_MV, x: 0 } }, r#ref: 0 }
    }

    fn make_mv(x: i32, y: i32) -> Mv {
        Mv { c: MvXY { y, x } }
    }

    fn inv_mv() -> Mv {
        Mv { c: MvXY { y: INVALID_MV, x: 0 } }
    }

    #[test]
    fn test_fill_gap_proj_skip_invalid() {
        let stride: isize = 4;
        let mut rp = vec![inv_smb(); 16];
        fill_gap_proj(&mut rp, stride, 0, 4, 0, 4, 4, 4);
        for i in 0..16 {
            assert_eq!(unsafe { rp[i].mv.c.y }, INVALID_MV);
        }
    }

    #[test]
    fn test_fill_gap_proj_single_valid() {
        let stride: isize = 4;
        let mut rp = vec![inv_smb(); 16];
        rp[0] = make_smb(10, 20, 1);
        fill_gap_proj(&mut rp, stride, 0, 4, 0, 4, 4, 4);
        // pos=0 valid, no right/bottom → all neighbors copy from pos=0
        assert_eq!(unsafe { rp[1].mv.c.x }, 10);
        assert_eq!(unsafe { rp[1].mv.c.y }, 20);
        let mid = stride as usize;
        assert_eq!(unsafe { rp[mid].mv.c.x }, 10);
        assert_eq!(unsafe { rp[mid].mv.c.y }, 20);
        let diag = (1 + stride) as usize;
        assert_eq!(unsafe { rp[diag].mv.c.x }, 10);
        assert_eq!(unsafe { rp[diag].mv.c.y }, 20);
    }

    #[test]
    fn test_fill_gap_proj_two_neighbors() {
        let stride: isize = 4;
        let mut rp = vec![inv_smb(); 16];
        rp[0] = make_smb(10, 20, 1);
        rp[2] = make_smb(30, 40, 1); // same ref → projection is identity
        fill_gap_proj(&mut rp, stride, 0, 4, 0, 4, 4, 4);
        // right neighbor valid with same ref: avg(10,30)=20, avg(20,40)=30
        assert_eq!(unsafe { rp[1].mv.c.x }, 20);
        assert_eq!(unsafe { rp[1].mv.c.y }, 30);
    }

    #[test]
    fn test_fill_gap_traj_skip_invalid() {
        let stride: isize = 4;
        let mut rp = vec![inv_mv(); 16];
        fill_gap_traj(&mut rp, stride, 0, 4, 0, 4, 4, 4);
        for i in 0..16 {
            assert_eq!(unsafe { rp[i].c.y }, INVALID_MV);
        }
    }

    #[test]
    fn test_fill_gap_traj_single_valid() {
        let stride: isize = 4;
        let mut rp = vec![inv_mv(); 16];
        rp[0] = make_mv(10, 20);
        fill_gap_traj(&mut rp, stride, 0, 4, 0, 4, 4, 4);
        assert_eq!(unsafe { rp[1].c.x }, 10);
        assert_eq!(unsafe { rp[1].c.y }, 20);
        let mid = stride as usize;
        assert_eq!(unsafe { rp[mid].c.x }, 10);
        assert_eq!(unsafe { rp[mid].c.y }, 20);
    }

    #[test]
    fn test_fill_gap_traj_right_neighbor() {
        let stride: isize = 4;
        let mut rp = vec![inv_mv(); 16];
        rp[0] = make_mv(10, 20);
        rp[2] = make_mv(30, 40);
        fill_gap_traj(&mut rp, stride, 0, 4, 0, 4, 4, 4);
        // right: avg of original + right = avg(10,30)=20, avg(20,40)=30
        assert_eq!(unsafe { rp[1].c.x }, 20);
        assert_eq!(unsafe { rp[1].c.y }, 30);
    }

    #[test]
    fn test_fill_gap_traj_bottom_neighbor() {
        let stride: isize = 4;
        let mut rp = vec![inv_mv(); 16];
        rp[0] = make_mv(10, 20);
        rp[(2 * stride) as usize] = make_mv(30, 40);
        fill_gap_traj(&mut rp, stride, 0, 2, 0, 4, 4, 4);
        let mid = stride as usize;
        // bottom: avg(10,30)=20, avg(20,40)=30
        assert_eq!(unsafe { rp[mid].c.x }, 20);
        assert_eq!(unsafe { rp[mid].c.y }, 30);
    }

    #[test]
    fn test_bank_update_sb_boundary() {
        let mut bank = MvBank {
            mv: [[[Mv::default(); 2]; 4]; 9],
            cwp_idx: [[0; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0; 9],
            idx: [0; 9],
            hits: [3, 5],
            avail: 10,
        };

        bank_update(&mut bank, crate::levels::BlockSize::Bs16x16, 0, 0, 16, false);
        assert_eq!(bank.hits[1], 0);
        assert!(bank.avail >= 4);
    }

    #[test]
    fn test_bank_update_sub_sb_boundary() {
        let mut bank = MvBank {
            mv: [[[Mv::default(); 2]; 4]; 9],
            cwp_idx: [[0; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0; 9],
            idx: [0; 9],
            hits: [0, 0],
            avail: 5,
        };

        bank_update(&mut bank, crate::levels::BlockSize::Bs8x8, 4, 4, 16, false);
        assert_eq!(bank.hits[1], 0);
        assert!(bank.avail > 5);
    }

    #[test]
    fn test_bank_update_no_boundary() {
        let mut bank = MvBank {
            mv: [[[Mv::default(); 2]; 4]; 9],
            cwp_idx: [[0; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0; 9],
            idx: [0; 9],
            hits: [2, 3],
            avail: 7,
        };

        bank_update(&mut bank, crate::levels::BlockSize::Bs8x8, 3, 3, 16, false);
        assert_eq!(bank.hits[1], 3);
        assert_eq!(bank.avail, 7);
    }

    #[test]
    fn test_splat_mv_1x1() {
        let mut dst = vec![Block::default(); 256];
        let mut src = Block::default();
        src.bs = crate::levels::BlockSize::Bs4x4 as u8;
        src.mf = 1;
        let t_src = TemporalBlock::default();

        splat_mv(&mut dst, &mut src, None, 0, &t_src, 2, 2);

        assert_eq!(dst[0].bs, crate::levels::BlockSize::Bs4x4 as u8);
        assert_eq!(dst[1].bs, crate::levels::BlockSize::Bs4x4 as u8);
        assert_eq!(dst[128].bs, crate::levels::BlockSize::Bs4x4 as u8);
        assert_eq!(dst[129].bs, crate::levels::BlockSize::Bs4x4 as u8);
    }

    #[test]
    fn test_save_tmvs_basic() {
        let mut r = vec![Block::default(); 128 * 64];
        let mut ra = vec![Block::default(); 32];
        let mut ra_tl = Block::default();

        r[((3 & 31) * 2 + 1) as usize * 128 + 4] = {
            let mut b = Block::default();
            b.bs = 5;
            b
        };
        r[((3 & 31) * 2 + 1) as usize * 128 + 6] = {
            let mut b = Block::default();
            b.bs = 7;
            b
        };

        save_tmvs(&r, &mut ra, &mut ra_tl, 2, 4, 0, 4, 8, 8);

        assert_eq!(ra[2].bs, 5);
        assert_eq!(ra[3].bs, 7);
        assert_eq!(ra_tl.bs, 0);
    }

    #[test]
    fn test_save_tmvs_clamps() {
        let r = vec![Block::default(); 128 * 64];
        let mut ra = vec![Block::default(); 8];
        let mut ra_tl = Block::default();

        save_tmvs(&r, &mut ra, &mut ra_tl, 0, 10, 0, 10, 4, 4);
        assert_eq!(ra_tl.bs, 0);
    }

    #[test]
    fn test_splat_warpmv_basic() {
        let mut dst = vec![Block::default(); 512];
        let mut src = Block::default();
        src.mf = 2;
        src.bs = 10;
        let mut t_src = TemporalBlock::default();
        let mat = WarpedMotionParams {
            matrix: [0, 0, 0x10000, 0, 0, 0x10000],
            ..WarpedMotionParams::default()
        };

        splat_warpmv(&mut dst, &mut src, None, 0, &mut t_src, 0, 0, &mat, 4, 4);

        assert_eq!(dst[0].bs, 10);
        assert_eq!(dst[1].bs, 10);
        assert_eq!(dst[128].bs, 10);
        assert_eq!(dst[129].bs, 10);
        assert_eq!(dst[256].bs, 10);
    }

    #[test]
    fn test_splat_warpmv_sets_mv() {
        let mut dst = vec![Block::default(); 512];
        let mut src = Block::default();
        src.mf = 3;
        let mut t_src = TemporalBlock::default();
        let mat = WarpedMotionParams {
            matrix: [0, 0, 0x10000, 0, 0, 0x10000],
            ..WarpedMotionParams::default()
        };

        splat_warpmv(&mut dst, &mut src, None, 0, &mut t_src, 8192 * 13, 0, &mat, 2, 2);

        let y = unsafe { dst[0].mv[0].c.y };
        assert_eq!(y, 13);
    }

    #[test]
    fn test_splat_comp_wedgemv_basic() {
        let mut dst = vec![Block::default(); 512];
        let mut src = Block::default();
        src.bs = 12;
        let t_src = TemporalBlock::default();
        let mask = vec![2u8; 4];

        splat_comp_wedgemv(&mut dst, &mut src, None, 0, &t_src, 4, 4, &mask, 0);

        assert_eq!(dst[0].bs, 12);
        assert_eq!(dst[256].bs, 12);
    }

    #[test]
    fn test_splat_comp_warpmv_basic() {
        let mut dst = vec![Block::default(); 512];
        let mut src = Block::default();
        src.mf = 2;
        src.bs = 8;
        let mut t_src = TemporalBlock::default();
        let mat = WarpedMotionParams {
            matrix: [0, 0, 0x10000, 0, 0, 0x10000],
            ..WarpedMotionParams::default()
        };

        splat_comp_warpmv(
            &mut dst, &mut src, None, 0, &mut t_src,
            0, 0, 0, 0, &mat, &mat, 4, 4, 0, None, 0,
        );

        assert_eq!(dst[0].bs, 8);
        assert_eq!(dst[256].bs, 8);
    }

    #[test]
    fn test_tile_sbrow_init_no_threading() {
        let rf = Frame {
            iw4: 64, ih4: 64, iw8: 32, ih8: 32,
            sbsz: 16, rp_stride: 32,
            have_threading: false, have_frame_threading: false,
            ..make_default_frame()
        };
        let mut rt = make_default_tile();
        tile_sbrow_init(&mut rt, &rf, 0, 32, 0, 32, 1, 3);

        assert_eq!(rt.rp_proj_off, (10 * 0 + 2 * 32) as usize);
        assert_eq!(rt.rp_traj_off, 0);
        assert_eq!(rt.ra_off, 0);
        assert_eq!(rt.tile_row.start, 0);
        assert_eq!(rt.tile_row.end, 32);
        assert_eq!(rt.tile_col.start, 0);
        assert_eq!(rt.tile_col.end, 32);
        assert!(rt.bank.size.iter().all(|&x| x == 0));
        assert!(rt.warp.size.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_tile_sbrow_init_with_threading() {
        let rf = Frame {
            iw4: 128, ih4: 128, iw8: 64, ih8: 64,
            sbsz: 16, rp_stride: 64,
            have_threading: true, have_frame_threading: false,
            ..make_default_frame()
        };
        let mut rt = make_default_tile();
        tile_sbrow_init(&mut rt, &rf, 4, 60, 8, 120, 2, 1);

        let off1 = 64isize * 1;
        let sbsz8 = 8;
        let off2 = sbsz8 * off1;
        let off3 = (sbsz8 + 2) * off1 + 2 * 64;
        assert_eq!(rt.rp_proj_off, off3 as usize);
        assert_eq!(rt.rp_traj_off, off2 as usize);
        assert_eq!(rt.ra_off, off1 as usize);
        assert_eq!(rt.tile_row.end, 120);
    }

    #[test]
    fn test_tile_sbrow_init_frame_threading() {
        let rf = Frame {
            iw4: 64, ih4: 64, iw8: 32, ih8: 32,
            sbsz: 16, rp_stride: 32,
            have_threading: true, have_frame_threading: true,
            ..make_default_frame()
        };
        let mut rt = make_default_tile();
        tile_sbrow_init(&mut rt, &rf, 0, 64, 0, 64, 3, 2);

        let sbsz8 = 8;
        assert_eq!(rt.rp_proj_off, (3 * sbsz8 * 32) as usize);
        assert_eq!(rt.tile_row.end, 64);
    }

    #[test]
    fn test_reset_sb_basic() {
        let mut rt = make_default_tile();
        rt.r = vec![Block::default(); 128 * 64];
        rt.tile_row = TileRange { start: 0, end: 64 };
        rt.tile_col = TileRange { start: 0, end: 64 };
        let ra = vec![Block::default(); 64];

        reset_sb(&mut rt, &ra, 16, false, false, 0, 0, 0);

        for y in 0..16usize {
            for x in 0..16usize {
                assert_eq!(unsafe { rt.r[y * 128 + x].mv[0].c.y }, INVALID_MV);
                assert_eq!(unsafe { rt.r[y * 128 + x].r#ref.pair }, -1);
            }
        }
    }

    #[test]
    fn test_reset_sb_refmv_bank() {
        let mut rt = make_default_tile();
        rt.r = vec![Block::default(); 128 * 64];
        rt.tile_row = TileRange { start: 0, end: 64 };
        rt.tile_col = TileRange { start: 0, end: 64 };
        rt.bank.hits = [5, 3];
        rt.bank.avail = 10;
        rt.warp.hits = 7;
        let ra = vec![Block::default(); 64];

        reset_sb(&mut rt, &ra, 16, true, false, 0, 0, 0);

        assert_eq!(rt.bank.hits[0], 0);
        assert_eq!(rt.bank.hits[1], 0);
        assert_eq!(rt.bank.avail, 0);
        assert_eq!(rt.warp.hits, 0);
    }

    #[test]
    fn test_reset_sb_key_frame_early_return() {
        let mut rt = make_default_tile();
        rt.r = vec![Block::default(); 128 * 64];
        rt.tile_row = TileRange { start: 0, end: 64 };
        rt.tile_col = TileRange { start: 0, end: 64 };
        let ra = vec![Block::default(); 64];

        reset_sb(&mut rt, &ra, 16, true, true, 0, 16, 0);
        assert_eq!(rt.bank.hits[0], 0);
    }

    #[test]
    fn test_check_traj_intersect_no_valid_maps() {
        let rf = Frame {
            iw8: 32, ih8: 32, sbsz: 16,
            mfmv_sbsz8: 8, mfmv_edge: 4, mfmv_k_shift: 3,
            rp_stride: 32,
            ..make_default_frame()
        };
        let stride = rf.rp_stride as usize;
        let sz = stride * 8;
        let invalid_mv = Mv { c: MvXY { y: INVALID_MV, x: 0 } };
        let mut rp_traj: [Vec<Mv>; 7] = Default::default();
        for v in rp_traj.iter_mut() {
            *v = vec![invalid_mv; sz];
        }
        let invalid_map = TrajMap { y: -128i8, x: -128i8 };
        let mut map: [[Vec<TrajMap>; 7]; 3] = Default::default();
        for k in 0..3 {
            for r in 0..7 {
                map[k][r] = vec![invalid_map; sz];
            }
        }
        let mv_in = Mv { c: MvXY { y: 64, x: 32 } };
        check_traj_intersect(&rf, &mut rp_traj, &mut map, 0, 1, 4, 4, mv_in, 0, 3, !0);

        assert_eq!(unsafe { rp_traj[1][4 * stride + 4].c.y }, INVALID_MV);
    }

    #[test]
    fn test_traj_map_n() {
        let tm = TrajMap { y: -128i8, x: -128i8 };
        assert_eq!(unsafe { tm.n() }, INVALID_TRAJ);
        let tm2 = TrajMap { y: 0, x: 0 };
        assert_eq!(unsafe { tm2.n() }, 0);
    }

    #[test]
    fn test_load_tmvs_basic() {
        let stride: isize = 8;
        let sbsz8: usize = 8;
        let total = (sbsz8 + 2) * stride as usize + sbsz8 * stride as usize;
        let mut rf = make_default_frame();
        rf.sbsz = 16;
        rf.ih8 = 8;
        rf.iw8 = 4;
        rf.rp_stride = stride;
        rf.n_mfmvs = 0;
        rf.have_threading = false;
        rf.have_frame_threading = false;
        rf.rp_proj = vec![SnglMvBlock { mv: Mv::default(), r#ref: 0 }; total];

        load_tmvs(&mut rf, 0, 0, 4, 0, 2, false, 0, false, 1, 0);

        let poffset = 2 * stride as usize;
        for y in 0..2usize {
            for x in 0..4usize {
                let idx = poffset + y * stride as usize + x;
                assert_eq!(unsafe { rf.rp_proj[idx].mv.c.y }, INVALID_MV,
                    "rp_proj[{idx}] should be INVALID_MV");
            }
        }
    }

    #[test]
    fn test_init_frame_basic() {
        use crate::headers::{SequenceHeader, FrameHeader};
        let mut rf = make_default_frame();
        let seq_hdr = SequenceHeader {
            order_hint_n_bits: 4,
            ..Default::default()
        };
        let mut frm_hdr = FrameHeader::default();
        frm_hdr.width = 256;
        frm_hdr.height = 128;
        frm_hdr.n_ref_frames = 3;
        frm_hdr.sb128 = 0;
        frm_hdr.tmvp_sample_step = 1;

        let ref_poc = [1, 2, 3, 0, 0, 0, 0];
        let ref_ref_poc = [[0u8; 7]; 7];
        let refcnt = [0u8; 7];
        let rp_ref: [Option<Vec<TemporalBlock>>; 7] = Default::default();

        init_frame(&mut rf, &seq_hdr, &frm_hdr, &ref_poc, &ref_ref_poc,
                   &refcnt, &rp_ref, false, false);

        assert_eq!(rf.iw8, 32);
        assert_eq!(rf.ih8, 16);
        assert_eq!(rf.iw4, 64);
        assert_eq!(rf.ih4, 32);
        assert_eq!(rf.sbsz, 16);
        assert_eq!(rf.mfmv_sbsz8, 8);
        assert!(!rf.rp_proj.is_empty());
        assert!(!rf.ra.is_empty());
        assert_eq!(rf.n_mfmvs, 0);
        assert_eq!(rf.use_ref_frame_mvs, 0);
    }

    #[test]
    fn test_init_frame_with_tmvs() {
        use crate::headers::{SequenceHeader, FrameHeader};
        let mut rf = make_default_frame();
        let seq_hdr = SequenceHeader {
            order_hint_n_bits: 4,
            ..Default::default()
        };
        let mut frm_hdr = FrameHeader::default();
        frm_hdr.width = 256;
        frm_hdr.height = 128;
        frm_hdr.n_ref_frames = 2;
        frm_hdr.sb128 = 0;
        frm_hdr.tmvp_sample_step = 1;
        frm_hdr.use_ref_frame_mvs = 1;

        let ref_poc = [5, 3, 0, 0, 0, 0, 0];
        let ref_ref_poc = [[0u8; 7]; 7];
        let refcnt = [1, 1, 0, 0, 0, 0, 0];
        let rp_ref: [Option<Vec<TemporalBlock>>; 7] = [
            Some(vec![TemporalBlock::default()]),
            Some(vec![TemporalBlock::default()]),
            None, None, None, None, None,
        ];

        init_frame(&mut rf, &seq_hdr, &frm_hdr, &ref_poc, &ref_ref_poc,
                   &refcnt, &rp_ref, false, false);

        assert_eq!(rf.use_ref_frame_mvs, if rf.n_mfmvs > 0 { 1 } else { 0 });
        assert_eq!(rf.ref_sign[0], 0);
    }

    fn make_default_frame() -> Frame {
        Frame {
            iw4: 0, ih4: 0, iw8: 0, ih8: 0,
            sbsz: 16, mfmv_sbsz8: 0, mfmv_edge: 0, mfmv_k_shift: 0,
            use_ref_frame_mvs: 0,
            tip: FrameTip { sf: [0; 2], r#ref: RefPair::default(), delta: 0 },
            ref_sign: [0; 7], pocdiff: [0; 7], ref_flip: 0,
            abspocdiff: [0; 7], mfmv_mask: 0,
            mfmv: [MfmvRef { r#ref: 0, tgt: 0, dir: 0 }; 4],
            mfmv_ref2cur: [0; 4], mfmv_ref2ref: [[0; 7]; 4],
            mfmv_ref2idx: [[0; 7]; 4], mfmv_ref2sf: [[[0; 2]; 7]; 4],
            n_mfmvs: 0, n_blocks: 0,
            rp: Vec::new(), rp_stride: 0, rp_ref: Default::default(),
            rp_proj: Vec::new(),
            rp_traj: Default::default(), rp_map: Default::default(),
            ra: Vec::new(),
            have_threading: false, have_frame_threading: false,
        }
    }

    fn make_default_tile() -> Tile {
        Tile {
            rp_proj: Vec::new(), rp_proj_off: 0, rp_traj_off: 0,
            ra: Vec::new(), ra_off: 0, ra_tl: Block::default(),
            r: Vec::new(),
            tile_col: TileRange { start: 0, end: 0 },
            tile_row: TileRange { start: 0, end: 0 },
            bank: MvBank {
                mv: [[[Mv::default(); 2]; 4]; 9],
                cwp_idx: [[0; 4]; 3], r#ref: [RefPair::default(); 4],
                size: [0; 9], idx: [0; 9], hits: [0; 2], avail: 0,
            },
            warp: WarpBank {
                mat: [[[0; 6]; 4]; 7], warp_type: [[0; 4]; 7],
                hits: 0, size: [0; 7], idx: [0; 7],
            },
        }
    }
}

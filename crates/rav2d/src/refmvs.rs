use crate::headers::WarpedMotionParams;
use crate::intops::{apply_sign, iclip, iclip64to32, imax, imin, ulog2};
use crate::levels::INVALID_MV;
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
    pub ra: Vec<Block>,
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
}

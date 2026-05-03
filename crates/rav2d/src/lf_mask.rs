#[derive(Clone)]
pub struct Av2RestorationUnit {
    pub restoration_type: u8,
    pub ns_filter: [[i8; 32]; 16],
}

impl Default for Av2RestorationUnit {
    fn default() -> Self {
        Self {
            restoration_type: 0,
            ns_filter: [[0; 32]; 16],
        }
    }
}

#[derive(Clone)]
pub struct Av2Filter {
    pub filter_y: [[[[u16; 4]; 5]; 64]; 2],
    pub filter_uv: [[[[u16; 4]; 5]; 64]; 2],
    pub qidx: [u16; 16],
    pub gdf: [u8; 16],
    pub cdef_idx: [i8; 16],
    pub ccso: [u8; 3],
    pub noskip_mask: [[u16; 4]; 32],
    pub lr_noskip_mask: [[u16; 4]; 64],
    pub lossless_mask_y: [[u16; 4]; 64],
    pub lossless_mask_uv: [[u16; 4]; 64],
}

impl Default for Av2Filter {
    fn default() -> Self {
        Self {
            filter_y: [[[[0; 4]; 5]; 64]; 2],
            filter_uv: [[[[0; 4]; 5]; 64]; 2],
            qidx: [0; 16],
            gdf: [0; 16],
            cdef_idx: [-1; 16],
            ccso: [0; 3],
            noskip_mask: [[0; 4]; 32],
            lr_noskip_mask: [[0; 4]; 64],
            lossless_mask_y: [[0; 4]; 64],
            lossless_mask_uv: [[0; 4]; 64],
        }
    }
}

#[derive(Clone)]
pub struct Av2Restoration {
    pub lr: [[Av2RestorationUnit; 16]; 3],
}

impl Default for Av2Restoration {
    fn default() -> Self {
        Self {
            lr: std::array::from_fn(|_| std::array::from_fn(|_| Av2RestorationUnit::default())),
        }
    }
}

use crate::intops::imin;

pub type FilterMasks = [[[u16; 4]; 5]; 64];

pub fn mask_outer_edge_l(
    masks: &mut [[u16; 4]],
    by4: i32,
    h4: i32,
    bwl4c: u8,
    l: &mut [u8],
) {
    debug_assert!((bwl4c as u32) <= 3);
    let mut mask: u64 = 1 << by4;
    for y in 0..h4 as usize {
        let sidx = ((by4 as usize) + y) >> 4;
        let smask = (mask >> (sidx << 4)) as u16;
        let lvl = imin(bwl4c as i32, l[y] as i32) as usize;
        masks[lvl][sidx] |= smask;
        mask <<= 1;
    }
    for y in 0..h4 as usize {
        l[y] = bwl4c;
    }
}

pub fn mask_outer_edge_t(
    masks: &mut [[u16; 4]],
    bx4: i32,
    w4: i32,
    bhl4c: u8,
    a: &mut [u8],
) {
    debug_assert!((bhl4c as u32) <= 3);
    let mut mask: u64 = 1 << bx4;
    for x in 0..w4 as usize {
        let sidx = ((bx4 as usize) + x) >> 4;
        let smask = (mask >> (sidx << 4)) as u16;
        let lvl = imin(bhl4c as i32, a[x] as i32) as usize;
        masks[lvl][sidx] |= smask;
        mask <<= 1;
    }
    for x in 0..w4 as usize {
        a[x] = bhl4c;
    }
}

pub fn mask_inner_edges_v(
    masks: &mut [FilterMasks; 2],
    inner: u64,
    bx4: i32,
    w4: i32,
    twl4c: i32,
    xoff: i32,
    hstep: i32,
) {
    debug_assert!((twl4c as u32) <= 3);
    let inner1 = (inner & 0xffff) as u16;
    let inner2 = ((inner >> 16) & 0xffff) as u16;
    let inner3 = ((inner >> 32) & 0xffff) as u16;
    let inner4 = (inner >> 48) as u16;
    let t = twl4c as usize;
    let mut x = xoff;
    while x < w4 {
        let idx = (bx4 + x) as usize;
        if inner1 != 0 { masks[0][idx][t][0] |= inner1; }
        if inner2 != 0 { masks[0][idx][t][1] |= inner2; }
        if inner3 != 0 { masks[0][idx][t][2] |= inner3; }
        if inner4 != 0 { masks[0][idx][t][3] |= inner4; }
        x += hstep;
    }
}

pub fn mask_inner_edges_h(
    masks: &mut [FilterMasks; 2],
    inner: u64,
    by4: i32,
    h4: i32,
    thl4c: i32,
    yoff: i32,
    vstep: i32,
) {
    debug_assert!((thl4c as u32) <= 3);
    let inner1 = (inner & 0xffff) as u16;
    let inner2 = ((inner >> 16) & 0xffff) as u16;
    let inner3 = ((inner >> 32) & 0xffff) as u16;
    let inner4 = (inner >> 48) as u16;
    let t = thl4c as usize;
    let mut y = yoff;
    while y < h4 {
        let idx = (by4 + y) as usize;
        if inner1 != 0 { masks[1][idx][t][0] |= inner1; }
        if inner2 != 0 { masks[1][idx][t][1] |= inner2; }
        if inner3 != 0 { masks[1][idx][t][2] |= inner3; }
        if inner4 != 0 { masks[1][idx][t][3] |= inner4; }
        y += vstep;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av2_filter_default() {
        let f = Av2Filter::default();
        assert_eq!(f.cdef_idx[0], -1);
        assert_eq!(f.qidx[0], 0);
    }

    #[test]
    fn test_av2_restoration_default() {
        let r = Av2Restoration::default();
        assert_eq!(r.lr[0][0].restoration_type, 0);
    }

    #[test]
    fn test_mask_outer_edge_l_basic() {
        let mut masks = [[0u16; 4]; 5];
        let mut l = [2u8; 4];
        mask_outer_edge_l(&mut masks, 0, 4, 1, &mut l);
        assert_eq!(masks[1][0], 0b1111);
        for i in 0..4 { assert_eq!(l[i], 1); }
    }

    #[test]
    fn test_mask_outer_edge_l_min_level() {
        let mut masks = [[0u16; 4]; 5];
        let mut l = [1u8; 2];
        mask_outer_edge_l(&mut masks, 0, 2, 3, &mut l);
        assert_eq!(masks[1][0], 0b11);
        assert_eq!(masks[3][0], 0);
        for i in 0..2 { assert_eq!(l[i], 3); }
    }

    #[test]
    fn test_mask_outer_edge_l_crosses_segment() {
        let mut masks = [[0u16; 4]; 5];
        let mut l = [2u8; 4];
        mask_outer_edge_l(&mut masks, 14, 4, 2, &mut l);
        assert_eq!(masks[2][0] & (1 << 14), 1 << 14);
        assert_eq!(masks[2][0] & (1 << 15), 1 << 15);
        assert_eq!(masks[2][1] & 1, 1);
        assert_eq!(masks[2][1] & 2, 2);
    }

    #[test]
    fn test_mask_outer_edge_t_basic() {
        let mut masks = [[0u16; 4]; 5];
        let mut a = [3u8; 4];
        mask_outer_edge_t(&mut masks, 0, 4, 2, &mut a);
        assert_eq!(masks[2][0], 0b1111);
        for i in 0..4 { assert_eq!(a[i], 2); }
    }

    #[test]
    fn test_mask_outer_edge_t_min_level() {
        let mut masks = [[0u16; 4]; 5];
        let mut a = [0u8; 3];
        mask_outer_edge_t(&mut masks, 0, 3, 2, &mut a);
        assert_eq!(masks[0][0], 0b111);
    }

    #[test]
    fn test_mask_inner_edges_v_basic() {
        let mut masks = [[[[0u16; 4]; 5]; 64]; 2];
        let inner: u64 = 0x0001_0001_0001_0001;
        mask_inner_edges_v(&mut masks, inner, 0, 8, 1, 4, 4);
        assert_eq!(masks[0][4][1][0], 1);
        assert_eq!(masks[0][4][1][1], 1);
        assert_eq!(masks[0][4][1][2], 1);
        assert_eq!(masks[0][4][1][3], 1);
    }

    #[test]
    fn test_mask_inner_edges_v_zero_inner() {
        let mut masks = [[[[0u16; 4]; 5]; 64]; 2];
        mask_inner_edges_v(&mut masks, 0, 0, 8, 1, 4, 4);
        for row in &masks[0] {
            for lvl in row { for s in lvl { assert_eq!(*s, 0); } }
        }
    }

    #[test]
    fn test_mask_inner_edges_h_basic() {
        let mut masks = [[[[0u16; 4]; 5]; 64]; 2];
        let inner: u64 = 0x0003_0003_0003_0003;
        mask_inner_edges_h(&mut masks, inner, 0, 8, 2, 4, 4);
        assert_eq!(masks[1][4][2][0], 3);
        assert_eq!(masks[1][4][2][1], 3);
        assert_eq!(masks[1][4][2][2], 3);
        assert_eq!(masks[1][4][2][3], 3);
    }

    #[test]
    fn test_mask_inner_edges_h_zero_inner() {
        let mut masks = [[[[0u16; 4]; 5]; 64]; 2];
        mask_inner_edges_h(&mut masks, 0, 0, 8, 1, 4, 4);
        for row in &masks[1] {
            for lvl in row { for s in lvl { assert_eq!(*s, 0); } }
        }
    }

    #[test]
    fn test_mask_inner_edges_h_multiple_steps() {
        let mut masks = [[[[0u16; 4]; 5]; 64]; 2];
        let inner: u64 = 0x0000_0000_0000_00FF;
        mask_inner_edges_h(&mut masks, inner, 0, 16, 1, 2, 4);
        assert_ne!(masks[1][2][1][0], 0);
        assert_ne!(masks[1][6][1][0], 0);
        assert_ne!(masks[1][10][1][0], 0);
        assert_ne!(masks[1][14][1][0], 0);
    }
}

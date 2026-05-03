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
}

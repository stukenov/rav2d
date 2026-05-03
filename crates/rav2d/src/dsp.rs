use crate::levels::N_RECT_TX_SIZES;

pub const N_FILTERS: usize = 4;
pub const N_SWITCHABLE_FILTERS: usize = 3;

pub type PixelFn = unsafe extern "C" fn();

#[derive(Clone)]
pub struct FilmGrainDSPContext {
    pub generate_grain_y: Option<PixelFn>,
    pub generate_grain_uv: [Option<PixelFn>; 3],
    pub fgy_32x32xn: Option<PixelFn>,
    pub fguv_32x32xn: [Option<PixelFn>; 3],
}

impl Default for FilmGrainDSPContext {
    fn default() -> Self {
        Self {
            generate_grain_y: None,
            generate_grain_uv: [None; 3],
            fgy_32x32xn: None,
            fguv_32x32xn: [None; 3],
        }
    }
}

#[derive(Clone)]
pub struct IntraPredDSPContext {
    pub intra_pred: [Option<PixelFn>; 14],
    pub cfl_pred: [[Option<PixelFn>; 3]; 2],
    pub cfl_gen_y: [[Option<PixelFn>; 3]; 3],
    pub cfl_gen_mat: [Option<PixelFn>; 3],
    pub cfl_calc_alphas: Option<PixelFn>,
    pub cfl_mhccp_pred: [Option<PixelFn>; 3],
    pub pal_pred: Option<PixelFn>,
}

impl Default for IntraPredDSPContext {
    fn default() -> Self {
        Self {
            intra_pred: [None; 14],
            cfl_pred: [[None; 3]; 2],
            cfl_gen_y: [[None; 3]; 3],
            cfl_gen_mat: [None; 3],
            cfl_calc_alphas: None,
            cfl_mhccp_pred: [None; 3],
            pal_pred: None,
        }
    }
}

#[derive(Clone)]
pub struct MCDSPContext {
    pub mc: [Option<PixelFn>; N_FILTERS],
    pub mc_scaled: [Option<PixelFn>; N_FILTERS],
    pub mct: [Option<PixelFn>; N_FILTERS],
    pub mct_scaled: [Option<PixelFn>; N_FILTERS],
    pub avg: Option<PixelFn>,
    pub w_avg: Option<PixelFn>,
    pub mask: Option<PixelFn>,
    pub w_mask: [Option<PixelFn>; 3],
    pub blend: Option<PixelFn>,
    pub warp8x8: Option<PixelFn>,
    pub warp8x8t: Option<PixelFn>,
    pub ext_warp4x4: Option<PixelFn>,
    pub ext_warp4x4t: Option<PixelFn>,
    pub emu_edge: Option<PixelFn>,
    pub morph: Option<PixelFn>,
    pub opfl_derive_mv: Option<PixelFn>,
    pub sad_refine_mv: Option<PixelFn>,
    pub sad8x8: Option<PixelFn>,
}

impl Default for MCDSPContext {
    fn default() -> Self {
        Self {
            mc: [None; N_FILTERS],
            mc_scaled: [None; N_FILTERS],
            mct: [None; N_FILTERS],
            mct_scaled: [None; N_FILTERS],
            avg: None,
            w_avg: None,
            mask: None,
            w_mask: [None; 3],
            blend: None,
            warp8x8: None,
            warp8x8t: None,
            ext_warp4x4: None,
            ext_warp4x4t: None,
            emu_edge: None,
            morph: None,
            opfl_derive_mv: None,
            sad_refine_mv: None,
            sad8x8: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct InvTxfmDSPContext {
    pub cctx: Option<PixelFn>,
    pub itxfm_add: [Option<PixelFn>; N_RECT_TX_SIZES],
}

#[derive(Clone, Default)]
pub struct StxDSPContext {
    pub stxfm: Option<PixelFn>,
}

#[derive(Clone, Default)]
pub struct DeblockDSPContext {
    pub deblock_sb: [[Option<PixelFn>; 2]; 2],
}

#[derive(Clone, Default)]
pub struct CcsoDSPContext {
    pub prep: [Option<PixelFn>; 3],
    pub add: Option<PixelFn>,
}

#[derive(Clone, Default)]
pub struct CdefDSPContext {
    pub dir: Option<PixelFn>,
    pub fb: [Option<PixelFn>; 3],
}

#[derive(Clone)]
pub struct LoopRestorationDSPContext {
    pub ns_wiener_single: [Option<PixelFn>; 2],
    pub ns_wiener_multi: Option<PixelFn>,
    pub pc_wiener: Option<PixelFn>,
    pub gdf_prep: Option<PixelFn>,
    pub gdf_add: Option<PixelFn>,
}

impl Default for LoopRestorationDSPContext {
    fn default() -> Self {
        Self {
            ns_wiener_single: [None; 2],
            ns_wiener_multi: None,
            pc_wiener: None,
            gdf_prep: None,
            gdf_add: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct DSPContext {
    pub fg: FilmGrainDSPContext,
    pub ipred: IntraPredDSPContext,
    pub mc: MCDSPContext,
    pub itx: InvTxfmDSPContext,
    pub stx: StxDSPContext,
    pub lf: DeblockDSPContext,
    pub ccso: CcsoDSPContext,
    pub cdef: CdefDSPContext,
    pub lr: LoopRestorationDSPContext,
}

#[derive(Clone, Default)]
pub struct PalDSPContext {
    pub pal_idx_finish: Option<PixelFn>,
}

#[derive(Clone, Default)]
pub struct RefmvsDSPContext {
    pub splat_mv: Option<PixelFn>,
    pub splat_warpmv: Option<PixelFn>,
    pub splat_comp_warpmv: Option<PixelFn>,
    pub splat_comp_wedgemv: Option<PixelFn>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsp_context_default() {
        let dsp = DSPContext::default();
        assert!(dsp.mc.avg.is_none());
        assert!(dsp.itx.itxfm_add[0].is_none());
    }

    #[test]
    fn test_n_filters() {
        assert_eq!(N_FILTERS, 4);
        assert_eq!(N_SWITCHABLE_FILTERS, 3);
    }
}

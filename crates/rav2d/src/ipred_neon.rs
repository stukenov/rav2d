//! NEON (aarch64) dispatch for the intra-prediction DSP family.
//!
//! dav2d ships hand-written NEON kernels for intra prediction
//! (`dav2d/src/arm/64/ipred.S`). rav2d assembles that `.S` into its own
//! staticlib (see `build.rs`) and calls the whole-block predictor kernels
//! through the C ABI declared in `dav2d/src/arm/ipred.h`. Each wrapper here
//! mirrors a scalar `crate::ipred` function and falls back to scalar when NEON
//! is unavailable or the mode is not bit-exact with rav2d's scalar.
//!
//! ## What is dispatched to NEON, and what stays scalar
//!
//! AV2 (AVM) changes intra prediction substantially versus AV1, and dav2d's own
//! AV2 fork **disables most of its NEON ipred kernels** in `ipred.h`
//! (`#if ARCH_AARCH64 && 0`): DC, Paeth, full Smooth, the directional Z1/Z2/Z3
//! fillers, CfL and filter-intra are all gated off there because the AV2 changes
//! made them no longer bit-exact. For 8bpc on aarch64 dav2d only keeps:
//!   * `ipred_h` (HOR_PRED), `ipred_v` (VERT_PRED)
//!   * `ipred_smooth_h` (SMOOTH_H_PRED), `ipred_smooth_v` (SMOOTH_V_PRED)
//!   * `pal_pred`
//!
//! rav2d follows the same conservative set, minus `pal_pred`: AV2 packs **two
//! 3-bit palette indices per byte** (`crate::ipred::pal_pred_8bpc`), whereas the
//! AV1 NEON `pal_pred` expects one index per byte, so it is not bit-compatible
//! and stays scalar. The H/V kernels are routed to NEON only when the AV2
//! multi-reference-line averaging flag (`ANGLE_MULTI_MRL_FLAG`) is absent, since
//! the asm has no MRL path. Everything wired here is verified bit-exact against
//! the scalar reference by the NEON-vs-scalar guard tests below.

use crate::levels::ANGLE_MULTI_MRL_FLAG;

/// VERT_PRED: copy the top row down. NEON-handled unless the AV2 MRL-averaging
/// flag is set (no asm path), in which case it falls back to scalar.
#[allow(clippy::too_many_arguments)]
pub fn ipred_v(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if angle & ANGLE_MULTI_MRL_FLAG != 0
        || !neon::ipred_simple(neon::Mode::V, dst, stride, tl, o, width, height, angle)
    {
        crate::ipred::ipred_v(dst, stride, tl, o, width, height, angle);
    }
}

/// HOR_PRED: replicate the left column across each row.
#[allow(clippy::too_many_arguments)]
pub fn ipred_h(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if angle & ANGLE_MULTI_MRL_FLAG != 0
        || !neon::ipred_simple(neon::Mode::H, dst, stride, tl, o, width, height, angle)
    {
        crate::ipred::ipred_h(dst, stride, tl, o, width, height, angle);
    }
}

/// SMOOTH_V_PRED: vertical smooth interpolation between top row and bottom edge.
pub fn ipred_smooth_v(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    if !neon::ipred_simple(neon::Mode::SmoothV, dst, stride, tl, o, w, h, 0) {
        crate::ipred::ipred_smooth_v(dst, stride, tl, o, w, h);
    }
}

/// SMOOTH_H_PRED: horizontal smooth interpolation between left column and right edge.
pub fn ipred_smooth_h(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    if !neon::ipred_simple(neon::Mode::SmoothH, dst, stride, tl, o, w, h, 0) {
        crate::ipred::ipred_smooth_h(dst, stride, tl, o, w, h);
    }
}

#[cfg(all(target_arch = "aarch64", rav2d_neon_ipred))]
mod neon {
    use crate::cpu::{arm, get_cpu_flags};
    use std::os::raw::c_int;

    /// ABI from dav2d/src/arm/ipred.h (8bpc): `decl_angular_ipred_fn`.
    /// `(dst, stride, topleft, w, h, angle, max_width, max_height)`.
    pub type AngularFn = unsafe extern "C" fn(
        dst: *mut u8,
        stride: isize,
        topleft: *const u8,
        w: c_int,
        h: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
    );

    unsafe extern "C" {
        pub fn dav2d_ipred_h_8bpc_neon();
        pub fn dav2d_ipred_v_8bpc_neon();
        pub fn dav2d_ipred_smooth_h_8bpc_neon();
        pub fn dav2d_ipred_smooth_v_8bpc_neon();
    }

    #[derive(Clone, Copy)]
    pub enum Mode {
        H,
        V,
        SmoothH,
        SmoothV,
    }

    impl Mode {
        fn sym(self) -> AngularFn {
            unsafe {
                let f: unsafe extern "C" fn() = match self {
                    Mode::H => dav2d_ipred_h_8bpc_neon,
                    Mode::V => dav2d_ipred_v_8bpc_neon,
                    Mode::SmoothH => dav2d_ipred_smooth_h_8bpc_neon,
                    Mode::SmoothV => dav2d_ipred_smooth_v_8bpc_neon,
                };
                std::mem::transmute::<unsafe extern "C" fn(), AngularFn>(f)
            }
        }
        fn name(self) -> &'static str {
            match self {
                Mode::H => "ipred_h",
                Mode::V => "ipred_v",
                Mode::SmoothH => "ipred_smooth_h",
                Mode::SmoothV => "ipred_smooth_v",
            }
        }
    }

    #[inline]
    fn have_neon() -> bool {
        get_cpu_flags() & arm::CPU_FLAG_NEON != 0
    }
    fn disabled_kernels() -> &'static str {
        use std::sync::OnceLock;
        static OFF: OnceLock<String> = OnceLock::new();
        OFF.get_or_init(|| std::env::var("RAV2D_NEON_OFF").unwrap_or_default())
    }
    #[inline]
    fn kern_on(name: &str) -> bool {
        let off = disabled_kernels();
        if off.is_empty() {
            return true;
        }
        !off.split(',')
            .any(|k| k == name || k == "ipred" || k == "all")
    }

    /// The asm width dispatch covers the intra block widths 4,8,16,32,64. rav2d
    /// only ever predicts those widths for whole blocks, but guard anyway.
    #[inline]
    fn valid_w(w: usize) -> bool {
        matches!(w, 4 | 8 | 16 | 32 | 64)
    }

    /// Call a whole-block predictor kernel. `tl[o]` is the topleft sample;
    /// `tl[o+1+x]` is the top row and `tl[o-1-y]` the left column, matching the
    /// asm's `topleft` pointer convention exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn ipred_simple(
        mode: Mode,
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        w: usize,
        h: usize,
        angle: i32,
    ) -> bool {
        if !have_neon() || !kern_on(mode.name()) || !valid_w(w) {
            return false;
        }
        let f = mode.sym();
        unsafe {
            f(
                dst.as_mut_ptr(),
                stride as isize,
                tl.as_ptr().add(o),
                w as c_int,
                h as c_int,
                angle as c_int,
                0,
                0,
            );
        }
        true
    }
}

#[cfg(not(all(target_arch = "aarch64", rav2d_neon_ipred)))]
mod neon {
    #[derive(Clone, Copy)]
    pub enum Mode {
        H,
        V,
        SmoothH,
        SmoothV,
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ipred_simple(
        _: Mode,
        _: &mut [u8],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: usize,
        _: i32,
    ) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    //! Bit-exactness guard: each wired intra mode must produce identical output
    //! to the scalar reference across all whole-block sizes. On non-aarch64 the
    //! wrappers are scalar pass-throughs so the test is trivially true.

    /// Build a topleft edge buffer with `o` slack on both sides so `tl[o-1-y]`
    /// and `tl[o+1+x]` stay in bounds, plus the extra samples the smooth modes
    /// read (`tl[o+w+1]` right, `tl[o-h-1]` bottom).
    fn edge() -> (Vec<u8>, usize) {
        let o = 80usize;
        let mut tl = vec![0u8; o * 2 + 160];
        for (i, p) in tl.iter_mut().enumerate() {
            *p = ((i * 109 + 41) & 0xff) as u8;
        }
        (tl, o)
    }

    const SIZES: &[(usize, usize)] = &[
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (4, 8),
        (8, 4),
        (8, 16),
        (16, 8),
        (16, 32),
        (32, 16),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
        (4, 64),
        (64, 4),
    ];

    #[test]
    fn ipred_hv_matches_scalar() {
        crate::cpu::init_cpu();
        let (tl, o) = edge();
        for &(w, h) in SIZES {
            let stride = w + 8;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];

            crate::ipred::ipred_v(&mut a, stride, &tl, o, w, h, 0);
            super::ipred_v(&mut b, stride, &tl, o, w, h, 0);
            assert_eq!(a, b, "ipred_v {w}x{h}");

            a.iter_mut().for_each(|p| *p = 0);
            b.iter_mut().for_each(|p| *p = 0);
            crate::ipred::ipred_h(&mut a, stride, &tl, o, w, h, 0);
            super::ipred_h(&mut b, stride, &tl, o, w, h, 0);
            assert_eq!(a, b, "ipred_h {w}x{h}");
        }
    }

    #[test]
    fn ipred_smooth_hv_matches_scalar() {
        crate::cpu::init_cpu();
        let (tl, o) = edge();
        for &(w, h) in SIZES {
            let stride = w + 8;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];

            crate::ipred::ipred_smooth_v(&mut a, stride, &tl, o, w, h);
            super::ipred_smooth_v(&mut b, stride, &tl, o, w, h);
            assert_eq!(a, b, "ipred_smooth_v {w}x{h}");

            a.iter_mut().for_each(|p| *p = 0);
            b.iter_mut().for_each(|p| *p = 0);
            crate::ipred::ipred_smooth_h(&mut a, stride, &tl, o, w, h);
            super::ipred_smooth_h(&mut b, stride, &tl, o, w, h);
            assert_eq!(a, b, "ipred_smooth_h {w}x{h}");
        }
    }
}

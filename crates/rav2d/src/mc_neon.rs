//! NEON (aarch64) dispatch for the motion-compensation DSP family.
//!
//! dav2d ships hand-written NEON kernels for MC (`dav2d/src/arm/64/mc.S`,
//! `mc_dotprod.S`). Per the project's "assembly stays via FFI" rule, rav2d
//! assembles those `.S` into its own staticlib (see `build.rs`) and calls them
//! through the C ABI declared in `dav2d/src/arm/mc.h`. The functions here mirror
//! the scalar `crate::mc` signatures exactly, translating Rust slices into the
//! raw `(ptr, ptrdiff_t stride)` ABI the asm expects, then fall back to the
//! scalar Rust kernels when NEON is unavailable (non-aarch64, or the asm could
//! not be assembled) or for cases the asm kernels do not cover bit-exactly.
//!
//! Bit-exactness: the NEON kernels and rav2d's scalar kernels both implement the
//! AV1/AV2 spec MC (dav2d C is the reference rav2d was ported from), so they are
//! bit-identical given a matched ABI. The two cases NOT routed to NEON are the
//! `ext_warp` 4x4 tiles (`filter_type == -1`), which use a different filter
//! table (`EXT_WARP_FILTER`) than the asm's 8-tap `mc_subpel_filters`, and the
//! scaled MC kernels (not part of this family yet).

/// Length (in `i16` elements) to allocate for a `w`x`h` compound-prediction
/// scratch buffer that may be consumed by the NEON `avg`/`w_avg`/`mask`/`w_mask`
/// kernels. Those kernels read the packed `w*h` `tmp` buffer in fixed 16-`i16`
/// (32-byte) chunks (`ld1 {t0.8h,t1.8h}` per inner step), so when `w*h` is not a
/// multiple of 16 they over-read up to 15 elements past the logical end. dav2d's
/// own `compinter` buffers are a fixed 128*128 so this never matters there;
/// rav2d packs tightly, so callers round the allocation up with this helper. The
/// extra elements are read but never written to `dst`, so output is unaffected.
#[inline]
pub fn compound_tmp_len(w: usize, h: usize) -> usize {
    (w * h).next_multiple_of(16)
}

// On aarch64 with the asm assembled, route through NEON. Otherwise everything
// here is a thin pass-through to the scalar `crate::mc` kernels.
#[cfg(all(target_arch = "aarch64", rav2d_neon_mc))]
use neon::*;

#[cfg(all(target_arch = "aarch64", rav2d_neon_mc))]
mod neon {
    use crate::cpu::{arm, get_cpu_flags};
    use std::os::raw::c_int;

    // ABI from dav2d/src/arm/mc.h (8bpc: HIGHBD_DECL_SUFFIX is empty).
    // strides are ptrdiff_t (isize); pixel = u8.
    pub type McFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        w: c_int,
        h: c_int,
        mx: c_int,
        my: c_int,
    );
    pub type MctFn = unsafe extern "C" fn(
        tmp: *mut i16,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        w: c_int,
        h: c_int,
        mx: c_int,
        my: c_int,
    );
    pub type AvgFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        tmp1: *const i16,
        tmp2: *const i16,
        w: c_int,
        h: c_int,
    );
    pub type WAvgFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        tmp1: *const i16,
        tmp2: *const i16,
        w: c_int,
        h: c_int,
        weight: c_int,
    );
    pub type MaskFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        tmp1: *const i16,
        tmp2: *const i16,
        w: c_int,
        h: c_int,
        mask: *const u8,
    );
    pub type WMaskFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        tmp1: *const i16,
        tmp2: *const i16,
        w: c_int,
        h: c_int,
        mask: *mut u8,
        mask_stride: isize,
        sign: c_int,
    );
    pub type BlendFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        tmp: *const u8,
        w: c_int,
        h: c_int,
        mask: *const u8,
    );
    pub type Warp8x8Fn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        abcd: *const i16,
        mx: c_int,
        my: c_int,
    );
    pub type Warp8x8tFn = unsafe extern "C" fn(
        tmp: *mut i16,
        tmp_stride: isize,
        src: *const u8,
        src_stride: isize,
        abcd: *const i16,
        mx: c_int,
        my: c_int,
    );

    unsafe extern "C" {
        // 8-tap put (regular/smooth/sharp), plain NEON + dotprod + i8mm variants.
        pub fn dav2d_put_8tap_regular_8bpc_neon();
        pub fn dav2d_put_8tap_smooth_8bpc_neon();
        pub fn dav2d_put_8tap_sharp_8bpc_neon();
        pub fn dav2d_put_8tap_regular_8bpc_neon_dotprod();
        pub fn dav2d_put_8tap_smooth_8bpc_neon_dotprod();
        pub fn dav2d_put_8tap_sharp_8bpc_neon_dotprod();
        pub fn dav2d_put_8tap_regular_8bpc_neon_i8mm();
        pub fn dav2d_put_8tap_smooth_8bpc_neon_i8mm();
        pub fn dav2d_put_8tap_sharp_8bpc_neon_i8mm();
        // 8-tap prep (i16 intermediate).
        pub fn dav2d_prep_8tap_regular_8bpc_neon();
        pub fn dav2d_prep_8tap_smooth_8bpc_neon();
        pub fn dav2d_prep_8tap_sharp_8bpc_neon();
        pub fn dav2d_prep_8tap_regular_8bpc_neon_dotprod();
        pub fn dav2d_prep_8tap_smooth_8bpc_neon_dotprod();
        pub fn dav2d_prep_8tap_sharp_8bpc_neon_dotprod();
        pub fn dav2d_prep_8tap_regular_8bpc_neon_i8mm();
        pub fn dav2d_prep_8tap_smooth_8bpc_neon_i8mm();
        pub fn dav2d_prep_8tap_sharp_8bpc_neon_i8mm();
        // bilinear.
        pub fn dav2d_put_bilin_8bpc_neon();
        pub fn dav2d_prep_bilin_8bpc_neon();
        // compound blend kernels.
        pub fn dav2d_avg_8bpc_neon();
        pub fn dav2d_w_avg_8bpc_neon();
        pub fn dav2d_mask_8bpc_neon();
        pub fn dav2d_blend_8bpc_neon();
        pub fn dav2d_w_mask_444_8bpc_neon();
        pub fn dav2d_w_mask_422_8bpc_neon();
        pub fn dav2d_w_mask_420_8bpc_neon();
        // warp affine.
        pub fn dav2d_warp_affine_8x8_8bpc_neon();
        pub fn dav2d_warp_affine_8x8t_8bpc_neon();
    }

    #[inline]
    fn have_neon() -> bool {
        get_cpu_flags() & arm::CPU_FLAG_NEON != 0
    }
    // Debug bisect: RAV2D_NEON_OFF="put_8tap,avg,..." disables named kernels.
    // The env var is read exactly once and cached, so this stays off the hot
    // path; with the var unset (the normal case) it is a single atomic load.
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
        !off.split(',').any(|k| k == name || k == "all")
    }
    /// dav2d's NEON MC kernels dispatch block width through a `clz(w)`-indexed
    /// jump table whose entries cover only the spec MC widths 4,8,16,32,64 (the
    /// `put` paths additionally have a 2xN entry, the `prep`/compound paths do
    /// not, and NONE has a 128-wide entry — dav2d always tiles wider predictions
    /// into <=64 sub-blocks). rav2d sometimes calls MC with w=128 (whole-SB TIP)
    /// or non-power-of-two scratch (the OPFL 12x12 prefetch); both must use the
    /// scalar path. We conservatively require w in {4,8,16,32,64} for every
    /// kernel — w=2 (rare 2-wide chroma) also falls back to scalar, which is
    /// always bit-exact. Heights are processed by a plain loop counter and may
    /// be any value, so only the width is constrained.
    #[inline]
    fn valid_w(w: usize) -> bool {
        w.is_power_of_two() && (4..=64).contains(&w)
    }
    /// Compound blend kernels share the same width constraint as `valid_w`.
    #[inline]
    fn valid_compound_w(w: usize) -> bool {
        valid_w(w)
    }
    /// The compound kernels consume their packed input buffers (the two `i16`
    /// predictors, and for `mask`/`blend` the `u8` weight buffer) in fixed
    /// 16-element chunks, so each must have at least `ceil(w*h/16)*16` elements
    /// for the asm not to over-read. Returns false (→ scalar fallback) when any
    /// buffer is too short, which is always safe.
    #[inline]
    fn chunk_fits(len: usize, w: usize, h: usize) -> bool {
        len >= (w * h).next_multiple_of(16)
    }
    #[inline]
    fn have_dotprod() -> bool {
        get_cpu_flags() & arm::CPU_FLAG_DOTPROD != 0
    }
    #[inline]
    fn have_i8mm() -> bool {
        get_cpu_flags() & arm::CPU_FLAG_I8MM != 0
    }

    /// Resolve the best `put` 8-tap symbol for `filter_type` (0/1/2) honoring the
    /// available ISA extensions, matching dav2d's `mc_dsp_init_arm` escalation.
    fn put_8tap_sym(ft: i32) -> McFn {
        // i8mm > dotprod > neon, same selection order as dav2d.
        unsafe {
            let f: unsafe extern "C" fn() = if have_i8mm() {
                match ft {
                    0 => dav2d_put_8tap_regular_8bpc_neon_i8mm,
                    1 => dav2d_put_8tap_smooth_8bpc_neon_i8mm,
                    _ => dav2d_put_8tap_sharp_8bpc_neon_i8mm,
                }
            } else if have_dotprod() {
                match ft {
                    0 => dav2d_put_8tap_regular_8bpc_neon_dotprod,
                    1 => dav2d_put_8tap_smooth_8bpc_neon_dotprod,
                    _ => dav2d_put_8tap_sharp_8bpc_neon_dotprod,
                }
            } else {
                match ft {
                    0 => dav2d_put_8tap_regular_8bpc_neon,
                    1 => dav2d_put_8tap_smooth_8bpc_neon,
                    _ => dav2d_put_8tap_sharp_8bpc_neon,
                }
            };
            std::mem::transmute::<unsafe extern "C" fn(), McFn>(f)
        }
    }

    fn prep_8tap_sym(ft: i32) -> MctFn {
        unsafe {
            let f: unsafe extern "C" fn() = if have_i8mm() {
                match ft {
                    0 => dav2d_prep_8tap_regular_8bpc_neon_i8mm,
                    1 => dav2d_prep_8tap_smooth_8bpc_neon_i8mm,
                    _ => dav2d_prep_8tap_sharp_8bpc_neon_i8mm,
                }
            } else if have_dotprod() {
                match ft {
                    0 => dav2d_prep_8tap_regular_8bpc_neon_dotprod,
                    1 => dav2d_prep_8tap_smooth_8bpc_neon_dotprod,
                    _ => dav2d_prep_8tap_sharp_8bpc_neon_dotprod,
                }
            } else {
                match ft {
                    0 => dav2d_prep_8tap_regular_8bpc_neon,
                    1 => dav2d_prep_8tap_smooth_8bpc_neon,
                    _ => dav2d_prep_8tap_sharp_8bpc_neon,
                }
            };
            std::mem::transmute::<unsafe extern "C" fn(), MctFn>(f)
        }
    }

    macro_rules! as_fn {
        ($sym:expr, $ty:ty) => {
            std::mem::transmute::<unsafe extern "C" fn(), $ty>($sym)
        };
    }

    /// 8-tap or bilinear `put` MC. Returns true if NEON handled it; false means
    /// the caller must use the scalar path (filter_type == -1 ext-warp tiles).
    #[allow(clippy::too_many_arguments)]
    pub fn put_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_off: usize,
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        filter_type: i32,
    ) -> bool {
        if !have_neon() || filter_type < 0 || !kern_on("put_8tap") || !valid_w(w) {
            return false;
        }
        let f = put_8tap_sym(filter_type);
        unsafe {
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                src.as_ptr().add(src_off),
                src_stride as isize,
                w as c_int,
                h as c_int,
                mx,
                my,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prep_8tap(
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u8],
        src_off: usize,
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        filter_type: i32,
    ) -> bool {
        if !have_neon() || filter_type < 0 || !kern_on("prep_8tap") || !valid_w(w) {
            return false;
        }
        let f = prep_8tap_sym(filter_type);
        unsafe {
            f(
                tmp.as_mut_ptr(),
                tmp_stride as isize,
                src.as_ptr().add(src_off),
                src_stride as isize,
                w as c_int,
                h as c_int,
                mx,
                my,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn put_bilin(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
    ) -> bool {
        if !have_neon() || !kern_on("put_bilin") || !valid_w(w) {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_put_bilin_8bpc_neon, McFn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                src.as_ptr(),
                src_stride as isize,
                w as c_int,
                h as c_int,
                mx,
                my,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prep_bilin(
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
    ) -> bool {
        if !have_neon() || !kern_on("prep_bilin") || !valid_w(w) {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_prep_bilin_8bpc_neon, MctFn);
            f(
                tmp.as_mut_ptr(),
                tmp_stride as isize,
                src.as_ptr(),
                src_stride as isize,
                w as c_int,
                h as c_int,
                mx,
                my,
            );
        }
        true
    }

    pub fn avg(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
    ) -> bool {
        if !have_neon()
            || !kern_on("avg")
            || !valid_compound_w(w)
            || !chunk_fits(tmp1.len(), w, h)
            || !chunk_fits(tmp2.len(), w, h)
        {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_avg_8bpc_neon, AvgFn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                tmp1.as_ptr(),
                tmp2.as_ptr(),
                w as c_int,
                h as c_int,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn w_avg(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
        weight: i32,
    ) -> bool {
        if !have_neon()
            || !kern_on("w_avg")
            || !valid_compound_w(w)
            || !chunk_fits(tmp1.len(), w, h)
            || !chunk_fits(tmp2.len(), w, h)
        {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_w_avg_8bpc_neon, WAvgFn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                tmp1.as_ptr(),
                tmp2.as_ptr(),
                w as c_int,
                h as c_int,
                weight,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mask(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
        m: &[u8],
    ) -> bool {
        if !have_neon()
            || !kern_on("mask")
            || !valid_compound_w(w)
            || !chunk_fits(tmp1.len(), w, h)
            || !chunk_fits(tmp2.len(), w, h)
            || !chunk_fits(m.len(), w, h)
        {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_mask_8bpc_neon, MaskFn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                tmp1.as_ptr(),
                tmp2.as_ptr(),
                w as c_int,
                h as c_int,
                m.as_ptr(),
            );
        }
        true
    }

    pub fn blend(
        dst: &mut [u8],
        dst_stride: usize,
        tmp: &[u8],
        w: usize,
        h: usize,
        m: &[u8],
    ) -> bool {
        if !have_neon()
            || !kern_on("blend")
            || !valid_compound_w(w)
            || !chunk_fits(tmp.len(), w, h)
            || !chunk_fits(m.len(), w, h)
        {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_blend_8bpc_neon, BlendFn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                tmp.as_ptr(),
                w as c_int,
                h as c_int,
                m.as_ptr(),
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn w_mask(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
        mask: &mut [u8],
        mask_stride: usize,
        sign: i32,
        ss_hor: bool,
        ss_ver: bool,
    ) -> bool {
        if !have_neon()
            || !kern_on("w_mask")
            || !valid_compound_w(w)
            || !chunk_fits(tmp1.len(), w, h)
            || !chunk_fits(tmp2.len(), w, h)
        {
            return false;
        }
        // dav2d picks the kernel by chroma subsampling of the *mask*:
        // 444 (no ss), 422 (h ss only), 420 (h+v ss).
        let sym: unsafe extern "C" fn() = match (ss_hor, ss_ver) {
            (false, false) => dav2d_w_mask_444_8bpc_neon,
            (true, false) => dav2d_w_mask_422_8bpc_neon,
            (true, true) => dav2d_w_mask_420_8bpc_neon,
            // (false, true) does not occur in AV2 chroma layouts; fall back.
            (false, true) => return false,
        };
        unsafe {
            let f = as_fn!(sym, WMaskFn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                tmp1.as_ptr(),
                tmp2.as_ptr(),
                w as c_int,
                h as c_int,
                mask.as_mut_ptr(),
                mask_stride as isize,
                sign,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn warp_affine_8x8(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        src_off: usize,
        abcd: &[i16; 4],
        mx: i32,
        my: i32,
    ) -> bool {
        if !have_neon() || !kern_on("warp_affine") {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_warp_affine_8x8_8bpc_neon, Warp8x8Fn);
            f(
                dst.as_mut_ptr(),
                dst_stride as isize,
                src.as_ptr().add(src_off),
                src_stride as isize,
                abcd.as_ptr(),
                mx,
                my,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn warp_affine_8x8t(
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u8],
        src_stride: usize,
        src_off: usize,
        abcd: &[i16; 4],
        mx: i32,
        my: i32,
    ) -> bool {
        if !have_neon() || !kern_on("warp_affinet") {
            return false;
        }
        unsafe {
            let f = as_fn!(dav2d_warp_affine_8x8t_8bpc_neon, Warp8x8tFn);
            f(
                tmp.as_mut_ptr(),
                tmp_stride as isize,
                src.as_ptr().add(src_off),
                src_stride as isize,
                abcd.as_ptr(),
                mx,
                my,
            );
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Public dispatch surface. These have the SAME signatures as the matching
// `crate::mc` functions; each tries NEON first (aarch64) and falls back to the
// scalar Rust kernel. Call sites use these instead of `crate::mc::*`.
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "aarch64", rav2d_neon_mc)))]
#[allow(clippy::too_many_arguments)]
mod neon {
    // Stubs so the dispatch wrappers compile on non-aarch64; always "not handled".
    pub fn put_8tap(
        _: &mut [u8],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: i32,
        _: i32,
        _: i32,
    ) -> bool {
        false
    }
    pub fn prep_8tap(
        _: &mut [i16],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: i32,
        _: i32,
        _: i32,
    ) -> bool {
        false
    }
    pub fn put_bilin(
        _: &mut [u8],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: usize,
        _: i32,
        _: i32,
    ) -> bool {
        false
    }
    pub fn prep_bilin(
        _: &mut [i16],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: usize,
        _: i32,
        _: i32,
    ) -> bool {
        false
    }
    pub fn avg(_: &mut [u8], _: usize, _: &[i16], _: &[i16], _: usize, _: usize) -> bool {
        false
    }
    pub fn w_avg(_: &mut [u8], _: usize, _: &[i16], _: &[i16], _: usize, _: usize, _: i32) -> bool {
        false
    }
    pub fn mask(
        _: &mut [u8],
        _: usize,
        _: &[i16],
        _: &[i16],
        _: usize,
        _: usize,
        _: &[u8],
    ) -> bool {
        false
    }
    pub fn blend(_: &mut [u8], _: usize, _: &[u8], _: usize, _: usize, _: &[u8]) -> bool {
        false
    }
    pub fn w_mask(
        _: &mut [u8],
        _: usize,
        _: &[i16],
        _: &[i16],
        _: usize,
        _: usize,
        _: &mut [u8],
        _: usize,
        _: i32,
        _: bool,
        _: bool,
    ) -> bool {
        false
    }
    pub fn warp_affine_8x8(
        _: &mut [u8],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: &[i16; 4],
        _: i32,
        _: i32,
    ) -> bool {
        false
    }
    pub fn warp_affine_8x8t(
        _: &mut [i16],
        _: usize,
        _: &[u8],
        _: usize,
        _: usize,
        _: &[i16; 4],
        _: i32,
        _: i32,
    ) -> bool {
        false
    }
}

#[cfg(not(all(target_arch = "aarch64", rav2d_neon_mc)))]
use neon::*;

#[allow(clippy::too_many_arguments)]
pub fn put_8tap_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    if !put_8tap(
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    ) {
        crate::mc::put_8tap_8bpc(
            dst,
            dst_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter_type,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prep_8tap_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    if !prep_8tap(
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    ) {
        crate::mc::prep_8tap_8bpc(
            tmp,
            tmp_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter_type,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn put_bilin_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    if !put_bilin(dst, dst_stride, src, src_stride, w, h, mx, my) {
        crate::mc::put_bilin_8bpc(dst, dst_stride, src, src_stride, w, h, mx, my);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prep_bilin_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    if !prep_bilin(tmp, tmp_stride, src, src_stride, w, h, mx, my) {
        crate::mc::prep_bilin_8bpc(tmp, tmp_stride, src, src_stride, w, h, mx, my);
    }
}

pub fn avg_8bpc(dst: &mut [u8], dst_stride: usize, tmp1: &[i16], tmp2: &[i16], w: usize, h: usize) {
    if !avg(dst, dst_stride, tmp1, tmp2, w, h) {
        crate::mc::avg_8bpc(dst, dst_stride, tmp1, tmp2, w, h);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn w_avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    if !w_avg(dst, dst_stride, tmp1, tmp2, w, h, weight) {
        crate::mc::w_avg_8bpc(dst, dst_stride, tmp1, tmp2, w, h, weight);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &[u8],
) {
    if !mask(dst, dst_stride, tmp1, tmp2, w, h, m) {
        crate::mc::mask_8bpc(dst, dst_stride, tmp1, tmp2, w, h, m);
    }
}

pub fn blend_8bpc(dst: &mut [u8], dst_stride: usize, tmp: &[u8], w: usize, h: usize, m: &[u8]) {
    if !blend(dst, dst_stride, tmp, w, h, m) {
        crate::mc::blend_8bpc(dst, dst_stride, tmp, w, h, m);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn w_mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    if !w_mask(
        dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        m,
        mask_stride,
        sign,
        ss_hor,
        ss_ver,
    ) {
        crate::mc::w_mask_8bpc(
            dst,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            m,
            mask_stride,
            sign,
            ss_hor,
            ss_ver,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn warp_affine_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    if !warp_affine_8x8(dst, dst_stride, src, src_stride, src_off, abcd, mx, my) {
        crate::mc::warp_affine_8x8_8bpc(dst, dst_stride, src, src_stride, src_off, abcd, mx, my);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn warp_affine_8x8t_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    if !warp_affine_8x8t(tmp, tmp_stride, src, src_stride, src_off, abcd, mx, my) {
        crate::mc::warp_affine_8x8t_8bpc(tmp, tmp_stride, src, src_stride, src_off, abcd, mx, my);
    }
}

#[cfg(test)]
mod tests {
    //! Bit-exactness guard: the NEON dispatch must produce identical output to
    //! the scalar reference for every width/height/subpel the asm handles. On
    //! aarch64 the dispatch wrappers take the NEON path for valid widths; on
    //! other arches they are scalar pass-throughs (so the test is trivially true
    //! but still compiles).
    fn ref_src() -> (Vec<u8>, usize, usize) {
        let stride = 192usize;
        let mut src = vec![0u8; stride * 192];
        for (i, p) in src.iter_mut().enumerate() {
            *p = ((i * 131 + 17) & 0xff) as u8;
        }
        // origin with 8 rows/cols of slack on every side for 8-tap reach.
        (src, 8 * stride + 8, stride)
    }

    #[test]
    fn put_8tap_matches_scalar() {
        crate::cpu::init_cpu();
        let (src, off, sstride) = ref_src();
        for &w in &[4usize, 8, 16, 32, 64] {
            for &h in &[4usize, 8, 16] {
                for ft in 0..3i32 {
                    for &(mx, my) in &[(0, 0), (5, 0), (0, 9), (3, 11), (15, 15)] {
                        let dstride = w;
                        let mut a = vec![0u8; dstride * h];
                        let mut b = vec![0u8; dstride * h];
                        crate::mc::put_8tap_8bpc(
                            &mut a, dstride, &src, off, sstride, w, h, mx, my, ft,
                        );
                        super::put_8tap_8bpc(&mut b, dstride, &src, off, sstride, w, h, mx, my, ft);
                        assert_eq!(a, b, "put_8tap w={w} h={h} ft={ft} mx={mx} my={my}");
                    }
                }
            }
        }
    }

    #[test]
    fn prep_8tap_matches_scalar() {
        crate::cpu::init_cpu();
        let (src, off, sstride) = ref_src();
        for &w in &[4usize, 8, 16, 32, 64] {
            for &h in &[4usize, 8, 16] {
                for ft in 0..3i32 {
                    for &(mx, my) in &[(0, 0), (5, 0), (0, 9), (3, 11), (15, 15)] {
                        let mut a = vec![0i16; w * h];
                        let mut b = vec![0i16; w * h];
                        crate::mc::prep_8tap_8bpc(&mut a, w, &src, off, sstride, w, h, mx, my, ft);
                        super::prep_8tap_8bpc(&mut b, w, &src, off, sstride, w, h, mx, my, ft);
                        assert_eq!(a, b, "prep_8tap w={w} h={h} ft={ft} mx={mx} my={my}");
                    }
                }
            }
        }
    }

    #[test]
    fn bilin_matches_scalar() {
        crate::cpu::init_cpu();
        let (src, off, sstride) = ref_src();
        for &w in &[4usize, 8, 16, 32, 64] {
            for &h in &[4usize, 8, 16] {
                for &(mx, my) in &[(0, 0), (5, 0), (0, 9), (7, 13), (15, 1)] {
                    let mut pa = vec![0u8; w * h];
                    let mut pb = vec![0u8; w * h];
                    crate::mc::put_bilin_8bpc(&mut pa, w, &src[off..], sstride, w, h, mx, my);
                    super::put_bilin_8bpc(&mut pb, w, &src[off..], sstride, w, h, mx, my);
                    assert_eq!(pa, pb, "put_bilin w={w} h={h} mx={mx} my={my}");

                    let mut ta = vec![0i16; w * h];
                    let mut tb = vec![0i16; w * h];
                    crate::mc::prep_bilin_8bpc(&mut ta, w, &src[off..], sstride, w, h, mx, my);
                    super::prep_bilin_8bpc(&mut tb, w, &src[off..], sstride, w, h, mx, my);
                    assert_eq!(ta, tb, "prep_bilin w={w} h={h} mx={mx} my={my}");
                }
            }
        }
    }

    #[test]
    fn compound_matches_scalar() {
        crate::cpu::init_cpu();
        let mut t1 = vec![0i16; 64 * 64];
        let mut t2 = vec![0i16; 64 * 64];
        for i in 0..t1.len() {
            t1[i] = ((i as i32 * 7 - 1000) % 4096 - 2048) as i16;
            t2[i] = ((i as i32 * 13 + 500) % 4096 - 2048) as i16;
        }
        for &w in &[4usize, 8, 16, 32, 64] {
            for &h in &[4usize, 8, 16] {
                let len = super::compound_tmp_len(w, h);
                let a1 = &t1[..len];
                let a2 = &t2[..len];
                // avg
                let mut da = vec![0u8; w * h];
                let mut db = vec![0u8; w * h];
                crate::mc::avg_8bpc(&mut da, w, a1, a2, w, h);
                super::avg_8bpc(&mut db, w, a1, a2, w, h);
                assert_eq!(da, db, "avg w={w} h={h}");
                // w_avg
                for weight in [1, 8, 15] {
                    let mut wa = vec![0u8; w * h];
                    let mut wb = vec![0u8; w * h];
                    crate::mc::w_avg_8bpc(&mut wa, w, a1, a2, w, h, weight);
                    super::w_avg_8bpc(&mut wb, w, a1, a2, w, h, weight);
                    assert_eq!(wa, wb, "w_avg w={w} h={h} weight={weight}");
                }
                // mask
                let mut msk = vec![0u8; len];
                for (i, m) in msk.iter_mut().enumerate() {
                    *m = ((i * 5 + 3) % 65) as u8;
                }
                let mut ma = vec![0u8; w * h];
                let mut mb = vec![0u8; w * h];
                crate::mc::mask_8bpc(&mut ma, w, a1, a2, w, h, &msk);
                super::mask_8bpc(&mut mb, w, a1, a2, w, h, &msk);
                assert_eq!(ma, mb, "mask w={w} h={h}");
            }
        }
    }

    #[test]
    fn warp_matches_scalar() {
        crate::cpu::init_cpu();
        let (src, off, sstride) = ref_src();
        let abcd: [i16; 4] = [120, -8, 16, 132];
        for &(mx, my) in &[(0, 0), (64, -128), (-256, 320), (512, 512)] {
            let mut a = vec![0u8; 8 * 8];
            let mut b = vec![0u8; 8 * 8];
            crate::mc::warp_affine_8x8_8bpc(&mut a, 8, &src, sstride, off, &abcd, mx, my);
            super::warp_affine_8x8_8bpc(&mut b, 8, &src, sstride, off, &abcd, mx, my);
            assert_eq!(a, b, "warp8x8 mx={mx} my={my}");

            let mut ta = vec![0i16; 8 * 8];
            let mut tb = vec![0i16; 8 * 8];
            crate::mc::warp_affine_8x8t_8bpc(&mut ta, 8, &src, sstride, off, &abcd, mx, my);
            super::warp_affine_8x8t_8bpc(&mut tb, 8, &src, sstride, off, &abcd, mx, my);
            assert_eq!(ta, tb, "warp8x8t mx={mx} my={my}");
        }
    }
}

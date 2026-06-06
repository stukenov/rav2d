//! Portable SIMD kernels (via the `wide` crate) for the hot scalar inner loops.
//!
//! `wide` compiles one source to NEON (aarch64) / AVX2 (x86_64) / SSE / a scalar
//! fallback, on **stable** Rust with a **safe** API (no `unsafe`, no nightly).
//! Every kernel here is bit-exact with its scalar reference — the unit tests in
//! this module assert SIMD output == an independent scalar reimplementation over
//! randomized inputs, and the end-to-end conformance oracle (`tests/conformance.rs`,
//! `bit_exact_*` sweeps vs dav2d) is the final net.
//!
//! Bit-exactness relies on `wide::i32x8`'s `>>` being an **arithmetic**
//! (sign-propagating) shift; `shr_is_arithmetic` below is the pre-flight gate.

use wide::{CmpLt, i32x8};

use crate::pixel::{BitDepth, Pixel};

/// Load 8 consecutive `i16` (sign-extended) into an `i32x8`.
#[inline(always)]
fn load8_i16(s: &[i16]) -> i32x8 {
    i32x8::from([
        s[0] as i32,
        s[1] as i32,
        s[2] as i32,
        s[3] as i32,
        s[4] as i32,
        s[5] as i32,
        s[6] as i32,
        s[7] as i32,
    ])
}

/// Load 8 consecutive `u8` (zero-extended) into an `i32x8`.
#[inline(always)]
fn load8_u8(s: &[u8]) -> i32x8 {
    i32x8::from([
        s[0] as i32,
        s[1] as i32,
        s[2] as i32,
        s[3] as i32,
        s[4] as i32,
        s[5] as i32,
        s[6] as i32,
        s[7] as i32,
    ])
}

/// Load 8 consecutive `i32` into an `i32x8`.
#[inline(always)]
fn load8_i32(s: &[i32]) -> i32x8 {
    i32x8::from([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// Store an `i32x8` to 8 consecutive `i32`.
#[inline(always)]
fn store8_i32(dst: &mut [i32], v: i32x8) {
    let a = v.to_array();
    dst[..8].copy_from_slice(&a);
}

/// Load 8 consecutive pixels (`u8`/`u16`, zero-extended) into an `i32x8`.
#[inline(always)]
fn load8_pix<P: Pixel>(s: &[P]) -> i32x8 {
    i32x8::from([
        s[0].into(),
        s[1].into(),
        s[2].into(),
        s[3].into(),
        s[4].into(),
        s[5].into(),
        s[6].into(),
        s[7].into(),
    ])
}

/// Store an `i32x8`, clamped to `[0, bitdepth_max]` then narrowed — reproduces
/// [`BitDepth::pixel_clip`] for 8 lanes.
#[inline(always)]
fn store8_clip<BD: BitDepth>(bd: BD, dst: &mut [BD::Pixel], v: i32x8) {
    let c = v.max(i32x8::splat(0)).min(i32x8::splat(bd.bitdepth_max()));
    let a = c.to_array();
    for k in 0..8 {
        dst[k] = BD::Pixel::from_i32(a[k]);
    }
}

/// Store an `i32x8`, truncated with no clamp — reproduces [`Pixel::from_i32`]
/// (used by `blend`, whose scalar path does not clamp).
#[inline(always)]
fn store8_trunc<P: Pixel>(dst: &mut [P], v: i32x8) {
    let a = v.to_array();
    for k in 0..8 {
        dst[k] = P::from_i32(a[k]);
    }
}

/// Arithmetic right shift of an `i32x8` by a uniform amount, matching scalar `>>`
/// on `i32`. (`wide`'s `Shr` on a signed vector is sign-propagating.)
#[inline(always)]
fn sra(v: i32x8, sh: i32) -> i32x8 {
    v >> i32x8::splat(sh)
}

/// `avg` row: `dst[x] = clip((tmp1[x] + tmp2[x] + rnd) >> sh)` for `x in 0..n`.
#[inline]
pub fn avg_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = i32x8::splat(rnd);
    let mut x = 0;
    while x + 8 <= n {
        let r = sra(load8_i16(&tmp1[x..]) + load8_i16(&tmp2[x..]) + rnd_v, sh);
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        dst[x] = bd.pixel_clip((tmp1[x] as i32 + tmp2[x] as i32 + rnd) >> sh);
        x += 1;
    }
}

/// `w_avg` row: `dst[x] = clip((tmp1[x]*weight + tmp2[x]*(16-weight) + rnd) >> sh)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn w_avg_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    let w1 = i32x8::splat(weight);
    let w2 = i32x8::splat(16 - weight);
    let rnd_v = i32x8::splat(rnd);
    let mut x = 0;
    while x + 8 <= n {
        let r = sra(
            load8_i16(&tmp1[x..]) * w1 + load8_i16(&tmp2[x..]) * w2 + rnd_v,
            sh,
        );
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        dst[x] =
            bd.pixel_clip((tmp1[x] as i32 * weight + tmp2[x] as i32 * (16 - weight) + rnd) >> sh);
        x += 1;
    }
}

/// `mask` row: `dst[x] = clip((tmp1[x]*m + tmp2[x]*(64-m) + rnd) >> sh)`,
/// `m = mask[x]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn mask_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = i32x8::splat(rnd);
    let c64 = i32x8::splat(64);
    let mut x = 0;
    while x + 8 <= n {
        let m = load8_u8(&mask[x..]);
        let r = sra(
            load8_i16(&tmp1[x..]) * m + load8_i16(&tmp2[x..]) * (c64 - m) + rnd_v,
            sh,
        );
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        dst[x] = bd.pixel_clip((tmp1[x] as i32 * m + tmp2[x] as i32 * (64 - m) + rnd) >> sh);
        x += 1;
    }
}

/// `blend` row: `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6` (truncate, no clamp),
/// `m = mask[x]`.
#[inline]
pub fn blend_row<P: Pixel>(dst: &mut [P], tmp: &[P], mask: &[u8], n: usize) {
    let c64 = i32x8::splat(64);
    let rnd_v = i32x8::splat(32);
    let mut x = 0;
    while x + 8 <= n {
        let m = load8_u8(&mask[x..]);
        let d = load8_pix(&dst[x..]);
        let t = load8_pix(&tmp[x..]);
        let r = sra(d * (c64 - m) + t * m + rnd_v, 6);
        store8_trunc(&mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        let d: i32 = dst[x].into();
        let t: i32 = tmp[x].into();
        dst[x] = P::from_i32((d * (64 - m) + t * m + 32) >> 6);
        x += 1;
    }
}

/// `morph` row: `dst[x] = clip((alpha*dst[x] + beta) >> 8)`.
#[inline]
pub fn morph_row<BD: BitDepth>(bd: BD, dst: &mut [BD::Pixel], alpha: i32, beta: i32, n: usize) {
    let a_v = i32x8::splat(alpha);
    let b_v = i32x8::splat(beta);
    let mut x = 0;
    while x + 8 <= n {
        let r = sra(load8_pix(&dst[x..]) * a_v + b_v, 8);
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let d: i32 = dst[x].into();
        dst[x] = bd.pixel_clip((alpha * d + beta) >> 8);
        x += 1;
    }
}

/// itx DC-only row: `dst[x] = clip(dst[x] + dc)` for `x in 0..n`.
#[inline]
pub fn dc_add_row<BD: BitDepth>(bd: BD, dst: &mut [BD::Pixel], dc: i32, n: usize) {
    let dc_v = i32x8::splat(dc);
    let mut x = 0;
    while x + 8 <= n {
        let r = load8_pix(&dst[x..]) + dc_v;
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let p: i32 = dst[x].into();
        dst[x] = bd.pixel_clip(p + dc);
        x += 1;
    }
}

/// itx row-clip pass: `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)` in place.
#[inline]
pub fn row_clip(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = i32x8::splat(rnd);
    let min_v = i32x8::splat(min);
    let max_v = i32x8::splat(max);
    let mut i = 0;
    while i + 8 <= n {
        let v = sra(load8_i32(&tmp[i..]) + rnd_v, shift)
            .max(min_v)
            .min(max_v);
        store8_i32(&mut tmp[i..], v);
        i += 8;
    }
    while i < n {
        tmp[i] = (((tmp[i] + rnd) >> shift).max(min)).min(max);
        i += 1;
    }
}

/// itx plain residual-add row (dpcm_flag 0): `dst[x] = clip(dst[x] + ((c[x]+rnd)>>shift))`.
#[inline]
pub fn residual_add_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = i32x8::splat(rnd);
    let mut x = 0;
    while x + 8 <= n {
        let cf = sra(load8_i32(&c[x..]) + rnd_v, shift);
        let r = load8_pix(&dst[x..]) + cf;
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let p: i32 = dst[x].into();
        dst[x] = bd.pixel_clip(p + ((c[x] + rnd) >> shift));
        x += 1;
    }
}

/// `cctx` row: cross-component-transform rotate + clip over two i32 planes.
/// `u'[i] = iclip((u*cosa - v*sina + 128 - (a<0)) >> 8, min, max)`,
/// `v'[i] = iclip((u*sina + v*cosa + 128 - (b<0)) >> 8, min, max)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn cctx_row(u: &mut [i32], v: &mut [i32], sina: i32, cosa: i32, sz: usize, min: i32, max: i32) {
    let sina_v = i32x8::splat(sina);
    let cosa_v = i32x8::splat(cosa);
    let c128 = i32x8::splat(128);
    let zero = i32x8::splat(0);
    let min_v = i32x8::splat(min);
    let max_v = i32x8::splat(max);
    let mut i = 0;
    while i + 8 <= sz {
        let uu = load8_i32(&u[i..]);
        let vv = load8_i32(&v[i..]);
        let a = uu * cosa_v - vv * sina_v;
        let b = uu * sina_v + vv * cosa_v;
        // `a.cmp_lt(0)` yields -1 lanes where a<0, i.e. `+ (-1)` == `- (a<0)`.
        let ra = sra(a + c128 + a.cmp_lt(zero), 8).max(min_v).min(max_v);
        let rb = sra(b + c128 + b.cmp_lt(zero), 8).max(min_v).min(max_v);
        store8_i32(&mut u[i..], ra);
        store8_i32(&mut v[i..], rb);
        i += 8;
    }
    while i < sz {
        let a = u[i] * cosa - v[i] * sina;
        let b = u[i] * sina + v[i] * cosa;
        u[i] = (((a + 128 - (a < 0) as i32) >> 8).max(min)).min(max);
        v[i] = (((b + 128 - (b < 0) as i32) >> 8).max(min)).min(max);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Loop-restoration FIR kernels.
//
// Each loop-restoration filter computes, per output pixel `x`, a separable-ish
// FIR over a small set of taps. Every tap reads ONE byte from a (per-tap) row
// buffer at a (per-tap) signed column offset. Across a run of consecutive `x`
// the access for a fixed tap is a *contiguous* 8-wide load from that row, so
// the MAC vectorizes cleanly — provided the whole run uses the SAME filter
// coefficients and is not skipped (the caller guarantees this by only batching
// runs of uniform class / no-skip).
//
// Rounding is a PLAIN arithmetic round `(s + 64) >> 7` (NOT CDEF's asymmetric
// `-(sum<0)` correction). Final clip is `[0, bitdepth_max]`.
// ---------------------------------------------------------------------------

/// One symmetric FIR tap: `a` is read from `row_p` at `+dx`, `b` from `row_m`
/// at `-dx` (relative to the per-pixel column `o + x`); both rows are u8 buffers
/// and `dx` is added to the run's base column offset.
pub(crate) struct WienerTap<'a> {
    pub row_p: &'a [u8],
    pub row_m: &'a [u8],
    pub dx: i32,
    pub coef: i32,
}

/// User ("NS") Wiener FIR over a run of `n` consecutive pixels.
///
/// `dst` is the destination row (already offset to the run's first pixel).
/// `center` is the center row buffer; `col0 = o + x0` is the column of the
/// run's first pixel inside every row buffer. Each tap contributes
/// `(a + b - 2*m) * coef`; accumulator starts at `m << 7`; output is
/// `clip((s + 64) >> 7, 0, 255)`.
#[inline]
pub(crate) fn ns_wiener_fir_run(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    let rnd = i32x8::splat(64);
    let mut x = 0;
    while x + 8 <= n {
        let c = col0 + x;
        let m = load8_u8(&center[c..]);
        let mut s = m << i32x8::splat(7);
        let two_m = m + m;
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            let a = load8_u8(&t.row_p[cp..]);
            let b = load8_u8(&t.row_m[cm..]);
            s += (a + b - two_m) * i32x8::splat(t.coef);
        }
        let v = sra(s + rnd, 7).max(i32x8::splat(0)).min(i32x8::splat(255));
        let arr = v.to_array();
        for k in 0..8 {
            dst[x + k] = arr[k] as u8;
        }
        x += 8;
    }
    while x < n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m << 7;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b - 2 * m) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
        x += 1;
    }
}

/// Pretrained ("PC") Wiener FIR over a run of `n` consecutive pixels.
///
/// Identical access pattern to [`ns_wiener_fir_run`] but each tap contributes
/// `coef * (a + b)` and the accumulator starts at `m * center_coef`.
#[inline]
pub(crate) fn pc_wiener_fir_run(
    dst: &mut [u8],
    center: &[u8],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    let rnd = i32x8::splat(64);
    let cc = i32x8::splat(center_coef);
    let mut x = 0;
    while x + 8 <= n {
        let c = col0 + x;
        let m = load8_u8(&center[c..]);
        let mut s = m * cc;
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            let a = load8_u8(&t.row_p[cp..]);
            let b = load8_u8(&t.row_m[cm..]);
            s += (a + b) * i32x8::splat(t.coef);
        }
        let v = sra(s + rnd, 7).max(i32x8::splat(0)).min(i32x8::splat(255));
        let arr = v.to_array();
        for k in 0..8 {
            dst[x + k] = arr[k] as u8;
        }
        x += 8;
    }
    while x < n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m * center_coef;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
        x += 1;
    }
}

/// GDF residual add over a run of `n` consecutive pixels in one 4x4 row.
/// `dst` is offset to the run's first pixel; `err` is offset likewise.
/// `dst[x] = clip(dst[x] + apply_sign((|err[x]*scale| + rnd) >> shift, err[x]*scale))`
/// with `shift = 4`, `rnd = 8`. `apply_sign(v, s)` returns `-v` if `s < 0`.
#[inline]
pub(crate) fn gdf_add_run(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let rnd = i32x8::splat(8);
    let sc = i32x8::splat(scale);
    let zero = i32x8::splat(0);
    let mut x = 0;
    while x + 8 <= n {
        // load 8 i8 (sign-extended)
        let diff = i32x8::from([
            err[x] as i32,
            err[x + 1] as i32,
            err[x + 2] as i32,
            err[x + 3] as i32,
            err[x + 4] as i32,
            err[x + 5] as i32,
            err[x + 6] as i32,
            err[x + 7] as i32,
        ]) * sc;
        let mag = sra(diff.abs() + rnd, 4);
        // apply_sign: negate where diff < 0. cmp_lt yields the all-ones mask
        // where diff<0; blend(mask, t, f) picks `t` there, `f` elsewhere.
        let neg = diff.cmp_lt(zero);
        let adj = neg.blend(zero - mag, mag);
        let d = load8_u8(&dst[x..]) + adj;
        let v = d.max(zero).min(i32x8::splat(255));
        let arr = v.to_array();
        for k in 0..8 {
            dst[x + k] = arr[k] as u8;
        }
        x += 8;
    }
    while x < n {
        let diff = err[x] as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let adj = if diff < 0 { -mag } else { mag };
        dst[x] = (dst[x] as i32 + adj).clamp(0, 255) as u8;
        x += 1;
    }
}

/// GDF gradient: for one direction, accumulate the per-column gradient
/// `|b*2 - a - c|` over 2 `y` rows into 8 lanes (one per input column), then
/// pair-reduce adjacent lanes into 4 output gradients.
///
/// For lane `j` (column `col0 + j`): `b = center_rows[y][col0+j-1]`,
/// `a = a_rows[y][col0+j-1-dx]`, `c = c_rows[y][col0+j-1+dx]` (all `>> shift`),
/// where `(a_rows[y], c_rows[y])` are the up/down rows for direction `d`.
/// Output `out[k] = lane(2k) + lane(2k+1)` summed over both `y`. Writes
/// `dst[k][d]` for k in 0..ncells.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn gdf_gradient_group(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    center_rows: [&[u8]; 2],
    a_rows: [&[u8]; 2],
    c_rows: [&[u8]; 2],
    col0: usize,
    dx: i32,
    shift: u32,
) {
    let sh = i32x8::splat(shift as i32);
    let mut acc = i32x8::splat(0);
    for y in 0..2 {
        let bcol = col0 - 1;
        let b = load8_u8(&center_rows[y][bcol..]) >> sh;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let a = load8_u8(&a_rows[y][acol..]) >> sh;
        let c = load8_u8(&c_rows[y][ccol..]) >> sh;
        acc += (b + b - a - c).abs();
    }
    let arr = acc.to_array();
    for k in 0..ncells {
        dst[base_cell + k][d] = (arr[2 * k] + arr[2 * k + 1]) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{BitDepth8, BitDepth16};

    /// Pre-flight gate for kernels that shift signed sums: `wide::i32x8 >>` MUST
    /// be arithmetic (sign-propagating), like scalar `i32 >>`.
    #[test]
    fn shr_is_arithmetic() {
        let v = i32x8::splat(-8);
        assert_eq!(
            sra(v, 1).to_array(),
            [-4i32; 8],
            "i32x8 >> must be arithmetic"
        );
        let v = i32x8::from([-1, -2, -3, -16, 15, 256, -257, 1024]);
        let got = sra(v, 4).to_array();
        let want = [-1, -1, -1, -1, 0, 16, -17, 64];
        assert_eq!(got, want, "i32x8 >> 4 must match scalar arithmetic shift");
    }

    /// Small deterministic PRNG (no external dep).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn i16(&mut self) -> i16 {
            self.next() as i16
        }
        fn u8(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
    }

    // --- independent scalar references (copies of the scalar kernel math) ---

    fn ref_avg<BD: BitDepth>(bd: BD, t1: &[i16], t2: &[i16], rnd: i32, sh: i32) -> Vec<BD::Pixel> {
        (0..t1.len())
            .map(|i| bd.pixel_clip((t1[i] as i32 + t2[i] as i32 + rnd) >> sh))
            .collect()
    }
    fn ref_wavg<BD: BitDepth>(
        bd: BD,
        t1: &[i16],
        t2: &[i16],
        wt: i32,
        rnd: i32,
        sh: i32,
    ) -> Vec<BD::Pixel> {
        (0..t1.len())
            .map(|i| bd.pixel_clip((t1[i] as i32 * wt + t2[i] as i32 * (16 - wt) + rnd) >> sh))
            .collect()
    }
    fn ref_mask<BD: BitDepth>(
        bd: BD,
        t1: &[i16],
        t2: &[i16],
        m: &[u8],
        rnd: i32,
        sh: i32,
    ) -> Vec<BD::Pixel> {
        (0..t1.len())
            .map(|i| {
                let mm = m[i] as i32;
                bd.pixel_clip((t1[i] as i32 * mm + t2[i] as i32 * (64 - mm) + rnd) >> sh)
            })
            .collect()
    }

    /// Run a kernel over every length 0..=40 (covers the 8-wide body + every tail).
    fn lens() -> impl Iterator<Item = usize> {
        0..=40
    }

    #[test]
    fn avg_matches_scalar() {
        let mut rng = Rng(0x1234_5678);
        for bd_max in [(8u8, 255i32), (10, 1023), (12, 4095)] {
            for n in lens() {
                let t1: Vec<i16> = (0..n).map(|_| rng.i16() >> 2).collect();
                let t2: Vec<i16> = (0..n).map(|_| rng.i16() >> 2).collect();
                let (rnd, sh) = (16, 5);
                if bd_max.0 == 8 {
                    let bd = BitDepth8;
                    let mut got = vec![0u8; n];
                    avg_row(bd, &mut got, &t1, &t2, n, rnd, sh);
                    assert_eq!(got, ref_avg(bd, &t1, &t2, rnd, sh), "avg n={n} 8bpc");
                } else {
                    let bd = BitDepth16::new(bd_max.0);
                    let mut got = vec![0u16; n];
                    avg_row(bd, &mut got, &t1, &t2, n, rnd, sh);
                    assert_eq!(
                        got,
                        ref_avg(bd, &t1, &t2, rnd, sh),
                        "avg n={n} {}b",
                        bd_max.0
                    );
                }
            }
        }
    }

    #[test]
    fn w_avg_matches_scalar() {
        let mut rng = Rng(0xABCD_1234);
        for wt in [0i32, 1, 5, 8, 11, 16] {
            for n in lens() {
                let t1: Vec<i16> = (0..n).map(|_| rng.i16() >> 2).collect();
                let t2: Vec<i16> = (0..n).map(|_| rng.i16() >> 2).collect();
                let (rnd, sh) = (128, 8);
                let bd = BitDepth16::new(10);
                let mut got = vec![0u16; n];
                w_avg_row(bd, &mut got, &t1, &t2, n, wt, rnd, sh);
                assert_eq!(
                    got,
                    ref_wavg(bd, &t1, &t2, wt, rnd, sh),
                    "wavg n={n} wt={wt}"
                );
            }
        }
    }

    #[test]
    fn mask_matches_scalar() {
        let mut rng = Rng(0x5555_AAAA);
        for n in lens() {
            let t1: Vec<i16> = (0..n).map(|_| rng.i16() >> 2).collect();
            let t2: Vec<i16> = (0..n).map(|_| rng.i16() >> 2).collect();
            let m: Vec<u8> = (0..n).map(|_| rng.u8() & 63).collect();
            let (rnd, sh) = (512, 10);
            let bd = BitDepth8;
            let mut got = vec![0u8; n];
            mask_row(bd, &mut got, &t1, &t2, &m, n, rnd, sh);
            assert_eq!(got, ref_mask(bd, &t1, &t2, &m, rnd, sh), "mask n={n}");
        }
    }

    #[test]
    fn blend_matches_scalar() {
        let mut rng = Rng(0x0F0F_F0F0);
        for n in lens() {
            let tmp: Vec<u8> = (0..n).map(|_| rng.u8()).collect();
            let m: Vec<u8> = (0..n).map(|_| rng.u8() & 63).collect();
            let dst0: Vec<u8> = (0..n).map(|_| rng.u8()).collect();
            let mut got = dst0.clone();
            blend_row(&mut got, &tmp, &m, n);
            let want: Vec<u8> = (0..n)
                .map(|i| {
                    let mm = m[i] as i32;
                    let d = dst0[i] as i32;
                    let t = tmp[i] as i32;
                    <u8 as Pixel>::from_i32((d * (64 - mm) + t * mm + 32) >> 6)
                })
                .collect();
            assert_eq!(got, want, "blend n={n}");
        }
    }

    #[test]
    fn row_clip_matches_scalar() {
        let mut rng = Rng(0x12_ABCD_77);
        for &(min, max) in &[(i16::MIN as i32, i16::MAX as i32), (-524288, 524287)] {
            for &(rnd, shift) in &[(8i32, 4i32), (1, 1), (2048, 12)] {
                for n in [0usize, 8, 16, 32, 64, 256, 1024] {
                    let v0: Vec<i32> = (0..n).map(|_| (rng.next() as i32) >> 8).collect();
                    let mut got = v0.clone();
                    row_clip(&mut got, n, rnd, shift, min, max);
                    let want: Vec<i32> = v0
                        .iter()
                        .map(|&v| ((v + rnd) >> shift).clamp(min, max))
                        .collect();
                    assert_eq!(got, want, "row_clip n={n} sh={shift}");
                }
            }
        }
    }

    #[test]
    fn residual_add_matches_scalar() {
        let mut rng = Rng(0x5A5A_3C3C);
        for &(bpc, max) in &[(8u8, 255i32), (10, 1023), (12, 4095)] {
            for &(rnd, shift) in &[(8i32, 4i32), (2048, 12)] {
                for n in lens() {
                    let c: Vec<i32> = (0..n).map(|_| (rng.next() as i32) >> 12).collect();
                    if bpc == 8 {
                        let bd = BitDepth8;
                        let d0: Vec<u8> = (0..n).map(|_| rng.u8()).collect();
                        let mut got = d0.clone();
                        residual_add_row(bd, &mut got, &c, n, rnd, shift);
                        let want: Vec<u8> = (0..n)
                            .map(|i| bd.pixel_clip(d0[i] as i32 + ((c[i] + rnd) >> shift)))
                            .collect();
                        assert_eq!(got, want, "resadd 8bpc n={n}");
                    } else {
                        let bd = BitDepth16::new(bpc);
                        let d0: Vec<u16> = (0..n)
                            .map(|_| (rng.next() % (max as u64 + 1)) as u16)
                            .collect();
                        let mut got = d0.clone();
                        residual_add_row(bd, &mut got, &c, n, rnd, shift);
                        let want: Vec<u16> = (0..n)
                            .map(|i| bd.pixel_clip(d0[i] as i32 + ((c[i] + rnd) >> shift)))
                            .collect();
                        assert_eq!(got, want, "resadd {bpc}b n={n}");
                    }
                }
            }
        }
    }

    #[test]
    fn dc_add_matches_scalar() {
        let mut rng = Rng(0xDC_ADD_111);
        for &(bpc, max) in &[(8u8, 255i32), (10, 1023), (12, 4095)] {
            for dc in [-300i32, -1, 0, 1, 50, 5000] {
                for n in lens() {
                    if bpc == 8 {
                        let bd = BitDepth8;
                        let d0: Vec<u8> = (0..n).map(|_| rng.u8()).collect();
                        let mut got = d0.clone();
                        dc_add_row(bd, &mut got, dc, n);
                        let want: Vec<u8> =
                            (0..n).map(|i| bd.pixel_clip(d0[i] as i32 + dc)).collect();
                        assert_eq!(got, want, "dc 8bpc n={n} dc={dc}");
                    } else {
                        let bd = BitDepth16::new(bpc);
                        let d0: Vec<u16> = (0..n)
                            .map(|_| (rng.next() % (max as u64 + 1)) as u16)
                            .collect();
                        let mut got = d0.clone();
                        dc_add_row(bd, &mut got, dc, n);
                        let want: Vec<u16> =
                            (0..n).map(|i| bd.pixel_clip(d0[i] as i32 + dc)).collect();
                        assert_eq!(got, want, "dc {bpc}b n={n} dc={dc}");
                    }
                }
            }
        }
    }

    #[test]
    fn cctx_matches_scalar() {
        let mut rng = Rng(0xCC_7700_3311);
        for bitdepth in [8i32, 10, 12] {
            let min = -(1 << (bitdepth + 7));
            let max = (1 << (bitdepth + 7)) - 1;
            // angle coeffs scaled by ~2^8 (rotation cos/sin); keep small like real data.
            for &(sina, cosa) in &[
                (0i32, 256i32),
                (181, 181),
                (256, 0),
                (-181, 181),
                (100, 236),
            ] {
                // sizes are powers of two in 16..=1024; also test a few + an odd tail.
                for sz in [16usize, 32, 64, 256, 1024, 17, 23] {
                    let u0: Vec<i32> = (0..sz).map(|_| (rng.next() as i32) % (max + 1)).collect();
                    let v0: Vec<i32> = (0..sz).map(|_| (rng.next() as i32) % (max + 1)).collect();
                    let (mut us, mut vs) = (u0.clone(), v0.clone());
                    cctx_row(&mut us, &mut vs, sina, cosa, sz, min, max);
                    // independent scalar reference
                    let (mut ur, mut vr) = (u0.clone(), v0.clone());
                    for i in 0..sz {
                        let a = ur[i] * cosa - vr[i] * sina;
                        let b = ur[i] * sina + vr[i] * cosa;
                        let na = ((a + 128 - (a < 0) as i32) >> 8).clamp(min, max);
                        let nb = ((b + 128 - (b < 0) as i32) >> 8).clamp(min, max);
                        ur[i] = na;
                        vr[i] = nb;
                    }
                    assert_eq!(us, ur, "cctx u bd={bitdepth} sz={sz} ang=({sina},{cosa})");
                    assert_eq!(vs, vr, "cctx v bd={bitdepth} sz={sz} ang=({sina},{cosa})");
                }
            }
        }
    }

    #[test]
    fn morph_matches_scalar() {
        let mut rng = Rng(0x9E37_79B9);
        for n in lens() {
            let dst0: Vec<u16> = (0..n).map(|_| (rng.next() % 1024) as u16).collect();
            let alpha = 300i32;
            let beta = -100i32;
            let bd = BitDepth16::new(10);
            let mut got = dst0.clone();
            morph_row(bd, &mut got, alpha, beta, n);
            let want: Vec<u16> = (0..n)
                .map(|i| bd.pixel_clip((alpha * dst0[i] as i32 + beta) >> 8))
                .collect();
            assert_eq!(got, want, "morph n={n}");
        }
    }

    impl Rng {
        fn u8b(&mut self) -> u8 {
            (self.next() >> 32) as u8
        }
        fn i8b(&mut self) -> i8 {
            (self.next() >> 40) as i8
        }
    }

    // Build a set of random u8 row buffers wide enough for a `cols`-pixel run
    // with column offsets in [-OFF, +OFF] starting at base `col0`.
    fn rand_rows(rng: &mut Rng, nrows: usize, len: usize) -> Vec<Vec<u8>> {
        (0..nrows)
            .map(|_| (0..len).map(|_| rng.u8b()).collect())
            .collect()
    }

    #[test]
    fn ns_wiener_fir_matches_scalar() {
        let mut rng = Rng(0x4E57_1234);
        let col0 = 6usize; // o
        let bufw = col0 + 64 + 8; // room for run + offsets + 8-wide load slack
        for ntaps in [6usize, 16] {
            for n in 0..=40usize {
                let nrows = 2 * ntaps + 1;
                let rows = rand_rows(&mut rng, nrows, bufw);
                let center = &rows[0];
                // random taps: each picks two rows, a dx in [-4,4], coef in [-128,127]
                let mut tap_meta = Vec::new();
                for i in 0..ntaps {
                    let rp = 1 + (i % (nrows - 1));
                    let rm = 1 + ((i + 1) % (nrows - 1));
                    let dx = (rng.next() % 9) as i32 - 4;
                    let coef = rng.i8b() as i32;
                    tap_meta.push((rp, rm, dx, coef));
                }
                let taps: Vec<WienerTap> = tap_meta
                    .iter()
                    .map(|&(rp, rm, dx, coef)| WienerTap {
                        row_p: &rows[rp],
                        row_m: &rows[rm],
                        dx,
                        coef,
                    })
                    .collect();
                let mut got = vec![0u8; n];
                ns_wiener_fir_run(&mut got, center, col0, &taps, n);
                let want: Vec<u8> = (0..n)
                    .map(|x| {
                        let c = col0 + x;
                        let m = center[c] as i32;
                        let mut s = m << 7;
                        for &(rp, rm, dx, coef) in &tap_meta {
                            let a = rows[rp][(c as i32 + dx) as usize] as i32;
                            let b = rows[rm][(c as i32 - dx) as usize] as i32;
                            s += (a + b - 2 * m) * coef;
                        }
                        ((s + 64) >> 7).clamp(0, 255) as u8
                    })
                    .collect();
                assert_eq!(got, want, "ns_wiener taps={ntaps} n={n}");
            }
        }
    }

    #[test]
    fn pc_wiener_fir_matches_scalar() {
        let mut rng = Rng(0x9C77_3311);
        let col0 = 6usize;
        let bufw = col0 + 64 + 8;
        let ntaps = 12usize;
        for n in 0..=40usize {
            let nrows = 2 * ntaps + 1;
            let rows = rand_rows(&mut rng, nrows, bufw);
            let center = &rows[0];
            let center_coef = rng.i16() as i32; // pretrained filters are i16
            let mut tap_meta = Vec::new();
            for i in 0..ntaps {
                let rp = 1 + (i % (nrows - 1));
                let rm = 1 + ((i + 1) % (nrows - 1));
                let dx = (rng.next() % 9) as i32 - 4;
                let coef = rng.i16() as i32;
                tap_meta.push((rp, rm, dx, coef));
            }
            let taps: Vec<WienerTap> = tap_meta
                .iter()
                .map(|&(rp, rm, dx, coef)| WienerTap {
                    row_p: &rows[rp],
                    row_m: &rows[rm],
                    dx,
                    coef,
                })
                .collect();
            let mut got = vec![0u8; n];
            pc_wiener_fir_run(&mut got, center, center_coef, col0, &taps, n);
            let want: Vec<u8> = (0..n)
                .map(|x| {
                    let c = col0 + x;
                    let m = center[c] as i32;
                    let mut s = m * center_coef;
                    for &(rp, rm, dx, coef) in &tap_meta {
                        let a = rows[rp][(c as i32 + dx) as usize] as i32;
                        let b = rows[rm][(c as i32 - dx) as usize] as i32;
                        s += (a + b) * coef;
                    }
                    ((s + 64) >> 7).clamp(0, 255) as u8
                })
                .collect();
            assert_eq!(got, want, "pc_wiener n={n}");
        }
    }

    #[test]
    fn gdf_gradient_group_matches_scalar() {
        let mut rng = Rng(0x67AD_9911);
        // mirrors compute_gradient_row_8bpc's per-cell math for one full group.
        let offs: [[i32; 2]; 4] = [[1, 0], [0, 1], [1, 1], [-1, 1]];
        let bufw = 6 + 64 + 16;
        for shift in [0u32, 2] {
            for row_off in [6usize, 8] {
                // need rows indexed [row_off-1-1 .. row_off+1]; allocate plenty.
                let nrows = row_off + 4;
                let rows: Vec<Vec<u8>> = (0..nrows)
                    .map(|_| (0..bufw).map(|_| rng.u8b()).collect())
                    .collect();
                let rowrefs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
                for col_off in [6usize, 7] {
                    let base_cell = 0usize;
                    let col0 = col_off + base_cell * 2;
                    let mut got = vec![[0u16; 4]; 8];
                    for d in 0..4 {
                        let dy = offs[d][0];
                        let dx = offs[d][1];
                        let center_rows = [rowrefs[row_off - 1], rowrefs[row_off]];
                        let a_rows = [
                            rowrefs[(row_off as i32 - 1 - dy) as usize],
                            rowrefs[(row_off as i32 - dy) as usize],
                        ];
                        let c_rows = [
                            rowrefs[(row_off as i32 - 1 + dy) as usize],
                            rowrefs[(row_off as i32 + dy) as usize],
                        ];
                        gdf_gradient_group(
                            &mut got,
                            d,
                            base_cell,
                            4,
                            center_rows,
                            a_rows,
                            c_rows,
                            col0,
                            dx,
                            shift,
                        );
                    }
                    // scalar reference for 4 cells (x1 = 0,2,4,6)
                    let mut want = vec![[0u16; 4]; 8];
                    let mut x1 = 0usize;
                    while x1 < 8 {
                        for d in 0..4 {
                            let mut grad = 0i32;
                            for x2 in 0..2usize {
                                let x = col_off + x1 + x2;
                                for y in 0..2 {
                                    let dy = offs[d][0];
                                    let dx = offs[d][1];
                                    let ry = row_off + y;
                                    let a = (rowrefs[(ry as i32 - 1 - dy) as usize]
                                        [(x as i32 - 1 - dx) as usize]
                                        >> shift)
                                        as i32;
                                    let b = (rowrefs[ry - 1][x - 1] >> shift) as i32;
                                    let c = (rowrefs[(ry as i32 - 1 + dy) as usize]
                                        [(x as i32 - 1 + dx) as usize]
                                        >> shift)
                                        as i32;
                                    grad += (b * 2 - a - c).abs();
                                }
                            }
                            want[x1 >> 1][d] = grad as u16;
                        }
                        x1 += 2;
                    }
                    assert_eq!(
                        got, want,
                        "gdf_grad shift={shift} ro={row_off} co={col_off}"
                    );
                }
            }
        }
    }

    #[test]
    fn gdf_add_matches_scalar() {
        let mut rng = Rng(0x6DF_ADD7);
        for scale in [5i32, 8] {
            for n in 0..=40usize {
                let dst0: Vec<u8> = (0..n).map(|_| rng.u8b()).collect();
                let err: Vec<i8> = (0..n).map(|_| rng.i8b()).collect();
                let mut got = dst0.clone();
                gdf_add_run(&mut got, &err, scale, n);
                let want: Vec<u8> = (0..n)
                    .map(|x| {
                        let diff = err[x] as i32 * scale;
                        let mag = (diff.abs() + 8) >> 4;
                        let adj = if diff < 0 { -mag } else { mag };
                        (dst0[x] as i32 + adj).clamp(0, 255) as u8
                    })
                    .collect();
                assert_eq!(got, want, "gdf_add scale={scale} n={n}");
            }
        }
    }
}

# SIMD optimization plan

Correctness is done (bit-exact vs dav2d, 8-bit + 10-bit). This is the **speed**
phase. Today rav2d runs scalar Rust for everything except motion compensation
(NEON via FFI), so it's ~0.03–0.25× dav2d. The plan: replace the hot scalar
kernels with **SIMD written in Rust**, keeping memory-safety.

## Why not just reuse dav2d's asm?
Most of dav2d's AV2-fork assembly (inverse transforms, MSAC, loop filters, CDEF,
film grain) is still AV1-era and is **not bit-exact for AV2** — can't reuse it.
Only MC + a few intra modes have AV2-valid NEON (already wired). So the rest must
be our own.

## Approach: `std::simd` (portable SIMD in Rust)
- Write each kernel once with `core::simd::Simd<T, N>` → compiles to NEON (ARM),
  AVX2 (x86), SVE, etc. One source, all arches.
- Stays memory-safe (no raw asm); `unsafe` only where an intrinsic is needed.
- LLVM inlines it with surrounding Rust; tested with the same `#[test]` + the FFI
  oracle catches any drift from bit-exactness.
- `std::simd` is nightly today → either pin nightly for the SIMD build, or use a
  stable shim crate (`wide` / `pulp`) or `core::arch` intrinsics (per-arch, `unsafe`).
- Expected: ~80–95% of hand-tuned asm, ~×3–6 over current scalar on hot kernels.
  Realistic target: same order of magnitude as dav2d (≈0.5–1×).

## Pattern (keep 8bpc bit-exact, like the M6 NEON wiring)
Generic `fn foo_simd<...>()` + a scalar fallback; gate behind the existing
`RAV2D_NEON_OFF` / a `simd` feature; verify each kernel with a NEON-vs-scalar guard
test AND the full conformance sweeps (`bit_exact_full_clip_sweep` etc.) before/after.

## Priority order (hottest first)
1. **Inverse transforms (itx)** — the #1 bottleneck for intra/small clips; no AV2
   asm exists upstream, so this is the biggest win. `itx.rs` / `itx_1d.rs` butterflies.
2. **Residual add + blend** — `avg`/`w_avg`/`mask`/`w_mask` (compound), `residual_add`.
   Simple, very hot, easy SIMD (load i16 lanes, add, clamp, narrow).
3. **Intra prediction** — DC/smooth/paeth/directional in `ipred.rs` (the modes dav2d
   doesn't provide AV2 NEON for).
4. **In-loop filters** — deblock / CDEF / loop-restoration (Wiener/PC-Wiener/GDF).
5. **Film grain** — fgy/fguv apply.
6. **MSAC** — the entropy decoder dominates the small 64x64 clips; hardest to SIMD
   (serial dependency), maybe scalar micro-opt instead (branch removal, table layout).

## Quick wins outside SIMD (do alongside)
- Reuse scratch buffers instead of per-block `Vec::new()` allocations in the recon
  path (decode.rs allocates i16 tmp per block — a known cost).
- `#[inline]` + avoid bounds checks in inner loops via slice windows.

## Verification gate (non-negotiable)
Every kernel change must keep all bit-exact oracles green (8-bit corpus filters
off+on, 10-bit, film grain) on NEON and scalar. The FFI oracle is the safety net.

## Example — `avg` compound blend
```rust
#![feature(portable_simd)]
use std::simd::{Simd, num::SimdInt, cmp::SimdOrd};
fn avg_simd(dst: &mut [u8], a: &[i16], b: &[i16]) {
    const N: usize = 8;
    for i in (0..a.len()).step_by(N) {
        let x = Simd::<i16, N>::from_slice(&a[i..]).cast::<i32>();
        let y = Simd::<i16, N>::from_slice(&b[i..]).cast::<i32>();
        let s = ((x + y + Simd::splat(1)) >> Simd::splat(1))
            .simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>();
        s.copy_to_slice(&mut dst[i..i + N]);
    }
    // + scalar tail for len % N
}
```
NEON intrinsic equivalent (`core::arch::aarch64`): `vld1q_s16` → `vhaddq_s16`
(avg+round in one op) → `vqmovun_s16` → `vst1_u8`.

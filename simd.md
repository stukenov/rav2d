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

## Approach: `wide` (portable SIMD on stable Rust) — DECIDED
Chosen over `std::simd` (nightly, would force downstream crates.io users onto
nightly) and raw `core::arch` (per-arch + `unsafe`). `wide` gives:
- One source compiles to NEON (aarch64) / AVX2 (x86_64) / SSE / etc.
- **Stable** toolchain — published crate stays buildable for everyone.
- **Safe API** — no `unsafe` in our kernels.
- `wide` types: `i16x8`, `i32x8`, `u8x16`, `f32x8`, … with lane ops.
- LLVM inlines it with surrounding Rust; tested with the same `#[test]` + the FFI
  oracle catches any drift from bit-exactness.
- Bit-exactness note: `wide` doesn't expose fused rounding-halving-add; write the
  rounding explicitly (`(a+b+1)>>1` in wider lanes) so it matches dav2d exactly.
- Expected: ~80–95% of hand-tuned asm, ~×3–6 over current scalar on hot kernels.
  Realistic target: same order of magnitude as dav2d (≈0.5–1×).
- Dep added: `wide = "0.7"` in crates/rav2d/Cargo.toml.

## Pattern (keep 8bpc bit-exact, like the M6 NEON wiring)
Per kernel: keep the existing scalar fn as the reference + add a `*_simd` variant
using `wide`. A small guard `#[test]` asserts SIMD output == scalar output over
randomized inputs (cheap, runs every `cargo test`). The full conformance sweeps
(`bit_exact_full_clip_sweep` etc.) are the end-to-end net — must stay green on
NEON and `RAV2D_NEON_OFF=all` before/after every kernel.

## Baseline (pre-SIMD: scalar everything except MC-NEON) — 2026-06-06
Single-thread, filters off, dav2d/rav2d throughput ratio:
| clip | rav MP/s | dav MP/s | dav/rav |
|---|---|---|---|
| bus.64x64.l5 | 4.1 | 70.1 | 0.06x |
| bus.64x64.l5.lossless | 3.4 | 27.2 | 0.13x |
| bus.64x64.l5.opfl0 | 4.4 | 80.0 | 0.06x |
| bus.64x64.l1.sdp0 | 3.4 | 34.3 | 0.10x |
| bus.352x288.l5.seg1 | 72.2 | 616.3 | 0.12x |
| bus.352x288.l10.deltaq1 | 18.6 | 196.4 | 0.09x |
| bus.352x288.l1.partial_lossless | 34.6 | 152.4 | 0.23x |
| hm.64x64.l5.filmgrain | 4.0 | 166.4 | 0.02x |
Re-run `cargo bench -p rav2d --bench decode -- --measurement-time 0.5 --sample-size 10`
after each kernel to track the ratio climbing.

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

## Ranked implementation plan (from kernel-map workflow, 2026-06-06)
Ranking = (throughput win × bit-exact safety) / effort, favoring kernels with no
NEON coverage + clean data-parallelism. **Pre-flight gate (kernels #1–#9):** the
first avg test MUST confirm `wide::i32x8 >>` is arithmetic (sign-propagating).

**Phase 1 (prove pattern + infra, all S):**
1. `mc.rs` compound-blend family — `avg`(:85) → `w_avg`(:126) → `mask_fn`(:170) →
   `blend`(:205) → `morph`(:236). Simplest vector shape; exercises every primitive
   (i16→i32x8, splat, arith `>>`, clamp[0,max], narrow). Wins on ALL HBD + non-aarch64
   + `morph`(BAWP) scalar-everywhere. Build the scalar-vs-SIMD guard harness here.
2. `cctx` (itx_1d.rs:258) — 2D rotate+clip over two i32 planes, sz%16==0 (no tail),
   loop-invariant coeffs. Tests the `-(a<0)` sign-mask machinery.
3. `inv_txfm_add` DC-only (itx.rs:61) — broadcast-add+clamp; hit on most TUs.

**Phase 2 (itx core, M–L):**
4. `inv_dct_1d`(itx_1d.rs:96) + `inv_dst_1d`(:151) — SoA column-batch 8 cols/call →
   vertical i32x8 butterflies, no gather. Biggest itx lever. Restructure caller.
5. `inv_txfm_add` row-clip(itx.rs:150) + plain `residual_add`(itx_1d.rs:332, dpcm==0).
   Keep 2× upsample + dpcm 1/2 accumulator paths SCALAR.

**Phase 3 (filters):**
6. CDEF `cdef_filter_block` sec-only(cdef.rs:428) → pri+sec(:313). Asymmetric round
   `(sum-(sum<0)+8)>>4` per-lane; reuse #1/#2 sign machinery.
7. Loop-restoration Wiener FIRs: `ns_wiener_single_y`(looprestoration.rs:400) +
   `wiener_multi`(:556). Plain arith `>>7` (NOT CDEF's asym round). Class-uniform 8-wide.

**Phase 4 (conditional/portable):**
8. Film grain apply `fgy_32x32xn`(filmgrain.rs:300)+`fguv`(:484). LUT gather stays scalar.
9. Intra `ipred_paeth`(ipred.rs:654)→`ipred_smooth`2D(:689)→`ipred_z1`(:825). z2/z3 defer.
10. msac `decode_symbol_adapt` CDF-update half(msac.rs:312) u16x8. Small absolute win.

**NOT worth SIMD (leave scalar):** msac symbol-search/`decode_bool`/`ctx_norm`/`ctx_refill`
(serial state machine), grain generate (serial LFSR/AR), `generate_scaling` (one-time LUT),
`filter_choice` deblock (control-flow-bound), `cfl_mhccp`/`ipred_dip` dot (per-lane dyn shift).

## PROFILING PIVOT (2026-06-06, `sample` on deltaq1, filters ON)
Real (filtered) decode self-time leaders — loop restoration dominates, itx is noise:
- `looprestoration::lr_stripe_8bpc` — **1765** (the Wiener/PC-Wiener/GDF FIR) ← #1 BY FAR
- `looprestoration::compute_gradient_row_8bpc` — 94 (GDF gradient)
- `decode::recon_b_inter_tip` 159, `mc::opfl_derive_mv` 121, `refmvs::load_tmvs` 121
- `deblock::deblock_bd` 29; `mc::mask_fn` 13; `mc::avg` 6
- `itx_1d::inv_dct_1d` 7, `inv_dct_1d_x8` 5  ← itx now negligible in filtered decode ✓
(madvise/init_wedge_masks/subsample_420 noise = profile harness re-opens Decoder each
iter; amortized in real streams — discount.)

=> **Reprioritize: loop restoration (#7) is actually the #1 lever for filtered playback.**
Phase-2 itx done (bit-exact, helps transform-heavy/intra). Next: SIMD the LR FIRs
(pc_wiener, ns_wiener single/multi, gdf compute_gradient/add), then inter MC helpers
(opfl_derive_mv, sad_refine_mv), then deblock.

## Filtered-decode throughput (after itx + LR SIMD, 2026-06-06) — the real metric
With ALL in-loop filters on (deblock+cdef+ccso+wiener+gdf = real playback), dav/rav:
| clip | rav MP/s | dav MP/s | dav/rav (ON) | (was OFF) |
|---|---|---|---|---|
| bus.352x288.l5.seg1 | 39.3 | 111.9 | **0.35x** | 0.11x |
| bus.352x288.l10.deltaq1 | 11.7 | 31.7 | **0.37x** | 0.10x |
| bus.352x288.l1.partial_lossless | 8.8 | 19.9 | **0.44x** | 0.24x |
| bus.64x64.l5 | 3.7 | 26.9 | 0.14x | 0.06x |
Filtered path is where decoding actually happens; rav2d is ~0.35–0.44x of C+NEON
dav2d there on representative clips. LR SIMD gave ~7.5% end-to-end filtered (more in
real streams). Remaining filtered hot spots to profile next: inter MC helpers
(opfl_derive_mv, sad_refine_mv, mc_opfl), deblock, recon_b_inter_tip orchestration.

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

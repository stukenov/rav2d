# rav2d

[![crates.io](https://img.shields.io/crates/v/rav2d.svg)](https://crates.io/crates/rav2d)
[![docs.rs](https://img.shields.io/docsrs/rav2d)](https://docs.rs/rav2d)
[![CI](https://github.com/stukenov/rav2d/actions/workflows/ci.yml/badge.svg)](https://github.com/stukenov/rav2d/actions/workflows/ci.yml)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**rav2d** is a memory-safe **AV2** video decoder in Rust, ported from [dav2d](https://code.videolan.org/videolan/dav2d) (the C reference decoder, an AV2 fork of dav1d).

> The entire C decode path has been ported to Rust and is **bit-exact** with dav2d: every coding-order frame of every shipped conformance clip matches byte-for-byte, with in-loop filters off **and** on, for both 8-bit and 10-bit streams. **846 library + 22 conformance tests pass**, on the NEON and the all-scalar path alike.

## Status

Bit-exact against the dav2d C reference (verified by an FFI oracle that decodes the same bitstream with both decoders and byte-compares every plane of every frame):

| Capability | Status |
|---|---|
| Intra (DC/directional/smooth/paeth, CfL, MHCCP, MRL, DIP, palette, IntraBC) | ✅ bit-exact |
| Inter (single-ref, compound, warp-affine, OBMC, interintra, BAWP) | ✅ bit-exact |
| TIP (block-level + whole-frame), OPFL optical-flow refinement | ✅ bit-exact |
| In-loop filters (deblock, CDEF, CCSO, Wiener / PC-Wiener / GDF) | ✅ bit-exact |
| Segmentation, delta-Q, lossless (WHT) | ✅ bit-exact |
| Film grain synthesis | ✅ bit-exact |
| High bit depth — 10-bit | ✅ bit-exact |
| High bit depth — 12-bit | ⚠️ code path present, not yet verified against a vector |
| Scaled references (svc/resize) | ✅ bit-exact |
| Assembly DSP dispatch (aarch64 NEON via FFI) | ✅ motion compensation + intra H/V/smooth |
| Assembly DSP dispatch (x86) | ❌ CPU features are detected, no kernel uses them — x86 runs the scalar path |
| Multithreading | ⚠️ output copy and film grain only; parsing, reconstruction and filters are single-threaded |

The full corpus (`bit_exact_full_clip_sweep`), the filtered corpus (`bit_exact_full_clip_filtered_sweep`), the 10-bit vectors (`bit_exact_hbd_sweep`), film grain (`bit_exact_filmgrain_applied`), scaled references (`bit_exact_scaled_ref_sweep`) and the coverage vectors (`bit_exact_coverage_sweep`, 4:0:0 and 4:2:0) are all enforced as tests.

One shipped vector is deliberately not a bit-exact gate: `cov-multitile-416x240.obu`, whose keyframe entropy stream is malformed in a way dav2d cannot decode deterministically either — its own output differs between thread configurations. rav2d must decode it without panicking, which `coverage_decode_no_panic` enforces.

## Why Rust?

Video decoders parse untrusted bitstreams from the internet — a prime target for memory-corruption exploits. Historical CVEs in C decoders (libvpx, dav1d, ffmpeg) are overwhelmingly buffer overflows, use-after-free, and integer overflows in parsing code.

**rav2d eliminates these bug classes at compile time** while reusing dav2d's hand-written SIMD for the hottest pixel kernels:

| | dav2d (C) | rav2d (Rust) |
|---|---|---|
| Bitstream parsing | C (unsafe) | Rust (bounds-checked) |
| Decode orchestration | C (unsafe) | Rust (safe, typed) |
| Filter pipeline | C (unsafe) | Rust (bounds-checked) |
| DSP kernels | Assembly + C | Assembly via FFI (where AV2-valid) + Rust |
| Type safety | Weak (enums as ints) | Strong (enum variants, pattern matching) |

## Quick Start

```sh
cargo add rav2d
```

```rust
use rav2d::{Decoder, Settings, Data, Rav2dError};

let mut decoder = Decoder::open(&Settings::default()).unwrap();

let obu_data: Vec<u8> = std::fs::read("input.obu").unwrap();
decoder.send_data(Some(Data::wrap(obu_data))).unwrap();

loop {
    match decoder.get_picture() {
        Ok(pic) => {
            // Planes come out as ordinary slices — no `unsafe` in consumer code.
            let (w, h) = pic.plane_dimensions(0).unwrap();   // luma size in samples
            for row in pic.plane_rows_u8(0) { /* row: &[u8], w samples, no padding */ }
            // 10/12-bit frames: pic.plane_rows_u16(0) -> &[u16]
        }
        Err(Rav2dError::Again) => break, // need more data
        Err(e) => panic!("{e}"),
    }
}
```

### CLI

```sh
cargo install rav2d-cli
rav2d input.ivf -o output.y4m      # decode IVF → Y4M
rav2d input.ivf                    # decode-only benchmark
rav2d input.ivf -o out.y4m --limit 100 --no-grain
```

## Performance

rav2d ports the C *logic* to Rust; hand-written assembly stays via FFI. On aarch64 the **motion-compensation** kernels and four intra-prediction modes dispatch to dav2d's NEON (run `RAV2D_NEON_OFF=all` to force the scalar Rust path). All other DSP families run scalar Rust — **not by choice**: dav2d's AV2 fork still ships AV1-era assembly for inverse transforms, the entropy decoder, loop filters, CDEF and film grain, which is not bit-exact for AV2, so those kernels cannot be reused and the correct scalar Rust is used instead.

Where that leaves throughput, single-threaded, against C+SIMD dav2d:

| | rav2d / dav2d |
|---|---|
| In-loop filters on (what playback actually does) | **0.14–0.44×** |
| Filters off | 0.03–0.25× |

The filtered figure is the one that matters, and it is the one portable-SIMD work has been moving (see [`simd.md`](simd.md)). Two things stand between here and parity: the decode core is still single-threaded, and on x86 every kernel is scalar. Closing the rest needs either AV2-updated assembly upstream or more of our own SIMD.

```sh
DYLD_LIBRARY_PATH=dav2d/build/src cargo bench -p rav2d   # prints a rav2d-vs-dav2d table
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| [`rav2d`](crates/rav2d/) | Main decoder library — safe Rust API |
| [`rav2d-sys`](crates/rav2d-sys/) | Raw FFI bindings to dav2d (bindgen) + NEON asm dispatch |
| [`rav2d-cli`](crates/rav2d-cli/) | Command-line decoder (IVF → Y4M) |

> **Note:** `rav2d` itself is pure Rust and does **not** link the C library — the NEON assembly it dispatches to is vendored under `crates/rav2d/vendor/dav2d-asm`, so `cargo add rav2d` works with no submodule and no C toolchain. `rav2d-sys` is the conformance oracle: it binds `dav2d.h` so the test suite can decode the same bitstream with both decoders and compare. It is a path-only dev-dependency, dropped from the published manifest entirely, and CI keeps it that way with a `cargo publish --dry-run` job that runs against a checkout with the submodule removed.

## Building

### Prerequisites

Using the library needs only Rust 1.85+ (edition 2024). Running the conformance suite needs the C reference too:

- meson + ninja, to build the `dav2d` submodule
- LLVM/clang, for bindgen

### Build & test

```sh
git submodule update --init --recursive

# 1. Build the dav2d C reference (used for linking + the conformance oracle)
cd dav2d && meson setup build && ninja -C build && cd ..

# 2. Build + test rav2d
cargo build --workspace
DYLD_LIBRARY_PATH=dav2d/build/src cargo test -p rav2d           # macOS
LD_LIBRARY_PATH=dav2d/build/src  cargo test -p rav2d            # Linux

# Force the all-scalar path (no NEON):
RAV2D_NEON_OFF=all DYLD_LIBRARY_PATH=dav2d/build/src cargo test -p rav2d
```

## Conformance

`crates/rav2d/tests/conformance.rs` is an FFI oracle: it decodes each clip with **both** rav2d and the dav2d C library and asserts byte-equal output. Test clips live in `crates/rav2d/tests/data` — the 8-bit vectors under `media/` came from dav2d, which has since stopped bundling them, and the rest were staged here. The dav2d submodule is kept pristine — it is the source of truth for the port.

A second, independent gate compares against **avmdec**, the AOM reference decoder: every vector under `tests/data/media` ships the md5 avmdec produces for it, and `bit_exact_avm_reference_md5` hashes rav2d's visible frames, in display order, against it. Matching dav2d cannot catch a bug faithfully ported from the C — both agree and the gate stays green — so this one does not go through dav2d at all.

## Approach

Following the [rav1d](https://github.com/memorysafety/rav1d) strategy:

1. Assembly stays via FFI (reused, not rewritten) — where the AV2 fork's asm is actually AV2-valid.
2. All C decoder logic is ported to Rust, validated bit-exact against dav2d at every step.
3. Data tables are extracted from C and validated via FFI comparison.

## Safety

- The public API hands out pixels as ordinary slices (`plane_rows_u8`, `plane_row_u16`, …), so decoding a frame and reading it needs no `unsafe` in calling code. Raw pointers remain available via `plane_ptr`/`plane_stride_bytes` for FFI.
- `PicAllocator` is an `unsafe trait`: supplying the decoder's pixel buffers is the one place a caller can undermine those safe accessors, so the contract is stated and opted into explicitly.
- All `unsafe impl Send/Sync` documented with SAFETY comments.
- Enum transmutes replaced with validated `from_raw()` helpers + debug assertions.
- `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide.
- Remaining `unsafe` is concentrated in FFI calls, the NEON dispatch, and performance-critical inner loops.

### Fuzzing and sanitizers

Bit-exactness proves the decoder computes the right picture; it says nothing about what the `unsafe` pixel paths do on input no encoder would produce. Two workflows cover that, both nightly and on demand:

- **`fuzz`** runs the `decode` and `decode_settings` [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets. `decode_settings` derives the decoder configuration from the input, so grain, header-only parsing, invisible-frame output and the per-filter combinations get fuzzed too. The corpus is seeded from the conformance vectors and carried between runs, and every crash it has found is replayed by `fuzz_regression_corpus_no_panic` in ordinary CI.
- **`sanitizers`** runs the unit tests, the regression corpus and the conformance sweeps under AddressSanitizer (with leak detection where the C reference is not involved), and the multi-threaded decode under ThreadSanitizer.

```sh
cd crates/rav2d/fuzz
./seed-corpus.sh decode
cargo +nightly fuzz run decode -- -max_total_time=900
```

## Development

rav2d was ported from dav2d with heavy use of AI coding tools (Claude Code). This is
disclosed here rather than left implicit in the commit history.

The methodology is built so correctness does not depend on trusting the tooling: every
ported step is checked bit-exact against the dav2d C reference by the FFI oracle described
in [Conformance](#conformance), and the whole test suite gates the result. Assembly is
reused via FFI, not regenerated. Where the port and the C reference disagree, the C
reference wins.

## Related Projects

- [rav1d](https://github.com/memorysafety/rav1d) — Rust port of dav1d (AV1)
- [dav2d](https://code.videolan.org/videolan/dav2d) — the C original (AV2)
- [dav1d](https://code.videolan.org/videolan/dav1d) — the AV1 predecessor

## License

BSD 2-Clause, same as dav2d.

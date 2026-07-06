# rav2d

[![crates.io](https://img.shields.io/crates/v/rav2d.svg)](https://crates.io/crates/rav2d)
[![docs.rs](https://img.shields.io/docsrs/rav2d)](https://docs.rs/rav2d)
[![CI](https://github.com/stukenov/rav2d/actions/workflows/ci.yml/badge.svg)](https://github.com/stukenov/rav2d/actions/workflows/ci.yml)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**rav2d** is a memory-safe **AV2** video decoder in Rust, ported from [dav2d](https://code.videolan.org/videolan/dav2d) (the C reference decoder, an AV2 fork of dav1d).

> The entire C decode path has been ported to Rust and is **bit-exact** with dav2d: every coding-order frame of every shipped conformance clip matches byte-for-byte, with in-loop filters off **and** on, for both 8-bit and 10-bit streams. **818 library + 16 conformance tests pass.**

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
| Assembly DSP dispatch (aarch64 NEON via FFI) | ✅ motion compensation + intra H/V/smooth |
| Multithreading | ✅ disjoint display passes; recon core single-threaded |

The full corpus (`bit_exact_full_clip_sweep`), the filtered corpus (`bit_exact_full_clip_filtered_sweep`), the 10-bit vectors (`bit_exact_hbd_sweep`), and film grain (`bit_exact_filmgrain_applied`) are all enforced as tests.

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
        Ok(pic) => { /* pic.data planes, pic.p.w, pic.p.h, pic.p.bpc */ }
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

Consequently single-thread throughput today is roughly **0.03–0.25× dav2d** (scalar Rust vs C+SIMD), with NEON MC narrowing the gap on motion-heavy clips by ~1.4–1.6×. Closing the rest requires either AV2-updated assembly upstream or optimizing the scalar Rust kernels.

```sh
DYLD_LIBRARY_PATH=dav2d/build/src cargo bench -p rav2d   # prints a rav2d-vs-dav2d table
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| [`rav2d`](crates/rav2d/) | Main decoder library — safe Rust API |
| [`rav2d-sys`](crates/rav2d-sys/) | Raw FFI bindings to dav2d (bindgen) + NEON asm dispatch |
| [`rav2d-cli`](crates/rav2d-cli/) | Command-line decoder (IVF → Y4M) |

> **Note:** `rav2d-sys` (and therefore `rav2d`) builds against the bundled `dav2d` C submodule — it binds `dav2d.h` and assembles dav2d's NEON `.S` files. Build the submodule first (see below). This is a workspace/source build; the crate is not a drop-in standalone crates.io dependency without the submodule present.

## Building

### Prerequisites

- Rust 1.85+ (edition 2024)
- meson + ninja (to build the dav2d submodule)
- LLVM/clang (for bindgen)

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

`crates/rav2d/tests/conformance.rs` is an FFI oracle: it decodes each clip with **both** rav2d and the dav2d C library and asserts byte-equal output. Test clips live in `dav2d/media` (8-bit) and `crates/rav2d/tests/data` (10-bit). The dav2d submodule is kept pristine — it is the source of truth for the port.

## Approach

Following the [rav1d](https://github.com/memorysafety/rav1d) strategy:

1. Assembly stays via FFI (reused, not rewritten) — where the AV2 fork's asm is actually AV2-valid.
2. All C decoder logic is ported to Rust, validated bit-exact against dav2d at every step.
3. Data tables are extracted from C and validated via FFI comparison.

## Safety

- All `unsafe impl Send/Sync` documented with SAFETY comments.
- Enum transmutes replaced with validated `from_raw()` helpers + debug assertions.
- `#![warn(unsafe_op_in_unsafe_fn)]` crate-wide.
- Remaining `unsafe` is concentrated in FFI calls, the NEON dispatch, and performance-critical inner loops.

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

# rav2d

[![crates.io](https://img.shields.io/crates/v/rav2d.svg)](https://crates.io/crates/rav2d)
[![docs.rs](https://img.shields.io/docsrs/rav2d)](https://docs.rs/rav2d)
[![CI](https://github.com/stukenov/rav2d/actions/workflows/ci.yml/badge.svg)](https://github.com/stukenov/rav2d/actions/workflows/ci.yml)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**rav2d** is a Rust port of [dav2d](https://code.videolan.org/videolan/dav2d), a cross-platform **AV2** video decoder focused on speed, correctness, and memory safety.

> All C decoder logic has been ported to Rust. Assembly-optimized DSP kernels remain via FFI. 790 tests pass across the workspace.

## Why Rust?

Video decoders parse untrusted bitstreams from the internet — they are a prime target for memory corruption exploits. Historical CVEs in video decoders (libvpx, dav1d, ffmpeg) are overwhelmingly buffer overflows, use-after-free, and integer overflows in C parsing code.

**rav2d eliminates these classes of bugs at compile time** while keeping the same hand-optimized assembly for actual pixel processing:

| | dav2d (C) | rav2d (Rust) |
|---|---|---|
| Bitstream parsing | C (unsafe) | Rust (bounds-checked) |
| Decode orchestration | C (unsafe) | Rust (safe, typed) |
| Filter pipeline | C (unsafe) | Rust (bounds-checked) |
| DSP kernels | Assembly | Assembly (shared via FFI) |
| Type safety | Weak (enums as ints) | Strong (enum variants, pattern matching) |

## Quick Start

Add the decoder library to your project:

```sh
cargo add rav2d
```

```rust
use rav2d::{Decoder, Settings, Data, Rav2dError};

let mut decoder = Decoder::open(&Settings::default()).unwrap();

// Feed compressed data
let obu_data: Vec<u8> = std::fs::read("input.obu").unwrap();
decoder.send_data(Some(Data::wrap(obu_data))).unwrap();

// Retrieve decoded pictures
loop {
    match decoder.get_picture() {
        Ok(pic) => { /* use pic.data, pic.p.w, pic.p.h */ }
        Err(Rav2dError::Again) => break, // need more data
        Err(e) => panic!("{e}"),
    }
}
```

### CLI

```sh
cargo install rav2d-cli
```

```sh
# Decode IVF to Y4M
rav2d input.ivf -o output.y4m

# Decode-only benchmark (no output)
rav2d input.ivf

# Limit frames and skip film grain
rav2d input.ivf -o out.y4m --limit 100 --no-grain
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| [`rav2d`](crates/rav2d/) | Main decoder library — safe Rust API |
| [`rav2d-sys`](crates/rav2d-sys/) | Raw FFI bindings to dav2d C/asm (bindgen) |
| [`rav2d-cli`](crates/rav2d-cli/) | Command-line decoder tool (IVF → Y4M) |

## What's Ported

| Module | Status |
|--------|--------|
| OBU parsing | Complete |
| Entropy decoding (MSAC) | Complete |
| Block decoding (intra/inter/compound) | Complete |
| Deblocking filter | Complete |
| CDEF | Complete |
| Loop restoration (NS/PC Wiener, GDF) | Complete |
| Film grain synthesis | Complete |
| Motion compensation | Complete |
| Inverse transforms | Complete |
| Reference management | Complete |
| Thread task scheduling | Complete |

**47 Rust source files**, ~47,000 lines of ported decoder logic.

## Building

### Prerequisites

- Rust 1.85+ (edition 2024)
- meson + ninja (to build dav2d)
- LLVM/clang (for bindgen)

### Build

```sh
# 1. Build the dav2d C library
cd dav2d
meson setup build
ninja -C build
cd ..

# 2. Build rav2d
cargo build

# 3. Run tests
DYLD_LIBRARY_PATH=dav2d/build/src cargo test --workspace    # macOS
LD_LIBRARY_PATH=dav2d/build/src cargo test --workspace       # Linux

# 4. Run clippy
cargo clippy --workspace

# 5. Build CLI
cargo build -p rav2d-cli --release
```

### Benchmarks

```sh
DYLD_LIBRARY_PATH=dav2d/build/src cargo bench -p rav2d
```

## Architecture

```
rav2d/
├── crates/
│   ├── rav2d/           # Decoder library
│   │   ├── src/
│   │   │   ├── lib.rs         # Public API re-exports
│   │   │   ├── decoder.rs     # Decoder struct, Settings, open/send/get
│   │   │   ├── obu.rs         # OBU parsing (sequence/frame/tile headers)
│   │   │   ├── decode.rs      # Block-level decoding (5600 lines)
│   │   │   ├── recon.rs       # Reconstruction (MC, prediction, ITX)
│   │   │   ├── refmvs.rs      # Reference motion vectors
│   │   │   ├── cdef.rs        # Constrained directional enhancement
│   │   │   ├── looprestoration.rs  # Wiener/GDF loop restoration
│   │   │   ├── filmgrain.rs   # Film grain synthesis
│   │   │   └── ...            # 38 more modules
│   │   └── benches/
│   │       └── decode.rs      # Criterion benchmarks
│   ├── rav2d-sys/       # FFI bindings (auto-generated via bindgen)
│   └── rav2d-cli/       # CLI binary
│       └── src/
│           ├── main.rs        # Argument parsing, decode loop
│           ├── ivf.rs         # IVF demuxer
│           └── y4m.rs         # Y4M writer
├── dav2d/               # C submodule (source of truth)
└── .github/workflows/   # CI (ubuntu + macOS matrix)
```

## Approach

Following the proven [rav1d](https://github.com/memorysafety/rav1d) strategy:

1. FFI bindings to dav2d's hand-optimized assembly (shared, not rewritten)
2. Progressive C-to-Rust port of the core decoder
3. Conformance testing at every step against dav2d test data

## Safety

- All `unsafe impl Send/Sync` blocks are documented with SAFETY comments
- Enum transmutes replaced with validated `from_raw()` helpers with debug assertions
- `#![warn(unsafe_op_in_unsafe_fn)]` enabled crate-wide
- Remaining `unsafe` blocks are concentrated in FFI calls and performance-critical inner loops

## Related Projects

- [rav1d](https://github.com/memorysafety/rav1d) — Rust port of dav1d (AV1), funded by Prossimo/ISRG
- [dav2d](https://code.videolan.org/videolan/dav2d) — The C original (AV2)
- [dav1d](https://code.videolan.org/videolan/dav1d) — The AV1 predecessor

## License

BSD 2-Clause, same as dav2d.

# rav2d

**rav2d** is a Rust port of [dav2d](https://code.videolan.org/videolan/dav2d), a cross-platform **AV2** video decoder focused on speed, correctness, and memory safety.

> **All C decoder logic has been ported to Rust.** Assembly-optimized DSP kernels remain via FFI. 786 tests pass.

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

## Status

- **47 Rust source files**, ~47,000 lines
- **786 unit tests** passing
- Full filter pipeline: deblock, CDEF, loop restoration (Wiener/GDF), film grain
- Full reconstruction: motion compensation, compound prediction, optical flow, inverse transform
- Core infrastructure: CPU detection, memory pools, ref counting, threading

### What's ported

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

## Crate Structure

| Crate | Description |
|-------|-------------|
| `rav2d` | Main decoder library with safe Rust API |
| `rav2d-sys` | Raw FFI bindings to dav2d C/asm |
| `rav2d-cli` | Command-line decoder tool |

## Approach

Following the proven [rav1d](https://github.com/memorysafety/rav1d) strategy:

1. FFI bindings to dav2d's hand-optimized assembly (shared, not rewritten)
2. Progressive C-to-Rust port of the core decoder
3. Conformance testing at every step against dav2d test data

## Building

### Prerequisites

- Rust 1.85+
- dav2d built locally (via meson/ninja)
- LLVM/clang (for bindgen)

### Build

```sh
# First, build dav2d
cd dav2d
mkdir build && cd build
meson setup .. && ninja

# Then build rav2d
cd ../..
cargo build

# Run tests
DYLD_LIBRARY_PATH=dav2d/build/src cargo test -p rav2d
```

## Related Projects

- [rav1d](https://github.com/memorysafety/rav1d) — Rust port of dav1d (AV1), funded by Prossimo/ISRG
- [dav2d](https://code.videolan.org/videolan/dav2d) — The C original (AV2)
- [dav1d](https://code.videolan.org/videolan/dav1d) — The AV1 predecessor

## License

BSD 2-Clause, same as dav2d.

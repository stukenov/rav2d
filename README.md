# rav2d

**rav2d** is a Rust port of [dav2d](https://code.videolan.org/videolan/dav2d), a cross-platform **AV2** video decoder focused on speed, correctness, and memory safety.

> **Status: Early development.** The AV2 specification is not yet finalized. Do not use in production.

## Goals

- Bit-exact AV2 decoding, matching dav2d output
- Memory safety for all parsing and decode logic (where most CVEs occur)
- Shared assembly optimizations with dav2d via FFI (x86 SSE/AVX2/AVX-512, ARM Neon, RISC-V, LoongArch)
- Drop-in C API compatibility for existing dav2d consumers
- Native Rust API for Rust ecosystem integration

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
- dav2d installed on system (via meson/ninja), or set `PKG_CONFIG_PATH`
- LLVM/clang (for bindgen)

### Build

```sh
# First, build and install dav2d
cd dav2d
mkdir build && cd build
meson setup .. && ninja
sudo ninja install

# Then build rav2d
cd ../..
cargo build
```

## License

BSD 2-Clause, same as dav2d.

## Acknowledgments

- [dav2d](https://code.videolan.org/videolan/dav2d) by VideoLAN / Alliance for Open Media
- [rav1d](https://github.com/memorysafety/rav1d) by Prossimo / ISRG — the reference for this porting approach
- [dav1d](https://code.videolan.org/videolan/dav1d) — the AV1 predecessor

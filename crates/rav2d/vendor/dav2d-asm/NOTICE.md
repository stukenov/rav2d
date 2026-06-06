# Vendored dav2d aarch64 assembly

The files under this directory are copied **verbatim** from
[dav2d](https://code.videolan.org/videolan/dav2d) (the AV2 reference decoder)
so that the published `rav2d` crate can assemble dav2d's hand-written NEON DSP
kernels without requiring the `dav2d` git submodule at build time.

When the `dav2d` submodule is present (workspace development), `build.rs` uses it
directly; otherwise it falls back to the copies here.

## Contents

| Vendored path                         | dav2d source                  |
| ------------------------------------- | ----------------------------- |
| `src/arm/64/mc.S`                     | `src/arm/64/mc.S`             |
| `src/arm/64/mc_dotprod.S`             | `src/arm/64/mc_dotprod.S`     |
| `src/arm/64/ipred.S`                  | `src/arm/64/ipred.S`          |
| `src/arm/64/util.S`                   | `src/arm/64/util.S`           |
| `src/arm/asm.S`                       | `src/arm/asm.S`               |
| `src/tables.c`                        | `src/tables.c`                |
| `include/common/attributes.h`         | `include/common/attributes.h` |
| `include/dav2d/headers.h`             | `include/dav2d/headers.h`     |
| `build/config.h`                      | generated `build/config.h`    |

`build.rs` extracts only the read-only data tables the kernels reference
(`dav2d_mc_subpel_filters`, `dav2d_mc_warp_filter`, `dav2d_sm_weights`,
`dav2d_filter_intra_taps`) out of `tables.c`.

## License

dav2d is licensed under the BSD 2-Clause License (see the upstream `COPYING`),
which is compatible with rav2d's own BSD-2-Clause license. The original
copyright notices in the vendored files are retained.

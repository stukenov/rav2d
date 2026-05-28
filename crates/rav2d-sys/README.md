# rav2d-sys

Raw FFI bindings to the [dav2d](https://code.videolan.org/videolan/dav2d) AV2 decoder C library.

Generated via [bindgen](https://github.com/rust-lang/rust-bindgen) from dav2d's C headers. Provides access to dav2d's assembly-optimized DSP kernels (motion compensation, inverse transforms, loop filters, etc.) which are shared with the pure-Rust `rav2d` decoder.

## What's Exposed

- DSP function tables (CDEF, loop restoration, film grain, MC, inverse transforms)
- CPU feature detection
- Low-level data structures matching dav2d's internal types

## Building

Requires dav2d to be built locally via meson:

```sh
cd dav2d
meson setup build
ninja -C build
```

The `build.rs` script runs bindgen against dav2d headers and links to the built library.

## License

BSD-2-Clause

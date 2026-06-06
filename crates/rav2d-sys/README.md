# rav2d-sys

Raw FFI bindings to the [dav2d](https://code.videolan.org/videolan/dav2d) AV2 decoder C library, generated via [bindgen](https://github.com/rust-lang/rust-bindgen) from dav2d's public `dav2d.h`.

These bindings link the dav2d C decoder so the pure-Rust [`rav2d`](../rav2d/) crate can run it side by side as the **bit-exact conformance reference** (re-exported as `rav2d::sys`). They wrap the public decode API — `dav2d_open` / `dav2d_send_data` / `dav2d_get_picture` / settings / picture types — not dav2d's internal DSP tables (those are not in the public header).

> The reusable NEON assembly dispatch lives in the [`rav2d`](../rav2d/) crate (`build.rs` + `mc_neon`/`ipred_neon`), which assembles dav2d's `.S` kernels directly; it is not exposed here.

## What's Exposed

- The dav2d public C decode API (`dav2d_*` functions)
- `Dav2dSettings`, `Dav2dContext`, `Dav2dData`, `Dav2dPicture` and related types
- `DAV2D_*` constants

## Building

Requires the dav2d submodule built locally via meson (the `build.rs` runs bindgen against its headers and links the built library):

```sh
cd dav2d
meson setup build
ninja -C build
```

## License

BSD-2-Clause

# rav2d

Memory-safe **AV2** video decoder library in Rust, ported from [dav2d](https://code.videolan.org/videolan/dav2d), the C reference decoder.

The output is checked byte for byte against dav2d on every frame of the conformance clips (8-bit and 10-bit, in-loop filters off and on), and against the md5 of `avmdec`, the AOM reference decoder, on vectors made with the AOM encoder. See the [repository README](https://github.com/stukenov/rav2d#status) for what is covered and what is not yet.

## Features

- Full OBU parsing: sequence, frame, tile group and show-existing-frame headers
- Intra: directional, smooth, Paeth, CfL, MHCCP, MRL, DIP, palette, IntraBC
- Inter: single reference, compound, warp, OBMC, inter-intra, BAWP, TIP, optical-flow refinement, scaled references
- In-loop filters: deblock, CDEF, CCSO, Wiener / PC-Wiener / GDF; film grain synthesis
- 8-bit and 10-bit, 4:2:0 and 4:0:0
- Frames come out in display order
- aarch64: dav2d's NEON assembly for 8-bit motion compensation and some intra modes, built from vendored sources; everything else runs Rust (`RAV2D_NEON_OFF=all` forces the Rust path)

No C library is linked and no submodule is needed: `cargo add rav2d` is enough. Requires Rust 1.88+.

## Usage

```rust
use rav2d::{Data, Decoder, Rav2dError, Settings};

let mut decoder = Decoder::open(&Settings::default())?;
decoder.send_data(Some(Data::wrap(std::fs::read("input.obu")?)))?;

loop {
    match decoder.get_picture() {
        Ok(pic) => {
            let (w, h) = pic.plane_dimensions(0).unwrap(); // luma, in samples
            for row in pic.plane_rows_u8(0) {
                // row: &[u8], `w` samples, no padding
            }
            // 10-bit frames: pic.plane_rows_u16(0) yields &[u16] rows
        }
        Err(Rav2dError::Again) => break, // needs more data
        Err(e) => return Err(e.into()),
    }
}
```

Decoded pixels are ordinary slices, so reading a frame needs no `unsafe`. Raw pointers stay available through `plane_ptr` / `plane_stride_bytes` for FFI.

## Public API

| Type | Description |
|------|-------------|
| `Decoder` | Decoder handle: `open`, `send_data`, `get_picture`, `flush` |
| `Settings` | Threads, film grain, operating point, frame size limit, filters |
| `Data` | Reference-counted compressed input |
| `Picture` | Decoded frame: plane accessors, headers, metadata |
| `Rav2dError` | `Again`, `Eof`, `InvalidData`, `FrameTooLarge`, `InvalidParam`, `OutOfMemory` |
| `SequenceHeader`, `FrameHeader` | Parsed headers |
| `PicAllocator` | `unsafe trait` for supplying picture memory |
| `Logger` | Logging callback |

## Status

Multithreading covers only the output copy and film grain; parsing, reconstruction and filters run on one thread. On x86 there is no assembly. 12-bit streams have a code path but no test vector. Details and the list of vectors that still differ from `avmdec` are in the [repository README](https://github.com/stukenov/rav2d#status).

## License

BSD-2-Clause, same as dav2d.

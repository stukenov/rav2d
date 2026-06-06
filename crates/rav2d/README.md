# rav2d

Memory-safe **AV2** video decoder library in Rust, ported from [dav2d](https://code.videolan.org/videolan/dav2d). **Bit-exact** with the dav2d C reference across the full conformance corpus (8-bit, filters off and on) and 10-bit streams.

## Features

- Full OBU bitstream parsing (sequence, frame, tile group headers)
- Entropy decoding (MSAC) with CDF adaptation
- Intra (CfL, MHCCP, MRL, DIP, palette, IntraBC) and inter (compound, warp-affine, OBMC, interintra, BAWP, TIP, OPFL) block decoding
- Complete filter pipeline: deblock, CDEF, CCSO, loop restoration (Wiener / PC-Wiener / GDF), film grain
- Motion compensation with optical-flow refinement
- Inverse transforms (all sizes/types), segmentation, delta-Q, lossless
- High bit depth (10-bit) decode, bit-exact
- Reference frame management and motion vector prediction
- aarch64 NEON dispatch (motion compensation + intra H/V/smooth) via FFI to dav2d, scalar Rust fallback (`RAV2D_NEON_OFF=all`)

## Usage

```rust
use rav2d::{Decoder, Settings, Data, Rav2dError};

let mut decoder = Decoder::open(&Settings::default())?;

// Feed OBU data
decoder.send_data(Some(Data::wrap(obu_bytes)))?;

// Get decoded frames
loop {
    match decoder.get_picture() {
        Ok(pic) => {
            println!("{}x{} bpc={}", pic.p.w, pic.p.h, pic.p.bpc);
            // Access pixel data via pic.data[0..3] (Y, U, V planes)
        }
        Err(Rav2dError::Again) => break,
        Err(e) => return Err(e.into()),
    }
}
```

## Public API

| Type | Description |
|------|-------------|
| `Decoder` | Main decoder handle — open, send_data, get_picture, flush |
| `Settings` | Configuration: threads, film grain, operating point |
| `Data` | Reference-counted byte buffer for compressed input |
| `Picture` | Decoded frame with pixel planes, headers, and metadata |
| `Rav2dError` | Error enum: Eof, Again, InvalidData, FrameTooLarge, InvalidParam, OutOfMemory |
| `SequenceHeader` | Parsed sequence-level parameters |
| `FrameHeader` | Parsed frame-level parameters |
| `Logger` | Configurable logging callback |
| `PicAllocator` | Custom picture memory allocator trait |

## Building

Requires dav2d built locally (for FFI assembly kernels):

```sh
# macOS
DYLD_LIBRARY_PATH=../../dav2d/build/src cargo test -p rav2d

# Linux
LD_LIBRARY_PATH=../../dav2d/build/src cargo test -p rav2d
```

## License

BSD-2-Clause

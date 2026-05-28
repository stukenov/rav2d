# rav2d

AV2 video decoder library in Rust, ported from [dav2d](https://code.videolan.org/videolan/dav2d).

## Features

- Full OBU bitstream parsing (sequence, frame, tile group headers)
- Entropy decoding (MSAC) with CDF adaptation
- Intra and inter block decoding with compound prediction
- Complete filter pipeline: deblock, CDEF, loop restoration (Wiener/GDF), film grain
- Motion compensation with optical flow refinement
- Inverse transforms (all sizes/types)
- Reference frame management and motion vector prediction
- Assembly-optimized DSP kernels via FFI to dav2d

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

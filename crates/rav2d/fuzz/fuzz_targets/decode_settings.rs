//! Same contract as the `decode` target — no panic, no sanitizer fault, errors
//! on malformed input are correct — but the decoder is opened with settings
//! derived from the first two bytes of the input.
//!
//! The plain `decode` target only ever exercises the default configuration, so
//! whole regions of the decoder never see a fuzzer: the film-grain output path,
//! header-only parsing, invisible-frame output, the per-filter combinations, and
//! the frame-type filters. Letting the fuzzer pick those reaches them, and lets
//! it find inputs that only misbehave under one particular configuration.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rav2d::{Data, DecodeFrameType, Decoder, InloopFilterType, Rav2dError, Settings};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let (cfg, stream) = data.split_at(2);
    let seed = u16::from_le_bytes([cfg[0], cfg[1]]);

    let settings = Settings {
        // Two threads exercise the parallel output/grain path; more would only
        // slow the fuzzer down without reaching new code.
        n_threads: 1 + ((seed >> 12) & 1) as u32,
        apply_grain: seed & 1 != 0,
        output_invisible_frames: seed & 2 != 0,
        all_layers: seed & 4 != 0,
        strict_std_compliance: seed & 8 != 0,
        // Header-only parsing is a supported mode and has its own error paths.
        run_decode: seed & 16 == 0,
        inloop_filters: match (seed >> 5) & 7 {
            0 => InloopFilterType::None,
            1 => InloopFilterType::Deblock,
            2 => InloopFilterType::Cdef,
            3 => InloopFilterType::Restoration,
            4 => InloopFilterType::Wiener,
            5 => InloopFilterType::Gdf,
            _ => InloopFilterType::All,
        },
        decode_frame_type: match (seed >> 8) & 3 {
            0 => DecodeFrameType::All,
            1 => DecodeFrameType::Reference,
            2 => DecodeFrameType::Intra,
            _ => DecodeFrameType::Key,
        },
        operating_point: ((seed >> 10) & 3) as u32,
        // As in the `decode` target: a caller that cares about memory sets a
        // limit, so an absurd declared frame size is rejected rather than
        // allocated.
        frame_size_limit: 8192 * 8192,
        ..Settings::default()
    };

    let Ok(mut decoder) = Decoder::open(&settings) else {
        return;
    };

    if decoder
        .send_data(Some(Data::wrap(stream.to_vec())))
        .is_err()
    {
        return;
    }

    let mut drained = false;
    loop {
        match decoder.get_picture() {
            Ok(_picture) => {}
            Err(Rav2dError::Again) if !drained => {
                drained = true;
                if decoder.send_data(None).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
});

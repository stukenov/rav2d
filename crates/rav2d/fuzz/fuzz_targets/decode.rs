//! Feeds arbitrary bytes to the decoder as a raw OBU stream and drains all
//! output. Any panic, OOM, or sanitizer fault is a finding; decode errors on
//! malformed input are the expected, correct outcome.
//!
//! A `frame_size_limit` is set (as any memory-conscious application should),
//! so a malformed stream declaring an enormous frame is rejected with
//! `FrameTooLarge` instead of triggering a by-design giant allocation. Without
//! it the decoder mirrors dav2d's default (0 = unlimited) and would OOM on such
//! input — that is the caller's responsibility, not a decoder defect.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rav2d::{Data, Decoder, Rav2dError, Settings};

fuzz_target!(|data: &[u8]| {
    let mut settings = Settings::default();
    settings.frame_size_limit = 8192 * 8192;
    let Ok(mut decoder) = Decoder::open(&settings) else {
        return;
    };

    if decoder.send_data(Some(Data::wrap(data.to_vec()))).is_err() {
        return;
    }

    let mut drained = false;
    loop {
        match decoder.get_picture() {
            Ok(_picture) => {}
            Err(Rav2dError::Again) if !drained => {
                // No more input to give: signal end-of-stream and drain
                // any frames still buffered for reorder delay.
                drained = true;
                if decoder.send_data(None).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
});

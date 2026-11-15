// clippy pedantic exception policy for rav2d (audit `-D clippy::pedantic`).
// rav2d is a faithful port of dav2d (C): the mechanical findings (~350 sites)
// were already fixed; the groups below are intentional port-of-dav2d idioms
// and remaining MaybeIncorrect-suggestion stragglers (Phase-2 candidates).
// Source attributes are used instead of cargo [lints] tables because cargo
// injects the latter as CLI --allow flags that a later `-D` would override.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::ptr_cast_constness)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::inline_always)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::stable_sort_primitive)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::decimal_bitwise_operands)]
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::single_match_else)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unnecessary_box_returns)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::format_push_string)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::same_functions_in_if_condition)]
#![allow(clippy::manual_slice_fill)]
#![allow(clippy::manual_is_variant_and)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::assert_is_empty)]
#![allow(clippy::manual_assert)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::format_collect)]

//! Deterministic robustness ("fuzz") harness: a memory-safe decoder must never
//! panic or crash on malformed input — it must return a graceful error. This
//! feeds the decoder (1) pure pseudo-random bytes and (2) bit/byte-mutated and
//! truncated copies of the valid conformance clips, and asserts every input is
//! handled without a panic. The PRNG is seeded, so any failure is reproducible
//! (the seed + mutation are printed).
//!
//! Full decode is enabled (`run_decode = true`) so the reconstruction path —
//! where most `unsafe` pixel handling lives — is exercised, not just the parser.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use rav2d::{Data, DecodeFrameType, Decoder, InloopFilterType, Rav2dError, Settings};

fn media(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/media")).join(name)
}
fn data(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data")).join(name)
}

/// xorshift64* — small deterministic PRNG (no external dep).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
}

/// Settings for the deterministic sweeps in this file.
fn sweep_settings() -> Settings {
    Settings {
        n_threads: 1,
        apply_grain: false,
        run_decode: true,
        // Match the fuzz target: cap frame size so a malformed stream
        // declaring an enormous frame is rejected (FrameTooLarge) rather
        // than allocating gigabytes. This is what a memory-conscious
        // application does.
        frame_size_limit: 8192 * 8192,
        ..Settings::default()
    }
}

/// The settings the `decode` fuzz target opens the decoder with.
fn decode_target_settings() -> Settings {
    Settings {
        frame_size_limit: 8192 * 8192,
        ..Settings::default()
    }
}

/// The settings the `decode_settings` fuzz target derives from the first two
/// bytes of its input. Keep in sync with `fuzz/fuzz_targets/decode_settings.rs`:
/// a reproducer from that target only crashes under the configuration it chose.
fn decode_settings_target_settings(seed: u16) -> Settings {
    Settings {
        n_threads: 1 + ((seed >> 12) & 1) as u32,
        apply_grain: seed & 1 != 0,
        output_invisible_frames: seed & 2 != 0,
        all_layers: seed & 4 != 0,
        strict_std_compliance: seed & 8 != 0,
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
        frame_size_limit: 8192 * 8192,
        ..Settings::default()
    }
}

/// Decode `bytes` to completion. Returns `Err(panic_msg)` if it panicked,
/// `Ok(())` if it finished (decoded or returned a graceful error).
fn decode_catch(bytes: Vec<u8>) -> Result<(), String> {
    decode_catch_with(bytes, &sweep_settings())
}

fn decode_catch_with(bytes: Vec<u8>, s: &Settings) -> Result<(), String> {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut dec = match Decoder::open(s) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut sent = false;
        let mut frames = 0u32;
        loop {
            if !sent {
                match dec.send_data(Some(Data::wrap(bytes.clone()))) {
                    Ok(()) => sent = true,
                    Err(Rav2dError::Again) => {}
                    Err(_) => break,
                }
            }
            match dec.get_picture() {
                Ok(_) => {
                    frames += 1;
                    if frames > 64 {
                        break; // adversarial loop guard
                    }
                }
                Err(Rav2dError::Again) => {
                    if sent {
                        let _ = dec.send_data(None);
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }));
    res.map_err(|e| {
        e.downcast_ref::<&str>()
            .map(std::string::ToString::to_string)
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic>".to_string())
    })
}

fn corpus() -> Vec<Vec<u8>> {
    let names_media = [
        "avm-v14.1.0-bus.64x64.l5.obu",
        "avm-v14.1.0-bus.352x288.l5.seg1.obu",
        "avm-v14.1.0-hm.64x64.l5.filmgrain.obu",
    ];
    let names_data = ["cov-monochrome-128x128.obu", "cov-multitile-416x240.obu"];
    let mut v = Vec::new();
    for n in names_media {
        if let Ok(b) = std::fs::read(media(n)) {
            v.push(b);
        }
    }
    for n in names_data {
        if let Ok(b) = std::fs::read(data(n)) {
            v.push(b);
        }
    }
    v
}

/// Pure random byte streams of varied lengths must not panic the decoder.
#[test]
fn fuzz_random_bytes_no_panic() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for iter in 0..6000u64 {
        let len = rng.below(4096);
        let buf: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        if let Err(msg) = decode_catch(buf) {
            panic!("panic on random input (iter {iter}, len {len}): {msg}");
        }
    }
}

/// Bit/byte-flip, truncation and region-zeroing mutations of valid streams must
/// not panic — they should decode or fail gracefully.
#[test]
fn fuzz_mutated_streams_no_panic() {
    let corpus = corpus();
    assert!(!corpus.is_empty(), "no corpus clips found");
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for (ci, base) in corpus.iter().enumerate() {
        if base.is_empty() {
            continue;
        }
        for iter in 0..2500u64 {
            let mut b = base.clone();
            match rng.below(4) {
                // single/multi byte flips
                0 => {
                    for _ in 0..=rng.below(8) {
                        let i = rng.below(b.len());
                        b[i] ^= 1 << rng.below(8);
                    }
                }
                // random byte overwrites
                1 => {
                    for _ in 0..=rng.below(16) {
                        let i = rng.below(b.len());
                        b[i] = rng.byte();
                    }
                }
                // truncate
                2 => {
                    let keep = rng.below(b.len());
                    b.truncate(keep);
                }
                // zero a region (corrupt a payload run)
                _ => {
                    let start = rng.below(b.len());
                    let end = (start + 1 + rng.below(64)).min(b.len());
                    for x in &mut b[start..end] {
                        *x = 0;
                    }
                }
            }
            if let Err(msg) = decode_catch(b) {
                panic!("panic on mutated clip {ci} (iter {iter}): {msg}");
            }
        }
    }
}

/// Replay every fuzzer-discovered crashing input (checked into
/// `tests/data/fuzz-regressions/`). Each was a real panic on malformed input
/// that has since been fixed; this guards against reintroducing any of them.
///
/// Inputs at the top level came from the `decode` target and are replayed with
/// its settings as well as the sweep settings. Inputs under `decode_settings/`
/// carry the decoder configuration in their first two bytes, exactly as that
/// target reads them, so they are replayed under that configuration.
#[test]
fn fuzz_regression_corpus_no_panic() {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/fuzz-regressions"
    ));
    let mut n = 0;
    for path in regression_inputs(&dir) {
        let bytes = std::fs::read(&path).unwrap();
        for s in [sweep_settings(), decode_target_settings()] {
            if let Err(msg) = decode_catch_with(bytes.clone(), &s) {
                panic!("regression: {} still panics: {msg}", path.display());
            }
        }
        n += 1;
    }
    for path in regression_inputs(&dir.join("decode_settings")) {
        let bytes = std::fs::read(&path).unwrap();
        if bytes.len() < 2 {
            continue;
        }
        let (cfg, stream) = bytes.split_at(2);
        let s = decode_settings_target_settings(u16::from_le_bytes([cfg[0], cfg[1]]));
        if let Err(msg) = decode_catch_with(stream.to_vec(), &s) {
            panic!("regression: {} still panics: {msg}", path.display());
        }
        n += 1;
    }
    assert!(n > 0, "no regression inputs found in {}", dir.display());
    eprintln!("fuzz_regression_corpus_no_panic: {n} inputs replayed cleanly");
}

fn regression_inputs(dir: &std::path::Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    paths
}

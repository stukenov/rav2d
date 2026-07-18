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

use rav2d::{Data, Decoder, Rav2dError, Settings};

fn media(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../dav2d/media")).join(name)
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

/// Decode `bytes` to completion. Returns `Err(panic_msg)` if it panicked,
/// `Ok(())` if it finished (decoded or returned a graceful error).
fn decode_catch(bytes: Vec<u8>) -> Result<(), String> {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut s = Settings::default();
        s.n_threads = 1;
        s.apply_grain = false;
        s.run_decode = true;
        // Match the fuzz target: cap frame size so a malformed stream declaring
        // an enormous frame is rejected (FrameTooLarge) rather than allocating
        // gigabytes. This is what a memory-conscious application does.
        s.frame_size_limit = 8192 * 8192;
        let mut dec = match Decoder::open(&s) {
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
            .map(|s| s.to_string())
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
                    for _ in 0..1 + rng.below(8) {
                        let i = rng.below(b.len());
                        b[i] ^= 1 << rng.below(8);
                    }
                }
                // random byte overwrites
                1 => {
                    for _ in 0..1 + rng.below(16) {
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
#[test]
fn fuzz_regression_corpus_no_panic() {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/fuzz-regressions"
    ));
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut n = 0;
    for entry in entries {
        let path = entry.unwrap().path();
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        if let Err(msg) = decode_catch(bytes) {
            panic!("regression: {} still panics: {msg}", path.display());
        }
        n += 1;
    }
    assert!(n > 0, "no regression inputs found in {}", dir.display());
    eprintln!("fuzz_regression_corpus_no_panic: {n} inputs replayed cleanly");
}

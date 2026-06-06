//! Decode throughput benchmark: rav2d (pure Rust) vs dav2d (C reference, via the
//! `rav2d::sys` FFI bindings).
//!
//! Run with the dav2d dylib on the loader path:
//! ```text
//! DYLD_LIBRARY_PATH=$PWD/dav2d/build/src cargo bench -p rav2d
//! ```
//!
//! Two outputs:
//!   * a one-shot comparison table (printed once at startup) with median decode
//!     time, megapixels/s, and the dav2d/rav2d speedup ratio per clip;
//!   * criterion benchmark groups (`rav2d/<clip>`, `dav2d/<clip>`) reporting
//!     per-iteration time and `Throughput::Elements` (samples/s) so the HTML
//!     report can be diffed function-by-function.
//!
//! NOTE on fairness: dav2d is C with hand-written SIMD (NEON/AVX2). On aarch64
//! rav2d now wires the motion-compensation DSP family to dav2d's NEON kernels
//! (see `src/mc_neon.rs`); other DSP families (itx, intra, loop filters, …) are
//! still scalar Rust, so the ratio below mixes NEON MC with scalar everything
//! else and tightens further as more families are wired. Set `RAV2D_NEON_OFF=all`
//! to force the all-scalar path. Both run single-threaded, film grain off.

use std::path::PathBuf;
use std::sync::Once;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Clips that decode end-to-end in rav2d today (keyframe + early inter frames).
/// Larger / TIP-heavy clips are added as the decoder gains coverage.
const CLIPS: &[&str] = &[
    "avm-v14.1.0-bus.64x64.l5.obu",
    "avm-v14.1.0-bus.64x64.l5.lossless.obu",
    "avm-v14.1.0-bus.64x64.l5.opfl0-refinemv0.obu",
    "avm-v14.1.0-bus.64x64.l1.sdp0.obu",
    "avm-v14.1.0-bus.64x64.l1.sdp1.obu",
    "avm-v14.1.0-bus.352x288.l5.seg1.obu",
    "avm-v14.1.0-bus.352x288.l10.deltaq1.obu",
    "avm-v14.1.0-bus.352x288.l1.partial_lossless.obu",
    "avm-v14.1.0-hm.64x64.l5.filmgrain.obu",
];

#[cfg(target_os = "macos")]
const EAGAIN: i32 = -35;
#[cfg(not(target_os = "macos"))]
const EAGAIN: i32 = -11;

fn media(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../dav2d/media")).join(name)
}

fn ss(layout: i32) -> (i32, i32) {
    match layout {
        1 => (1, 1), // 420
        2 => (1, 0), // 422
        _ => (0, 0), // 444 / monochrome
    }
}

/// Total luma+chroma samples produced by a decoded frame (used for throughput).
fn frame_samples(w: i32, h: i32, layout: i32) -> u64 {
    let (ssh, ssv) = ss(layout);
    let luma = (w as u64) * (h as u64);
    if layout == 0 {
        return luma;
    }
    let cw = ((w + ssh) >> ssh) as u64;
    let ch = ((h + ssv) >> ssv) as u64;
    luma + 2 * cw * ch
}

/// Decode a clip with rav2d. Returns (frames decoded, total samples produced).
fn rav2d_run(bytes: &[u8], filters: rav2d::InloopFilterType) -> (u32, u64) {
    use rav2d::{Data, Decoder, Settings};
    let mut s = Settings::default();
    s.n_threads = 1;
    s.apply_grain = false;
    s.run_decode = true;
    s.inloop_filters = filters;
    let mut dec = match Decoder::open(&s) {
        Ok(d) => d,
        Err(_) => return (0, 0),
    };

    let mut frames = 0u32;
    let mut samples = 0u64;
    let mut sent = false;
    loop {
        if !sent {
            match dec.send_data(Some(Data::wrap(bytes.to_vec()))) {
                Ok(()) => sent = true,
                Err(rav2d::Rav2dError::Again) => {}
                Err(_) => break,
            }
        }
        match dec.get_picture() {
            Ok(pic) => {
                frames += 1;
                samples += frame_samples(pic.p.w, pic.p.h, pic.p.layout as i32);
            }
            Err(rav2d::Rav2dError::Again) => {
                if sent {
                    let _ = dec.send_data(None);
                } else {
                    break;
                }
            }
            Err(_) => break,
        }
        if frames > 4096 {
            break;
        }
    }
    (frames, samples)
}

/// Decode a clip with the dav2d C reference. Returns (frames, total samples).
fn dav2d_run(bytes: &[u8], inloop_filters: u32) -> (u32, u64) {
    use rav2d::sys;
    let mut frames = 0u32;
    let mut samples = 0u64;
    unsafe {
        let mut settings: sys::Dav2dSettings = std::mem::zeroed();
        sys::dav2d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.apply_grain = 0;
        settings.inloop_filters = inloop_filters;
        // rav2d emits every decoded frame in coding order; match that so the
        // frame counts (and thus the work compared) line up.
        settings.output_invisible_frames = 1;

        let mut ctx: *mut sys::Dav2dContext = std::ptr::null_mut();
        if sys::dav2d_open(&mut ctx, &settings) != 0 {
            return (0, 0);
        }

        let mut d: sys::Dav2dData = std::mem::zeroed();
        let buf = sys::dav2d_data_create(&mut d, bytes.len());
        assert!(!buf.is_null(), "dav2d_data_create failed");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());

        let mut drained = false;
        loop {
            loop {
                let mut pic: sys::Dav2dPicture = std::mem::zeroed();
                let pr = sys::dav2d_get_picture(ctx, &mut pic);
                if pr == EAGAIN || pr < 0 {
                    break;
                }
                frames += 1;
                samples += frame_samples(pic.p.w, pic.p.h, pic.p.layout as i32);
                sys::dav2d_picture_unref(&mut pic);
            }
            if d.sz > 0 {
                let sr = sys::dav2d_send_data(ctx, &mut d);
                if sr < 0 && sr != EAGAIN {
                    break;
                }
            } else if !drained {
                drained = true;
                sys::dav2d_send_data(ctx, std::ptr::null_mut());
            } else {
                let mut pic: sys::Dav2dPicture = std::mem::zeroed();
                if sys::dav2d_get_picture(ctx, &mut pic) == 0 {
                    frames += 1;
                    samples += frame_samples(pic.p.w, pic.p.h, pic.p.layout as i32);
                    sys::dav2d_picture_unref(&mut pic);
                } else {
                    break;
                }
            }
        }
        sys::dav2d_close(&mut ctx);
    }
    (frames, samples)
}

/// Median wall-clock of `iters` decode runs, plus frames/samples from one run.
fn median_decode(
    bytes: &[u8],
    iters: usize,
    run: impl Fn(&[u8]) -> (u32, u64),
) -> (Duration, u32, u64) {
    let (frames, samples) = run(bytes);
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = black_box(run(black_box(bytes)));
        times.push(t.elapsed());
    }
    times.sort();
    (times[times.len() / 2], frames, samples)
}

/// Print a once-only rav2d-vs-dav2d comparison table to stderr.
fn print_comparison_table() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!();
        eprintln!("=== rav2d vs dav2d decode throughput (single-thread, filters off) ===");
        eprintln!(
            "{:<46} {:>6} {:>10} {:>10} {:>9} {:>9} {:>8}",
            "clip", "frames", "rav2d ms", "dav2d ms", "rav MP/s", "dav MP/s", "dav/rav"
        );
        for name in CLIPS {
            let path = media(name);
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => {
                    eprintln!("{name:<46} (missing)");
                    continue;
                }
            };
            let iters = 15;
            let (r_ms, r_frames, r_samples) = median_decode(&bytes, iters, |b| {
                rav2d_run(b, rav2d::InloopFilterType::None)
            });
            let (d_ms, d_frames, d_samples) = median_decode(&bytes, iters, |b| dav2d_run(b, 0));

            let r_s = r_ms.as_secs_f64();
            let d_s = d_ms.as_secs_f64();
            let r_mps = if r_s > 0.0 {
                r_samples as f64 / 1e6 / r_s
            } else {
                0.0
            };
            let d_mps = if d_s > 0.0 {
                d_samples as f64 / 1e6 / d_s
            } else {
                0.0
            };
            let ratio = if r_s > 0.0 { d_s / r_s } else { 0.0 };
            let frames = if r_frames == d_frames {
                format!("{r_frames}")
            } else {
                format!("{r_frames}/{d_frames}")
            };
            eprintln!(
                "{:<46} {:>6} {:>10.3} {:>10.3} {:>9.1} {:>9.1} {:>7.2}x",
                name,
                frames,
                r_s * 1e3,
                d_s * 1e3,
                r_mps,
                d_mps,
                ratio
            );
        }
        eprintln!(
            "(frames shown as rav2d/dav2d when they differ — rav2d still gaining inter coverage)"
        );
        eprintln!("(dav2d uses SIMD; rav2d uses NEON for MC on aarch64, scalar elsewhere)");
        eprintln!();
    });
}

fn bench_decode(c: &mut Criterion) {
    print_comparison_table();

    let mut grp = c.benchmark_group("decode");
    for name in CLIPS {
        let path = media(name);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let (_, samples) = rav2d_run(&bytes, rav2d::InloopFilterType::None);
        if samples > 0 {
            grp.throughput(Throughput::Elements(samples));
        }
        grp.bench_with_input(format!("rav2d/{name}"), &bytes, |b, bytes| {
            b.iter(|| black_box(rav2d_run(black_box(bytes), rav2d::InloopFilterType::None)));
        });
        grp.bench_with_input(format!("dav2d/{name}"), &bytes, |b, bytes| {
            b.iter(|| black_box(dav2d_run(black_box(bytes), 0)));
        });
    }
    grp.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);

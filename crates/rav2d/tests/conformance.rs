//! Bit-exact conformance harness: decode a clip with the dav2d C reference (via
//! the rav2d-sys FFI bindings) and with rav2d, and compare the output planes
//! frame-by-frame.
//!
//! This is the acceptance gate reused by every reconstruction milestone. Run
//! under the dav2d shared library, per CLAUDE.md:
//!
//!   DYLD_LIBRARY_PATH=$PWD/dav2d/build/src cargo test -p rav2d --test conformance
//!
//! The full bit-exact comparison (`bit_exact_*`) is `#[ignore]`d until rav2d
//! reconstruction produces pixels; the un-ignored test validates the reference
//! decode path and the harness itself.

use std::path::PathBuf;

/// One decoded frame's planes, with stride padding stripped (tightly packed
/// rows), so two decoders' outputs are directly comparable regardless of stride.
#[derive(Clone)]
pub struct FramePlanes {
    pub w: i32,
    pub h: i32,
    pub bpc: i32,
    pub layout: i32,
    /// Y, U, V (U/V empty for monochrome). Bytes, row-packed.
    pub planes: [Vec<u8>; 3],
}

fn media(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../dav2d/media")).join(name)
}

/// Chroma subsampling (ss_hor, ss_ver) for a dav2d pixel layout.
/// I400=0, I420=1, I422=2, I444=3.
fn ss(layout: i32) -> (i32, i32) {
    match layout {
        1 => (1, 1), // I420
        2 => (1, 0), // I422
        3 => (0, 0), // I444
        _ => (0, 0), // I400 (no chroma)
    }
}

// ---------------------------------------------------------------------------
// dav2d C reference decode (in-process via rav2d-sys)
// ---------------------------------------------------------------------------

// EAGAIN on Darwin is 35; dav2d returns DAV2D_ERR(e) = -e. (Linux would be -11.)
#[cfg(target_os = "macos")]
const EAGAIN: i32 = -35;
#[cfg(not(target_os = "macos"))]
const EAGAIN: i32 = -11;

unsafe fn extract_planes(pic: &rav2d::sys::Dav2dPicture) -> FramePlanes {
    let w = pic.p.w;
    let h = pic.p.h;
    let bpc = pic.p.bpc;
    let layout = pic.p.layout as i32;
    let bytes_per_sample = if bpc > 8 { 2usize } else { 1usize };
    let (ssh, ssv) = ss(layout);

    let mut planes: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for pl in 0..3 {
        if pl > 0 && layout == 0 {
            break; // monochrome: no chroma
        }
        let pw = if pl == 0 { w } else { (w + ssh) >> ssh };
        let ph = if pl == 0 { h } else { (h + ssv) >> ssv };
        let stride = pic.stride[if pl == 0 { 0 } else { 1 }];
        let base = pic.data[pl] as *const u8;
        if base.is_null() {
            continue;
        }
        let row_bytes = pw as usize * bytes_per_sample;
        let mut buf = Vec::with_capacity(row_bytes * ph as usize);
        for y in 0..ph as isize {
            let row = unsafe { base.offset(y * stride) };
            let slice = unsafe { std::slice::from_raw_parts(row, row_bytes) };
            buf.extend_from_slice(slice);
        }
        planes[pl] = buf;
    }

    FramePlanes {
        w,
        h,
        bpc,
        layout,
        planes,
    }
}

/// Decode a clip with the dav2d C reference library. Returns one `FramePlanes`
/// per output frame.
pub fn dav2d_decode(path: &PathBuf) -> Vec<FramePlanes> {
    use rav2d::sys;
    let data = std::fs::read(path).expect("read clip");
    let mut frames = Vec::new();

    unsafe {
        let mut settings: sys::Dav2dSettings = std::mem::zeroed();
        sys::dav2d_default_settings(&mut settings);
        // Deterministic single-threaded reference. Disable in-loop filters and
        // film grain so the reference is PURE reconstruction — rav2d has no
        // post-filters yet, so comparing pre-filter pixels isolates recon.
        settings.n_threads = 1;
        settings.apply_grain = 0;
        settings.inloop_filters = 0; // DAV2D_INLOOPFILTER_NONE

        let mut ctx: *mut sys::Dav2dContext = std::ptr::null_mut();
        let r = sys::dav2d_open(&mut ctx, &settings);
        assert_eq!(r, 0, "dav2d_open failed: {r}");

        // dav2d-owned data buffer.
        let mut d: sys::Dav2dData = std::mem::zeroed();
        let buf = sys::dav2d_data_create(&mut d, data.len());
        assert!(!buf.is_null(), "dav2d_data_create failed");
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len());

        let mut drained = false;
        loop {
            // Drain all currently-available pictures.
            loop {
                let mut pic: sys::Dav2dPicture = std::mem::zeroed();
                let pr = sys::dav2d_get_picture(ctx, &mut pic);
                if pr == EAGAIN {
                    break;
                }
                if pr < 0 {
                    break;
                }
                frames.push(extract_planes(&pic));
                sys::dav2d_picture_unref(&mut pic);
            }

            if d.sz > 0 {
                let sr = sys::dav2d_send_data(ctx, &mut d);
                if sr < 0 && sr != EAGAIN {
                    panic!("dav2d_send_data failed: {sr}");
                }
            } else if !drained {
                drained = true;
                // signal end-of-stream
                sys::dav2d_send_data(ctx, std::ptr::null_mut());
            } else {
                // final drain pass already done above with no new pictures
                let mut pic: sys::Dav2dPicture = std::mem::zeroed();
                let pr = sys::dav2d_get_picture(ctx, &mut pic);
                if pr == 0 {
                    frames.push(extract_planes(&pic));
                    sys::dav2d_picture_unref(&mut pic);
                } else {
                    break;
                }
            }
        }

        sys::dav2d_close(&mut ctx);
    }

    frames
}

// ---------------------------------------------------------------------------
// rav2d decode
// ---------------------------------------------------------------------------

/// Decode a clip with rav2d. Returns one `FramePlanes` per output frame.
/// (Until reconstruction + output queueing land, this yields no frames.)
pub fn rav2d_decode(path: &PathBuf) -> Vec<FramePlanes> {
    use rav2d::{Data, Decoder, Settings};
    let bytes = std::fs::read(path).expect("read clip");
    let mut s = Settings::default();
    s.n_threads = 1;
    s.apply_grain = false;
    s.run_decode = true;
    let mut dec = Decoder::open(&s).expect("open");

    let mut frames = Vec::new();
    let mut sent = false;
    loop {
        if !sent {
            match dec.send_data(Some(Data::wrap(bytes.clone()))) {
                Ok(()) => sent = true,
                Err(rav2d::Rav2dError::Again) => {}
                Err(_) => break,
            }
        }
        match dec.get_picture() {
            Ok(pic) => frames.push(rav2d_picture_planes(&pic)),
            Err(rav2d::Rav2dError::Again) => {
                if sent {
                    let _ = dec.send_data(None);
                } else {
                    break;
                }
            }
            Err(_) => break,
        }
        if frames.len() > 4096 {
            break; // safety
        }
    }
    frames
}

fn rav2d_picture_planes(pic: &rav2d::Picture) -> FramePlanes {
    let w = pic.p.w;
    let h = pic.p.h;
    let bpc = pic.p.bpc;
    let layout = pic.p.layout as i32;
    let bytes_per_sample = if bpc > 8 { 2usize } else { 1usize };
    let (ssh, ssv) = ss(layout);

    let mut planes: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for pl in 0..3 {
        if pl > 0 && layout == 0 {
            break;
        }
        let pw = if pl == 0 { w } else { (w + ssh) >> ssh };
        let ph = if pl == 0 { h } else { (h + ssv) >> ssv };
        let stride = pic.stride[if pl == 0 { 0 } else { 1 }];
        let base = match pic.data[pl] {
            Some(p) => p.as_ptr() as *const u8,
            None => continue,
        };
        let row_bytes = pw as usize * bytes_per_sample;
        let mut buf = Vec::with_capacity(row_bytes * ph as usize);
        for y in 0..ph as isize {
            let row = unsafe { base.offset(y * stride) };
            let slice = unsafe { std::slice::from_raw_parts(row, row_bytes) };
            buf.extend_from_slice(slice);
        }
        planes[pl] = buf;
    }

    FramePlanes {
        w,
        h,
        bpc,
        layout,
        planes,
    }
}

/// Assert two decode results are byte-identical, plane by plane.
pub fn assert_bit_exact(reference: &[FramePlanes], got: &[FramePlanes], clip: &str) {
    assert_eq!(
        reference.len(),
        got.len(),
        "{clip}: frame count mismatch (dav2d={}, rav2d={})",
        reference.len(),
        got.len()
    );
    for (i, (r, g)) in reference.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            (r.w, r.h, r.bpc, r.layout),
            (g.w, g.h, g.bpc, g.layout),
            "{clip}: frame {i} params differ"
        );
        for pl in 0..3 {
            assert_eq!(
                r.planes[pl], g.planes[pl],
                "{clip}: frame {i} plane {pl} bytes differ"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// Validates the dav2d C reference decode path and the harness extraction.
/// (Does not exercise rav2d reconstruction yet.)
#[test]
fn dav2d_reference_decodes_keyframe_clip() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    if !path.exists() {
        eprintln!("skip: {path:?} not found");
        return;
    }
    let frames = dav2d_decode(&path);
    assert!(!frames.is_empty(), "dav2d reference produced no frames");
    let f0 = &frames[0];
    assert_eq!((f0.w, f0.h), (64, 64), "unexpected dims");
    assert!(f0.bpc == 8 || f0.bpc == 10, "unexpected bpc {}", f0.bpc);
    assert!(!f0.planes[0].is_empty(), "empty luma plane");
}

/// M1 gate: frame-0 LUMA must match the dav2d reference (in-loop filters off,
/// so pure reconstruction). Chroma + later frames + filters are follow-ups.
#[test]
fn bit_exact_keyframe_luma() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    if !path.exists() {
        eprintln!("skip: {path:?} not found");
        return;
    }
    let reference = dav2d_decode(&path);
    let got = rav2d_decode(&path);
    assert!(!got.is_empty(), "rav2d produced no frames");
    assert!(!reference.is_empty(), "dav2d produced no frames");
    assert_eq!(
        (reference[0].w, reference[0].h),
        (got[0].w, got[0].h),
        "frame 0 dims differ"
    );

    let r = &reference[0].planes[0];
    let g = &got[0].planes[0];
    assert_eq!(r.len(), g.len(), "frame 0 luma size differs");
    let diff = r.iter().zip(g.iter()).filter(|(a, b)| a != b).count();
    if diff != 0 {
        let first = r.iter().zip(g.iter()).position(|(a, b)| a != b).unwrap();
        let stride = reference[0].w as usize;
        panic!(
            "frame 0 luma differs in {diff}/{} bytes; first @ ({},{}) ref={} got={}",
            r.len(),
            first % stride,
            first / stride,
            r[first],
            g[first]
        );
    }
}

/// M1b gate: frame-0 ALL PLANES (Y+U+V) must match the dav2d reference
/// (in-loop filters off, film grain off), validating intra chroma reconstruction
/// including CfL (explicit/implicit/MHCCP) and CCTX.
#[test]
fn bit_exact_keyframe_allplanes() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    if !path.exists() {
        eprintln!("skip: {path:?} not found");
        return;
    }
    let reference = dav2d_decode(&path);
    let got = rav2d_decode(&path);
    assert!(!got.is_empty(), "rav2d produced no frames");
    assert!(!reference.is_empty(), "dav2d produced no frames");
    assert_eq!(
        (reference[0].w, reference[0].h),
        (got[0].w, got[0].h),
        "frame 0 dims differ"
    );

    let (ssh, ssv) = ss(reference[0].layout);
    let plane_names = ["luma", "U", "V"];
    let mut failures = Vec::new();
    for pl in 0..3 {
        let r = &reference[0].planes[pl];
        let g = &got[0].planes[pl];
        assert_eq!(
            r.len(),
            g.len(),
            "frame 0 plane {} size differs",
            plane_names[pl]
        );
        let diff = r.iter().zip(g.iter()).filter(|(a, b)| a != b).count();
        if diff != 0 {
            let first = r.iter().zip(g.iter()).position(|(a, b)| a != b).unwrap();
            let stride = if pl == 0 {
                reference[0].w as usize
            } else {
                ((reference[0].w + ssh) >> ssh) as usize
            };
            let _ = ssv;
            failures.push(format!(
                "plane {} differs in {diff}/{} bytes; first @ ({},{}) ref={} got={}",
                plane_names[pl],
                r.len(),
                first % stride,
                first / stride,
                r[first],
                g[first]
            ));
        }
    }
    if !failures.is_empty() {
        panic!("frame 0 not bit-exact:\n  {}", failures.join("\n  "));
    }
}

/// Full bit-exact comparison rav2d vs dav2d (all planes, all frames, filters on).
/// Enabled once chroma recon + post-filters + inter support land.
#[test]
#[ignore = "enabled later: needs chroma recon + post-filters + inter + filters-on"]
fn bit_exact_keyframe_clip() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    let reference = dav2d_decode(&path);
    let got = rav2d_decode(&path);
    assert_bit_exact(&reference, &got, "bus.64x64.l5");
}

/// Informational frame-0 (keyframe, all-intra) sweep across the media clips:
/// for each, compare rav2d's first frame's planes to dav2d (filters/grain off).
/// Catches panics per clip so one failure doesn't mask the rest. Prints a table;
/// run with `--ignored --nocapture`. Flushes the intra bug surface across
/// sdp/lossless/deltaq/seg/partial-lossless features.
#[test]
#[ignore = "intra frame-0 conformance sweep (run with --ignored --nocapture)"]
fn intra_frame0_sweep() {
    let clips = [
        "avm-v14.1.0-bus.64x64.l5.obu",
        "avm-v14.1.0-bus.64x64.l1.sdp0.obu",
        "avm-v14.1.0-bus.64x64.l1.sdp1.obu",
        "avm-v14.1.0-bus.64x64.l5.lossless.obu",
        "avm-v14.1.0-bus.64x64.l5.opfl0-refinemv0.obu",
        "avm-v14.1.0-bus.352x288.l1.partial_lossless.obu",
        "avm-v14.1.0-bus.352x288.l10.deltaq1.obu",
        "avm-v14.1.0-bus.352x288.l5.seg1.obu",
        "avm-v14.1.0-hm.64x64.l5.filmgrain.obu",
    ];
    let mut summary = Vec::new();
    for clip in clips {
        let path = media(clip);
        if !path.exists() {
            summary.push(format!("{clip}: MISSING"));
            continue;
        }
        let reference = dav2d_decode(&path);
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rav2d_decode(&path)));
        let got = match got {
            Ok(g) => g,
            Err(_) => {
                summary.push(format!("{clip}: rav2d PANIC"));
                continue;
            }
        };
        if reference.is_empty() {
            summary.push(format!("{clip}: dav2d no frames"));
            continue;
        }
        if got.is_empty() {
            summary.push(format!("{clip}: rav2d NO FRAMES"));
            continue;
        }
        let r = &reference[0];
        let g = &got[0];
        if (r.w, r.h) != (g.w, g.h) {
            summary.push(format!(
                "{clip}: dims differ ref={}x{} got={}x{}",
                r.w, r.h, g.w, g.h
            ));
            continue;
        }
        let mut diffs = [0usize; 3];
        let mut total = [0usize; 3];
        for pl in 0..3 {
            total[pl] = r.planes[pl].len();
            diffs[pl] = r.planes[pl]
                .iter()
                .zip(g.planes[pl].iter())
                .filter(|(a, b)| a != b)
                .count();
        }
        if diffs == [0, 0, 0] {
            summary.push(format!("{clip}: BIT-EXACT ({}x{})", r.w, r.h));
        } else {
            summary.push(format!(
                "{clip}: DIFF Y={}/{} U={}/{} V={}/{}",
                diffs[0], total[0], diffs[1], total[1], diffs[2], total[2]
            ));
        }
    }
    eprintln!("\n=== intra frame-0 sweep ===\n{}", summary.join("\n"));
}

/// Debug helper: decode one clip (env CLIP) with both decoders so RAV2D_TRACE
/// emits matched LUMATX/CHROMATX traces to stderr.
#[test]
#[ignore = "debug trace harness"]
fn trace_clip() {
    let clip = std::env::var("CLIP").unwrap_or_else(|_| "avm-v14.1.0-bus.64x64.l5.lossless.obu".into());
    let path = media(&clip);
    let which = std::env::var("WHICH").unwrap_or_else(|_| "both".into());
    if which == "dav2d" || which == "both" {
        eprintln!("@@@DAV2D@@@");
        let _ = dav2d_decode(&path);
    }
    if which == "rav2d" || which == "both" {
        eprintln!("@@@RAV2D@@@");
        let _ = rav2d_decode(&path);
    }
}

/// Debug helper: print the bounding box / first coordinates of per-plane diffs
/// for frame 0 of CLIP (rav2d vs dav2d).
#[test]
#[ignore = "debug diff-location harness"]
fn diff_loc() {
    let clip = std::env::var("CLIP").unwrap();
    let path = media(&clip);
    let r = dav2d_decode(&path);
    let g = rav2d_decode(&path);
    let (r, g) = (&r[0], &g[0]);
    for pl in 0..3 {
        let (ssh, ssv) = ss(r.layout);
        let pw = if pl == 0 { r.w } else { (r.w + ssh) >> ssh } as usize;
        let mut coords = Vec::new();
        for (i, (a, b)) in r.planes[pl].iter().zip(g.planes[pl].iter()).enumerate() {
            if a != b {
                coords.push((i % pw, i / pw, *a, *b));
            }
        }
        if coords.is_empty() {
            eprintln!("plane {pl}: EXACT");
        } else {
            let minx = coords.iter().map(|c| c.0).min().unwrap();
            let maxx = coords.iter().map(|c| c.0).max().unwrap();
            let miny = coords.iter().map(|c| c.1).min().unwrap();
            let maxy = coords.iter().map(|c| c.1).max().unwrap();
            eprintln!(
                "plane {pl}: {} diffs, bbox x[{minx}..{maxx}] y[{miny}..{maxy}], first 6: {:?}",
                coords.len(),
                &coords[..coords.len().min(6)]
            );
        }
    }
}

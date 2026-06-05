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

/// DAV2D_INLOOPFILTER_ALL (deblock | cdef | ccso | wiener | gdf).
const DAV2D_INLOOPFILTER_ALL: u32 = 31;

/// Decode a clip with the dav2d C reference library, with in-loop filters off
/// (pure reconstruction). Returns one `FramePlanes` per output frame.
pub fn dav2d_decode(path: &PathBuf) -> Vec<FramePlanes> {
    dav2d_decode_filters(path, 0)
}

/// Like `dav2d_decode`, but also emits invisibly-coded (non-shown) frames in
/// CODING order (`output_invisible_frames`). rav2d currently outputs every
/// decoded frame in coding order without visibility filtering, so this gives a
/// frame list whose keyframe (a non-shown keyframe lands here too) can be
/// matched 1:1 against rav2d for the all-intra sweep. Filters/grain off.
pub fn dav2d_decode_invisible(path: &PathBuf) -> Vec<FramePlanes> {
    use rav2d::sys;
    let data = std::fs::read(path).expect("read clip");
    let mut frames = Vec::new();

    unsafe {
        let mut settings: sys::Dav2dSettings = std::mem::zeroed();
        sys::dav2d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.apply_grain = 0;
        settings.inloop_filters = 0;
        settings.output_invisible_frames = 1;

        let mut ctx: *mut sys::Dav2dContext = std::ptr::null_mut();
        let r = sys::dav2d_open(&mut ctx, &settings);
        assert_eq!(r, 0, "dav2d_open failed: {r}");

        let mut d: sys::Dav2dData = std::mem::zeroed();
        let buf = sys::dav2d_data_create(&mut d, data.len());
        assert!(!buf.is_null(), "dav2d_data_create failed");
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len());

        let mut drained = false;
        loop {
            loop {
                let mut pic: sys::Dav2dPicture = std::mem::zeroed();
                let pr = sys::dav2d_get_picture(ctx, &mut pic);
                if pr == EAGAIN || pr < 0 {
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
                sys::dav2d_send_data(ctx, std::ptr::null_mut());
            } else {
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

/// Decode a clip with the dav2d C reference library at the given in-loop-filter
/// flag word (0 = none, 31 = all). Single-threaded, film grain off.
pub fn dav2d_decode_filters(path: &PathBuf, inloop_filters: u32) -> Vec<FramePlanes> {
    use rav2d::sys;
    let data = std::fs::read(path).expect("read clip");
    let mut frames = Vec::new();

    unsafe {
        let mut settings: sys::Dav2dSettings = std::mem::zeroed();
        sys::dav2d_default_settings(&mut settings);
        // Deterministic single-threaded reference. Film grain off so only the
        // in-loop filters differ between the pure-recon and filtered harnesses.
        settings.n_threads = 1;
        settings.apply_grain = 0;
        settings.inloop_filters = inloop_filters;

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

/// Like `dav2d_decode_filters`, but also emits invisibly-coded (non-shown)
/// frames in CODING order (`output_invisible_frames`). This lets a non-shown
/// keyframe be matched 1:1 against rav2d's coding-order output even with filters
/// on. Film grain off.
pub fn dav2d_decode_filters_invisible(path: &PathBuf, inloop_filters: u32) -> Vec<FramePlanes> {
    use rav2d::sys;
    let data = std::fs::read(path).expect("read clip");
    let mut frames = Vec::new();

    unsafe {
        let mut settings: sys::Dav2dSettings = std::mem::zeroed();
        sys::dav2d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.apply_grain = 0;
        settings.inloop_filters = inloop_filters;
        settings.output_invisible_frames = 1;

        let mut ctx: *mut sys::Dav2dContext = std::ptr::null_mut();
        let r = sys::dav2d_open(&mut ctx, &settings);
        assert_eq!(r, 0, "dav2d_open failed: {r}");

        let mut d: sys::Dav2dData = std::mem::zeroed();
        let buf = sys::dav2d_data_create(&mut d, data.len());
        assert!(!buf.is_null(), "dav2d_data_create failed");
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len());

        let mut drained = false;
        loop {
            loop {
                let mut pic: sys::Dav2dPicture = std::mem::zeroed();
                let pr = sys::dav2d_get_picture(ctx, &mut pic);
                if pr == EAGAIN || pr < 0 {
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
                sys::dav2d_send_data(ctx, std::ptr::null_mut());
            } else {
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

/// Decode a clip with rav2d, in-loop filters off (pure reconstruction). Returns
/// one `FramePlanes` per output frame.
pub fn rav2d_decode(path: &PathBuf) -> Vec<FramePlanes> {
    rav2d_decode_filters(path, rav2d::InloopFilterType::None)
}

/// Decode a clip with rav2d at the given in-loop filter setting.
pub fn rav2d_decode_filters(
    path: &PathBuf,
    inloop_filters: rav2d::InloopFilterType,
) -> Vec<FramePlanes> {
    use rav2d::{Data, Decoder, Settings};
    let bytes = std::fs::read(path).expect("read clip");
    let mut s = Settings::default();
    s.n_threads = 1;
    s.apply_grain = false;
    s.run_decode = true;
    s.inloop_filters = inloop_filters;
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

/// M2 gate: frame-0 ALL PLANES must match the dav2d reference WITH in-loop
/// filters ON (both decoders run deblock + CDEF (+CCSO) + loop restoration). The
/// bus.64x64.l5 keyframe exercises CDEF (chroma strength 1), so this validates
/// the per-superblock-row post-filter pipeline. Film grain stays off (M4).
#[test]
fn bit_exact_keyframe_filtered() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    if !path.exists() {
        eprintln!("skip: {path:?} not found");
        return;
    }
    let reference = dav2d_decode_filters(&path, DAV2D_INLOOPFILTER_ALL);
    let got = rav2d_decode_filters(&path, rav2d::InloopFilterType::All);
    assert!(!got.is_empty(), "rav2d produced no frames");
    assert!(!reference.is_empty(), "dav2d produced no frames");
    assert_eq!(
        (reference[0].w, reference[0].h),
        (got[0].w, got[0].h),
        "frame 0 dims differ"
    );

    let (ssh, _ssv) = ss(reference[0].layout);
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
        panic!(
            "frame 0 not bit-exact (filters on):\n  {}",
            failures.join("\n  ")
        );
    }
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
        // Emit invisible frames so a non-shown keyframe (e.g. delta-q clips)
        // appears in dav2d's coding-order output and can be matched to rav2d's.
        let reference = dav2d_decode_invisible(&path);
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
        // rav2d emits frames in DECODE order without visibility filtering, so
        // got[0] is always the keyframe (the all-intra frame this sweep targets).
        // dav2d emits in DISPLAY order honouring show_frame, so a non-shown
        // keyframe lands at a different output index (or is re-shown later). Pick
        // the dav2d output frame whose pixels best match rav2d's keyframe; for a
        // correct decoder that is the same keyframe content, so a residual diff
        // here is a real reconstruction bug rather than an output-ordering quirk.
        let g = &got[0];
        let r = reference
            .iter()
            .filter(|r| (r.w, r.h) == (g.w, g.h))
            .min_by_key(|r| {
                (0..3)
                    .map(|pl| {
                        r.planes[pl]
                            .iter()
                            .zip(g.planes[pl].iter())
                            .filter(|(a, b)| a != b)
                            .count()
                    })
                    .sum::<usize>()
            });
        let r = match r {
            Some(r) => r,
            None => {
                summary.push(format!(
                    "{clip}: no dav2d frame matches dims {}x{}",
                    g.w, g.h
                ));
                continue;
            }
        };
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

/// M2 acceptance gate: for each clip whose keyframe is already intra-bit-exact
/// (filters off), decode frame-0 with BOTH decoders' in-loop filters fully ON
/// (deblock + CDEF + CCSO + Wiener/GDF) and assert all planes are bit-exact.
///
/// dav2d emits invisible frames so a non-shown keyframe appears in coding order;
/// rav2d emits coding order too, so got[0] is the keyframe. We pick the dav2d
/// frame whose dims match and whose pixels best match rav2d's keyframe (the
/// best-match-keyframe approach used by the intra sweep), then require zero diff.
///
/// Film grain stays off (M4). All listed clips' keyframes must be bit-exact with
/// filters on; this is the M2 done-condition.
/// WIP: deblock+CDEF clips pass; LR/CCSO (352x288 seg1/deltaq1/partial_lossless)
/// still pending — un-ignore once those land.
#[test]
#[ignore = "M2 WIP: deblock+CDEF bit-exact; LR/CCSO pending"]
fn bit_exact_filtered_sweep() {
    let clips = [
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
    let plane_names = ["luma", "U", "V"];
    let mut failures = Vec::new();
    let mut summary = Vec::new();
    for clip in clips {
        let path = media(clip);
        if !path.exists() {
            summary.push(format!("{clip}: MISSING"));
            continue;
        }
        let reference = dav2d_decode_filters_invisible(&path, DAV2D_INLOOPFILTER_ALL);
        let got = rav2d_decode_filters(&path, rav2d::InloopFilterType::All);
        if got.is_empty() {
            failures.push(format!("{clip}: rav2d produced no frames"));
            continue;
        }
        if reference.is_empty() {
            failures.push(format!("{clip}: dav2d produced no frames"));
            continue;
        }
        let g = &got[0];
        // Best-match the dav2d coding-order keyframe by pixel similarity.
        let r = reference
            .iter()
            .filter(|r| (r.w, r.h, r.bpc, r.layout) == (g.w, g.h, g.bpc, g.layout))
            .min_by_key(|r| {
                (0..3)
                    .map(|pl| {
                        r.planes[pl]
                            .iter()
                            .zip(g.planes[pl].iter())
                            .filter(|(a, b)| a != b)
                            .count()
                    })
                    .sum::<usize>()
            });
        let r = match r {
            Some(r) => r,
            None => {
                failures.push(format!("{clip}: no dav2d frame matches {}x{}", g.w, g.h));
                continue;
            }
        };
        let (ssh, _) = ss(r.layout);
        let mut clip_fail = Vec::new();
        for pl in 0..3 {
            if r.planes[pl].len() != g.planes[pl].len() {
                clip_fail.push(format!(
                    "plane {} size differs ({} vs {})",
                    plane_names[pl],
                    r.planes[pl].len(),
                    g.planes[pl].len()
                ));
                continue;
            }
            let diff = r.planes[pl]
                .iter()
                .zip(g.planes[pl].iter())
                .filter(|(a, b)| a != b)
                .count();
            if diff != 0 {
                let first = r.planes[pl]
                    .iter()
                    .zip(g.planes[pl].iter())
                    .position(|(a, b)| a != b)
                    .unwrap();
                let stride = if pl == 0 {
                    r.w as usize
                } else {
                    ((r.w + ssh) >> ssh) as usize
                };
                clip_fail.push(format!(
                    "plane {} differs in {diff}/{} bytes; first @ ({},{}) ref={} got={}",
                    plane_names[pl],
                    r.planes[pl].len(),
                    first % stride,
                    first / stride,
                    r.planes[pl][first],
                    g.planes[pl][first]
                ));
            }
        }
        if clip_fail.is_empty() {
            summary.push(format!("{clip}: BIT-EXACT ({}x{})", r.w, r.h));
        } else {
            summary.push(format!("{clip}: DIFF"));
            failures.push(format!("{clip}:\n    {}", clip_fail.join("\n    ")));
        }
    }
    eprintln!("\n=== filtered frame-0 sweep ===\n{}", summary.join("\n"));
    if !failures.is_empty() {
        panic!(
            "filtered frame-0 not bit-exact:\n  {}",
            failures.join("\n  ")
        );
    }
}

/// Debug helper: bisect which filter stage diverges for CLIP by decoding both
/// decoders with the same filter mask, comparing the keyframe, for each of:
/// none, deblock, deblock+cdef, deblock+cdef+ccso, +wiener, +gdf, all.
#[test]
#[ignore = "filter bisect harness"]
fn filter_bisect() {
    use rav2d::InloopFilterType as F;
    let clip = std::env::var("CLIP").unwrap();
    let path = media(&clip);
    // Cumulative filter masks (dav2d DAV2D_INLOOPFILTER_* bits). rav2d uses the
    // RAV2D_FILT_OVERRIDE env to run the exact same bit mask, since the public
    // InloopFilterType enum only exposes single bits / All.
    let combos: [(u32, &str); 6] = [
        (0, "none"),
        (1, "deblock"),
        (3, "deblock+cdef"),
        (7, "deblock+cdef+ccso"),
        (15, "deblock+cdef+ccso+wiener"),
        (31, "all"),
    ];
    for (bits, name) in combos {
        let reference = dav2d_decode_filters_invisible(&path, bits);
        // SAFETY: single-threaded test; set the override env for the rust decode.
        unsafe {
            std::env::set_var("RAV2D_FILT_OVERRIDE", bits.to_string());
        }
        let got = rav2d_decode_filters(&path, F::All);
        unsafe {
            std::env::remove_var("RAV2D_FILT_OVERRIDE");
        }
        if got.is_empty() || reference.is_empty() {
            eprintln!("{name}: empty");
            continue;
        }
        let g = &got[0];
        let r = reference
            .iter()
            .filter(|r| (r.w, r.h, r.bpc, r.layout) == (g.w, g.h, g.bpc, g.layout))
            .min_by_key(|r| {
                (0..3)
                    .map(|pl| {
                        r.planes[pl]
                            .iter()
                            .zip(g.planes[pl].iter())
                            .filter(|(a, b)| a != b)
                            .count()
                    })
                    .sum::<usize>()
            });
        let r = match r {
            Some(r) => r,
            None => {
                eprintln!("{name}: no match");
                continue;
            }
        };
        let (ssh, _) = ss(r.layout);
        let mut parts = Vec::new();
        for pl in 0..3 {
            if r.planes[pl].len() != g.planes[pl].len() {
                parts.push(format!("p{pl} sizediff"));
                continue;
            }
            let diff = r.planes[pl]
                .iter()
                .zip(g.planes[pl].iter())
                .filter(|(a, b)| a != b)
                .count();
            if diff != 0 {
                let first = r.planes[pl]
                    .iter()
                    .zip(g.planes[pl].iter())
                    .position(|(a, b)| a != b)
                    .unwrap();
                let stride = if pl == 0 {
                    r.w as usize
                } else {
                    ((r.w + ssh) >> ssh) as usize
                };
                parts.push(format!(
                    "p{pl} diff={diff} @({},{}) r={} g={}",
                    first % stride,
                    first / stride,
                    r.planes[pl][first],
                    g.planes[pl][first]
                ));
            }
        }
        if parts.is_empty() {
            eprintln!("{name}: BIT-EXACT");
        } else {
            eprintln!("{name}: {}", parts.join("  "));
        }
        // Optional first-N diff coords for the combo named by env DIFFCOMBO.
        if std::env::var("DIFFCOMBO").ok().as_deref() == Some(name) {
            let pl = std::env::var("DIFFPL")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0usize);
            let pw = if pl == 0 {
                r.w as usize
            } else {
                ((r.w + ssh) >> ssh) as usize
            };
            let nmax = std::env::var("DIFFN")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(60);
            let mut n = 0;
            for (i, (a, b)) in r.planes[pl].iter().zip(g.planes[pl].iter()).enumerate() {
                if a != b {
                    eprintln!("  diff @({},{}) r={} g={}", i % pw, i / pw, a, b);
                    n += 1;
                    if n >= nmax {
                        break;
                    }
                }
            }
        }
        // Optional grid map for the combo named by env GRIDCOMBO.
        if std::env::var("GRIDCOMBO").ok().as_deref() == Some(name) {
            let cell = std::env::var("CELL")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);
            for pl in 0..3 {
                if r.planes[pl].len() != g.planes[pl].len() || r.planes[pl].is_empty() {
                    continue;
                }
                let pw = if pl == 0 {
                    r.w as usize
                } else {
                    ((r.w + ssh) >> ssh) as usize
                };
                let ph = r.planes[pl].len() / pw;
                let gw = pw.div_ceil(cell);
                let gh = ph.div_ceil(cell);
                let mut grid = vec![0usize; gw * gh];
                for (i, (a, b)) in r.planes[pl].iter().zip(g.planes[pl].iter()).enumerate() {
                    if a != b {
                        grid[(i / pw / cell) * gw + (i % pw / cell)] += 1;
                    }
                }
                eprintln!("  plane {pl} grid (cell={cell}, {gw}x{gh}):");
                for gy in 0..gh {
                    let mut row = String::new();
                    for gx in 0..gw {
                        let n = grid[gy * gw + gx];
                        row.push(if n == 0 {
                            '.'
                        } else if n < cell {
                            '+'
                        } else {
                            '#'
                        });
                    }
                    eprintln!("    {row}");
                }
            }
        }
    }
}

/// Debug helper: decode one clip (env CLIP) with both decoders so RAV2D_TRACE
/// emits matched LUMATX/CHROMATX traces to stderr.
#[test]
#[ignore = "debug trace harness"]
fn trace_clip() {
    let clip =
        std::env::var("CLIP").unwrap_or_else(|_| "avm-v14.1.0-bus.64x64.l5.lossless.obu".into());
    let path = media(&clip);
    let which = std::env::var("WHICH").unwrap_or_else(|_| "both".into());
    // FILTERS env: bit word (dav2d DAV2D_INLOOPFILTER_*); default 0 (off).
    let filt: u32 = std::env::var("FILTERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let rav_filt = match filt {
        0 => rav2d::InloopFilterType::None,
        1 => rav2d::InloopFilterType::Deblock,
        3 => rav2d::InloopFilterType::Cdef,
        7 => rav2d::InloopFilterType::Restoration,
        15 => rav2d::InloopFilterType::Wiener,
        _ => rav2d::InloopFilterType::All,
    };
    if which == "dav2d" || which == "both" {
        eprintln!("@@@DAV2D@@@");
        let _ = dav2d_decode_filters_invisible(&path, filt);
    }
    if which == "rav2d" || which == "both" {
        eprintln!("@@@RAV2D@@@");
        let _ = rav2d_decode_filters(&path, rav_filt);
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
        let (ssh, _ssv) = ss(r.layout);
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
            if std::env::var("GRID").is_ok() {
                let ph = r.planes[pl].len() / pw;
                let cell = std::env::var("CELL")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let gw = pw.div_ceil(cell);
                let gh = ph.div_ceil(cell);
                let mut grid = vec![0usize; gw * gh];
                for c in &coords {
                    grid[(c.1 / cell) * gw + (c.0 / cell)] += 1;
                }
                eprintln!("plane {pl} grid (cell={cell}, {gw}x{gh}):");
                for gy in 0..gh {
                    let mut row = String::new();
                    for gx in 0..gw {
                        let n = grid[gy * gw + gx];
                        row.push(if n == 0 {
                            '.'
                        } else if n < cell {
                            '+'
                        } else {
                            '#'
                        });
                    }
                    eprintln!("  {row}");
                }
            }
        }
    }
}

/// Diagnostic: for CLIP, match each rav2d (decode-order) frame to the
/// best-matching dav2d (display-order) output frame and report per-frame,
/// per-plane diff counts. Reveals output-ordering mismatches.
#[test]
#[ignore = "frame-alignment diagnostic"]
fn frame_align() {
    let clip = std::env::var("CLIP").unwrap();
    let path = media(&clip);
    let d = dav2d_decode(&path);
    let r = rav2d_decode(&path);
    eprintln!("rav2d frames={} dav2d frames={}", r.len(), d.len());
    for (ri, rf) in r.iter().enumerate() {
        let mut best = (usize::MAX, usize::MAX);
        for (di, df) in d.iter().enumerate() {
            if (rf.w, rf.h) != (df.w, df.h) {
                continue;
            }
            let diff: usize = (0..3)
                .map(|pl| {
                    rf.planes[pl]
                        .iter()
                        .zip(df.planes[pl].iter())
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .sum();
            if diff < best.1 {
                best = (di, diff);
            }
        }
        eprintln!(
            "rav2d[{ri}] best-match dav2d[{}] totaldiff={}",
            best.0, best.1
        );
    }
}

/// Diagnostic: rav2d emits frames in DECODE order (no POC reorder yet) while
/// dav2d emits in DISPLAY order. For the first inter frame (rav2d decode index
/// 1) find the closest-matching dav2d output frame and report per-plane diffs.
/// Run with `--ignored --nocapture`.
#[test]
#[ignore = "block-grid diff map for first inter frame"]
fn inter_frame1_blockmap() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    let reference = dav2d_decode(&path);
    let got = rav2d_decode(&path);
    let g = &got[1];
    let r = reference.last().unwrap();
    let w = r.w as usize;
    let h = r.h as usize;
    for pl in 0..3 {
        let (ssh, ssv) = ss(r.layout);
        let (ssh, ssv) = (ssh as usize, ssv as usize);
        let pw = if pl == 0 { w } else { (w + ssh) >> ssh };
        let ph = if pl == 0 { h } else { (h + ssv) >> ssv };
        let rp = &r.planes[pl];
        let gp = &g.planes[pl];
        eprintln!("== plane {pl} {pw}x{ph} ==");
        // 8x8 grid of diff counts
        let mut total = 0;
        for by in (0..ph).step_by(8) {
            let mut row = String::new();
            for bx in (0..pw).step_by(8) {
                let mut c = 0;
                for y in by..(by + 8).min(ph) {
                    for x in bx..(bx + 8).min(pw) {
                        if rp[y * pw + x] != gp[y * pw + x] {
                            c += 1;
                        }
                    }
                }
                total += c;
                row.push_str(&format!("{c:3} "));
            }
            eprintln!("y={by:3}: {row}");
        }
        eprintln!("plane {pl} total diffs {total}");
        if pl == 0 {
            eprintln!("block(0,4) cols16-31 rows0-3 REF then GOT:");
            for y in 0..4 {
                let mut rs = String::from("R");
                let mut gs = String::from("G");
                for x in 16..32 {
                    rs.push_str(&format!(" {}", rp[y * pw + x]));
                    gs.push_str(&format!(" {}", gp[y * pw + x]));
                }
                eprintln!("{rs}");
                eprintln!("{gs}");
            }
        }
    }
}

#[test]
#[ignore = "first-inter-frame diff diagnostic"]
fn inter_frame1_diag() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    let reference = dav2d_decode(&path);
    let got = rav2d_decode(&path);
    eprintln!(
        "dav2d frames={} rav2d frames={}",
        reference.len(),
        got.len()
    );
    let g = &got[1]; // rav2d decode order: [key, inter(poc4), ...]
    let mut best = (usize::MAX, [0usize; 3], [0usize; 3]);
    for (ri, r) in reference.iter().enumerate() {
        if (r.w, r.h) != (g.w, g.h) {
            continue;
        }
        let mut diffs = [0usize; 3];
        let mut total = [0usize; 3];
        for pl in 0..3 {
            total[pl] = r.planes[pl].len().min(g.planes[pl].len());
            diffs[pl] = r.planes[pl]
                .iter()
                .zip(g.planes[pl].iter())
                .filter(|(a, b)| a != b)
                .count();
        }
        let sum: usize = diffs.iter().sum();
        if sum < best.0 {
            best = (sum, diffs, total);
        }
        eprintln!(
            "  vs dav2d[{ri}]: Y={}/{} U={}/{} V={}/{}",
            diffs[0], total[0], diffs[1], total[1], diffs[2], total[2]
        );
    }
    eprintln!(
        "BEST: total_diff={} Y={}/{} U={}/{} V={}/{}",
        best.0, best.1[0], best.2[0], best.1[1], best.2[1], best.1[2], best.2[2]
    );
}

/// M3 gate: the FIRST inter frame of bus.64x64.l5 (decode order index 1,
/// poc=4) must reconstruct bit-exact vs dav2d (filters off). rav2d emits in
/// decode order; the matching dav2d output frame is poc=4, the last output
/// frame for this clip's display order.
///
/// All planes (Y/U/V) are bit-exact: single-ref + same-ref compound MC,
/// warp-affine / warp-delta / warp-extend, inter-intra blend, sub-8x8 chroma
/// MC and the inter-chroma CCTX residual all match the C decoder.
#[test]
fn bit_exact_first_inter_frame() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    if !path.exists() {
        eprintln!("skip: {path:?} not found");
        return;
    }
    let reference = dav2d_decode(&path);
    let got = rav2d_decode(&path);
    assert!(got.len() > 1, "rav2d produced too few frames");
    let g = &got[1];
    // dav2d display order: poc=4 is the last output frame for this clip.
    let r = reference.last().expect("dav2d frames");
    assert_eq!((r.w, r.h), (g.w, g.h), "first inter frame dims differ");

    let plane_names = ["luma", "U", "V"];
    let (ssh, _ssv) = ss(r.layout);
    let mut failures = Vec::new();
    for pl in 0..3 {
        let rp = &r.planes[pl];
        let gp = &g.planes[pl];
        assert_eq!(rp.len(), gp.len(), "plane {} size differs", plane_names[pl]);
        let diff = rp.iter().zip(gp.iter()).filter(|(a, b)| a != b).count();
        if diff != 0 {
            let first = rp.iter().zip(gp.iter()).position(|(a, b)| a != b).unwrap();
            let stride = if pl == 0 { r.w } else { (r.w + ssh) >> ssh } as usize;
            failures.push(format!(
                "{} differs in {diff}/{} bytes; first @ ({},{}) ref={} got={}",
                plane_names[pl],
                rp.len(),
                first % stride,
                first / stride,
                rp[first],
                gp[first]
            ));
        }
    }
    if !failures.is_empty() {
        panic!(
            "first inter frame not bit-exact:\n  {}",
            failures.join("\n  ")
        );
    }
}

/// Full-clip gate: EVERY coding-order frame of bus.64x64.l5 must reconstruct
/// bit-exact vs dav2d (in-loop filters off on both decoders). The clip is a
/// keyframe (poc0) plus four inter frames (poc4 single-ref, then poc2/poc1/poc3
/// which use cross-reference compound + temporal-interpolated-prediction blocks).
///
/// rav2d emits in DECODE order without visibility filtering; dav2d is run with
/// `output_invisible_frames` so its non-shown reference frames are emitted in
/// coding order too. The two frame lists therefore contain the same set of
/// reconstructed frames but in different orders, so each rav2d frame is matched
/// to the dav2d frame with identical dimensions and the fewest pixel diffs (for
/// a correct decoder that is its true counterpart, diff 0) before the bit-exact
/// assertion. A residual diff is a real reconstruction bug, not a reorder.
#[test]
#[ignore = "WIP: full-clip inter (TIP/compound poc1-3) not yet bit-exact"]
fn bit_exact_full_clip_bus64() {
    let path = media("avm-v14.1.0-bus.64x64.l5.obu");
    if !path.exists() {
        eprintln!("skip: {path:?} not found");
        return;
    }
    let reference = dav2d_decode_invisible(&path);
    let got = rav2d_decode(&path);
    assert!(!reference.is_empty(), "dav2d produced no frames");
    assert_eq!(
        got.len(),
        reference.len(),
        "frame count mismatch (rav2d={}, dav2d={})",
        got.len(),
        reference.len()
    );

    let mut used = vec![false; reference.len()];
    let mut failures = Vec::new();
    for (gi, g) in got.iter().enumerate() {
        // Pick the unused dav2d frame of matching dims with the fewest diffs.
        let mut best: Option<(usize, usize)> = None; // (ref_index, total_diff)
        for (ri, r) in reference.iter().enumerate() {
            if used[ri] || (r.w, r.h, r.bpc, r.layout) != (g.w, g.h, g.bpc, g.layout) {
                continue;
            }
            let diff: usize = (0..3)
                .map(|pl| {
                    r.planes[pl]
                        .iter()
                        .zip(g.planes[pl].iter())
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .sum();
            if best.is_none_or(|(_, bd)| diff < bd) {
                best = Some((ri, diff));
            }
        }
        let (ri, _) = best.unwrap_or_else(|| panic!("rav2d frame {gi}: no dav2d match"));
        used[ri] = true;
        let r = &reference[ri];
        let (ssh, _ssv) = ss(r.layout);
        for pl in 0..3 {
            let rp = &r.planes[pl];
            let gp = &g.planes[pl];
            if rp.len() != gp.len() {
                failures.push(format!(
                    "rav2d[{gi}]~dav2d[{ri}] plane {pl} size differs ({} vs {})",
                    rp.len(),
                    gp.len()
                ));
                continue;
            }
            let diff = rp.iter().zip(gp.iter()).filter(|(a, b)| a != b).count();
            if diff != 0 {
                let first = rp.iter().zip(gp.iter()).position(|(a, b)| a != b).unwrap();
                let stride = if pl == 0 { r.w } else { (r.w + ssh) >> ssh } as usize;
                failures.push(format!(
                    "rav2d[{gi}]~dav2d[{ri}] plane {pl}: {diff}/{} bytes differ; first @ ({},{}) ref={} got={}",
                    rp.len(),
                    first % stride,
                    first / stride,
                    rp[first],
                    gp[first]
                ));
            }
        }
    }
    if !failures.is_empty() {
        panic!("full clip not bit-exact:\n  {}", failures.join("\n  "));
    }
}

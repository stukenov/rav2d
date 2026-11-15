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

//! End-to-end checks of the `rav2d` binary on the library's test vectors.
//! Skipped when the vectors are not there (a crates.io download has none).

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_rav2d");

fn media(name: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rav2d/tests/data/media"
    ))
    .join(name)
}

const BUS: &str = "avm-v14.1.0-bus.64x64.l5.obu";
/// One 64x64 4:2:0 8-bit frame as raw planes.
const BUS_FRAME: usize = 64 * 64 * 3 / 2;

fn have_vectors() -> bool {
    let ok = media(BUS).exists();
    if !ok {
        eprintln!("skip: test vectors not found");
    }
    ok
}

fn rav2d(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().expect("run rav2d")
}

fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    dir.join(name)
}

/// The hash avmdec printed for a vector.
fn avmdec_md5(clip: &Path) -> String {
    std::fs::read_to_string(format!("{}.md5", clip.display()))
        .unwrap()
        .trim()
        .to_string()
}

#[test]
fn md5_matches_avmdec_on_every_vector() {
    if !have_vectors() {
        return;
    }
    let dir = media("");
    let mut clips: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "obu"))
        .filter(|p| PathBuf::from(format!("{}.md5", p.display())).exists())
        .collect();
    clips.sort();
    assert_ne!(clips, [] as [std::path::PathBuf; 0]);
    for clip in &clips {
        let out = rav2d(&["--md5", "-q", path(clip)]);
        assert!(out.status.success(), "{clip:?}: {out:?}");
        let got = String::from_utf8(out.stdout).unwrap();
        assert_eq!(got.trim(), avmdec_md5(clip), "{clip:?}");

        // --verify takes the .md5 file as it ships.
        let md5_file = format!("{}.md5", clip.display());
        let out = rav2d(&["-q", path(clip), "--verify", &md5_file]);
        assert!(out.status.success(), "{clip:?}: {out:?}");
    }
}

#[test]
fn verify_mismatch_exits_with_1() {
    if !have_vectors() {
        return;
    }
    let out = rav2d(&[
        path(&media(BUS)),
        "--verify",
        "00000000000000000000000000000000",
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("MD5 mismatch"));
}

#[test]
fn limit_stops_a_raw_obu_stream() {
    if !have_vectors() {
        return;
    }
    let out = rav2d(&[path(&media(BUS)), "-q", "-l", "1", "-o", "-", "-m", "yuv"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(out.stdout.len(), BUS_FRAME);
}

#[test]
fn stdout_y4m_equals_the_file() {
    if !have_vectors() {
        return;
    }
    let file = tmp("stdout_y4m_equals_the_file.y4m");
    let out = rav2d(&[path(&media(BUS)), "-q", "-o", path(&file)]);
    assert!(out.status.success(), "{out:?}");
    let piped = rav2d(&[path(&media(BUS)), "-q", "-o", "-"]);
    assert!(piped.status.success(), "{piped:?}");
    assert!(piped.stdout.starts_with(b"YUV4MPEG2 W64 H64 "));
    assert_eq!(piped.stdout, std::fs::read(&file).unwrap());
}

#[test]
fn yuv_is_the_y4m_without_headers() {
    if !have_vectors() {
        return;
    }
    let y4m = rav2d(&[path(&media(BUS)), "-q", "-o", "-"]).stdout;
    let yuv_file = tmp("yuv_is_the_y4m_without_headers.yuv");
    let out = rav2d(&[path(&media(BUS)), "-q", "-o", path(&yuv_file)]);
    assert!(out.status.success(), "{out:?}");
    let yuv = std::fs::read(&yuv_file).unwrap();

    let body = &y4m[y4m.iter().position(|&b| b == b'\n').unwrap() + 1..];
    let stripped: Vec<u8> = body
        .chunks(b"FRAME\n".len() + BUS_FRAME)
        .flat_map(|f| {
            assert!(f.starts_with(b"FRAME\n"));
            f[6..].to_vec()
        })
        .collect();
    assert_eq!(stripped, yuv);
    assert_eq!(yuv.len() % BUS_FRAME, 0);
}

#[test]
fn skip_drops_leading_frames() {
    if !have_vectors() {
        return;
    }
    let all = rav2d(&[path(&media(BUS)), "-q", "-o", "-", "-m", "yuv"]).stdout;
    let third = rav2d(&[
        path(&media(BUS)),
        "-q",
        "--skip",
        "2",
        "-l",
        "1",
        "-o",
        "-",
        "-m",
        "yuv",
    ])
    .stdout;
    assert_eq!(third, all[2 * BUS_FRAME..3 * BUS_FRAME]);
}

#[test]
fn reads_stdin() {
    if !have_vectors() {
        return;
    }
    let file = std::fs::File::open(media(BUS)).unwrap();
    let out = Command::new(BIN)
        .args(["-", "--md5", "-q"])
        .stdin(Stdio::from(file))
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        avmdec_md5(&media(BUS))
    );
}

#[test]
fn info_lists_frames_and_the_stream() {
    if !have_vectors() {
        return;
    }
    let out = rav2d(&[path(&media(BUS)), "--info", "-q"]);
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with("frame 0: 64x64 8-bit 4:2:0 key"), "{text}");
    assert!(text.contains("stream: profile "), "{text}");
    assert!(text.contains("frames: "), "{text}");
}

#[test]
fn unwritable_output_exits_with_2_without_panicking() {
    if !have_vectors() {
        return;
    }
    let out = rav2d(&[path(&media(BUS)), "-o", "/nonexistent-dir/out.y4m"]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("cannot create"), "{err}");
    assert!(!err.contains("panicked"), "{err}");
}

#[test]
fn unknown_extension_is_a_usage_error() {
    let out = rav2d(&["whatever.obu", "-o", "out.mp4"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn missing_input_is_a_usage_error() {
    let out = rav2d(&["/nonexistent-dir/in.obu"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn corrupt_stream_exits_with_1() {
    if !have_vectors() {
        return;
    }
    // Keep the sequence header, then flip bits through the frame data.
    let mut data = std::fs::read(media(BUS)).unwrap();
    for b in data.iter_mut().skip(40).step_by(3) {
        *b ^= 0x5a;
    }
    let file = tmp("corrupt_stream_exits_with_1.obu");
    std::fs::write(&file, &data).unwrap();
    let out = rav2d(&[path(&file), "-q"]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
}

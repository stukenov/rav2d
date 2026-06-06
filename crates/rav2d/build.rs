//! Build script for rav2d.
//!
//! On aarch64 it assembles dav2d's hand-written NEON DSP kernels and the
//! read-only data tables they reference, linking them into the rav2d staticlib
//! so the decoder can dispatch the corresponding DSP families to NEON:
//!
//!   * motion compensation: `dav2d/src/arm/64/mc.S` + `mc_dotprod.S`
//!     (tables `dav2d_mc_subpel_filters`, `dav2d_mc_warp_filter`) → `src/mc_neon.rs`.
//!   * intra prediction: `dav2d/src/arm/64/ipred.S` (tables `dav2d_sm_weights`,
//!     `dav2d_filter_intra_taps`) → `src/ipred_neon.rs`.
//!
//! The inverse-transform family (`itx.S`) is deliberately NOT assembled: dav2d's
//! NEON `itx.S` still implements the AV1 DCT/ADST butterflies (constants 2896,
//! 1567, 3784, …), whereas AV2 redefined the inverse transforms to a matrix form
//! (constants 64, 83, 35, … — see `dav2d/src/itx_1d.c` and `src/itx_1d.rs`). The
//! two are not bit-identical, so itx stays on the scalar Rust kernels.
//!
//! The remaining dav2d NEON DSP families were surveyed for AV2-bit-exact wiring
//! and all left on the scalar Rust kernels. Each diverges from AV2 in a way that
//! makes the dav2d NEON unsafe to call (it would either compute the wrong result
//! for AV2, or require an invasive struct-layout / decomposition rewrite that
//! the bit-exact corpus cannot risk). The scalar Rust kernels ARE the correct
//! AV2 implementation; matching dav2d's MC/ipred wins came precisely because
//! those leaf kernels are pure pixel math AV2 never touched.
//!
//!   * msac (`arm/64/msac.S`): the AV2 fork DROPPED the multi-symbol decoder
//!     (`decode_symbol_adapt4/8/16`) — the dominant entropy primitive — because
//!     AV2 changed the symbol decoder to the `dav2d_msac_min_prob` form
//!     (`v = (r*p>>10)<<3`); no `*symbol_adapt*_neon` symbol exists. Only the 4
//!     bool/bypass primitives remain, and they read `MsacContext` by fixed byte
//!     offset (BUF_POS=0, BUF_END=8, DIF=16, RNG=24, CNT=28, ALLOW_UPDATE_CDF=32)
//!     — rav2d's `MsacContext` holds a borrow-checked `&[u8]` (fat ptr+len) with
//!     no raw `buf_end`, so the layout cannot be matched without a pervasive,
//!     load-bearing `#[repr(C)]` rewrite for only the minority bool paths.
//!   * filmgrain (`arm/64/filmgrain.S`): the asm-offsets (`arm/asm-offsets.h`:
//!     `FGD_SEED=0, FGD_SCALING_SHIFT=88, FGD_AR_COEFFS_Y=96, FGD_AR_COEFFS_UV=120`)
//!     describe the AV1 `Dav2dFilmGrainData` (leading `seed`, split 24-entry
//!     `ar_coeffs_y`/`ar_coeffs_uv`). AV2's `Dav2dFilmGrainData` was reorganized
//!     (leading `chroma_scaling_from_luma`, unified `ar_coeffs[3][28]`, `seed`
//!     moved into `Dav2dFrameHeader.film_grain`), so the offsets are STALE — the
//!     NEON would read the wrong fields. Grain synthesis stays scalar.
//!   * loopfilter/deblock (`arm/64/loopfilter.S`): dav2d DID port deblock to AV2
//!     (the `lpf_*_sb_*` ABI takes a packed `vmask` + `l[4]` level grid + an
//!     `Av2FilterLUT` with separate q/side thresholds), but rav2d's scalar
//!     deblock uses a different decomposition (`deblock_sbrow_cols/rows` with
//!     per-edge q_thr/side_thr LUTs, no `vmask` bitmask). Wiring would require
//!     restructuring the most complex filter module to emit that exact mask/lut
//!     layout — too high a bit-exact risk for the win.
//!   * cdef (`arm/64/cdef.S`): dav2d splits CDEF into `cdef_padding{4,8}` (writes
//!     a `uint16_t` tmp buffer) + `cdef_filter{4,8}`; rav2d uses a fused
//!     `cdef_filter_block_8bpc` and folds CCSO into the CDEF superblock-row loop.
//!     The decompositions don't line up at a kernel boundary that can be guard-
//!     tested cheaply; left scalar.
//!   * refmvs (`arm/64/refmvs.S`): `splat_mv` loads a 16-byte `refmvs_block`
//!     template and writes 48-byte rows assuming the AV1 packing, but rav2d's
//!     `Block` is the 64-byte AV2-extended struct (adds `subpel_filter`,
//!     `warp_type`, `lmv[2]`, `m[6]`), and rav2d's `splat_mv` also splats the
//!     temporal MV grid in the same pass. Incompatible; left scalar. (`load_tmvs`
//!     is `#if ARCH_AARCH64 && 0`-disabled in dav2d itself.)
//!
//! There is no `looprestoration.S`: AV2's Wiener/PC-Wiener/GDF loop restoration
//! has no NEON in dav2d, so LR is scalar by construction.
//!
//! The dav2d submodule is referenced read-only; nothing under `dav2d/src` or
//! `dav2d/include` is modified.
//!
//! On other architectures this is a no-op and the decoder uses the scalar Rust
//! kernels in `src/mc.rs` / `src/itx.rs` / `src/ipred.rs`.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Declare the cfgs we may set so rustc's unexpected-cfg lint stays quiet.
    println!("cargo:rustc-check-cfg=cfg(rav2d_neon_mc)");
    println!("cargo:rustc-check-cfg=cfg(rav2d_neon_ipred)");
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
        build_neon_mc();
        build_neon_ipred();
    }
}

fn build_neon_mc() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dav2d = manifest.join("../../dav2d");
    let dav2d = dav2d.canonicalize().unwrap_or(dav2d).to_path_buf();
    let src = dav2d.join("src");
    let include = dav2d.join("include");
    let build = dav2d.join("build");
    let build_src = build.join("src");

    let mc_s = src.join("arm/64/mc.S");
    let mc_dotprod_s = src.join("arm/64/mc_dotprod.S");
    let config_h = build.join("config.h");

    // If the dav2d build artifacts (config.h, asm-offsets.h) aren't present we
    // can't assemble the kernels; fall back to scalar by skipping cleanly.
    if !mc_s.exists() || !config_h.exists() {
        println!(
            "cargo:warning=dav2d asm sources not found ({}); rav2d MC will use scalar Rust",
            mc_s.display()
        );
        return;
    }

    println!("cargo:rerun-if-changed={}", mc_s.display());
    println!("cargo:rerun-if-changed={}", mc_dotprod_s.display());
    println!("cargo:rerun-if-changed={}", src.join("arm/asm.S").display());
    println!(
        "cargo:rerun-if-changed={}",
        src.join("arm/64/util.S").display()
    );
    println!("cargo:rerun-if-changed={}", config_h.display());
    println!("cargo:rerun-if-changed=build.rs");

    // Generate the data-table C file from dav2d/src/tables.c (verbatim bytes) so
    // the asm's `dav2d_mc_subpel_filters` / `dav2d_mc_warp_filter` references
    // resolve. Regenerated each build; dav2d source is never modified.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tables_c = out_dir.join("mc_tables.c");
    generate_mc_tables(&src.join("tables.c"), &tables_c);
    println!("cargo:rerun-if-changed={}", src.join("tables.c").display());

    // Shared include dirs matching dav2d's own build (see build.ninja).
    let includes: [&Path; 6] = [
        &src.join("arm/64"), // util.S
        &src,                // src/arm/asm.S, src/mc.h
        &dav2d,              // "src/..." relative includes
        &include,            // common/attributes.h, dav2d/headers.h
        &build,              // config.h (#include "config.h")
        &build_src,          // generated headers
    ];

    let mut cc = cc::Build::new();
    cc.file(&mc_s).file(&mc_dotprod_s).file(&tables_c);
    for inc in includes {
        cc.include(inc);
    }
    // Match dav2d's release flags for the asm/data. NDEBUG keeps assert() out of
    // the asm macros; PIC is already defined in config.h so we don't pass -DPIC.
    cc.define("NDEBUG", None);
    cc.flag_if_supported("-std=c99");
    // The .S use the .arch/.arch_extension directives gated by config.h; clang
    // handles them. Silence the benign config.h-controlled warnings.
    cc.warnings(false);
    cc.compile("rav2d_mc_neon");

    // Tell the linker NEON MC is available so src/mc_neon.rs enables dispatch.
    println!("cargo:rustc-cfg=rav2d_neon_mc");
}

/// Shared include dirs matching dav2d's own asm build (see build.ninja).
fn dav2d_includes(
    src: &Path,
    dav2d: &Path,
    include: &Path,
    build: &Path,
    build_src: &Path,
) -> Vec<PathBuf> {
    vec![
        src.join("arm/64"), // util.S
        src.to_path_buf(),  // src/arm/asm.S, src/*.h
        dav2d.to_path_buf(),
        include.to_path_buf(),
        build.to_path_buf(),
        build_src.to_path_buf(),
    ]
}

/// Assemble dav2d's NEON intra-prediction kernels (`ipred.S`) plus the two
/// read-only tables they reference (`dav2d_sm_weights`, `dav2d_filter_intra_taps`),
/// copied verbatim from `dav2d/src/tables.c` into a generated C TU.
fn build_neon_ipred() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dav2d = manifest.join("../../dav2d");
    let dav2d = dav2d.canonicalize().unwrap_or(dav2d).to_path_buf();
    let src = dav2d.join("src");
    let include = dav2d.join("include");
    let build = dav2d.join("build");
    let build_src = build.join("src");

    let ipred_s = src.join("arm/64/ipred.S");
    let config_h = build.join("config.h");
    if !ipred_s.exists() || !config_h.exists() {
        println!(
            "cargo:warning=dav2d ipred asm not found ({}); rav2d ipred will use scalar Rust",
            ipred_s.display()
        );
        return;
    }

    println!("cargo:rerun-if-changed={}", ipred_s.display());
    println!("cargo:rerun-if-changed={}", src.join("tables.c").display());
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tables_c = out_dir.join("ipred_tables.c");
    generate_ipred_tables(&src.join("tables.c"), &tables_c);

    let mut cc = cc::Build::new();
    cc.file(&ipred_s).file(&tables_c);
    for inc in dav2d_includes(&src, &dav2d, &include, &build, &build_src) {
        cc.include(inc);
    }
    cc.define("NDEBUG", None);
    cc.flag_if_supported("-std=c99");
    cc.warnings(false);
    cc.compile("rav2d_ipred_neon");

    println!("cargo:rustc-cfg=rav2d_neon_ipred");
}

/// Copy the two intra-prediction data tables out of `dav2d/src/tables.c`
/// verbatim into a small standalone C translation unit so the asm's
/// `dav2d_sm_weights` / `dav2d_filter_intra_taps` references resolve.
fn generate_ipred_tables(tables_c: &Path, out: &Path) {
    let src = std::fs::read_to_string(tables_c).expect("read dav2d tables.c");
    let lines: Vec<&str> = src.lines().collect();

    let grab = |marker: &str| -> String {
        let start = lines
            .iter()
            .position(|l| l.contains(marker))
            .unwrap_or_else(|| panic!("table {marker} not found in tables.c"));
        let mut out = Vec::new();
        for l in &lines[start..] {
            out.push(*l);
            if l.trim() == "};" {
                break;
            }
        }
        out.join("\n")
    };

    let sm_weights = grab("dav2d_sm_weights[3 /* scale */][64]");
    // `dav2d_filter_intra_taps` is built with a local `F(...)` initializer macro
    // (and `ATTR_MCMODEL_SMALL`); grab from the `#if ARCH_X86` guard that defines
    // it so the verbatim copy compiles standalone.
    let filter_taps = grab("#if ARCH_X86");

    let content = format!(
        "/* Generated by rav2d build.rs from dav2d/src/tables.c (verbatim). Provides\n\
         * the read-only data tables dav2d's NEON ipred kernels reference. Do not edit. */\n\
         #include \"config.h\"\n\
         #include <stdint.h>\n\
         #include \"common/attributes.h\"\n\
         #include \"dav2d/headers.h\"\n\n\
         {sm_weights}\n\n\
         {filter_taps}\n"
    );
    std::fs::write(out, content).expect("write generated ipred_tables.c");
}

/// Copy the two MC data tables out of `dav2d/src/tables.c` verbatim into a small
/// standalone C translation unit. Only `dav2d_mc_subpel_filters` and
/// `dav2d_mc_warp_filter` are needed by the asm.
fn generate_mc_tables(tables_c: &Path, out: &Path) {
    let src = std::fs::read_to_string(tables_c).expect("read dav2d tables.c");
    let lines: Vec<&str> = src.lines().collect();

    let grab = |marker: &str| -> String {
        let start = lines
            .iter()
            .position(|l| l.contains(marker))
            .unwrap_or_else(|| panic!("table {marker} not found in tables.c"));
        let mut out = Vec::new();
        for l in &lines[start..] {
            out.push(*l);
            if l.trim() == "};" {
                break;
            }
        }
        out.join("\n")
    };

    let subpel = grab("dav2d_mc_subpel_filters[6][15][8]");
    let warp = grab("dav2d_mc_warp_filter[7*64+1][8]");

    let content = format!(
        "/* Generated by rav2d build.rs from dav2d/src/tables.c (verbatim). Provides\n\
         * the read-only data tables dav2d's NEON MC kernels reference. Do not edit. */\n\
         #include \"config.h\"\n\
         #include <stdint.h>\n\
         #include \"common/attributes.h\"\n\
         #include \"dav2d/headers.h\"\n\n\
         {subpel}\n\n\
         {warp}\n"
    );
    std::fs::write(out, content).expect("write generated mc_tables.c");
}

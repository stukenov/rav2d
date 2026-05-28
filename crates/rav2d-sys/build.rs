use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dav2d_dir = manifest_dir.join("../../dav2d");
    let dav2d_build = dav2d_dir.join("build");
    let include_dir = dav2d_dir.join("include");

    println!("cargo:rerun-if-changed=../../dav2d/include");

    // bindgen needs both source include dir and build dir (for generated headers)
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("dav2d/dav2d.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", dav2d_build.join("include").display()))
        .allowlist_function("dav2d_.*")
        .allowlist_type("Dav2d.*")
        .allowlist_var("DAV2D_.*")
        .generate()
        .expect("failed to generate dav2d bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings");

    // Try local build first, then pkg-config, then fail
    if link_local_build(&dav2d_build) {
        return;
    }
    if pkg_config_probe() {
        return;
    }

    eprintln!(
        "dav2d not found. Build it first: cd dav2d && mkdir build && cd build && meson setup .. && ninja"
    );
    std::process::exit(1);
}

fn link_local_build(build_dir: &std::path::Path) -> bool {
    let lib_dir = build_dir.join("src");
    if lib_dir.join("libdav2d.dylib").exists()
        || lib_dir.join("libdav2d.so").exists()
        || lib_dir.join("libdav2d.a").exists()
    {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dav2d");
        return true;
    }
    false
}

fn pkg_config_probe() -> bool {
    match std::process::Command::new("pkg-config")
        .args(["--libs", "--cflags", "dav2d"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for flag in stdout.split_whitespace() {
                if let Some(lib) = flag.strip_prefix("-l") {
                    println!("cargo:rustc-link-lib={lib}");
                } else if let Some(path) = flag.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={path}");
                }
            }
            true
        }
        _ => false,
    }
}

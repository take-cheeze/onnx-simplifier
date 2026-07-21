//! Build script for `onnxsim-sys`.
//!
//! It links the `onnxsim_c` shared library, obtaining it in one of three ways:
//!
//! 1. **Skip** — if `DOCS_RS` or `ONNXSIM_NO_BUILD` is set, emit nothing and let
//!    the crate type-check without linking (used by `cargo check` and docs.rs).
//! 2. **Pre-built** — if `ONNXSIM_LIB_DIR` is set, link against a `onnxsim_c`
//!    library already present in the listed directories (`:`-separated). This is
//!    the fast path for CI and for users who built onnxsim separately.
//! 3. **From source** — otherwise, configure and build the C API target with
//!    CMake. This compiles the full onnxsim stack (ONNX Runtime, onnx-optimizer,
//!    protobuf) and is correspondingly slow the first time.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ONNXSIM_NO_BUILD");
    println!("cargo:rerun-if-env-changed=ONNXSIM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ONNXSIM_SOURCE_DIR");

    // Mode 1: skip building entirely (docs.rs / `cargo check`).
    if env::var_os("DOCS_RS").is_some() || env::var_os("ONNXSIM_NO_BUILD").is_some() {
        return;
    }

    // Mode 2: link against a pre-built library.
    if let Some(lib_dirs) = env::var_os("ONNXSIM_LIB_DIR") {
        for dir in env::split_paths(&lib_dirs) {
            add_search_dir(&dir);
        }
        println!("cargo:rustc-link-lib=dylib=onnxsim_c");
        return;
    }

    // Mode 3: build from source with CMake.
    let source_dir = source_dir();
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("onnxsim/capi/onnxsim_c_api.cpp").display()
    );

    let dst = cmake::Config::new(&source_dir)
        .define("ONNXSIM_C_API", "ON")
        .define("ONNXSIM_BUILTIN_ORT", "ON")
        .define("ONNXSIM_PYTHON", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        // No install() rules exist for onnxsim_c, so build the target in place
        // and locate the artifacts under the CMake build tree ourselves.
        .build_target("onnxsim_c")
        .very_verbose(false)
        .build();

    let build_dir = dst.join("build");
    for dir in find_lib_dirs(&build_dir) {
        add_search_dir(&dir);
    }
    println!("cargo:rustc-link-lib=dylib=onnxsim_c");

    // Expose where the shared libraries live so downstream crates / test
    // harnesses can set LD_LIBRARY_PATH if needed.
    println!("cargo:root={}", build_dir.display());
}

/// Root of the onnxsim C++ project (the directory holding the top-level
/// `CMakeLists.txt`). Defaults to two levels up from this crate.
fn source_dir() -> PathBuf {
    if let Some(dir) = env::var_os("ONNXSIM_SOURCE_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("onnxsim-sys must live at <repo>/rust/onnxsim-sys")
}

/// Emit link-search and rpath directives for a directory.
fn add_search_dir(dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    // Help the dynamic loader find onnxsim_c and its transitive dependencies at
    // runtime without requiring LD_LIBRARY_PATH to be set by hand.
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

/// Recursively find directories under `root` that contain a built shared
/// library, so both `onnxsim_c` and its transitive dependencies (ONNX Runtime,
/// onnx, ...) can be found at link and run time.
fn find_lib_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    collect_lib_dirs(root, &mut dirs);
    dirs.sort();
    dirs.dedup();
    dirs
}

fn collect_lib_dirs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    let mut has_lib = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_lib_dirs(&path, out);
        } else if !has_lib && is_shared_library(&path) {
            has_lib = true;
        }
    }
    if has_lib {
        out.push(dir.to_path_buf());
    }
}

fn is_shared_library(path: &Path) -> bool {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => return false,
    };
    if cfg!(target_os = "windows") {
        name.ends_with(".lib") || name.ends_with(".dll")
    } else if cfg!(target_os = "macos") {
        name.contains(".dylib")
    } else {
        name.contains(".so")
    }
}

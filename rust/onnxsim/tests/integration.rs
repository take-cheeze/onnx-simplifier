//! End-to-end tests that require the native `onnxsim_c` library to be linked.
//!
//! They are `#[ignore]`d so they don't run (or force a native build) during a
//! plain `cargo test`. Once the library is available, run them with:
//!
//! ```sh
//! cargo test -- --ignored
//! ```
//!
//! `roundtrip_simplify_path` additionally needs a real model; point it at one
//! with `ONNXSIM_TEST_MODEL=/path/to/model.onnx`.

#[test]
#[ignore = "requires the native onnxsim_c library"]
fn list_optimizers_is_non_empty() {
    let passes = onnxsim::list_optimizers();
    assert!(
        !passes.is_empty(),
        "expected at least one optimizer pass to be reported"
    );
}

#[test]
#[ignore = "requires the native onnxsim_c library and a model file"]
fn roundtrip_simplify_path() {
    let model = match std::env::var("ONNXSIM_TEST_MODEL") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("set ONNXSIM_TEST_MODEL to run this test");
            return;
        }
    };
    let dir = std::env::temp_dir();
    let out = dir.join("onnxsim_rust_test_out.onnx");
    onnxsim::simplify_path(&model, &out).expect("simplify_path should succeed");
    let simplified = std::fs::read(&out).expect("output model should exist");
    assert!(
        !simplified.is_empty(),
        "simplified model should not be empty"
    );
    let _ = std::fs::remove_file(&out);
}

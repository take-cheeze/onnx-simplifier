//! Simplify an ONNX model from the command line.
//!
//! Usage:
//!     cargo run --example simplify -- <input.onnx> <output.onnx>
//!
//! Requires the native `onnxsim_c` library to be built/linked (see the crate
//! README for build modes).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.onnx> <output.onnx>", args[0]);
        return ExitCode::from(2);
    }
    let input = &args[1];
    let output = &args[2];

    match onnxsim::simplify_path(input, output) {
        Ok(()) => {
            println!("simplified {input} -> {output}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

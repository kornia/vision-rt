//! Minimal TensorRT example: load any .engine, print I/O tensor info, run a zero-input inference.
//!
//! Usage:
//!   cargo run --example engine_info -- <path/to/model.engine>
//!
//! Does NOT require a specific model — works with any serialized TensorRT engine.
//! Good first smoke test to confirm the binding, CUDA, and TRT runtime all work.

use std::sync::Arc;
use trt::{Engine, Logger, Runtime, Session};
use trt::logger::Severity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: engine_info <engine_path>");
        eprintln!("       cargo run --example engine_info -- model.fp16.engine");
        std::process::exit(1);
    }
    let engine_path = &args[1];

    // ── Load engine ─────────────────────────────────────────────────────────
    let logger  = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine  = Engine::from_file(Arc::clone(&runtime), engine_path)?;

    // ── Print I/O tensor info ────────────────────────────────────────────────
    println!("\n=== TensorRT Engine: {} ===", engine_path);
    println!("I/O tensors ({}):", engine.specs().len());
    for spec in engine.specs() {
        println!("  [{:?}] {:?}  shape={:?}  name={}",
                 spec.mode, spec.dtype, spec.dims, spec.name);
    }

    // ── Create session ───────────────────────────────────────────────────────
    let mut session = Session::new(Arc::clone(&engine))?;

    // ── Build zero-filled inputs ─────────────────────────────────────────────
    let inputs: Vec<(String, Vec<f32>)> = engine
        .inputs()
        .map(|spec| {
            let n: usize = spec.dims.iter()
                .filter(|&&d| d > 0)
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            (spec.name.clone(), vec![0.0f32; n])
        })
        .collect();

    let input_refs: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();

    // ── Run inference ────────────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let outputs = session.run(&input_refs)?;
    let elapsed_ms = t0.elapsed().as_millis();

    // ── Print output shapes ──────────────────────────────────────────────────
    println!("\nInference time: {}ms", elapsed_ms);
    println!("Outputs:");
    // Sort for deterministic output
    let mut sorted: Vec<_> = outputs.iter().collect();
    sorted.sort_by_key(|(name, _)| name.as_str());
    for (name, tensor) in &sorted {
        let n_elems = tensor.data.len() / match tensor.dtype {
            trt::DataType::Float32 | trt::DataType::Int32 => 4,
            trt::DataType::Float16 => 2,
            _ => 1,
        };
        println!("  {} shape={:?} dtype={:?} ({} elements)",
                 name, tensor.shape, tensor.dtype, n_elems);
    }

    println!("\nOK — TensorRT binding works.");
    Ok(())
}

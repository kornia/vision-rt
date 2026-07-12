---
name: vrt-add-model-crate
description: Use when adding a new model crate to vision-rt (a new detector/segmenter/depth/pose net) — crate layout, the async caller-owned pattern, classifying engine outputs by shape, from_engine_file/from_onnx/from_hub constructors, vrt-hub registration, export/build scripts, and the build/clippy discipline. Copy vrt-rfdetr-seg or vrt-depth-anything as the template.
---

# Adding a vrt Model Crate

Every model crate is the **same shape**. Don't invent structure — copy the
closest sibling and adapt:

- **`vrt-rfdetr-seg`** (`crates/vrt-rfdetr-seg/src/lib.rs`) — multi-output engine
  (boxes + labels + masks), two GPU decode kernels, survivor-count readback.
  Copy this for detectors/segmenters.
- **`vrt-depth-anything`** (`crates/vrt-depth-anything/src/lib.rs`) — single dense
  output, plus GPU **fusion** kernels that consume another model's device output.
  Copy this for dense-map models or anything that fuses with a detector.

## Crate layout

```
crates/vrt-<model>/
  Cargo.toml           # [lib] name = "vrt_<model>", thiserror, feature gates
  src/lib.rs           # the whole crate: error enum, *Result, payload type, kernels
  examples/<demo>.rs   # a runnable demo (from_engine_file, one loop)
  scripts/
    export_<model>.py  # PyTorch/ONNX export (documents transformers pins etc.)
    build_engine.sh    # trtexec → vrt-named .engine, on-device
  README.md
```

`Cargo.toml` essentials (see `crates/vrt-rfdetr-seg/Cargo.toml`):

```toml
[lib]
name = "vrt_<model>"           # short underscore name — code does `use vrt_<model>::`

[features]
default = []
hub     = ["dep:vrt-hub", "vrt-hub/hub"]       # auto-pull pinned ONNX from HF
builder = ["dep:vrt-hub", "vrt-hub/builder"]    # build engine in-process (nvonnxparser)

[dependencies]
vrt = { path = "../vrt" }
vrt-types = { path = "../vrt-types" }           # share Detection/Mask/DepthImage
vrt-hub = { path = "../vrt-hub", optional = true }
kornia-image = { workspace = true }
kornia-tensor = { workspace = true }
kornia-imgproc = { workspace = true }
cudarc = { workspace = true }
thiserror = "2.0"

[dev-dependencies]
vrt-hub = { path = "../vrt-hub", features = ["builder"] }
```

## The payload type (build once, reuse per frame)

A plain struct owning `ModelSession` + a kornia `Preprocessor` + the reused
`input: Tensor<f32,4>` + any decode/fusion `CudaKernel`s + the shared
`Arc<CudaStream>` + the shapes read from the engine. No framework, no trait to
implement. Errors: a per-crate `thiserror` enum wrapping `vrt::TrtError`,
`DriverError`, `kornia CudaError`, `PreprocessError`, plus `MissingOutput` (see
`crates/vrt-rfdetr-seg/src/lib.rs:37-50`). Constructors that aggregate kinds
return `vrt::BoxError`.

Preprocessor is built with the exact resize/normalize the export expects, e.g.
`PreprocessorBuilder::new().mode(ResizeMode::Stretch).normalize(Normalize::imagenet()).build_cuda(stream.clone())`
(`crates/vrt-depth-anything/src/lib.rs:236`). Full-frame `Stretch` is what makes
cross-model coordinates a plain scalar ratio downstream.

## Classify engine outputs by SHAPE, not name

Export tools rename tensors; **bind outputs by their dims**, with positive-dim
guards so a dynamic `-1` (which wraps to a huge `usize`) is rejected. RF-DETR-Seg
disambiguates three outputs purely by shape
(`crates/vrt-rfdetr-seg/src/lib.rs:296-309`):

```rust
for s in engine.outputs() {
    match s.dims.as_slice() {
        [1, nq, nh, nw] if *nq > 0 && *nh > 0 && *nw > 0 => { /* masks */ }
        [1, nq, 4]      if *nq > 0                       => { /* boxes cxcywh */ }
        [1, nq, ncl]    if *nq > 0 && *ncl > 0 && *ncl != 4 => { /* labels */ }
        _ => {}
    }
}
```

Read the static input `[1,3,H,W]` the same way and reject non-static shapes early.

## Caller-owned, GPU-resident `*Result` (VPI-style)

Mirror `SegResult` / `DepthResult`: allocate device buffers **once** via
`Model::alloc_result()`, fill them in `submit(&img, &mut result)` with **no sync
and no host copy** (only a tiny count scalar async-copied to a
`vrt::PinnedBuffer`). Expose:

- host accessors that copy **on request** (`detections()`, `masks_host()`,
  `depth_host()`), bounded by `count()`;
- GPU-resident `*_slice() -> &CudaSlice<T>` for downstream fusion.

`submit` = `preproc.run(img, &mut self.input)` → `self.model.run(&self.input)` →
launch decode kernels on the shared stream (`crates/vrt-rfdetr-seg/src/lib.rs:414`).
TRT output pointers come from `tmap.get(name)?.f32_ptr()? as usize as CUdeviceptr`
(see `rust-cuda-patterns` for the pointer/launch mechanics). A caller-owned copy
is mandatory when a result must outlive the next `run` — TRT output views alias
session memory (`crates/vrt-depth-anything/src/lib.rs:60`).

## Constructors + feature gating

Provide the standard three, gated exactly like the siblings
(`crates/vrt-rfdetr-seg/src/lib.rs:283-391`):

- `new(engine: Arc<Engine>, stream, …)` — the real constructor.
- `from_engine_file(path, stream, …)` — always available, `Engine::load`.
- `from_onnx(path, stream, …)` — `#[cfg(any(feature = "hub", feature = "builder"))]`;
  builds+caches via `vrt_hub::EngineCache::default().resolve(name, onnx, &Self::engine_profile())`.
- `from_hub(stream, …)` — `#[cfg(feature = "hub")]`;
  `vrt_hub::resolve_engine(name, &Self::engine_profile())`.
- `engine_profile()` under the same cfg returns `vrt_hub::EngineProfile { input: None, fp16: true, workspace_mb: 2048 }` for a static-shape model.

## Scripts (vrt engine naming)

- `scripts/export_<model>.py` — produce the ONNX; document exact library pins in
  the docstring (e.g. RF-DETR-Seg needs `transformers>=5.1`, installed in
  isolation on `PYTHONPATH` — `crates/vrt-rfdetr-seg/scripts/export_rfdetr_seg.py:1-20`).
- `scripts/build_engine.sh` — trtexec → engine named to the vrt convention
  `<model>-trt<M.m.p.b>-sm<cc>-fp16.engine` (e.g.
  `rfdetr-seg-preview-trt10.3.0.30-sm87-fp16.engine`), deriving TRT version from
  `NvInferVersion.h` and `sm` from torch (`crates/vrt-rfdetr-seg/scripts/build_engine.sh`).
  Engines are machine-locked — always build on-device (see `trt-engine-rebuild`).

## Register in vrt-hub (sha256-pinned)

Add a `ModelSpec` to `REGISTRY` in `crates/vrt-hub/src/lib.rs:106` — HF repo +
revision, `files` (entry ONNX first, sidecars after, each with a `sha256sum`
pin), and optional `engines` (prebuilt, guarded by exact `trt_version` + `sm`;
downloaded only on a matching box, else built from ONNX). Copy the `rfdetr-seg`
entry (`crates/vrt-hub/src/lib.rs:168-186`) verbatim and swap names/hashes.

## Build & check discipline

- **Off-Jetson / pre-commit:** `TRT_STUB=1 cargo clippy --workspace --all-targets -- -D warnings`.
  `TRT_STUB=1` uses committed bindings so nothing native compiles; kornia checks
  via cudarc `fallback-*`. This catches almost everything without CUDA.
- **On-device build:** `cargo build --release -j2` — **never** uncapped: parallel
  template builds OOM-kill the 7.4 GB box (CLAUDE.md "Hard constraints").
- **Tests:** CPU unit tests plainly; GPU kernel tests `#[ignore]`d, run with
  `cargo test -p vrt-<model> --release -- --ignored` on-device.

## Related skills

- `vrt-pipeline-compose` — how the new crate gets composed into a fast pipeline.
- `rust-cuda-patterns` / `cuda-kernel-craft` — the decode/fusion kernels.
- `trt-engine-rebuild` — building/debugging the `.engine`.
- `model-tensor-semantics` — a worked example of tensor shapes/coordinate spaces.

# vision-rt

Real-time neural-vision libraries for NVIDIA Jetson — TensorRT inference + GPU
pre/post-processing as plain Rust types. No orchestration framework: threading
and messaging are the application's job. GPU image/tensor types come from
[`kornia-rs`](https://github.com/kornia/kornia-rs); each model is its own crate
over a shared safe core.

**Target:** Jetson Orin (aarch64), JetPack 6.x, TensorRT 10.3.x, CUDA 12.6.
**Internals:** see [ARCHITECTURE.md](ARCHITECTURE.md).

## Workspace

| Crate | Role |
|---|---|
| `trt-sys` | Raw FFI: pure-C shim over TensorRT (bindgen) |
| `vrt` | Safe core: `Logger→Runtime→Engine→Session`, `ModelSession`, CUDA helpers |
| `vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |
| `vrt-xfeat` | XFeat keypoints + descriptors + GPU mutual-NN matching |
| `vrt-rfdetr` | RF-DETR object detector (NMS-free) + GPU decode |

## Usage

Each model is a payload + a caller-owned output. The library is **fully async —
you own the one sync per frame** (VPI-style):

```rust
let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
let mut xfeat = XFeat::from_hub(stream.clone(), XFeatParams::new(2048, 0.05))?;

let mut res = xfeat.alloc_result()?;    // reuse across frames
xfeat.submit(&image, &mut res)?;        // enqueue (resize → backbone → top-K), no sync
stream.synchronize()?;                   // you own the one sync
let kpts = res.kpts_to_host()?;          // original-image pixels
```

Construct with `from_hub` (pull from HF), `from_onnx` (local ONNX),
`from_engine_file` (prebuilt engine), or `new(engine, …)`. RF-DETR is the same
shape (`RfDetr` → `Vec<Detection>`).

## Models & engines

- **ONNX is the portable artifact** — hosted on Hugging Face (`kornia/*`),
  sha256-pinned, never committed. Private/gated repo → export `HF_TOKEN`.
- **Engines are machine-locked** (TRT version + GPU arch), built **on-device**
  into `~/.cache/vision-rt/engines/…` (first run only); a matching prebuilt
  engine may instead be pulled from HF.

Model credit belongs to the upstream authors — see each crate's README.

## Examples

Per-crate `examples/`:

```bash
cargo run --release -p vrt-xfeat  --example xfeat_match  -- <onnx|engine> map.jpg query.jpg out.png
cargo run --release -p vrt-xfeat  --example xfeat_detect -- <onnx|engine> image.jpg out.png
cargo run --release -p vrt-rfdetr --example rfdetr_detect -- <onnx|engine> image.jpg 0.5
```

## Building

```bash
cargo build --release -j2                         # -j2: the Orin Nano OOM-kills parallel builds
cargo test -p vrt-xfeat --release -- --ignored    # on-device GPU tests
TRT_STUB=1 cargo clippy --all-targets             # off-Jetson (committed bindings, no CUDA/TRT)
```

## License

Apache-2.0

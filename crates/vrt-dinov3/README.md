# vrt-dinov3

DINOv3 **global image descriptors** on TensorRT: one L2-normed vector per frame, plus a
GPU cosine bank for retrieval.

`DinoV3` is an `Image<u8,3> → [f32; D]` whole-frame embedder on a DINOv3 ViT-S/16 export.
The descriptor is the model's own **CLS token** (HF `pooler_output`) — what DINOv3 pools
into by default — L2-normalized on device so every downstream cosine is a plain dot
product. Useful for retrieval, visual place recognition, relocalization candidates and
scene-change detection. Model credit to the upstream authors (Siméoni et al., Meta AI).

Async / caller-owned (VPI-style) like the sibling crates: `submit` enqueues
stretch+normalize → TRT → L2-norm into a caller-owned `DinoV3Result` with **no sync and
no host copy**; the caller syncs the shared stream once, then reads.

```rust
let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
let mut dino = DinoV3::from_engine_file(engine, stream.clone())?;
let mut bank = DescriptorBank::new(512, dino.dim(), stream.clone())?;
let mut r = dino.alloc_result()?;                     // allocated ONCE
let mut scores = stream.alloc_zeros::<f32>(512)?;     // allocated ONCE

dino.submit(&img, &mut r)?;                           // enqueue, no sync
bank.match_into(r.descriptor_slice(), &mut scores)?;  // enqueue, no sync
stream.synchronize()?;                                 // the ONE sync
let s = stream.clone_dtoh(&scores.slice(0..bank.len()))?;  // 2 KB, post-sync
```

## Engine I/O

Input `[1,3,S,S]` with `S % 16 == 0` (Stretch + ImageNet norm). Outputs are bound **by
shape**, not name — export tools rename tensors, and the two ranks are unambiguous:

| Shape | Role |
|-------|------|
| `[1,D]` | the global descriptor (`pooler_output`) — **required** |
| `[1,N,D]` | the full token sequence (`last_hidden_state`) — optional |

At `S=336`: `D=384`, `N=446` = 1 CLS + 4 registers + 441 patches (21×21 grid).

When the token output is present the crate copies out the **patch grid only** — the CLS
and register tokens are skipped, so `patch_tokens_slice()` never contains DINOv3's 4
register artifacts. The prefix length is *derived* from `N - (S/16)²` rather than
hardcoded to 5, so register-free variants work with no code change.

Binding the token output costs an output buffer (677 KB at S=336) and **zero FLOPs** —
the ViT computes every patch token regardless; only the binding differs. That is why the
export ships both: recovering dense features later would otherwise mean a re-export, an
engine rebuild and a fresh sha256 pin.

The crate reads `S` from the engine, so a 224 build works unchanged.

## Precision: use bf16. fp16 is broken for this model.

All **measured** on this box (TRT 10.3.0.30 / SM87), descriptor cosine vs the PyTorch
reference:

| build | GPU compute | throughput | cosine vs PyTorch | |
|-------|------------:|-----------:|------------------:|--|
| **bf16 (recommended)** | **7.64 ms** | 130 qps | **0.999568** | ✅ 1.75× faster than fp32 |
| fp32 | 13.39 ms | 74 qps | 0.999998 | ✅ |
| fp16 | 5.23 ms | 191 qps | **all NaN** | ❌ |
| fp16 + bf16 (TRT picks per layer) | 5.24 ms | 190 qps | **all NaN** | ❌ |
| fp16 + `--layerPrecisions=…:fp32` | 5.24 ms | 190 qps | **all NaN** | ❌ pins ignored — see below |

### Why fp16 fails

Dumping every intermediate of the fp32 graph under onnxruntime finds **exactly one**
tensor outside fp16 range: the attention logits `Q·Kᵀ`, shape `(1, 6, 446, 446)`, max
|value| = **2,138,656**. fp16 tops out at 65,504, so it overflows to `inf` in the *first*
block, `softmax(inf − inf)` = NaN, and NaN reaches every output. Every other tensor in
the graph stays under 1800.

Checked on both a random reference image and a natural photo; the natural one is *worse*
(**2,499,626**), so this is not an artefact of a synthetic input.

Those are already-scaled logits — DINOv3 genuinely produces massive attention values (the
"attention sink" its 4 register tokens exist to absorb). So this is a **dynamic-range**
problem, not a precision one, which is exactly what bf16 fixes: bf16 keeps fp32's 8-bit
exponent (max ~3.4e38) and gives up mantissa bits instead. Hence 0.999568 rather than
fp32's 0.999998 — still comfortably past the 0.999 gate.

### Don't try to rescue fp16 with `--layerPrecisions`

TensorRT fuses the whole attention block into a myelin kernel, so the ONNX node names are
**gone** from the engine — real layer names look like
`__myl_ResTraConCasMeaSubMulMeaAddSqrDivMulCasMulAdd_myl2_1`. Patterns like `*norm*`, and
even explicit `node_MatMul_148:fp32` for all 12 QK matmuls, match nothing and are silently
ignored: the build succeeds, runs at identical speed, and still emits NaN. Verified both
ways. This is worth knowing generally — a `--precisionConstraints=obey` build that does
not change the timing at all is a sign the pins never applied.

### All four constructors give you bf16

`vrt_hub::EngineProfile` carries a `bf16: bool` (threaded down through
`vrt::builder::EngineBuilder::bf16` to `BuilderFlag::kBF16` in the `trt-sys` shim), and
`DinoV3::engine_profile()` sets `fp16: false, bf16: true`. So `from_onnx()` and
`from_hub()` build the 7.6 ms engine on-device, matching what `build_engine.sh` produces —
verified end-to-end by `from_onnx_builds_a_correct_bf16_engine` (cosine 0.999568, the same
number as the trtexec-built engine).

`EngineProfile::cache_tag()` hashes `bf16` too, so a bf16 engine can never collide with a
previously-cached fp32 one.

## Building the weights

The upstream HF repo is **gated** — accept the licence at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> and export an `HF_TOKEN`
first.

```bash
HF_TOKEN=hf_... python3 crates/vrt-dinov3/scripts/export_dinov3.py \
    --input-size 336 \
    --out models/onnx/dinov3-vits16-336.onnx \
    --dump-ref models/onnx/dinov3-ref

crates/vrt-dinov3/scripts/build_engine.sh models/onnx/dinov3-vits16-336.onnx
```

The export script reports the patch grid, the derived prefix-token count, and
`cos(pooler_output, last_hidden_state[:,0])` — the crate's premise is that the descriptor
*is* the CLS token, so it verifies that rather than assuming it. It also runs
`onnx.checker` and scans for TRT-hostile ops (`NonZero`/`If`/`Loop`/…), which is far
cheaper than debugging a failed `trtexec` parse.

`--dump-ref` writes the parity fixtures: an `S×S` u8 RGB image plus the reference
descriptor. The image is exactly the engine's input size **on purpose** — kornia's Stretch
resize is then the identity, so the Rust test can push it through the crate's real public
path (preprocessor included) and any difference is genuine numerics rather than a
resampling mismatch.

The export produces **two** files: `dinov3-vits16-336.onnx` (a 1.2 MB graph) and
`dinov3-vits16-336.onnx.data` (86 MB of weights). torch's dynamo exporter externalizes
weights regardless of the 2 GB protobuf limit, so the sidecar is mandatory — keep the pair
together, and note the ONNX parser finds it next to the `.onnx` (no `pushd` needed, unlike
the older `scripts/trtexec.sh` xfeat path).

You do not have to do any of that, though — the export is published at
<https://huggingface.co/kornia/dinov3> (both files, sha256-pinned in `vrt-hub`, shipped
with the DINOv3 Licence and attribution as its redistribution terms require). So:

```rust
let dino = DinoV3::from_hub(stream.clone())?;   // downloads, verifies, builds bf16, caches
```

The first call downloads ~87 MB and builds the engine on-device (~2 min); later calls are
cache hits. `from_onnx` on the same file lands on the *same* cached engine — the key is
`(name, onnx_sha8, profile_tag, trt_version, sm)`, so the route you took to it does not
matter.

Build it yourself only if you want a different `--input-size` or precision.

## Tests

```bash
# CPU, no GPU — the shape/binding matcher
TRT_STUB=1 cargo test -p vrt-dinov3

# On-device: parity (the fp16 gate), discrimination, and the bank kernel
DINOV3_ENGINE=models/engines/dinov3-vits16-336-trt10.3.0.30-sm87-bf16.engine \
DINOV3_REF_DIR=models/onnx/dinov3-ref \
    cargo test -p vrt-dinov3 --release -- --ignored
```

`descriptor_matches_pytorch_reference` asserts **cosine > 0.999**, not an element-wise
tolerance — cosine is what the application actually uses. Do not skip it: a
silently-wrong low-precision ViT still emits a plausible-looking 384-d unit vector, and
the failure surfaces only as mysteriously poor retrieval.

`descriptor_discriminates` catches what parity alone can miss — a mis-bound output that
is numerically stable but carries no scene information.

**The suite is entirely synthetic**, by choice: no image fixtures are committed. The
parity reference is a fixed-seed random image (fine by construction — it compares the
engine against PyTorch on the *same* input), and the discrimination test uses a generated
structured image. The real-image separation numbers below came from a manual
`dinov3_match` run and are **not** regression-tested. If you want that coverage, point
`dinov3_match` at your own data rather than adding fixtures here.

## Examples

```bash
# Two images (or N — prints the full similarity matrix)
cargo run --release -p vrt-dinov3 --example dinov3_match -- <engine> a.png b.png

# Live: RTSP → descriptor → keyframe bank retrieval (workspace-excluded, needs sensor-rtsp)
export CARGO_NET_GIT_FETCH_WITH_CLI=true CARGO_BUILD_JOBS=2
cargo run --release --manifest-path examples/rtsp_dinov3/Cargo.toml \
    -- <engine> rtsp://<camera>/stream --port 8080
```

The RTSP example enrolls keyframes and re-matches them. **The test that matters is not the
frame rate**: walk the camera away from a scene and back, and confirm it re-matches the
original keyframe on return. Its live view is opt-in (`--port`) precisely because it is
the only full-frame D2H in the program — without it the pipeline stays entirely on the GPU.

## Benchmark

Jetson Orin (SM87, TRT 10.3.0.30, fp32, `trtexec --iterations=200`, engine-only GPU
compute). **Measured at the box's default power mode, not MAXN** — re-run under
`sudo nvpmodel -m 2 && sudo jetson_clocks` before quoting these as peak:

| Input | Precision | GPU compute | Throughput | Note |
|-------|-----------|------------:|-----------:|------|
| **336×336** (21×21 patches) | **bf16** | **7.64 ms** | ~130 fps | recommended — what `from_onnx`/`from_hub` build; p99 7.69, tight |
| 336×336 | fp32 | 13.39 ms | ~74 fps | reference precision (cosine 0.999998) |
| 224×224 (14×14, native) | — | not built | — | ~2.2× cheaper if 336 is more than you need |

Attention is O(N²) in token count, so the 336→224 saving is larger than the pixel ratio
suggests. Resolution generally buys more for *dense* tasks than for a whole-image
descriptor, so benchmark both and keep whichever retrieves as well — the crate reads `S`
from the engine, so switching costs one export flag.

### End-to-end, live (`examples/rtsp_dinov3`)

1280×720 Tapo RTSP camera → descriptor → 512-slot bank, bf16 engine, live view off:

| Stage | ms | note |
|-------|---:|------|
| source (recv + enqueue) | 56.1 | **the bottleneck** — a 15 fps camera, blocking receive |
| enqueue (`submit`) | 1.93 | CPU kernel-launch, ≪ sync → genuinely async |
| match (512-slot bank) | 0.015 | free, as predicted |
| **sync (GPU)** | **6.8** | the real GPU wall |
| readout (K floats D2H) | 0.09 | on-demand, post-sync |
| **end-to-end** | | **15.0 fps — source-gated, not GPU-gated** |

The GPU spends 6.8 ms of a 66 ms frame interval, so this is **~10× GPU headroom**: room
for a much faster sensor, several cameras, or another model on the same stream. Note
`sync` here (6.8 ms) comes in *below* the 7.64 ms `trtexec` figure — the standalone
benchmark includes host-side binding overhead the in-pipeline path does not.

Retrieval behaved correctly on a static scene: one keyframe enrolled on frame 0, then
`best` held 0.994–0.997 for the whole run without spurious re-enrollment, which is exactly
what `tau = 0.75` should do given the measured separation.

First run on a cold page cache takes >70 s to reach the loop (deserializing the 47 MB
engine); warm it is ~7 s. Budget for that before assuming a stall.

### Measured descriptor separation

`examples/dinov3_match` on real images (EuRoC MH01 frames, a dog, an AprilTag board, a
kitchen scene) — this is what makes the retrieval demo's `--tau 0.75` default reasonable:

| pair | cosine |
|------|-------:|
| MH01 frame 1 ↔ frame 2 (same place, consecutive) | 0.9606 |
| MH01 frame 1 ↔ its 90 % centre crop (same place, shifted pose) | 0.9581 |
| MH01 ↔ dog / AprilTag / kitchen (unrelated) | −0.001 … 0.11 |

A very wide empty band between ~0.95 and ~0.11, so the enrollment threshold is not
delicately tuned — anything in roughly 0.5–0.9 behaves the same.

Worth knowing: DINOv3 maps **white noise** to nearly a single point (0.9934 same vs
0.9923 different). Synthetic-noise fixtures therefore make a discrimination test look
like it passes while proving nothing — which is why `tests/gpu.rs` uses a *structured*
synthetic image and asserts a margin rather than a bare inequality.

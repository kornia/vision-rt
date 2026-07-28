# vrt-dinov3

DINOv3 **global image descriptors** on TensorRT: one L2-normed vector per frame, plus a
GPU cosine bank for retrieval.

`DinoV3` is an `Image<u8,3> → [f32; D]` whole-frame embedder on a DINOv3 ViT-S/16 export.
The descriptor is the model's own **CLS token** (HF `pooler_output`), L2-normalized on
device so every downstream cosine is a plain dot product. Useful for retrieval, visual
place recognition, relocalization candidates and scene-change detection. Model credit to
the upstream authors (Siméoni et al., Meta AI).

Async / caller-owned (VPI-style) like the sibling crates: `submit` enqueues
stretch+normalize → TRT → L2-norm into a caller-owned `DinoV3Result` with **no sync and
no host copy**; the caller syncs the shared stream once, then reads.

```rust
let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
let mut dino = DinoV3::from_hub(stream.clone())?;      // or from_engine_file / from_onnx
let mut bank = DescriptorBank::new(512, dino.dim(), stream.clone())?;
let mut r = dino.alloc_result()?;                      // allocated ONCE
let mut scores = stream.alloc_zeros::<f32>(512)?;      // allocated ONCE

dino.submit(&img, &mut r)?;                            // enqueue, no sync
bank.match_into(r.descriptor_slice(), &mut scores)?;   // enqueue, no sync
stream.synchronize()?;                                  // the ONE sync
let s = stream.clone_dtoh(&scores.slice(0..bank.len()))?;   // 2 KB, post-sync
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
hardcoded, so register-free variants work with no code change.

Binding the token output costs an output buffer (677 KB at S=336) and **zero FLOPs** — the
ViT computes every patch token regardless. The crate reads `S` from the engine, so a 224
build works unchanged.

## Precision: build bf16, never fp16

**fp16 produces all-NaN for this model.** DINOv3's attention logits reach ~2.1e6 against
fp16's 65,504 ceiling, so they overflow to `inf` in the first block and `softmax(inf − inf)`
poisons every output. It is a dynamic-*range* problem, which is what bf16 fixes: it keeps
fp32's exponent and gives up mantissa instead.

Measured on Jetson Orin (TRT 10.3.0.30 / SM87), descriptor cosine vs the PyTorch reference:

| build | GPU compute | cosine | |
|-------|------------:|-------:|--|
| **bf16** | **7.64 ms** | **0.999568** | ✅ recommended |
| fp32 | 13.39 ms | 0.999998 | ✅ |
| fp16 | 5.23 ms | all NaN | ❌ |

Do **not** try to rescue fp16 with `--layerPrecisions`: TensorRT fuses the attention block
into a myelin kernel, so ONNX node names are absent from the engine and the pins are
silently ignored — the build succeeds, runs at identical speed, and still emits NaN.

`EngineProfile` carries `bf16` alongside `fp16` (they are mutually exclusive, and setting
both is rejected), `engine_profile()` requests bf16, and `cache_tag()` hashes it — so all
four constructors give you the same bf16 engine and it can never collide with a cached
fp32 one.

## Weights

Published at <https://huggingface.co/kornia/dinov3>, sha256-pinned in `vrt-hub` and shipped
with the DINOv3 Licence and attribution. `DinoV3::from_hub()` downloads, verifies and
resolves a bf16 engine; the first call builds on-device (~2 min), later ones are cache hits.

To build your own — e.g. a different `--input-size` — the upstream HF repo is **gated**:
accept the licence at <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>,
export an `HF_TOKEN`, then:

```bash
python3 crates/vrt-dinov3/scripts/export_dinov3.py \
    --input-size 336 --out models/onnx/dinov3-vits16-336.onnx \
    --dump-ref models/onnx/dinov3-ref
crates/vrt-dinov3/scripts/build_engine.sh models/onnx/dinov3-vits16-336.onnx
```

The export writes **two** files — the graph and a `.onnx.data` sidecar holding the weights
(torch externalizes them regardless of the 2 GB protobuf limit). Keep the pair together.
`--dump-ref` writes the parity fixtures used by the on-device tests.

## Tests

```bash
# CPU, no GPU — the shape/binding matcher
TRT_STUB=1 cargo test -p vrt-dinov3

# On-device: parity, discrimination, and the bank kernel
DINOV3_ENGINE=models/engines/dinov3-vits16-336-trt10.3.0.30-sm87-bf16.engine \
DINOV3_REF_DIR=models/onnx/dinov3-ref \
    cargo test -p vrt-dinov3 --release -- --ignored
```

The parity test asserts descriptor **cosine > 0.999** against a PyTorch reference — cosine
rather than an element-wise tolerance, because cosine is what the application uses and a
wrong low-precision ViT still emits a plausible unit vector. It is what caught the fp16 NaN.

The suite is entirely synthetic; no image fixtures are committed. The real-image separation
below came from a manual `dinov3_match` run and is not regression-tested.

## Examples

```bash
# Two images (or N — prints the full similarity matrix)
cargo run --release -p vrt-dinov3 --example dinov3_match -- <engine> a.png b.png

# Live: RTSP → descriptor → keyframe bank retrieval (workspace-excluded, needs sensor-rtsp)
export CARGO_NET_GIT_FETCH_WITH_CLI=true CARGO_BUILD_JOBS=2
cargo run --release --manifest-path examples/rtsp_dinov3/Cargo.toml \
    -- <engine> rtsp://<camera>/stream --port 8080
```

The RTSP example enrols keyframes and re-matches them. The test that matters is not the
frame rate: walk the camera away from a scene and back, and confirm it re-matches the
original keyframe on return.

## Benchmark

Jetson Orin, bf16, `trtexec --iterations=200` (engine-only), at the box's default power
mode — re-run under `sudo nvpmodel -m 2 && sudo jetson_clocks` before quoting as peak:

| Input | GPU compute | Throughput |
|-------|------------:|-----------:|
| **336×336** (21×21 patches) | **7.64 ms** | ~130 fps |

Attention is O(N²) in token count, so 224×224 is ~2.2× cheaper if 336 is more than you
need — and resolution buys more for dense tasks than for a whole-image descriptor.

End-to-end on a 1280×720 RTSP camera: **15.0 fps, camera-bound** — the GPU uses 6.8 ms of
the 66 ms frame interval (~10× headroom), with `enqueue` 1.9 ms and a 512-slot bank match
at 0.015 ms.

Measured descriptor separation on real images: two views of one place score **~0.96**,
unrelated scenes **−0.001 … 0.11**. That wide, empty band is why the retrieval demo's
`--tau 0.75` default is not delicately tuned.

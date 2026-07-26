#!/usr/bin/env bash
# Build a TensorRT engine from the DINOv3 ONNX, named to the vrt convention used by
# vrt-hub prebuilts:
#
#     <model>-trt<M.m.p.b>-sm<cc>[-<prec>].engine
#     e.g. dinov3-vits16-336-trt10.3.0.30-sm87-bf16.engine
#
# Engines are machine-locked (TensorRT version + GPU arch) — always build on the target
# device, never copy across hosts.
#
# ── PRECISION: use bf16. fp16 is BROKEN for this model. ───────────────────────────────
#
# Measured on this box (TRT 10.3.0.30 / SM87), descriptor cosine vs the PyTorch reference:
#
#   PREC=fp32   13.39 ms   0.999998   ✅
#   PREC=bf16    7.64 ms   0.999568   ✅  1.75x faster than fp32 — the default
#   PREC=fp16    5.23 ms   all NaN    ❌
#   fp16+bf16    5.24 ms   all NaN    ❌  (TRT picks fp16 for attention on speed)
#
# Root cause, found by dumping every intermediate of the fp32 graph under onnxruntime:
# exactly ONE tensor exceeds fp16 range — the attention logits Q·K^T, shape
# (1, 6, 446, 446), max |value| = 2,138,656. fp16 tops out at 65,504, so it overflows to
# inf in the FIRST block, softmax(inf - inf) = NaN, and the NaN propagates to every
# output. Every other tensor in the graph stays under 1800.
#
# These are already-scaled logits: DINOv3 genuinely produces massive attention values
# (the "attention sink" that its 4 register tokens exist to absorb). It is a dynamic
# RANGE problem, not a precision one — which is exactly what bf16 fixes, since bf16 keeps
# fp32's 8-bit exponent (max ~3.4e38) and only gives up mantissa bits.
#
# Do NOT try to rescue fp16 with --layerPrecisions: TensorRT fuses the whole attention
# block into a myelin kernel, so the ONNX node names are gone from the engine (real layer
# names look like `__myl_ResTraConCasMeaSubMulMeaAddSqrDivMulCasMulAdd_myl2_1`). Patterns
# such as `*norm*` or explicit `node_MatMul_148:fp32` match nothing and are silently
# ignored — the build "succeeds" at identical speed and still emits NaN. Verified.
#
# Whatever you pick, run the parity test before trusting it:
#   DINOV3_ENGINE=<engine> DINOV3_REF_DIR=models/onnx/dinov3-ref \
#       cargo test -p vrt-dinov3 --release --test gpu -- --ignored
#
# NOTE: `DinoV3::engine_profile()` sets `bf16: true`, so `from_onnx`/`from_hub` build the
# same precision as this script's default — you do not have to use this script to get
# bf16, it is just the offline path.
#
# NOTE: the export writes weights to a `<model>.onnx.data` sidecar. Keep it beside the
# .onnx — the parser resolves it there, so no `pushd` is needed.
#
# Usage:
#   crates/vrt-dinov3/scripts/build_engine.sh <model.onnx> [out_dir]
#   PREC=fp32 crates/vrt-dinov3/scripts/build_engine.sh <model.onnx>
set -euo pipefail

ONNX="${1:?usage: build_engine.sh <model.onnx> [out_dir]}"
OUT_DIR="${2:-models/engines}"
MODEL="dinov3-vits16-336"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PREC="${PREC:-bf16}"

case "$PREC" in
    bf16) PREC_ARGS=(--bf16); SUFFIX="-bf16" ;;
    fp32) PREC_ARGS=();       SUFFIX="" ;;
    fp16) PREC_ARGS=(--fp16); SUFFIX="-fp16"
          echo "WARNING: fp16 emits all-NaN for this model (attention logits reach 2.1e6," >&2
          echo "         fp16 max is 65504). See the header. Run the parity test." >&2 ;;
    *)    echo "PREC must be one of: bf16 (default), fp32, fp16 — got '$PREC'" >&2; exit 1 ;;
esac

# TRT version exactly as trt-sys parses NvInferVersion.h (MAJOR.MINOR.PATCH.BUILD);
# GPU compute capability (sm) from torch — together they key the engine.
HDR="$(ls /usr/include/*/NvInferVersion.h 2>/dev/null | head -1)"
[ -n "$HDR" ] || { echo "NvInferVersion.h not found — is TensorRT installed?" >&2; exit 1; }
ver() { grep -E "define NV_TENSORRT_$1 " "$HDR" | awk '{print $3}'; }
TRT="$(ver MAJOR).$(ver MINOR).$(ver PATCH).$(ver BUILD)"
SM="$(python3 -c 'import torch;print("%d%d"%torch.cuda.get_device_capability())')"

ENGINE="$OUT_DIR/${MODEL}-trt${TRT}-sm${SM}${SUFFIX}.engine"
mkdir -p "$OUT_DIR"
echo "building $ENGINE ($PREC, workspace 2 GB)…"
# shellcheck disable=SC2086  # TRTEXEC_EXTRA is intentionally word-split
"$TRTEXEC" \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    "${PREC_ARGS[@]}" \
    --memPoolSize=workspace:2048 \
    ${TRTEXEC_EXTRA:-}
echo "built $ENGINE"

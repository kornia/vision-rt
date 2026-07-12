#!/usr/bin/env bash
# Build a fp16 TensorRT engine from the Depth Anything V2 metric ONNX, named to the
# vrt-hub convention:
#
#     <model>-trt<M.m.p.b>-sm<cc>-fp16.engine
#     e.g. depth-anything-v2-metric-small-indoor-trt10.3.0.30-sm87-fp16.engine
#
# Engines are machine-locked (TensorRT version + GPU arch) — always build on the
# target device.
#
# ⚠️ fp16 gotcha: Depth Anything V2 has a known ONNX→TRT fp16 wrong-output bug
# (overflow in the normalization / final layers). After building, VALIDATE the depth
# numerically against the PyTorch reference on one image. If it is wrong, keep those
# layers in fp32 — re-run trtexec with e.g.:
#     --precisionConstraints=obey --layerPrecisions='*norm*:fp32,*head*:fp32'
# (exact layer names depend on the export; inspect with `polygraphy inspect model`).
#
# Usage:
#   crates/vrt-depth-anything/scripts/build_engine.sh <model.onnx> [out_dir]
set -euo pipefail

ONNX="${1:?usage: build_engine.sh <model.onnx> [out_dir]}"
OUT_DIR="${2:-models/engines}"
MODEL="depth-anything-v2-metric-small-indoor"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

# TRT version exactly as trt-sys parses NvInferVersion.h (MAJOR.MINOR.PATCH.BUILD);
# GPU compute capability (sm) from torch — together they key the engine.
HDR="$(ls /usr/include/*/NvInferVersion.h 2>/dev/null | head -1)"
[ -n "$HDR" ] || { echo "NvInferVersion.h not found — is TensorRT installed?" >&2; exit 1; }
ver() { grep -E "define NV_TENSORRT_$1 " "$HDR" | awk '{print $3}'; }
TRT="$(ver MAJOR).$(ver MINOR).$(ver PATCH).$(ver BUILD)"
SM="$(python3 -c 'import torch;print("%d%d"%torch.cuda.get_device_capability())')"

ENGINE="$OUT_DIR/${MODEL}-trt${TRT}-sm${SM}-fp16.engine"
mkdir -p "$OUT_DIR"
echo "building $ENGINE (fp16, workspace 2 GB)…"
"$TRTEXEC" \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    --memPoolSize=workspace:2048
echo "built $ENGINE — now VALIDATE depth vs the PyTorch reference (fp16 gotcha above)"

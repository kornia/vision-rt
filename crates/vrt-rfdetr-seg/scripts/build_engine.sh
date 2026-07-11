#!/usr/bin/env bash
# Build a fp16 TensorRT engine from the RF-DETR-Seg ONNX, named to the vrt
# convention used by vrt-hub prebuilts:
#
#     <model>-trt<M.m.p.b>-sm<cc>-fp16.engine
#     e.g. rfdetr-seg-preview-trt10.3.0.30-sm87-fp16.engine
#
# Engines are machine-locked (TensorRT version + GPU arch) — always build on the
# target device, never copy across hosts. Mirrors the profile the crate's
# `from_onnx`/`vrt-hub` cache uses (fp16, 2 GB workspace).
#
# Usage:
#   crates/vrt-rfdetr-seg/scripts/build_engine.sh <model.onnx> [out_dir]
set -euo pipefail

ONNX="${1:?usage: build_engine.sh <model.onnx> [out_dir]}"
OUT_DIR="${2:-models/engines}"
MODEL="rfdetr-seg-preview"
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
echo "built $ENGINE"

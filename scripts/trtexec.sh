#!/usr/bin/env bash
# Build TensorRT engines for all models in this repo.
#
# Usage:
#   cd /home/nvidia/trt-rs
#   bash scripts/trtexec.sh [model_name]
#
# Without arguments, builds everything.  With a model name, builds just that
# one (e.g. "bash scripts/trtexec.sh xfeat").
#
# Output engines are written alongside their ONNX sources in models/.
# Build is always on-device (engines are tied to this TRT version + SM87).

set -euo pipefail

TRTEXEC=/usr/src/tensorrt/bin/trtexec
REPO=/home/nvidia/trt-rs
MODELS=$REPO/models

build() {
    local name="$1"
    local onnx="$2"
    local engine="$3"
    shift 3          # remaining args forwarded to trtexec

    echo ""
    echo "════════════════════════════════════════"
    echo "  Building: $name"
    echo "  ONNX   : $onnx"
    echo "  Engine : $engine"
    echo "════════════════════════════════════════"

    if [[ ! -f "$onnx" ]]; then
        echo "  SKIP — ONNX not found: $onnx"
        return
    fi

    "$TRTEXEC" \
        --onnx="$onnx" \
        --saveEngine="$engine" \
        --fp16 \
        --memPoolSize=workspace:2048 \
        "$@" \
        2>&1 | grep -E "^\[|PASSED|FAILED|Timing|Latency|Throughput|Error|error" \
             | grep -v "^\[.\] \[V\]" || true

    if [[ -f "$engine" ]]; then
        size=$(du -sh "$engine" | cut -f1)
        echo "  → engine saved ($size)"
    else
        echo "  → engine build FAILED"
        return 1
    fi
}

TARGET="${1:-all}"

# ── YOLO11n ─────────────────────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "yolo" ]]; then
    build "YOLO11n (static 640×640, FP16)" \
        "$REPO/yolo11n.onnx" \
        "$REPO/yolo11n.fp16.engine" \
        --shapes=images:1x3x640x640
fi

# ── XFeat backbone (dynamic H×W, FP16) ──────────────────────────────────────
# Must run from the models/xfeat directory — weights are in xfeat_backbone.onnx.data
# (external data sidecar written by the dynamo exporter).
if [[ "$TARGET" == "all" || "$TARGET" == "xfeat" ]]; then
    pushd "$MODELS/xfeat" > /dev/null
    build "XFeat backbone (dynamic 240–768×320–1024, FP16)" \
        "xfeat_backbone.onnx" \
        "xfeat_backbone_fp16.engine" \
        --minShapes=image:1x3x240x320 \
        --optShapes=image:1x3x480x640 \
        --maxShapes=image:1x3x768x1024
    popd > /dev/null
fi

echo ""
echo "Done."

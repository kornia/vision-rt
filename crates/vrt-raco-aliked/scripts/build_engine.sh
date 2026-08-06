#!/usr/bin/env bash
# Build a TensorRT engine from either half of the split RaCo-ALIKED-LightGlue+ pipeline,
# named to the vrt convention used by vrt-hub prebuilts:
#
#     <model>-trt<M.m.p.b>-sm<cc>[-<prec>].engine
#     e.g. raco-aliked-extractor-k1024-trt10.3.0.30-sm87-fp16.engine
#
# Engines are machine-locked (TensorRT version + GPU arch) — always build on the target
# device, never copy across hosts.
#
# Feed it the output of split_raco_pipeline.py; the half is detected from the ONNX input
# names, and K is read out of the graph, so there is nothing to keep in sync by hand:
#
#   crates/vrt-raco-aliked/scripts/build_engine.sh models/onnx/raco/raco_aliked_extractor_k1024.onnx
#   crates/vrt-raco-aliked/scripts/build_engine.sh models/onnx/raco/lightglue_matcher_k1024.onnx
#
# ── PRECISION ─────────────────────────────────────────────────────────────────────────
#
# fp16 is the default and is what upstream recommends ("These ONNX models have been
# specially optimized for the TensorRT Execution Provider. It is recommended to enable
# FP16 for the best performance"). Note the released graph already contains the fix for
# the one known fp16 hazard: RaCo's flattened image indices exceed fp16's 65504 limit, so
# the export lowers that division to integer floor-division instead of routing it through
# a float. Do NOT convert the ONNX itself to fp16 (upstream's exporter warns against it);
# let TensorRT do the precision selection from the fp32 graph, which is what this does.
#
# If parity fails at fp16, try `PREC=fp16-strict` (adds --precisionConstraints=obey) or
# fall back to `PREC=fp32` before assuming the split is at fault.
#
# Always verify before trusting an engine:
#   python3 crates/vrt-raco-aliked/scripts/check_split_parity.py ... --dump-ref <dir>
#
# ── SHAPE PROFILES ────────────────────────────────────────────────────────────────────
#
# Extractor input `images` is (B,3,H,W) with H,W multiples of 32 (RaCo's
# input_dim_divisor). Matcher inputs are (2P,1,K,2) and (2P,1,K,128) — rank-4 padded so
# vrt's ModelSession::run_inputs, which binds &Tensor<f32,4>, can drive them. Images are
# stacked interleaved [L0,R0,L1,R1,...], hence the leading dim is always even.
#
# Override any of MIN_HW / OPT_HW / MAX_HW / MAX_PAIRS to widen or narrow the profile.
#
# Keep the defaults tight on a 7.4 GB Orin Nano. A wide profile is not free: TensorRT
# sizes its tactic workspaces from the MAX shape, so an over-generous max makes the
# builder skip most tactics ("Tactic Device request: 1296MB Available: 642MB. Device
# memory is insufficient") — which both stretches the build out enormously and leaves
# you with a *slower* engine, since the fast tactics are the ones that got skipped.
# MAX_PAIRS=1 (max batch 2, i.e. one image pair) is the right default here: the
# extractor is driven one image at a time and the matcher takes exactly one pair.
#
# Usage:
#   crates/vrt-raco-aliked/scripts/build_engine.sh <model.onnx> [out_dir]
#   PREC=fp32 MAX_HW=1024 crates/vrt-raco-aliked/scripts/build_engine.sh <model.onnx>
set -euo pipefail

ONNX="${1:?usage: build_engine.sh <model.onnx> [out_dir]}"
OUT_DIR="${2:-models/engines}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PREC="${PREC:-fp16}"
WORKSPACE="${WORKSPACE:-2048}"

MIN_HW="${MIN_HW:-256}"
OPT_HW="${OPT_HW:-512}"
MAX_HW="${MAX_HW:-640}"
MAX_PAIRS="${MAX_PAIRS:-1}"

case "$PREC" in
    fp16)        PREC_ARGS=(--fp16); SUFFIX="-fp16" ;;
    fp16-strict) PREC_ARGS=(--fp16 --precisionConstraints=obey); SUFFIX="-fp16" ;;
    fp32)        PREC_ARGS=();       SUFFIX="" ;;
    *) echo "PREC must be one of: fp16 (default), fp16-strict, fp32 — got '$PREC'" >&2; exit 1 ;;
esac

for v in MIN_HW OPT_HW MAX_HW; do
    if (( ${!v} % 32 != 0 )); then
        echo "$v must be a multiple of 32 (RaCo input_dim_divisor), got ${!v}" >&2; exit 1
    fi
done

# Identify the half and read K straight out of the graph.
read -r HALF K < <(python3 - "$ONNX" <<'PY'
import sys, onnx
g = onnx.load(sys.argv[1], load_external_data=False).graph
names = {i.name for i in g.input}
if names == {"images"}:
    k = next(o for o in g.output if o.name == "descriptors")
    print("extractor", k.type.tensor_type.shape.dim[1].dim_value)
elif names == {"normalized_keypoints", "descriptors"}:
    d = next(i for i in g.input if i.name == "descriptors")
    print("matcher", d.type.tensor_type.shape.dim[2].dim_value)
else:
    sys.exit(f"unrecognised graph inputs {sorted(names)} — not a split_raco_pipeline.py output")
PY
)

# TRT version exactly as trt-sys parses NvInferVersion.h (MAJOR.MINOR.PATCH.BUILD);
# GPU compute capability (sm) from torch — together they key the engine.
HDR="$(ls /usr/include/*/NvInferVersion.h 2>/dev/null | head -1)"
[ -n "$HDR" ] || { echo "NvInferVersion.h not found — is TensorRT installed?" >&2; exit 1; }
ver() { grep -E "define NV_TENSORRT_$1 " "$HDR" | awk '{print $3}'; }
TRT="$(ver MAJOR).$(ver MINOR).$(ver PATCH).$(ver BUILD)"
SM="$(python3 -c 'import torch;print("%d%d"%torch.cuda.get_device_capability())')"

if [ "$HALF" = "extractor" ]; then
    MODEL="raco-aliked-extractor-k${K}"
    SHAPE_ARGS=(
        "--minShapes=images:1x3x${MIN_HW}x${MIN_HW}"
        "--optShapes=images:2x3x${OPT_HW}x${OPT_HW}"
        "--maxShapes=images:$((MAX_PAIRS * 2))x3x${MAX_HW}x${MAX_HW}"
    )
else
    MODEL="lightglue-matcher-k${K}"
    MAXB=$((MAX_PAIRS * 2))
    SHAPE_ARGS=(
        "--minShapes=normalized_keypoints:2x1x${K}x2,descriptors:2x1x${K}x128"
        "--optShapes=normalized_keypoints:2x1x${K}x2,descriptors:2x1x${K}x128"
        "--maxShapes=normalized_keypoints:${MAXB}x1x${K}x2,descriptors:${MAXB}x1x${K}x128"
    )
fi

ENGINE="$OUT_DIR/${MODEL}-trt${TRT}-sm${SM}${SUFFIX}.engine"
mkdir -p "$OUT_DIR"
echo "building $ENGINE"
echo "  half=$HALF K=$K prec=$PREC workspace=${WORKSPACE}MB"
printf '  %s\n' "${SHAPE_ARGS[@]}"
# shellcheck disable=SC2086  # TRTEXEC_EXTRA is intentionally word-split
"$TRTEXEC" \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    "${PREC_ARGS[@]}" \
    "${SHAPE_ARGS[@]}" \
    --memPoolSize=workspace:"$WORKSPACE" \
    ${TRTEXEC_EXTRA:-}
echo "built $ENGINE"

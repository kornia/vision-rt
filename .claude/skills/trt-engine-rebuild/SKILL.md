---
name: trt-engine-rebuild
description: Use when building, rebuilding, or debugging TensorRT engines on this Jetson — shape profile errors ("does not satisfy any optimization profiles"), new ONNX exports, FP16 engines, or engine/version mismatch on deserialize.
---

# TensorRT Engine Rebuild (Jetson Orin, TRT 10.3.0.30, SM87)

## Hard rules

- Engines are tied to **this exact machine** (TRT 10.3.0.30 + sm87).
  Never copy `.engine` files between hosts. A `Deserialize` error usually
  means version/arch mismatch — rebuild from ONNX.
- `trtexec` lives at `/usr/src/tensorrt/bin/trtexec` (not on PATH).
- Set MAXN_SUPER before building or timing: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

## Shape profile errors

Error: `Set dimension [1,3,H,W] for tensor X does not satisfy any optimization
profiles. Valid range: [min]..[max]` → the input exceeds the engine's max
profile. Either resize the input upstream (VIC resize in `RtspSource::connect_resized`)
or rebuild with a bigger `--maxShapes`.

Current xfeat profile covers 1080p: min `1x3x240x320`, opt `1x3x640x640`,
max `1x3x1088x1920`. Inputs must be multiples of 32 (use `pad32`).

## Rebuild commands

```bash
# XFeat backbone (input tensor name: "image")
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/xfeat/xfeat_backbone.onnx \
    --saveEngine=models/xfeat/xfeat_backbone_fp16.engine \
    --fp16 \
    --minShapes=image:1x3x240x320 \
    --optShapes=image:1x3x640x640 \
    --maxShapes=image:1x3x1088x1920 \
    --memPoolSize=workspace:2048

# Timing sanity check on an existing engine
/usr/src/tensorrt/bin/trtexec --loadEngine=<engine> --shapes=image:1x3x640x640
```

## Inspecting an engine

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=<engine_path> --verbose 2>&1 | \
    grep -iA2 "input\|output\|profile" | head -30
```

Shows I/O tensor names, dtypes, and profile shapes — check this FIRST when
inference fails with "unknown tensor" or shape errors. (Programmatic access:
`Engine` exposes the same via the named-tensor API in `crates/vrt`.)

## Performance expectations (MAXN_SUPER)

- XFeat backbone @ 640×640 opt shape: ~3ms GPU
- Larger inputs scale ~linearly with pixel area; running far from opt shape
  is slower than the area ratio suggests — prefer VIC-resizing input toward opt.
- Build time: ~5 min. Keep workspace at 2048 MB (unified memory — don't go higher).

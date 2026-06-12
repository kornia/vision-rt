# engine

Rebuild a TensorRT engine from ONNX on this Jetson (sm87, TRT 10.3.0.30).

## XFeat backbone (default)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/xfeat/xfeat_backbone.onnx \
    --saveEngine=models/xfeat/xfeat_backbone_fp16.engine \
    --fp16 \
    --minShapes=image:1x3x240x320 \
    --optShapes=image:1x3x640x640 \
    --maxShapes=image:1x3x1088x1920 \
    --memPoolSize=workspace:2048
```

## Timing profile (load existing engine)

```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=models/xfeat/xfeat_backbone_fp16.engine \
    --shapes=image:1x3x640x640
```

## Notes
- Engines are tied to this machine (sm87 + TRT 10.3.0.30) — never copy across hosts.
- Build takes ~5 min; opt shape 640×640 gives ~3ms GPU latency.
- Jetson must be in MAXN_SUPER mode for best perf: `sudo nvpmodel -m 2 && sudo jetson_clocks`

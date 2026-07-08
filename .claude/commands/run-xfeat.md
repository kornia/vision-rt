# run-xfeat

Run the XFeat keypoint detector on an RTSP stream.

Usage: `/run-xfeat <rtsp_url> [save_dir]`

```bash
RUST_LOG=warn cargo run --release -p rtsp_xfeat -- \
    models/xfeat/xfeat_backbone_fp16.engine \
    "$ARGUMENTS" \
    /tmp/xfeat_out
```

**Default model**: `models/xfeat/xfeat_backbone_fp16.engine`  
**Default save dir**: `/tmp/xfeat_out` (PNG snapshots every 30 frames)  
**Resize**: VIC hardware resize to 1280×720 before CUDA  
**Profile**: engine opt shape 640×640; actual input after pad32 is 1280×736

To check saved keypoint images:
```bash
ls -lh /tmp/xfeat_out/*.png 2>/dev/null | tail -5
```

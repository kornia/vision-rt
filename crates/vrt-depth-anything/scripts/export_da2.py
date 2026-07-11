#!/usr/bin/env python3
"""
Export **Depth Anything V2 Metric-Small (indoor / Hypersim)** as a TRT-compatible ONNX.

Model source — Depth Anything V2 (Yang et al.); the metric-depth fine-tunes live in
the `DepthAnything/Depth-Anything-V2` repo under `metric_depth/`. All model credit
belongs to the original authors.

Setup (once):
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    # download the metric indoor (Hypersim) vits checkpoint into checkpoints/, e.g.
    #   depth_anything_v2_metric_hypersim_vits.pth
    export DA2_REPO=$PWD/Depth-Anything-V2

Export (fixed square input, multiple of 14):
    python3 crates/vrt-depth-anything/scripts/export_da2.py \
        --checkpoint $DA2_REPO/checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
        --out models/onnx/depth-anything-v2-metric-small \
        --input-size 518

Then build the fp16 engine on-device (see build_engine.sh):
    crates/vrt-depth-anything/scripts/build_engine.sh \
        models/onnx/depth-anything-v2-metric-small/depth_anything_v2_metric.onnx

Output (metric meters): input `[1,3,S,S]` (ImageNet-normalized) → `depth [1,1,S,S]`.
Notes:
- `--input-size 518` (37x14) = DA2 native, best accuracy; `392` (28x14) is faster on
  Orin Nano. The Rust crate reads the size from the engine, so either works.
- Indoor (Hypersim) metric range ~20 m; a separate outdoor (VKITTI) checkpoint would
  be a second registered model.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="metric vits .pth checkpoint")
    ap.add_argument(
        "--out",
        default="models/onnx/depth-anything-v2-metric-small",
        help="output directory for the ONNX",
    )
    ap.add_argument("--input-size", type=int, default=518, help="square input (mult of 14)")
    ap.add_argument("--max-depth", type=float, default=20.0, help="metric range cap (indoor=20)")
    ap.add_argument(
        "--repo",
        default=os.environ.get("DA2_REPO", "."),
        help="path to the Depth-Anything-V2 repo (or set DA2_REPO)",
    )
    args = ap.parse_args()
    if args.input_size % 14 != 0:
        raise SystemExit(f"--input-size must be a multiple of 14, got {args.input_size}")

    import sys

    sys.path.insert(0, os.path.join(args.repo, "metric_depth"))
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    # ViT-Small config.
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**{**cfg, "max_depth": args.max_depth})
    # weights_only=True — the checkpoint is a plain tensor state_dict; avoids
    # unpickling arbitrary objects.
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "depth_anything_v2_metric.onnx")
    dummy = torch.zeros(1, 3, args.input_size, args.input_size)
    print(f">> exporting ONNX → {onnx_path} (input [1,3,{args.input_size},{args.input_size}])", flush=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["depth"],
        opset_version=17,
        do_constant_folding=True,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export the XFeat backbone as a TRT-compatible ONNX.

Model source — XFeat (Potje et al., "XFeat: Accelerated Features for Lightweight
Image Matching", CVPR 2024): https://github.com/verlab/accelerated_features
This script imports that repo's `modules.xfeat.XFeat` and its `xfeat.pt` weights;
all model credit belongs to the original authors.

The full XFeat model uses `nonzero()` in NMS (data-dependent output shape),
which TRT cannot compile. This script exports only the conv backbone:

    image (B, 3, H, W) → descriptors  (B, 64, H/8, W/8)
                        → heatmap      (B,  1,   H,   W)
                        → reliability  (B,  1,   H,   W)

Post-processing (TopK keypoint selection + descriptor lookup) is done outside
TRT, e.g. in a CUDA kernel or on CPU.  The backbone is the expensive part
(~2.6 MB weights, all conv/pool); post-processing is cheap.

Usage (run from the repo root):
    git clone https://github.com/verlab/accelerated_features
    XFEAT_REPO=$PWD/accelerated_features \
        python3 crates/vrt-xfeat/scripts/export_xfeat_backbone.py \
            --weights $PWD/accelerated_features/weights/xfeat.pt \
            --out models/xfeat/xfeat_backbone.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import onnx
try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

# Point XFEAT_REPO at a checkout of github.com/verlab/accelerated_features so
# `modules.xfeat` is importable.
_xfeat_repo = os.environ.get("XFEAT_REPO")
if _xfeat_repo:
    sys.path.insert(0, _xfeat_repo)

from modules.xfeat import XFeat


class XFeatBackbone(torch.nn.Module):
    """Backbone-only wrapper: image → (descriptors, heatmap, reliability).

    All outputs have static spatial dimensions (H/8×W/8 or H×W), so the
    graph contains only conv/pool/norm ops — fully TRT-parseable.
    """

    def __init__(self, xfeat: XFeat):
        super().__init__()
        self.net = xfeat.net
        self.interpolator = xfeat.interpolator

        # Preprocess scale constants (XFeat pads to multiple of 32)
        self._div = 32

    def forward(self, x: torch.Tensor):
        # Normalize to [0, 1] if uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Convert RGB → grayscale (XFeat internally works on single channel,
        # but the public API accepts 3-channel; net.forward handles it).
        _, _, H, W = x.shape

        # Pad so H and W are multiples of 32
        ph = (self._div - H % self._div) % self._div
        pw = (self._div - W % self._div) % self._div
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        # Build pixel-level heatmap from 65-channel keypoint logits
        scores = F.softmax(K1, 1)[:, :64]
        Bs, _, Hf, Wf = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(Bs, Hf, Wf, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(Bs, 1, Hf * 8, Wf * 8)

        # Crop back to original size
        heatmap = heatmap[:, :, :H, :W]
        H1 = F.interpolate(H1, size=(H, W), mode="bilinear", align_corners=False)

        return M1, heatmap, H1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default="weights/xfeat.pt",
        help="path to upstream xfeat.pt (from verlab/accelerated_features)",
    )
    p.add_argument(
        "--out",
        default="models/xfeat/xfeat_backbone.onnx",
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--opset",  type=int, default=17)
    p.add_argument("--no-simplify", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading weights from {args.weights}")
    xfeat = XFeat(weights=args.weights)
    xfeat.eval()

    model = XFeatBackbone(xfeat)
    model.eval()

    H, W = args.height, args.width
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, H, W, device=device)

    with torch.no_grad():
        M1, heatmap, H1 = model(dummy)
    print(f"Backbone outputs at {H}×{W}:")
    print(f"  descriptors : {tuple(M1.shape)}")
    print(f"  heatmap     : {tuple(heatmap.shape)}")
    print(f"  reliability : {tuple(H1.shape)}")

    print(f"\nExporting ONNX (opset {args.opset}) → {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["descriptors", "heatmap", "reliability"],
        dynamic_axes={
            "image":       {0: "batch", 2: "height", 3: "width"},
            "descriptors": {0: "batch", 2: "height_8", 3: "width_8"},
            "heatmap":     {0: "batch", 2: "height", 3: "width"},
            "reliability": {0: "batch", 2: "height", 3: "width"},
        },
    )

    # Verify
    m = onnx.load(args.out)
    onnx.checker.check_model(m)

    if not args.no_simplify and HAS_ONNXSIM:
        print("Simplifying with onnxsim…")
        m_sim, ok = onnxsim.simplify(m, test_input_shapes={"image": [1, 3, H, W]})
        if ok:
            onnx.save(m_sim, args.out)
            print("  simplified OK")
        else:
            print("  simplification failed — keeping original")
    elif not HAS_ONNXSIM:
        print("onnxsim not installed — skipping simplification (pip install onnxsim)")

    # Print ops used
    ops = sorted(set(n.op_type for n in onnx.load(args.out).graph.node))
    print(f"\nOps in exported graph: {ops}")
    bad = [o for o in ops if o in ("NonZero", "If", "Loop")]
    if bad:
        print(f"WARNING: TRT-incompatible ops present: {bad}")
    else:
        print("No TRT-incompatible ops — ready for trtexec.")

    sz = os.path.getsize(args.out) / 1024**2
    print(f"\nSaved: {args.out} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export the RF-DETR Segmentation (instance-mask) model as a TRT-compatible ONNX.

Model source — RF-DETR (Roboflow): the `rfdetr` package's `RFDETRSegPreview`
(MaskDINO-style instance-mask head on RF-DETR). All model credit belongs to the
original authors.

Requires **transformers >= 5.1** — rfdetr 1.6.5 uses the transformers 5.x
`BackboneMixin` API (a real signature change, not just import paths). If your
environment is pinned to transformers 4.x, install 5.x in ISOLATION without
touching the system packages, then shadow it on PYTHONPATH:

    pip install --target=/tmp/tf5 "transformers>=5.1,<6"
    PYTHONPATH=/tmp/tf5 python3 crates/vrt-rfdetr-seg/scripts/export_rfdetr_seg.py \
        --out models/onnx/rfdetr-seg-preview

Then build the engine on-device (machine-locked; see build_engine.sh):

    crates/vrt-rfdetr-seg/scripts/build_engine.sh \
        models/onnx/rfdetr-seg-preview/inference_model.onnx

Outputs (input [1, 3, 432, 432]):
    dets   [1, 200,   4]      cxcywh, normalized
    labels [1, 200,  91]      class logits (class 0 = background, COCO 1-90)
    masks  [1, 200, 108, 108] raw per-query mask logits (einsum + bias, no sigmoid)
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="models/onnx/rfdetr-seg-preview",
        help="output directory for inference_model.onnx",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    # Import the concrete class (rfdetr re-exports lazily; the submodule is stable).
    from rfdetr.detr import RFDETRSegPreview

    print(">> instantiating RFDETRSegPreview (downloads checkpoint on first run)…", flush=True)
    model = RFDETRSegPreview()
    print(f">> exporting ONNX → {args.out}", flush=True)
    model.export(output_dir=args.out, batch_size=args.batch_size)
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export **DINOv3 ViT-S/16** as a TRT-compatible ONNX emitting a global image descriptor.

Model source — DINOv3 (Siméoni et al., Meta AI). All model credit belongs to the
original authors. The HF repo is **gated**: accept the licence at
    https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
and export a token (`huggingface-cli login`, or HF_TOKEN in the environment).

Requires transformers>=4.56 (DINOv3 support) + torch + onnx. If the system transformers
is older, install in isolation rather than breaking it:
    pip install --target=/tmp/tf456 "transformers>=4.56"
    PYTHONPATH=/tmp/tf456 python3 crates/vrt-dinov3/scripts/export_dinov3.py ...

Export (fixed square input, multiple of 16):
    HF_TOKEN=hf_... python3 crates/vrt-dinov3/scripts/export_dinov3.py \
        --input-size 336 \
        --out models/onnx/dinov3-vits16-336.onnx \
        --dump-ref models/onnx/dinov3-ref

Then build the engine on-device (see build_engine.sh):
    crates/vrt-dinov3/scripts/build_engine.sh models/onnx/dinov3-vits16-336.onnx

Outputs: `descriptor [1,D]` (the CLS token / `pooler_output` — the headline) and
`tokens [1,N,D]` (the full sequence: CLS + 4 registers + (S/16)^2 patches). Binding the
token output costs an output buffer and zero FLOPs — the ViT computes every patch token
regardless — so we export it and let the Rust crate ignore it until something wants it.

Notes:
- `--input-size 224` is DINOv3 native and ~2.2x cheaper than 336. Resolution buys more
  for dense tasks than for a whole-image descriptor, so benchmark both. The Rust crate
  reads the size from the engine, so either works with no code change.
- `--dump-ref` writes the raw preprocessed input and the reference descriptor as f32
  binaries, for the engine-vs-PyTorch parity test (see the crate README). That test is
  what decides fp16 vs fp32 — do not skip it.
"""
import argparse
import os

# ONNX ops TensorRT either cannot parse or handles badly. HF ViT exports can emit these
# (usually from a Python-level branch or an unfolded loop); catching them here is far
# cheaper than debugging a failed trtexec parse.
TRT_HOSTILE = {"NonZero", "If", "Loop", "Scan", "SequenceAt", "GridSample"}


def build_wrapper(torch):
    """Built lazily so `--help` works without torch installed."""
    class Wrap(torch.nn.Module):
        """Return (pooler_output, last_hidden_state) as a plain tuple.

        Exporting the raw HF model would emit a ModelOutput dataclass, whose ordering
        the ONNX graph does not preserve reliably. A tuple pins the output order.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return out.pooler_output, out.last_hidden_state

    return Wrap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="HF repo id (gated — accept the licence first)",
    )
    ap.add_argument("--input-size", type=int, default=336, help="square input (mult of 16)")
    ap.add_argument(
        "--out",
        default="models/onnx/dinov3-vits16-336.onnx",
        help="output .onnx path",
    )
    ap.add_argument(
        "--dump-ref",
        default=None,
        help="directory for the parity reference (ref_input.bin + ref_descriptor.bin)",
    )
    # 18, not the 17 the sibling exports use: torch has no opset-17 implementations for
    # this graph, so asking for 17 makes it export at 18 and then down-convert through the
    # onnx C API. TRT 10.3 handles 18 natively, so skipping the conversion removes a step
    # that "may not be successful" (torch's own words) for no benefit.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    if args.input_size % 16 != 0:
        raise SystemExit(f"--input-size must be a multiple of 16, got {args.input_size}")

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    print(f">> loading {args.model}", flush=True)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    s = args.input_size
    # The parity reference is an S x S **u8 RGB image**, not a raw tensor. At exactly the
    # engine's input size kornia's Stretch resize is the identity (scale 1 -> integer
    # taps, zero bilinear weight on the neighbours), so the Rust side can push this
    # through the crate's real public path — preprocessor included — and any difference
    # is genuine numerics rather than a resampling mismatch.
    #
    # Fixed seed, and deliberately not zeros: a broken engine still maps 0 to something
    # stable, which would make the check vacuous.
    g = torch.Generator().manual_seed(0)
    img_u8 = torch.randint(0, 256, (s, s, 3), generator=g, dtype=torch.uint8)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    chw = img_u8.permute(2, 0, 1).float() / 255.0
    dummy = ((chw - mean) / std).unsqueeze(0)

    with torch.no_grad():
        out = model(dummy)
    desc = out.pooler_output
    tokens = out.last_hidden_state
    dim = desc.shape[-1]
    ntok = tokens.shape[1]
    npatch = (s // 16) ** 2
    prefix = ntok - npatch

    print(f">> descriptor {tuple(desc.shape)}   tokens {tuple(tokens.shape)}")
    print(f">> patch grid {s//16}x{s//16} = {npatch}, prefix tokens = {prefix} (CLS + registers)")
    if prefix < 1:
        raise SystemExit(
            f"token count {ntok} <= patch count {npatch}: the model and --input-size disagree"
        )

    # The crate's whole premise is "the descriptor IS the CLS token". Report how true
    # that is for this checkpoint rather than assuming it — a final layernorm in the
    # pooler will pull the cosine below 1.0 without invalidating the export.
    cos = F.cosine_similarity(desc, tokens[:, 0], dim=-1).item()
    print(f">> cos(pooler_output, last_hidden_state[:,0]) = {cos:.6f}")
    if cos < 0.9:
        print(
            "!! WARNING: pooler_output is far from the raw CLS token — this checkpoint "
            "applies a non-trivial pooling head. The export is still valid (the crate "
            "uses pooler_output as-is), but re-read the model card before trusting it.",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    print(f">> exporting ONNX -> {args.out} (input [1,3,{s},{s}])", flush=True)
    # eval() on the WRAPPER, not just the inner model: nn.Module.__init__ defaults
    # `training = True`, and assigning an already-eval child does not clear the parent's
    # flag. The inner model would still behave correctly, but torch warns loudly about
    # exporting in training mode and relying on that asymmetry is asking for a dropout
    # or drop-path bug the day someone restructures this.
    wrapped = build_wrapper(torch)(model)
    wrapped.eval()
    torch.onnx.export(
        wrapped,
        (dummy,),
        args.out,
        input_names=["input"],
        output_names=["descriptor", "tokens"],
        opset_version=args.opset,
        do_constant_folding=True,
        # No dynamic_axes: static shapes, matching every other vrt model export.
    )

    import onnx

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    hostile = sorted(ops & TRT_HOSTILE)
    if hostile:
        print(
            f"!! WARNING: graph contains TRT-hostile ops {hostile} — trtexec may fail to "
            "parse or fall back to a slow path. Consider onnxsim, or a newer transformers.",
            flush=True,
        )
    else:
        print(f">> op scan clean ({len(ops)} distinct ops)")

    if args.dump_ref:
        os.makedirs(args.dump_ref, exist_ok=True)
        # L2-normalize here so the Rust side compares against exactly what the crate
        # produces (the crate normalizes on device).
        ref = F.normalize(desc, dim=-1).squeeze(0).contiguous()
        ip = os.path.join(args.dump_ref, "ref_image.bin")
        dp = os.path.join(args.dump_ref, "ref_descriptor.bin")
        img_u8.contiguous().numpy().tofile(ip)
        ref.numpy().astype("float32").tofile(dp)
        print(f">> parity reference: {ip} [{s},{s},3] u8 HWC RGB")
        print(f">> parity reference: {dp} [{dim}] f32, L2-normed")
        print(
            "   run it with:  DINOV3_ENGINE=<engine> DINOV3_REF_DIR="
            f"{args.dump_ref} cargo test -p vrt-dinov3 --release -- --ignored"
        )

    print("done", flush=True)


if __name__ == "__main__":
    main()

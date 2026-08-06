#!/usr/bin/env python3
"""
Verify that extractor+matcher reproduce the fused RaCo-ALIKED-LightGlue+ pipeline.

`split_raco_pipeline.py` cuts the released graph in two and drops the trailing
`NonZero` compaction. This script proves the cut is exact: it runs the fused model
and the two halves on the same input under onnxruntime (CPU EP is enough — we are
checking graph equivalence, not speed) and compares:

  * keypoints              — must match elementwise
  * the match set          — the fused `(M,3)` list, reconstructed from the halves'
                             per-query `matches0` (-1 = unmatched), must be identical
  * the match scores       — `mscores` vs the gathered `mscores0`

With `--dump-ref` it also writes raw little-endian f32/i32 binaries so the Rust side
can assert engine-vs-ONNX parity later (same pattern as vrt-dinov3's tests/gpu.rs and
its DINOV3_REF_DIR).

USAGE
    python3 crates/vrt-raco-aliked/scripts/check_split_parity.py \
        --fused     models/onnx/raco/raco_aliked_lightglue_pipeline_k1024.onnx \
        --extractor models/onnx/raco/raco_aliked_extractor_k1024.onnx \
        --matcher   models/onnx/raco/lightglue_matcher_k1024.onnx \
        --size 256 --dump-ref models/onnx/raco/ref_k1024

Input images may be supplied with `--images L R`; otherwise a deterministic synthetic
pair is generated (seeded), which exercises the graph just as well for equivalence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def synthetic_pair(size: int, seed: int = 0) -> np.ndarray:
    """A deterministic, structured image pair — RaCo needs real corners to fire on.

    Pure uniform noise gives a degenerate score map; a grid of squares with a small
    translation between the two views produces a normal keypoint distribution.
    """
    rng = np.random.default_rng(seed)
    imgs = np.zeros((2, 3, size, size), dtype=np.float32)
    step = max(size // 8, 8)
    for b in range(2):
        shift = 0 if b == 0 else max(size // 32, 2)
        canvas = rng.uniform(0.05, 0.15, size=(3, size, size)).astype(np.float32)
        for y in range(step, size - step, step):
            for x in range(step, size - step, step):
                yy, xx = y + shift, x + shift
                if yy + step // 2 < size and xx + step // 2 < size:
                    canvas[:, yy : yy + step // 2, xx : xx + step // 2] = rng.uniform(0.6, 1.0)
        imgs[b] = np.clip(canvas, 0.0, 1.0)
    return imgs


def load_pair(paths: list[str], size: int) -> np.ndarray:
    from PIL import Image  # optional; only needed with --images

    out = np.zeros((2, 3, size, size), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fused", required=True, type=Path)
    ap.add_argument("--extractor", required=True, type=Path)
    ap.add_argument("--matcher", required=True, type=Path)
    ap.add_argument("--size", type=int, default=256, help="square input side; multiple of 32")
    ap.add_argument("--images", nargs=2, default=None, help="optional real image pair")
    ap.add_argument("--dump-ref", type=Path, default=None)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    if args.size % 32 != 0:
        print(f"--size must be a multiple of 32 (RaCo input_dim_divisor), got {args.size}")
        return 2

    imgs = load_pair(args.images, args.size) if args.images else synthetic_pair(args.size)
    print(f"== input images {imgs.shape} range [{imgs.min():.3f}, {imgs.max():.3f}]")

    print(f"== running fused    {args.fused.name}")
    fused = session(args.fused)
    f_kpts, f_matches, f_mscores = fused.run(["keypoints", "matches", "mscores"], {"images": imgs})
    print(f"   keypoints {f_kpts.shape}  matches {f_matches.shape}  mscores {f_mscores.shape}")

    print(f"== running extractor {args.extractor.name}")
    ex = session(args.extractor)
    e_kpts, e_nkpts, e_descs = ex.run(
        ["keypoints", "normalized_keypoints", "descriptors"], {"images": imgs}
    )
    print(f"   keypoints {e_kpts.shape}  norm_kpts {e_nkpts.shape}  descriptors {e_descs.shape}")

    print(f"== running matcher   {args.matcher.name}")
    mt = session(args.matcher)
    n2b, k, _ = e_nkpts.shape
    m_matches, m_mscores = mt.run(
        ["matches0", "mscores0"],
        {
            "normalized_keypoints": e_nkpts.reshape(n2b, 1, k, 2),
            "descriptors": e_descs.reshape(n2b, 1, k, 128),
        },
    )
    print(f"   matches0 {m_matches.shape} {m_matches.dtype}  mscores0 {m_mscores.shape}")

    ok = True

    # -- keypoints ------------------------------------------------------------
    if not np.allclose(f_kpts, e_kpts, atol=args.atol, rtol=0):
        d = np.abs(f_kpts - e_kpts).max()
        print(f"FAIL keypoints differ, max abs {d}")
        ok = False
    else:
        print("PASS keypoints identical")

    # -- match set ------------------------------------------------------------
    # Rebuild the fused (M,3) list from the halves' per-query matches0.
    rows = []
    scores = []
    for p in range(m_matches.shape[0]):
        idx = np.nonzero(m_matches[p] >= 0)[0]
        for i in idx:
            rows.append((p, int(i), int(m_matches[p, i])))
            scores.append(float(m_mscores[p, i]))
    rebuilt = np.array(rows, dtype=np.int64).reshape(-1, 3)
    rebuilt_scores = np.array(scores, dtype=np.float32)

    print(f"   fused matches: {f_matches.shape[0]}   rebuilt: {rebuilt.shape[0]}")
    if rebuilt.shape != f_matches.shape:
        print(f"FAIL match count differs: {f_matches.shape} vs {rebuilt.shape}")
        ok = False
    elif not np.array_equal(np.asarray(f_matches, dtype=np.int64), rebuilt):
        n_diff = int((np.asarray(f_matches, dtype=np.int64) != rebuilt).any(axis=1).sum())
        print(f"FAIL match indices differ in {n_diff} rows")
        print(f"   fused[:5]   {np.asarray(f_matches)[:5].tolist()}")
        print(f"   rebuilt[:5] {rebuilt[:5].tolist()}")
        ok = False
    else:
        print(f"PASS match set identical ({rebuilt.shape[0]} matches)")
        if not np.allclose(f_mscores, rebuilt_scores, atol=args.atol, rtol=0):
            d = np.abs(f_mscores - rebuilt_scores).max()
            print(f"FAIL mscores differ, max abs {d}")
            ok = False
        else:
            print("PASS mscores identical")

    if rebuilt.shape[0] == 0:
        print("WARN zero matches — parity is vacuous; rerun with --images on a real pair")

    # -- reference dump -------------------------------------------------------
    if args.dump_ref:
        args.dump_ref.mkdir(parents=True, exist_ok=True)
        for name, arr in (
            ("images.f32", imgs),
            ("keypoints.f32", e_kpts),
            ("normalized_keypoints.f32", e_nkpts),
            ("descriptors.f32", e_descs),
            ("mscores0.f32", m_mscores),
            ("matches0.i32", m_matches.astype(np.int32)),
        ):
            (args.dump_ref / name).write_bytes(np.ascontiguousarray(arr).tobytes())
        (args.dump_ref / "shapes.txt").write_text(
            "\n".join(
                f"{n} {list(a.shape)}"
                for n, a in (
                    ("images", imgs),
                    ("keypoints", e_kpts),
                    ("normalized_keypoints", e_nkpts),
                    ("descriptors", e_descs),
                    ("mscores0", m_mscores),
                    ("matches0", m_matches),
                )
            )
            + "\n"
        )
        print(f"== wrote reference tensors to {args.dump_ref}")

    print("\n== PARITY OK" if ok else "\n== PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

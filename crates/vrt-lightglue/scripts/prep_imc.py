"""Convert the IMC 2021 phototourism validation metadata to plain text for `eval_imc`.

The dataset ships camera calibration as HDF5 and its co-visibility pair lists as .npy,
neither of which Rust reads without a heavy dependency. This script converts *only that
metadata*; it never touches an image. `eval_imc` loads the JPEGs itself through
`kornia_io` and resizes through `kornia_imgproc`, so there is no second decode path to
disagree with the one the models actually see.

Pairs are stratified by co-visibility. The dataset provides `new-vis-pairs/keys-th-X.npy`
listing pairs whose co-visibility is at least X, so a pair's band is the *highest* X at
which it still appears. Sampling evenly across bands is the point of the exercise: a
matcher's mean score over an unstratified pool is dominated by easy pairs and hides
exactly the regime where matchers differ.

Writes, under each scene's `set_100/`:
  calib_txt/<image>.txt   21 floats -- K (9, row-major), R (9, row-major), T (3)
and at the dataset root:
  imc_manifest.txt        `scene band image1 image2` per line

Usage: python3 prep_imc.py [phototourism_dir] [pairs_per_band] [seed]
Requires: h5py, numpy (metadata only).
"""

from __future__ import annotations

import pathlib
import random
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/vision-rt/models/datasets/imc2021/phototourism"
)
PER_BAND = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BANDS = ["0.1", "0.2", "0.3", "0.4", "0.5"]


def load_calib(path: pathlib.Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.concatenate(
            [np.array(f["K"]).ravel(), np.array(f["R"]).ravel(), np.array(f["T"]).ravel()]
        )


def main() -> None:
    manifest = []
    for scene in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        base = ROOT / scene / "set_100"
        vis = base / "new-vis-pairs"
        if not vis.is_dir():
            print(f"{scene}: no new-vis-pairs, skipped")
            continue

        # A pair's band is the strictest threshold it survives, so walk high -> low and
        # take the first hit. Reversing this would put every pair in the loosest band.
        seen: dict[str, str] = {}
        for th in reversed(BANDS):
            f = vis / f"keys-th-{th}.npy"
            if not f.exists():
                continue
            for key in np.load(f):
                seen.setdefault(str(key), th)

        by_band: dict[str, list[str]] = {b: [] for b in BANDS}
        for key, band in seen.items():
            by_band[band].append(key)

        wanted = set()
        for band in BANDS:
            # Sample rather than take a prefix. Keys are "<img1>-<img2>", so a sorted
            # prefix clusters on a handful of left images: measured over the real dataset
            # it touches 38 of 274 images, with reichstag's 30 pairs drawn from 10 left
            # images. A seeded sample over the same bands and the same budget reaches 135
            # — the determinism the prefix was there for costs nothing to keep.
            pool = sorted(by_band[band])
            picked = sorted(random.Random(SEED).sample(pool, min(PER_BAND, len(pool))))
            for key in picked:
                a, b = key.split("-")
                manifest.append(f"{scene} {band} {a} {b}")
                wanted.update((a, b))
            print(f"{scene} band {band}: {len(by_band[band]):5d} pairs, took {len(picked)}")

        out = base / "calib_txt"
        out.mkdir(exist_ok=True)
        for img in sorted(wanted):
            c = load_calib(base / "calibration" / f"calibration_{img}.h5")
            np.savetxt(out / f"{img}.txt", c.reshape(1, -1), fmt="%.12g")

    (ROOT / "imc_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\n{len(manifest)} evaluation pairs -> {ROOT}/imc_manifest.txt")


if __name__ == "__main__":
    main()

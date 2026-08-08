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


def decode_key(key: object) -> str:
    """Pair keys may arrive as a bytes dtype; `str(np.bytes_(b'a-b'))` is "b'a-b'"."""
    return key.decode() if isinstance(key, bytes) else str(key)


def main() -> None:
    # One generator for the whole run. Rebuilding it per band would reseed to the same
    # state each time, so two bands with equal-length pools select identical positional
    # indices -- correlated strata, which is what sampling was meant to avoid.
    rng = random.Random(SEED)
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
                seen.setdefault(decode_key(key), th)

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
            picked = sorted(rng.sample(pool, min(PER_BAND, len(pool))))
            taken = 0
            for key in picked:
                # "<image1>-<image2>". Unpacking blind raises ValueError and kills the run
                # after calib files are already written, so report and skip instead.
                parts = key.split("-")
                if len(parts) != 2:
                    print(f"  skipping unparseable pair key {key!r}")
                    continue
                a, b = parts
                manifest.append(f"{scene} {band} {a} {b}")
                wanted.update((a, b))
                taken += 1
            # `taken`, not `len(picked)`: a skipped key would otherwise be reported as
            # sampled, and the band would look fuller than the manifest actually is.
            print(f"{scene} band {band}: {len(by_band[band]):5d} pairs, took {taken}")

        out = base / "calib_txt"
        out.mkdir(exist_ok=True)
        for img in sorted(wanted):
            c = load_calib(base / "calibration" / f"calibration_{img}.h5")
            np.savetxt(out / f"{img}.txt", c.reshape(1, -1), fmt="%.12g")

    # An empty manifest must not be written: `eval_imc` skips blank lines, so a lone "\n"
    # produces a complete-looking all-zero table with no error anywhere to explain it.
    if not manifest:
        raise SystemExit(
            f"{ROOT}: no pairs found — no scene had a readable set_100/new-vis-pairs/. "
            "No manifest was written."
        )
    (ROOT / "imc_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\n{len(manifest)} evaluation pairs -> {ROOT}/imc_manifest.txt")


if __name__ == "__main__":
    main()

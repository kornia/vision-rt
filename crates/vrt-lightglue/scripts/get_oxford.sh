#!/usr/bin/env bash
# Fetch the Oxford/VGG affine covariant-regions benchmark sequences used by
# `examples/eval_oxford`.
#
#   bark, boat = rotation + zoom  (the sequences that test RaCo's rotation claim)
#   graf       = viewpoint/perspective (contrast case)
#
# Each sequence ships img1..img6 plus ground-truth homographies H1to2p..H1to6p
# mapping img1's pixels into each other image.
#
# Usage: ./get_oxford.sh [dest_dir]
set -euo pipefail
D="${1:-/mnt/data/vision-rt/models/datasets/oxford}"
mkdir -p "$D"
cd "$D"

failed=0
for seq in bark boat graf; do
  if [ -d "$seq" ]; then echo "have $seq"; continue; fi
  echo "=== fetching $seq"
  # -f is load-bearing: without it curl writes the server's HTML error page into the
  # tarball and exits 0, tar then fails inside an && list (which `set -e` does not catch).
  if ! curl -fsSL -m 300 -o "$seq.tar.gz" \
    "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/$seq.tar.gz"; then
    echo "FAILED to download $seq"
    rm -f "$seq.tar.gz"
    failed=$((failed + 1))
    continue
  fi
  # Extract into a staging directory and rename only on success. Extracting straight
  # into "$seq" means a run interrupted mid-tar (Ctrl-C, OOM, power) leaves a partly
  # populated directory that satisfies the "have $seq" guard above forever, and
  # prep_oxford then builds a quietly short manifest from it. The rename is atomic.
  rm -rf "$seq.partial"
  mkdir -p "$seq.partial"
  if ! tar xzf "$seq.tar.gz" -C "$seq.partial"; then
    echo "FAILED to extract $seq"
    rm -rf "$seq.partial" "$seq.tar.gz"
    failed=$((failed + 1))
    continue
  fi
  mv "$seq.partial" "$seq"
  rm -f "$seq.tar.gz"
done

echo
for seq in bark boat graf; do
  [ -d "$seq" ] || continue
  echo "$seq: $(ls "$seq" | tr '\n' ' ')"
done
echo
if [ "$failed" -gt 0 ]; then
  echo "$failed sequence(s) failed — the benchmark would be silently truncated" >&2
  exit 1
fi
echo "next: cargo run --release -p vrt-lightglue --example prep_oxford -- $D ${D}_prepared"

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
set -eu
D="${1:-/mnt/data/vision-rt/models/datasets/oxford}"
mkdir -p "$D"
cd "$D"

for seq in bark boat graf; do
  if [ -d "$seq" ]; then echo "have $seq"; continue; fi
  echo "=== fetching $seq"
  curl -sSL -m 300 -o "$seq.tar.gz" \
    "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/$seq.tar.gz" || {
    echo "FAILED $seq"
    continue
  }
  mkdir -p "$seq" && tar xzf "$seq.tar.gz" -C "$seq" && rm -f "$seq.tar.gz"
done

echo
for seq in bark boat graf; do
  [ -d "$seq" ] || continue
  echo "$seq: $(ls "$seq" | tr '\n' ' ')"
done
echo
echo "next: cargo run --release -p vrt-lightglue --example prep_oxford -- $D ${D}_prepared"

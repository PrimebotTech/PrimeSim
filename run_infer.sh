#!/usr/bin/env bash
# Run PrimeSim inference on the GPU cluster (A100 80GB).
# The 4090 cannot hold the 5B model plus the VAE and umt5 encoder at once.
set -euo pipefail

REPO=/path/to/worldmodel-wan-release
BASE=/path/to/wan2.2
CKPT=/path/to/checkpoint
EXAMPLE=${EXAMPLE:-example_data/g2_example_1}
OUT=${OUT:-outputs/$(basename "$EXAMPLE").mp4}

cd "$REPO"
mkdir -p outputs

python scripts/inference.py \
  --checkpoint    "$CKPT/g2_mixed/stage4/model.safetensors" \
  --text-encoder  "$BASE/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae           "$BASE/Wan2.2_VAE.pth" \
  --tokenizer     "$BASE/google/umt5-xxl" \
  --demo-dir      "$EXAMPLE" \
  --output        "$OUT"

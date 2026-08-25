#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_CHECKPOINT" >&2
  exit 2
fi

config_path=$1
output_checkpoint=$2

torchrun --standalone --nproc-per-node=8 -m xuannv_embedding.cli train \
  --config "${config_path}" \
  --profile npu-smoke \
  --output "${output_checkpoint}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [xuannv train arguments...]" >&2
  exit 2
fi

config_path=$1
shift

torchrun --standalone --nproc-per-node=6 -m xuannv_embedding.cli train \
  --config "${config_path}" "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_CHECKPOINT" >&2
  exit 2
fi

config_path=$1
output_checkpoint=$2

# Multiple jobs can share an Atlas A2 host. Let CANN allocate free NPU-side
# communicator ports unless the operator supplied an explicit isolated range.
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"

torchrun --standalone --nproc-per-node=8 -m xuannv_embedding.cli train \
  --config "${config_path}" \
  --profile npu-smoke \
  --npu-phase first \
  --output "${output_checkpoint}"

# A second launcher process proves that model, optimizer, per-rank RNG and
# sampler position survive a real process exit/restart at step 50.
torchrun --standalone --nproc-per-node=8 -m xuannv_embedding.cli train \
  --config "${config_path}" \
  --profile npu-smoke \
  --npu-phase resume \
  --output "${output_checkpoint}"

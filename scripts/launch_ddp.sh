#!/usr/bin/env bash
# 平台无关的 torchrun DDP 启动器；卡数与多节点参数通过环境变量覆盖。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ${LAUNCHER_NAME:-$0} CONFIG [xuannv train arguments...]" >&2
  exit 2
fi

config_path=$1
shift

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
nproc_per_node=${NPROC_PER_NODE:-8}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

torchrun \
  --nnodes="${nnodes}" \
  --node-rank="${node_rank}" \
  --nproc-per-node="${nproc_per_node}" \
  --master-addr="${master_addr}" \
  --master-port="${master_port}" \
  -m xuannv_embedding.cli train \
  --config "${config_path}" "$@"

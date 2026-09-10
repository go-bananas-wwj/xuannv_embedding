#!/usr/bin/env bash
# 多节点 24 卡训练入口；在每个节点各运行一次，只有 NODE_RANK 不同。
#
#   NODE_RANK=0 scripts/cuda/launch_24card.sh CONFIG --output /path/ckpt.pt
#
# 相对 scripts/cuda/launch.sh 只多两件事：预置本集群的 RoCE 环境，
# 以及从 scripts/cuda/cluster.env 读取节点拓扑（见 cluster.env.example）。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cuda/env_roce.sh
source "${script_dir}/env_roce.sh"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "未设置 MASTER_ADDR：复制 scripts/cuda/cluster.env.example 为 cluster.env 并填入 rank 0 的地址" >&2
  exit 2
fi

export NNODES=${NNODES:-3}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT:-29500}
export LAUNCHER_NAME=scripts/cuda/launch_24card.sh

exec "${script_dir}/../launch_ddp.sh" "$@"

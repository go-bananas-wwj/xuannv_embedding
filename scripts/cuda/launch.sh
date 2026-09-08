#!/usr/bin/env bash
# CUDA 入口：默认单机 8 卡；多节点用 NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT 覆盖。
set -euo pipefail

export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export LAUNCHER_NAME=scripts/cuda/launch.sh
exec "$(dirname "${BASH_SOURCE[0]}")/../launch_ddp.sh" "$@"

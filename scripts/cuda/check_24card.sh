#!/usr/bin/env bash
# 多节点 NCCL 自检入口；在每个节点各运行一次，只有 NODE_RANK 不同。
#
#   NODE_RANK=0 scripts/cuda/check_24card.sh    # rank 0 必须是 MASTER_ADDR 那台
#   NODE_RANK=1 scripts/cuda/check_24card.sh
#   NODE_RANK=2 scripts/cuda/check_24card.sh
#
# 节点地址来自 scripts/cuda/cluster.env（见 cluster.env.example），也可用
# 环境变量直接覆盖。只验证通信，不加载模型也不读数据：训练异常时先用它
# 区分网络问题与训练侧问题。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cuda/env_roce.sh
source "${script_dir}/env_roce.sh"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "未设置 MASTER_ADDR：复制 scripts/cuda/cluster.env.example 为 cluster.env 并填入 rank 0 的地址" >&2
  exit 2
fi

nnodes=${NNODES:-3}
node_rank=${NODE_RANK:-0}
nproc_per_node=${NPROC_PER_NODE:-8}
master_port=${MASTER_PORT:-29500}

echo "自检：NODE_RANK=${node_rank}/${nnodes}，每节点 ${nproc_per_node} 卡，master=${MASTER_ADDR}:${master_port}" >&2

launcher=(torchrun)
if [[ -n "${XUANNV_PYTHON:-}" ]]; then
  launcher=("${XUANNV_PYTHON}" -m torch.distributed.run)
fi

"${launcher[@]}" \
  --nnodes="${nnodes}" \
  --node-rank="${node_rank}" \
  --nproc-per-node="${nproc_per_node}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${master_port}" \
  "${script_dir}/check_multinode.py" "$@"

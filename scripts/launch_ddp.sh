#!/usr/bin/env bash
# 平台无关的 torchrun DDP 启动器；卡数与多节点参数通过环境变量覆盖。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ${LAUNCHER_NAME:-$0} CONFIG [xuannv train arguments...]" >&2
  exit 2
fi

config_path=$1
shift

# 解析启动器：裸 torchrun 可能来自系统 Python 或别的项目镜像，会以错误的解释器拉起
# 训练进程（典型症状是 ModuleNotFoundError: xuannv_embedding）。优先用 XUANNV_PYTHON
# 指定的解释器，其次用与本仓同装的那个，最后才回退到 PATH 上的 torchrun。
if [[ -n "${XUANNV_PYTHON:-}" ]]; then
  launcher=("${XUANNV_PYTHON}" -m torch.distributed.run)
elif python_bin=$(command -v python 2>/dev/null) \
  && "${python_bin}" -c "import xuannv_embedding" 2>/dev/null; then
  launcher=("${python_bin}" -m torch.distributed.run)
else
  launcher=(torchrun)
fi

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
nproc_per_node=${NPROC_PER_NODE:-8}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

"${launcher[@]}" \
  --nnodes="${nnodes}" \
  --node-rank="${node_rank}" \
  --nproc-per-node="${nproc_per_node}" \
  --master-addr="${master_addr}" \
  --master-port="${master_port}" \
  -m xuannv_embedding.cli train \
  --config "${config_path}" "$@"

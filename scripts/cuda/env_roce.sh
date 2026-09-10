#!/usr/bin/env bash
# 多节点 NCCL over RoCE 环境；用 `source` 载入而非执行。
#
# 这里只放与加速器/网卡相关的通用设置。节点地址等站点信息放在同目录的
# cluster.env（不入库，见 cluster.env.example）。换集群时先用
# scripts/cuda/check_multinode.py 复核下面每一项。

# 只使用高速 RoCE 网卡。不指定时 NCCL 可能选中低速的 bond 口，
# 用 `ibv_devinfo` 查看各设备速率后按实际情况覆盖。
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1}

# RoCE v2 对应的 GID index。用下面这行确认，选 "RoCE v2" 那一项的编号：
#   grep -r . /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/ 2>/dev/null
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-5}

# bootstrap 与 out-of-band 通信所用网卡；需是各节点互通的那一张。
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

# link_layer 是 Ethernet（RoCE）而非 InfiniBand，但仍走 IB verbs 路径。
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

# 首次联调建议 INFO，确认选中的网卡与 GID；稳定后设为 WARN 降噪。
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

_xuannv_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 站点参数（MASTER_ADDR 等）。已在环境里设置的值优先，不被本文件覆盖。
if [[ -f "${_xuannv_script_dir}/cluster.env" ]]; then
  while IFS='=' read -r _xuannv_key _xuannv_value; do
    [[ "${_xuannv_key}" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [[ -z "${!_xuannv_key:-}" ]] && export "${_xuannv_key}=${_xuannv_value}"
  done < "${_xuannv_script_dir}/cluster.env"
  unset _xuannv_key _xuannv_value
fi

# 训练来源证明：镜像内没有 .git 时 _git_sha() 会拒绝启动，这里预先固化。
if [[ -z "${XUANNV_GIT_SHA:-}" ]]; then
  _xuannv_repo_root="$(cd "${_xuannv_script_dir}/../.." && pwd)"
  if _xuannv_sha=$(git -C "${_xuannv_repo_root}" rev-parse HEAD 2>/dev/null); then
    export XUANNV_GIT_SHA="${_xuannv_sha}"
  fi
  unset _xuannv_repo_root _xuannv_sha
fi

# 解释器：各节点镜像可能自带别的 torch，裸 torchrun 会解析到系统 Python
# 而找不到 xuannv_embedding。这里锁定共享盘上的环境。
if [[ -z "${XUANNV_PYTHON:-}" ]]; then
  _xuannv_python=/data/heyuhang/xuannv_env/bin/python
  if [[ -x "${_xuannv_python}" ]]; then
    export XUANNV_PYTHON="${_xuannv_python}"
  fi
  unset _xuannv_python
fi

unset _xuannv_script_dir

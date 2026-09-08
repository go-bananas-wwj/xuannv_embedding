#!/usr/bin/env bash
# Ascend NPU 入口：默认 6 卡，对应已验收的 6×NPU DDP 发布门禁。
# 运行前请先加载 CANN set_env.sh。
set -euo pipefail

export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export LAUNCHER_NAME=scripts/npu/launch_6card.sh
exec "$(dirname "${BASH_SOURCE[0]}")/../launch_ddp.sh" "$@"

#!/usr/bin/env python3
"""多节点 NCCL 自检：只验证通信，不加载模型也不读数据。

用途是把「网络通不通」与「训练代码对不对」分开。真实训练失败时先跑它：
如果这里通过而训练失败，问题在训练侧；如果这里就失败，问题在网络或 NCCL 配置。

通过 torchrun 启动，例如三节点 24 卡：

    NODE_RANK=0 scripts/cuda/check_24card.sh

节点地址来自 scripts/cuda/cluster.env（见 cluster.env.example）。也可直接用
torchrun 启动本文件，自行传入 --nnodes/--node-rank/--master-addr 等参数。

检查项依次为：每个 rank 绑定到独立的本地卡、all_reduce 数值正确、
all_gather 覆盖全部 rank、以及一次粗略的 all_reduce 带宽测量。
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import sys
import time

import torch
import torch.distributed as dist


def _env(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise SystemExit(f"缺少环境变量 {name}；本脚本必须通过 torchrun 启动")
    return int(value)


def _log(rank: int, message: str) -> None:
    """只让 rank 0 打印汇总行，避免 24 份重复输出。"""
    if rank == 0:
        print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_multinode")
    parser.add_argument(
        "--payload-mb",
        type=int,
        default=256,
        help="带宽测量所用的 all_reduce 张量大小，默认 256 MiB",
    )
    parser.add_argument("--iters", type=int, default=10, help="带宽测量迭代次数，默认 10")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="进程组初始化超时；网络不通时用它换取明确报错而非无限等待",
    )
    args = parser.parse_args(argv)

    rank = _env("RANK")
    world_size = _env("WORLD_SIZE")
    local_rank = _env("LOCAL_RANK")

    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA 设备")
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise SystemExit(
            f"LOCAL_RANK={local_rank} 超出本节点可见卡数 {device_count}；"
            "检查 NPROC_PER_NODE 与 CUDA_VISIBLE_DEVICES"
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    start = time.perf_counter()
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=args.timeout_seconds),
    )
    handshake_seconds = time.perf_counter() - start

    _log(rank, f"[1/4] 进程组建立成功：world_size={world_size}，耗时 {handshake_seconds:.1f}s")

    # 每个 rank 贡献自己的编号，正确的 all_reduce 结果是 0+1+...+(world_size-1)。
    contribution = torch.full((1,), float(rank), device=device)
    dist.all_reduce(contribution)
    expected = float(world_size * (world_size - 1) // 2)
    actual = contribution.item()
    if actual != expected:
        raise SystemExit(f"all_reduce 数值错误：期望 {expected}，实得 {actual}")
    _log(rank, f"[2/4] all_reduce 数值正确：sum(rank) = {expected:.0f}")

    # 汇总每个 rank 的主机与本地卡，用来确认没有两个 rank 抢同一张卡。
    identity = f"{socket.gethostname()}:{local_rank}"
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, identity)
    if rank == 0:
        if len(set(gathered)) != world_size:
            raise SystemExit(f"存在重复的 (主机, 本地卡) 绑定：{gathered}")
        hosts = sorted({str(item).rsplit(":", 1)[0] for item in gathered})
        print(f"[3/4] {world_size} 个 rank 绑定到 {len(hosts)} 个节点，无重复占卡", flush=True)
        for host in hosts:
            ranks = [i for i, item in enumerate(gathered) if str(item).startswith(f"{host}:")]
            print(f"        {host}: rank {ranks[0]}-{ranks[-1]}（{len(ranks)} 卡）", flush=True)

    # 粗测 all_reduce 带宽。ring all_reduce 每个 rank 收发约 2*(N-1)/N 份数据。
    elements = args.payload_mb * 1024 * 1024 // 4
    payload = torch.ones(elements, dtype=torch.float32, device=device)
    for _ in range(3):  # 预热，排除首次通信的建链开销
        dist.all_reduce(payload)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if rank == 0:
        payload_bytes = elements * 4
        factor = 2 * (world_size - 1) / world_size
        bus_gbps = (payload_bytes * factor * args.iters) / elapsed / 1e9
        per_iter_ms = elapsed / args.iters * 1000
        print(
            f"[4/4] all_reduce {args.payload_mb} MiB × {args.iters} 次："
            f"{per_iter_ms:.1f} ms/次，bus 带宽约 {bus_gbps:.1f} GB/s",
            flush=True,
        )
        print("多节点 NCCL 自检通过", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

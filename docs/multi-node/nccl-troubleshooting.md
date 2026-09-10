# 跨节点故障速查

按症状查。每条给出判定依据与处置，不要凭猜测改环境变量。

首次联调时设 `NCCL_DEBUG=INFO`；`NCCL_DEBUG_SUBSYS=INIT,NET,ENV` 会额外打印后端选择与
网卡绑定，是定位网络问题的关键信息。稳定后设回 `WARN` 降噪。

**先看原始报错，不要看 `ChildFailedError`。** 后者只说明「有子进程退出了」，不说明原因。
真正的报错在它上面，或在其他 rank 的日志里。各节点日志若写在本地 `/tmp`，需逐台查看。

## 症状：`ibv_reg_mr_iova2 failed`

```
ibvwrap.c:115  NCCL WARN Call to ibv_reg_mr_iova2 failed
proxy.cc:1517  NCCL WARN [Service thread] Accept failed Cannot allocate memory
ncclSystemError: System call ... or external library call failed
```

RDMA 需要把通信缓冲区锁页注册，注册量受 `RLIMIT_MEMLOCK` 限制。rendezvous 走普通 socket
所以能成功（`[1/4]` 通过），第一次集合通信才失败。

判定：

```bash
ulimit -l; ulimit -Hl                      # hard limit 很小（如 8192 KB）即受限
grep CapEff /proc/self/status              # 0000000000000000 表示无任何 capability
```

`CAP_IPC_LOCK` 会**完全绕过** `RLIMIT_MEMLOCK`，所以严格来说只需要这一项。容器内无法自行
提升 hard limit，必须在实例/Pod 配置层面授予。

**注意失败位置。** 若报错来自 `transport/net.cc` 的连接建立路径，注册的是每条连接的
控制/FIFO 区，而不是 `NCCL_BUFFSIZE` 管的数据缓冲——此时调小 `NCCL_BUFFSIZE`、
`NCCL_MAX_NCHANNELS` **无效**。日志中若能看到部分 rank 已打出 `Connected all trees`，
说明前几条连接注册成功、池子中途耗尽，即固定开销已超限，没有调参空间。

处置：授予 `CAP_IPC_LOCK`（并将 memlock 设为 unlimited）；做不到则退到 TCP（见下条）。
Serverless 类容器产品通常不开放这些设置，见 `acp-migration.md`。

## 症状：设了 `NCCL_IB_DISABLE=1` 却仍在走 IB

```
NCCL INFO Using network IBext_v8          ← 仍是 IB
ibvwrap.c:115 NCCL WARN Call to ibv_reg_mr_iova2 failed
```

`NCCL_IB_DISABLE` **只关 NCCL 内置的 IB transport**。若环境装了厂商的外部网络插件
（`libnccl-net.so`，日志中显示为 `IBext_v8` 之类），插件优先级更高，置 1 也仍会走 IB verbs
并调用 `ibv_reg_mr`。

处置：换掉整个网络后端。

```bash
NCCL_NET=Socket NCCL_SOCKET_IFNAME=<互通网卡,可逗号分隔多张>
```

生效的标志是日志出现 `NET/Socket : Using [0]<网卡>:<地址>` 与 `Using network Socket`，
且 `ibv_reg_mr` 不再出现。

可先在单节点两进程上花几秒确认该变量有效，再去三台联调，省一轮往返：

```bash
NCCL_NET=Socket NCCL_SOCKET_IFNAME=<网卡> python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --master-port=<空闲端口> \
  scripts/cuda/check_multinode.py --payload-mb 8 --iters 2
```

**不要把 `NCCL_NET=Socket` 写进 `env_roce.sh` 的默认值。** 留空 NCCL 才能自行选优；在
提供 RDMA 的平台上写死 Socket 等于主动放弃 RDMA。

## 症状：`RuntimeError: 无法证明训练代码的 Git SHA`

训练拒绝启动，防止产出无来源 checkpoint。触发场景有两类：

- 镜像内没有 `.git`；
- 容器内以 root 运行，而仓库属主是普通用户，`git rev-parse` 因 dubious ownership 失败。

第二类容易误判成「代码有问题」，其实与代码无关。判定：

```bash
git -C <仓库> rev-parse HEAD; echo "exit=$?"
```

处置：显式传 `XUANNV_GIT_SHA=<40 位提交 SHA>`。这是设计意图，不是绕过。镜像化运行时应在
**构建期**固化该变量。

## 症状：`ModuleNotFoundError: xuannv_embedding`

裸 `torchrun` 解析到了系统 Python 或镜像自带的另一套 torch。`env_roce.sh` 会把
`XUANNV_PYTHON` 指向共享盘上的解释器，`scripts/launch_ddp.sh` 优先用它启动。镜像化运行时
按实际路径覆盖该变量。

## 症状：`Expected to have finished reduction in the prior iteration`

DDP 发现有参数没参与反向。本仓库的 `precision_scale=1` 会让 `AEFModel` 整体跳过
`upsample_head`，其 6 个参数因此拿不到梯度；这些键属于已登记的 431 键合同不能删除，所以
`training/cli.py` 里开了 `find_unused_parameters=True`。

该报错重新出现，说明前向路径变了。**不要直接删参数**——那会破坏旧 checkpoint 的加载合同。
先跑 `tests/test_training_runtime.py` 里对应的回归用例确认未用参数的范围。

## 症状：rendezvous 卡在 `[1/4]` 之前

各 rank 起来了但 world 凑不齐。依次检查：

- `MASTER_ADDR` 是否是 rank 0 那台，且各节点填的一致；
- `MASTER_PORT` 跨节点是否可达（`REFUSED` 说明没被防火墙拦，只是没人监听，属正常）；
- 上一轮残留进程是否仍占着端口——换个 `MASTER_PORT` 重试；
- `NNODES` 与实际启动的节点数是否一致，`NODE_RANK` 是否有重复或缺号。

## 带宽参考

同一套硬件上的实测量级，用于判断当前路径是否符合预期：

| 路径 | 量级 | 说明 |
| --- | --- | --- |
| 机内 NVLink（P2P/CUMEM） | ~168 GB/s | 单节点 8 卡 all_reduce bus 带宽 |
| RoCE 单流（`ibv_rc_pingpong`） | ~109 Gbit/s | 硬件与 RoCE 配置正常的证据 |
| 跨节点 TCP（`NCCL_NET=Socket`） | ~3.7 GB/s | 24 卡 all_reduce bus 带宽 |

TCP 约为 RoCE 的四分之一，够调试，不够正式训练。若跨节点带宽远低于该量级，先确认
`NCCL_SOCKET_IFNAME` 选中的是高速网卡而非管理口。

诊断 RDMA 硬件本身是否正常，用 `ibv_rc_pingpong` 而不是 NCCL——它不受 NCCL 配置影响，
且能通过逐步增大消息尺寸定位注册上限：

```bash
ibv_rc_pingpong -d <设备> -g <gid_index> -s <字节数>   # 服务端
ibv_rc_pingpong -d <设备> -g <gid_index> -s <字节数> <服务端地址>
```

# 24 卡环境分层验收

新集群、换镜像、换网络后按本清单逐层验证。**不要跳层**：上层失败时，下层的通过记录能
把排查范围缩小一个数量级。

节点地址等站点信息填在 `scripts/cuda/cluster.env`（复制 `cluster.env.example`）。该文件
不入库——节点地址属于内部拓扑，本仓库是公开的。环境变量优先于文件取值，临时改拓扑无需
改文件。

## 前置：三节点一致性

三个节点的以下四项必须完全一致，不一致会产生难以定位的故障：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

期望三台输出相同，例如 `2.6.0+cu124 True 8`。版本不同会在 NCCL 握手或算子层面出错；
卡数不同会让 `NPROC_PER_NODE` 与实际不符。

## L0–L1：单节点

```bash
scripts/cuda/launch.sh configs/examples/china_p10c_pilot.yaml \
  --synthetic --steps 5 --epochs 1 --output /tmp/l1.pt
```

单节点 8 卡通过后再上跨节点。这一层失败与网络无关。

## L2–L3：跨节点通信自检

先跑自检，不要直接跑训练。在每个节点各运行一次，**只有 `NODE_RANK` 不同**：

```bash
NODE_RANK=0 scripts/cuda/check_24card.sh   # 必须是 MASTER_ADDR 那台
NODE_RANK=1 scripts/cuda/check_24card.sh
NODE_RANK=2 scripts/cuda/check_24card.sh
```

期望 rank 0 输出四项全过：

```
[1/4] 进程组建立成功：world_size=24
[2/4] all_reduce 数值正确：sum(rank) = 276      ← 24×23/2，全 rank 都参与了
[3/4] 24 个 rank 绑定到 3 个节点，无重复占卡
[4/4] all_reduce 64 MiB × 5 次：… bus 带宽约 … GB/s
多节点 NCCL 自检通过
```

`sum(rank)` 是有效性校验：数值对不上说明有 rank 没参与归约。第 3 项防的是两个 rank 抢
同一张卡（`NPROC_PER_NODE` 或 `CUDA_VISIBLE_DEVICES` 配错时会发生）。

只到 `[1/4]` 就卡住或报错，说明 rendezvous 通了但集合通信没通——这是网络后端问题，
查 `nccl-troubleshooting.md`。

### 锁页内存受限的环境

容器 `ulimit -l` 很小且无 `CAP_IPC_LOCK` 时，RDMA 不可用（判定方法与原理见
`nccl-troubleshooting.md` 与 `acp-migration.md`）。此时退到 TCP：

```bash
NCCL_NET=Socket NCCL_SOCKET_IFNAME=<互通网卡> NODE_RANK=<n> scripts/cuda/check_24card.sh
```

注意必须用 `NCCL_NET=Socket`，**`NCCL_IB_DISABLE=1` 不够**——原因见速查表。

## L4：跨节点 synthetic 训练

自检通过后跑合成训练。它建真实模型、跑前反向、走 AMP 与梯度累积、落 checkpoint，
但不读数据，因此能在数据管线就绪前先把训练侧验完：

```bash
XUANNV_GIT_SHA=<40 位提交 SHA> \
NODE_RANK=<n> scripts/cuda/launch_24card.sh configs/examples/china_p10c_pilot.yaml \
  --synthetic --steps 20 --epochs 1 --batch-size 1 --spatial-size 128 \
  --output <共享盘路径>/ckpt.pt
```

三处容易出错：

- **`--epochs 1` 必须显式给**。不给会按配置里的 `epochs`（如 800）跑满。
- **`XUANNV_GIT_SHA` 建议显式给**。容器里常以 root 运行，而仓库属主是普通用户，
  `git rev-parse` 会因 dubious ownership 失败，导致训练拒绝启动。
- **`--output` 要落在共享盘**，且不要放进仓库目录。

期望 rank 0 输出汇总 JSON：

```json
{"batches": 20, "optimizer_steps": 10, "world_size": 24, "loss": …, "device_type": "cuda"}
```

`optimizer_steps` = `batches ÷ gradient_accumulation_steps`。对不上说明梯度累积在跨节点
下没生效。

### checkpoint 验收

多卡跑完必须验 checkpoint，确认 DDP 没有破坏已登记的键合同：

```python
import torch
ck = torch.load("<路径>/ckpt.pt", map_location="cpu", weights_only=False)
sd = ck["model"]
assert len(sd) == 431                                                    # 登记合同
assert len([k for k in sd if k.startswith("upsample_head.")]) == 6       # 未用参数仍在
print(ck["git_sha"], ck["metrics"]["world_size"])
```

第二条针对性很强：`precision_scale=1` 时 `upsample_head` 拿不到梯度，DDP 必须开
`find_unused_parameters=True`。该选项只影响归约记账，**不应影响 `state_dict`**——这条
断言就是在守这个性质。它失败意味着海淀 checkpoint 的加载合同被多卡破坏了。

## L5：真实数据

需要 `data-process` extra 与数据根目录就绪。见 `docs/data-processing.md`。

## 重跑前的清场

前一轮残留的进程会占着显存和端口，导致下一轮出现看似无关的故障。每轮之间：

```bash
pkill -f "xuannv_embedding.cli train" || true
pkill -f "torch.distributed.run" || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l   # 期望 0
```

并且**每轮换一个 `MASTER_PORT`**（29500 → 29501 → …），避免上一轮的 TIME_WAIT 连接干扰
新一轮的 rendezvous。

## 全局 batch 换算

改卡数会改变全局 batch，进而改变优化动力学：

```
全局 batch = 卡数 × data.batch_size × training.gradient_accumulation_steps
```

例如配置为 `batch_size: 3` + `gradient_accumulation_steps: 2` 时，8 卡是 48，24 卡是 144。
**扩卡时必须显式决定**是保持全局 batch 不变（相应调小上面两项），还是接受它变大并同步
调整 `lr` 与 `warmup_epochs`。默认什么都不改等于悄悄把 batch 放大了三倍。

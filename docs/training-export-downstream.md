# 训练、导出与下游评测

## 训练

CUDA 环境可直接运行单卡或 `torchrun`。训练命令默认优先使用 CUDA；`--synthetic` 只用于发布
smoke，必须显式给出 `--steps`。
真实栅格训练与导出需要安装 `data-process` extra：
`pip install ".[data-process]"`；基础安装仍可查看全部命令帮助并运行 synthetic CPU smoke。

CUDA 发布环境锁定为 `torch==2.6.0` 的 CUDA 12.4 构建；不要混装不同主次版本。
从不含 `.git` 的已安装 wheel 训练时，启动前必须把发布提交写入 `XUANNV_GIT_SHA`；非法或无法
证明的 SHA 会在设备初始化前被拒绝，避免产出无来源 checkpoint。

```bash
xuannv train --config configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt --device cuda:0

torchrun --standalone --nproc-per-node=8 -m xuannv_embedding.cli train \
  --config configs/production/mixed_haidian_harbin_p10c.yaml \
  --output /path/out/checkpoint.pt
```

仓内 `scripts/cuda/launch.sh` 默认启动单机 8 卡。两个平台入口（`scripts/cuda/launch.sh` 8 卡、
`scripts/npu/launch_6card.sh` 6 卡）都只设置默认卡数并转发给共用的 `scripts/launch_ddp.sh`；
`NPROC_PER_NODE` 可覆盖卡数。

### 多节点 24 卡

先复制 `scripts/cuda/cluster.env.example` 为 `scripts/cuda/cluster.env` 并填入 rank 0 的
`MASTER_ADDR`、`MASTER_PORT` 与节点数。该文件不入库：节点地址属于站点内部拓扑。环境变量优先于
文件取值，因此临时改拓扑无需改文件。

RoCE 相关设置集中在 `scripts/cuda/env_roce.sh`，两个入口脚本都会自动载入。换集群时逐项复核：
`NCCL_IB_HCA` 需指向高速网卡（不指定时 NCCL 可能选中低速 bond 口），`NCCL_IB_GID_INDEX` 需对应
RoCE v2（用 `grep -r . /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/` 确认编号），
`NCCL_SOCKET_IFNAME` 需是各节点互通的那张网卡。

先跑通信自检，再跑训练——它不加载模型也不读数据，能把网络问题与训练侧问题分开：

```bash
NODE_RANK=0 scripts/cuda/check_24card.sh   # 在 MASTER_ADDR 那台
NODE_RANK=1 scripts/cuda/check_24card.sh
NODE_RANK=2 scripts/cuda/check_24card.sh
```

自检通过后在每个节点各运行一次训练入口，只有 `NODE_RANK` 不同：

```bash
NODE_RANK=0 scripts/cuda/launch_24card.sh configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt
```

各节点镜像不一致时，裸 `torchrun` 可能解析到系统 Python 或镜像自带的另一套 torch，症状是
`ModuleNotFoundError: xuannv_embedding`。`env_roce.sh` 会把 `XUANNV_PYTHON` 指向共享盘上的
解释器，`scripts/launch_ddp.sh` 优先用它启动；镜像化运行时按实际路径覆盖该变量。
`env_roce.sh` 同时固化 `XUANNV_GIT_SHA`，因为镜像内通常没有 `.git`，缺少它训练会拒绝启动。

卡数变化时注意全局 batch：全局 batch = 卡数 × `data.batch_size` ×
`training.gradient_accumulation_steps`。要在 24 卡上保持与 8 卡基线相同的全局 batch（8×3×2=48），
可设 `data.batch_size: 2` 且 `gradient_accumulation_steps: 1`，此时 `lr` 与 `warmup_epochs`
无需改动；若全局 batch 变了，学习率需要同步复核。

分布式后端按设备类型自动选择：CUDA 用 nccl、NPU 用 hccl、CPU 用 gloo。CUDA AMP 在支持 bf16 的
硬件上使用 bfloat16 并且不启用 GradScaler（bf16 不需要 loss scaling），仅在硬件不支持 bf16 时
回退到 fp16 + GradScaler。因此 CUDA 的 loss 数值不会与 NPU fp16 记录逐位一致。

多区域使用相同模型和损失。区域 loader 按 `sampling_weight` 确定性轮转；每个 batch 保持同一区域，
以保留不同物理产品的原生高分尺寸。`--resume` 严格恢复模型、semantic probe、optimizer、scheduler
和 epoch。学习率按绝对 epoch 执行线性 warmup 与 cosine 退火；`save_every` 每隔指定 epoch 原子
写入带 `.epoch-NNNN` 后缀的恢复点，最终 checkpoint 始终写到 `--output`，两者不会互相覆盖。
恢复或导出新格式 checkpoint 时，还会逐项核对当前配置文件 SHA-256、source schema 和 region
清单；任一身份不一致都会在加载权重前失败，不能把别的运行状态静默套到当前数据合同。

## 导出

新格式 checkpoint 直接严格加载；原海淀 checkpoint 必须提供兼容 profile。导出文件为每 patch
一个 NPZ，含 `embedding[M,D,H,W]` 与 `timestamps[M]`，经过有限值检查后原子发布。

```bash
xuannv export --config configs/production/haidian_p10c_v1.yaml \
  --checkpoint /path/epoch_800.pt --compatibility-profile haidian_p10c_v1 \
  --region haidian --output-root /path/embeddings --device cuda:0
```

## 下游数据与协议

下游 NPZ 必须包含：`embeddings[N,D,H,W]`、`labels[N,H,W]`、唯一 `patch_ids[N]`。空间 fold
JSON 的每项包含互斥 `train / val / test`。

标准头只有 `linear`、`mlp`、`wide_mlp`、`deep_wide_mlp`、`conv3x3`、`unet` 和
`deeplab_lite`。few-shot 的 N 指 N 个正样本 patch，并确定性配对 N 个负样本 patch；不是任意
抽 N 个 patch。checkpoint 封存数据、标签、fold 和实际训练 patch 清单摘要。

阈值只用 validation 选择，test 报告 F1、AP、AUC。full-label 与 5/10/50-shot 必须分开报告；
更换标签、fold、shot、seed 或数据文件后，身份摘要不一致会直接拒绝评测。

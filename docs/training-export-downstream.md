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

### 8 卡训练

当前正式训练统一使用单节点 8 卡。年度 5 米方案与阶段门禁见
[年度技术报告](xuannv_annual5m_report.tex)。启动器固定为：

```bash
scripts/cuda/launch.sh CONFIG --output /path/out/checkpoint.pt
```

年度训练的吞吐瓶颈、必需的 GDAL 环境变量与实测加速倍数见
[训练加速](training-acceleration.md)。在网络文件系统上不设
`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` 会让训练慢 5 倍以上。

年度模型直接使用 `configs/production/annual5m_v1.yaml`：原生年度 manifest、共同地理遮挡、
日期/质量感知集合聚合、5m/PAN面积对齐和统一 64 维瓶颈均在同一前向中训练。真实训练必须
读取 `AnnualObservationDataset`；不提供 annual synthetic 替代路径。潜变量教师是可选第二阶段，
必须同时提供 `--teacher-checkpoint` 和 `--teacher-config`，两者通过配置摘要、source schema 和区域清单严格核验。
配置中的 `training.latent_prediction_weight` 默认是 0，只有显式打开该权重并提供教师 checkpoint
才会增加潜变量损失。教师权重不写入学生 checkpoint，避免恢复时把两套模型混在一起。

```bash
scripts/cuda/launch.sh CONFIG --output /path/out/student.pt \
  --teacher-checkpoint /path/out/teacher.pt \
  --teacher-config /path/configs/teacher.yaml
```

月度兼容模型可先用 synthetic smoke 验证运行时；年度模型直接对正式年度数据执行短跑。短跑至少记录每卡峰值显存、
有效样本与目标计数、数据等待时间、吞吐、有限 loss、梯度有限性、checkpoint 原子写入和严格恢复。
非有限 loss 直接终止训练；非有限梯度不终止，而是记入摘要的 `nonfinite_gradient_steps`
（每 rank 一份，也在 `rank_metrics` 中）。fp16/NPU 路径出现少量跳步属于 GradScaler 的正常
行为，但该计数接近 `optimizer_steps` 说明这一步实际没有学到东西，必须先查清再继续长训。
正式长训只有在缺模态 batch、PAN 原生网格和吉林一号原生网格都能稳定前反向后才启动。

全局 batch = 8 × `data.batch_size` × `training.gradient_accumulation_steps`。例如
`batch_size: 3`、`gradient_accumulation_steps: 2` 时为48。改变单卡 batch 或累积步数后，
必须重新核对学习率、warmup和有效目标数，不能只保持名义 epoch 数。

CUDA 使用 nccl。CUDA AMP 在支持 bf16 的硬件上使用 bfloat16 并且不启用 GradScaler（bf16 不需要
loss scaling），仅在硬件不支持 bf16 时回退到 fp16 + GradScaler。

多区域使用相同模型和损失。区域 loader 按 `sampling_weight` 确定性轮转；每个 batch 保持同一区域，
以保留不同物理产品的原生高分尺寸。`--resume` 严格恢复模型、semantic probe、optimizer、scheduler
和 epoch。学习率按绝对 epoch 执行线性 warmup 与 cosine 退火；`save_every` 每隔指定 epoch 原子
写入带 `.epoch-NNNN` 后缀的恢复点，最终 checkpoint 始终写到 `--output`，两者不会互相覆盖。
恢复或导出新格式 checkpoint 时，还会逐项核对当前配置文件 SHA-256、source schema 和 region
清单；任一身份不一致都会在加载权重前失败，不能把别的运行状态静默套到当前数据合同。

## P0 诊断

`xuannv diagnose` 在真实观测上核验月份绑定的高分融合合同，是启动正式长训前的门禁，不是精度
评测。它要求配置里 `data.highres_mode: observations` 且 `data.target_months` 非空，并逐项检查：

- 高分观测重排后 embedding 不变，重复同一观测后重排同样不变；
- 掩码全零的观测与完全不传该 source 逐位等价，且掩码外像元数值不影响结果；
- 未被任何观测覆盖的目标月份与无高分基线逐位相同；
- 缺测 source 与全部无效输入的重建损失为 0；
- 时序 stem、高分编码器和融合模块的梯度有限且非零，三类边界 batch 反传梯度有限。

```bash
xuannv diagnose --config /path/observations.yaml \
  --checkpoint /path/checkpoint.pt --output /path/reports/p0.json --device cuda:0
```

任一不变量失败直接抛错，不会写出 `passed: true` 的报告。报告显式记录
`landmark_registration` 和 `radiometric_and_cloud_QA` 为 `not_verified`、`p1_ready` 为
`false`：通过诊断只说明训练合同自洽，不代表绝对定位精度或云影准确率已经认证。

## 导出

新格式 checkpoint 直接严格加载；原海淀 checkpoint 必须提供兼容 profile。导出文件为每 patch
一个 NPZ，含 `embedding[M,D,H,W]`、`timestamps[M]`，以及可用时的
`validity_mask[M,1,H,W]`，经过有限值检查后原子发布。

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

# 年度 5m 一亿参数训练配置

本文说明 `configs/production/annual5m_100m_v1.yaml` 的由来、改动、资源需求和启动方式。
这是 `annual5m_v1`（401 万参数）的放大版：最终输出 embedding 仍是 64 维、5 m、每个 patch
`64 × 256 × 256`，下游数据格式和探针代码都不用改。

## 1. 与 `annual5m_v1` 的差异

其余字段（数据、source、掩蔽、学习率、月份、轮数）与 `annual5m_v1` 完全相同。

| 字段 | `annual5m_v1` | `annual5m_100m_v1` | 说明 |
|---|---|---|---|
| `experiment.name` | `annual5m_stp_ms_pan_v1` | `annual5m_stp_ms_pan_100m_v1` | 区分实验 |
| `model.stp.space_dim` | 128 | **1280** | STP 空间通道宽度 |
| `model.stp.time_dim` | 128 | **512** | STP 时间通道宽度 |
| `model.stp.num_blocks` | 2 | **4** | STP block 数 |
| `model.stp.reorder_resample` | false | **true** | 加速：先 1×1 卷积再上采样（见 §3.2） |
| `training.compile` | false | **true** | 加速：编译时序编码器，并开 TF32 与 fused AdamW |
| `training.highres_visible_targets` | false | **true** | 可见区域的高分像素也作为重建目标（见 §6） |
| `training.highres_loss_weight` | 1.0（默认） | 1.0 | 显式写出，便于后续调整 |

保持不变的关键字段：`embed_dim: 64`（最终输出）、`annual_feature_dim: 128`（融合层逐像素
特征）、`precision_dim: 128`、`num_heads: 8`、`stem_dim: 32`、`lr: 0.0001`、
`warmup_epochs: 5`、`epochs: 200`、`save_every: 50`、`batch_size: 4`、
`gradient_accumulation_steps: 1`、`gradient_checkpointing: false`。

有效 batch = 8 卡 × 每卡 4 = **32**，与 `annual5m_v1` 相同，所以学习率和 warmup 不需要重新调整。

## 2. 参数量

| 模块 | `annual5m_v1` | `annual5m_100m_v1` |
|---|---:|---:|
| 总参数 | 4,010,057 | **104,216,905** |
| 推理用（不含 S2/S1 与高分重建解码器） | 3,636,257 | 103,843,105 |

几乎所有新增参数都在 STP 时序编码器里。数字来自按配置实例化模型后的实际计数；两个加速开关
不改变参数和键名。

## 3. 形状与加速

### 3.1 为什么选这个形状

目标是约 1 亿参数、最终输出 64 维、训练尽量快。下表是单张 A100-SXM4-40GB 上用真实数据、
bf16 测得的结果（未开 §3.2 的加速；每个配置测 6 步，误差约 ±20%）。

| STP 形状（space/time/blocks） | 参数量 | 每卡 batch | 梯度检查点 | 每步 | 每样本 | 相对 `annual5m_v1` |
|---|---:|---:|---|---:|---:|---:|
| 128/128/2（`annual5m_v1`） | 401 万 | 4 | 否 | 0.39 s | 0.097 s | 1× |
| **1280/512/4（本配置）** | **1.04 亿** | **2** | **否** | **0.57 s** | **0.284 s** | **2.9×** |
| 1280/512/4 | 1.04 亿 | 2 | 是 | 0.72 s | 0.361 s | 3.7× |
| 1280/512/4 | 1.04 亿 | 4 | 是 | 1.70 s | 0.426 s | 4.4× |
| 1280/512/4 | 1.04 亿 | 4 | 否 | 显存不足（>40 GB） | — | — |
| 1024/512/5 | 9289 万 | 2 | 否 | 0.64 s | 0.319 s | 3.3× |
| 768/512/8（precision 256） | 1.01 亿 | 4 | 是 | 2.89 s | 0.723 s | 7.4× |
| 128/128/60 | 1.01 亿 | 4 | 是 | 14.65 s | 3.66 s | 37.7× |

宽而浅比窄而深快得多：每个 block 都要在 5 m 稠密特征图上完整算一遍，耗时基本与层数成正比；
加宽只增加矩阵乘法计算量，A100 算这类运算效率很高。

### 3.2 训练加速

剖析一亿参数模型的单步（真实数据、bf16）发现，时序编码器占前向约 90%；其中最大的开销
不是注意力，而是跨尺度交换里的双线性上采样：原代码先把最宽 1280 通道的特征上采样到
128×128 再用 1×1 卷积降到 128 通道，插值前反向合计约占全部 GPU 时间的 36%。

下表在单卡上逐项叠加（同一批真实数据、同一初始化，30 步取平均）：

| 方案 | 每卡 batch | 每样本 | 峰值显存 | 相对现状 |
|---|---:|---:|---:|---:|
| 现状（bf16 混合精度） | 2 | 0.296 s | 24.9 GB | 1× |
| + TF32 | 2 | 0.295 s | 24.9 GB | 1.00× |
| + 先 1×1 卷积再上采样 | 2 | 0.218 s | 18.4 GB | 1.36× |
| + fused AdamW | 2 | 0.211 s | 18.4 GB | 1.40× |
| + FlashAttention（SDPA） | 2 | 0.210 s | 18.4 GB | 1.41× |
| + `torch.compile` | 2 | 0.136 s | 13.1 GB | 2.18× |
| 同上，不开 compile、batch 4 | 4 | 0.194 s | 34.7 GB | 1.52× |
| **本配置：全部开启、batch 4** | **4** | **0.129 s** | **24.3 GB** | **2.29×** |

- **先降通道再上采样**：1×1 卷积与双线性插值可交换，双精度下两种顺序误差小于 1e-10，
  初始 loss 与原顺序一致到第 4 位小数。只改运算顺序，参数与键名不变。
- **`torch.compile`** 合并大量小算子，同时省显存，使每卡 batch 能从 2 提到 4、不再需要梯度累积。
  代价是每个进程启动时多约 2 分钟编译。只编译时序编码器：它的输入形状在 batch 间固定，
  不随变长的高分观测重新编译。编译后 checkpoint 键名不变，已验证可严格加载回未编译模型。
- **FlashAttention 与 TF32 基本无效**：空间注意力序列只有 64 个位置、时间注意力只有 12 个月，
  注意力本身很便宜；bf16 已开启，TF32 能覆盖的部分很少。FlashAttention 因此没有写进代码。
- 两个开关都默认关闭，`annual5m_v1` 及其已发布权重的行为不变。

## 4. 训练时长与资源

以下数字来自 8 卡实测（见 4.1），不是单卡外推。

| 项 | 数值 |
|---|---|
| 硬件 | 单机 8 × A100-SXM4-40GB（NVSwitch） |
| 吞吐 | 稳态 **48.7 samples/s**（未加速时为 24.1；`annual5m_v1` 为 59.9） |
| 每 epoch | 101,120 个样本，约 **0.58 小时** |
| 200 epoch | 约 **4.8 天** |
| 100 epoch | 约 2.4 天 |
| 峰值显存 | 28.0 GB / 卡（8 卡中最大值），40 GB 卡余量约 12 GB |
| checkpoint 大小 | 1.25 GB / 个（含 AdamW 状态；`annual5m_v1` 为 48 MB）；200 epoch 共 5 个，约 6.3 GB |
| 推理（导出 embedding） | 约为 `annual5m_v1` 的 2–3 倍耗时 |

### 4.1 8 卡实测（2026-10-03，含全部加速）

命令：`scripts/cuda/launch.sh configs/production/annual5m_100m_v1.yaml --output /tmp/.../ckpt.pt --steps 300 --epochs 1`

| 指标 | 结果 |
|---|---|
| 完成 | 300 次优化器更新，正常退出并写出 checkpoint；checkpoint 可严格加载回未编译模型 |
| 启动阶段 | 前 25 步用时 150 s（含数据加载与编译） |
| 稳态每 batch 耗时 | 0.669 s（batch 100→200）、0.646 s（batch 200→300） |
| 稳态吞吐 | 8 卡 × 4 / 0.657 s = 48.7 samples/s（含启动阶段的平均值为 29.1） |
| 数据等待占比 | 4.4%（14.5 s / 330.4 s），数据加载不是瓶颈 |
| 峰值显存 | 28.0 GB |
| 非有限梯度步 | 0 |

单卡实测每样本 0.129 s，折算 8 卡约 62 samples/s；实测 48.7，差额来自 DDP 梯度同步和
不同 batch 的高分观测数量差异。以 8 卡实测为准。

## 5. 启动

用 `setsid nohup` 脱离终端，断开 SSH 后训练继续：

```bash
cd /data/heyuhang/xuannv_embedding
export XUANNV_PYTHON=/data/heyuhang/xuannv_env/bin/python
RUN=outputs/annual5m/run_100m_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
cp configs/production/annual5m_100m_v1.yaml "$RUN/config.yaml"
git rev-parse HEAD > "$RUN/git_sha.txt"
setsid nohup scripts/cuda/launch.sh configs/production/annual5m_100m_v1.yaml \
  --output "$RUN/annual5m_100m_v1.pt" > "$RUN/train.log" 2>&1 < /dev/null &
```

- 周期 checkpoint 写在 `annual5m_100m_v1.epoch-0050.pt` 等位置（`save_every: 50`）。
- 必须在干净的 git 工作区启动；checkpoint 会记录 `git_sha` 和配置 SHA，导出时逐项核对。
- 查看进度：`grep '"progress"' "$RUN/train.log" | tail -1`。每 25 个 batch 打印一次 loss、
  梯度范数、累计耗时和数据等待时间。
- 中断后续训：同一条命令加 `--resume "$RUN/annual5m_100m_v1.epoch-0050.pt"`（或最近的周期
  checkpoint）。续训要求配置逐字节一致，会恢复模型、优化器、调度器与轮数。

## 6. 注意事项

1. **为什么打开 `highres_visible_targets`。** 默认只在被遮住的区域和月份上监督高分重建，梯度里
   没有要求嵌入保留可见高分细节的信号。10 epoch 对比实验（401 万参数，203 个城市 patch 的 5m OSM
   评测）中：
   - 只改融合结构（concat）时，5m 相对自身 10m 的增益和含高分 patch 的高频能量都没有变化；
   - 打开本选项后，含高分 patch 的高频能量占比由 1.58% 升至 2.01%（无高分 patch 不变），
     道路 5m 相对 10m 的增益由不显著变为 +0.0086 F1（3×3 卷积头，95% CI 不含 0），
     未见城市的水体 F1 +0.036；
   - 代价：含高分 patch 上的建筑 F1 下降（仅 24 个 patch，OSM 建筑标签稀疏，噪声较大）；
     权重调到 3 时道路与水体反而变差，所以保持 1.0。
   在 50 轮 checkpoint 上复查同一套 5m 评测，确认建筑没有明显退化再继续使用。
2. **不要只用 `--epochs` 缩短训练。** 学习率调度按 `training.epochs` 计算，只改命令行
   `--epochs` 会在 200 epoch 的余弦曲线上提前停下，学习率没有退火。
3. **数据规模仍是风险。** 参数多 26 倍，但训练数据只有约 5 万个位置 × 2 年；放大模型而数据
   不变，可能过拟合或提升有限，需以下游评测确认。
4. **输出格式不变。** 下游只需用新 checkpoint 重新导出 embedding。
5. **更换硬件时。** 显存小于 32 GB 的卡改 `batch_size: 2`、`gradient_accumulation_steps: 2`；
   仍不够时再开 `gradient_checkpointing: true`。不支持 CUDA 的设备会自动跳过 `compile`。

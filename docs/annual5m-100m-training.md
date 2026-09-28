# 年度 5m 一亿参数训练配置

本文说明 `configs/production/annual5m_100m_v1.yaml` 的由来、改动、资源需求和启动方式。
这是 `annual5m_v1`（401 万参数）的放大版：最终输出 embedding 仍是 64 维、5 m、每个 patch
`64 × 256 × 256`，下游数据格式和探针代码都不用改。

## 1. 与 `annual5m_v1` 的差异

两份配置只有以下 6 行不同，其余字段（数据、source、损失权重、掩蔽、学习率、月份）完全相同。

| 字段 | `annual5m_v1` | `annual5m_100m_v1` | 说明 |
|---|---|---|---|
| `experiment.name` | `annual5m_stp_ms_pan_v1` | `annual5m_stp_ms_pan_100m_v1` | 区分实验 |
| `model.stp.space_dim` | 128 | **1280** | STP 空间通道宽度 |
| `model.stp.time_dim` | 128 | **512** | STP 时间通道宽度 |
| `model.stp.num_blocks` | 2 | **4** | STP block 数 |
| `data.batch_size` | 4 | **2** | 每卡 batch |
| `training.gradient_accumulation_steps` | 1 | **2** | 补回有效 batch |

保持不变的关键字段：`embed_dim: 64`（最终输出）、`annual_feature_dim: 128`（融合层逐像素
特征）、`precision_dim: 128`、`num_heads: 8`、`stem_dim: 32`、`lr: 0.0001`、
`warmup_epochs: 5`、`epochs: 200`、`save_every: 50`、`gradient_checkpointing: false`。

有效 batch = 8 卡 × 每卡 2 × 累积 2 = **32**，与 `annual5m_v1` 相同，所以学习率和 warmup
不需要重新调整。

## 2. 参数量

| 模块 | `annual5m_v1` | `annual5m_100m_v1` |
|---|---:|---:|
| 总参数 | 4,010,057 | **104,216,905** |
| 推理用（不含 S2/S1 与高分重建解码器） | 3,636,257 | 103,843,105 |

几乎所有新增参数都在 STP 时序编码器里。数字来自按配置实例化模型后的实际计数。

## 3. 为什么选这个形状

目标是约 1 亿参数、最终输出 64 维、训练尽量快。下表是单张 A100-SXM4-40GB 上用真实数据、
bf16 测得的结果（每个配置测 6 步，误差约 ±20%）。

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

结论：

- **宽而浅比窄而深快得多。** 每个 block 都要在 5 m 稠密特征图上完整算一遍，耗时基本与层数
  成正比；加宽只增加矩阵乘法计算量，A100 算这类运算效率很高。同为 1 亿参数，60 层窄模型比
  本配置慢约 13 倍。
- **每卡 batch 2 + 梯度累积 2，比 batch 4 + 梯度检查点快约 1.5 倍。** batch 2 不开检查点时
  峰值显存 24.4 GB，40 GB 卡放得下，省掉了重算前向的开销。
- 只有在显存更小的卡上才需要把 `gradient_checkpointing` 改成 `true`（峰值降到 9.9 GB，
  速度慢约 25%），训练结果不变。

## 4. 训练时长与资源

以下数字来自 8 卡实测（见 4.1），不是单卡外推。

| 项 | 数值 |
|---|---|
| 硬件 | 单机 8 × A100-SXM4-40GB（NVSwitch） |
| 吞吐 | 稳态 **24.1 samples/s**（`annual5m_v1` 实际为 59.9，约为其 2.5 倍耗时） |
| 每 epoch | 101,120 个样本，约 **1.17 小时** |
| 200 epoch | 约 **9.7 天** |
| 100 epoch | 约 4.9 天 |
| 峰值显存 | 28.3 GB / 卡（8 卡中最大值），40 GB 卡余量约 11 GB |
| checkpoint 大小 | 1.25 GB / 个（含 AdamW 状态；`annual5m_v1` 为 48 MB）；200 epoch 共 5 个，约 6.3 GB |
| 推理（导出 embedding） | 约为 `annual5m_v1` 的 2.5–3 倍耗时 |

### 4.1 8 卡实测（2026-09-28）

命令：`scripts/cuda/launch.sh configs/production/annual5m_100m_v1.yaml --output /tmp/.../ckpt.pt --steps 300 --epochs 1`

| 指标 | 结果 |
|---|---|
| 完成 | 300 个 batch / 150 次优化器更新，正常退出并写出 checkpoint |
| 稳态每 batch 耗时 | 0.664 s（batch 100→200）、0.663 s（batch 200→300） |
| 稳态吞吐 | 8 卡 × 2 / 0.663 s = 24.1 samples/s（含启动阶段的平均值为 23.5） |
| 数据等待占比 | 4.4%（9.0 s / 203.9 s），数据加载不是瓶颈 |
| 峰值显存 | 28.3 GB（单卡 6 步测试为 24.4 GB；差异来自 DDP 通信缓冲和不同 batch 的高分观测数量） |
| 非有限梯度步 | 0 |
| loss | 0.656（batch 25）→ 0.620（batch 300），正常下降 |

单卡 benchmark 只测了 6 步，曾估计 200 epoch 约 11.4 天；以本节 8 卡实测的 9.7 天为准。
正式训练前仍建议在新环境上重跑这条命令核对吞吐。

## 5. 启动

与 `annual5m_v1` 相同，用 `setsid nohup` 脱离终端，断开 SSH 后训练继续：

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
- 正式训练前先跑一次短冒烟，确认吞吐和显存：

  ```bash
  scripts/cuda/launch.sh configs/production/annual5m_100m_v1.yaml \
    --output /tmp/smoke/ckpt.pt --steps 300 --epochs 1
  ```

- 查看进度：`grep '"progress"' "$RUN/train.log" | tail -1`。每 25 个 batch 打印一次 loss、
  梯度范数、累计耗时和数据等待时间。

## 6. 注意事项

1. **效果提升尚未验证。** 参数多 26 倍、训练时间约 2.5 倍，但训练数据只有约 5 万个位置 × 2 年。
   AEF 的消融显示训练数据规模对效果影响很大，放大模型而数据不变，可能过拟合或提升有限。
   建议先做短对比：`annual5m_v1` 与本配置各训 10 epoch（改 `training.epochs: 10`，使余弦
   退火在 10 epoch 内完成），用同一套下游快速评测比较，再决定是否投入完整训练。
2. **不要只用 `--epochs` 缩短训练。** 学习率调度按 `training.epochs` 计算，只改命令行
   `--epochs` 会在 200 epoch 的余弦曲线上提前停下，学习率没有退火。
3. **输出格式不变。** 下游只需用新 checkpoint 重新导出 embedding。
4. **更换硬件时。** A100 80GB 可改回 `batch_size: 4`、`gradient_accumulation_steps: 1`；
   显存小于 32 GB 的卡需开 `gradient_checkpointing: true`。

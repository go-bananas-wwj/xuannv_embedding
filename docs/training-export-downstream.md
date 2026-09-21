# 训练、导出与下游评测

## 训练

Ascend 环境先加载 CANN，再运行单卡或 `torchrun`。训练命令默认读取配置中的真实 manifest；
`--synthetic` 只用于发布 smoke，必须显式给出 `--steps`。
真实栅格训练与导出需要同时安装 `npu` 和 `data-process` extras：
`pip install ".[npu,data-process]"`；基础安装仍可查看全部命令帮助并运行 synthetic CPU smoke。

发布环境锁定为已验收的 `torch==2.6.0` 与 `torch-npu==2.6.0.post5`；不要混装不同主次版本。
从不含 `.git` 的已安装 wheel 训练时，启动前必须把发布提交写入 `XUANNV_GIT_SHA`；非法或无法
证明的 SHA 会在设备初始化前被拒绝，避免产出无来源 checkpoint。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
xuannv train --config configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt --device npu:0

torchrun --standalone --nproc-per-node=6 -m xuannv_embedding.cli train \
  --config configs/production/mixed_haidian_harbin_p10c.yaml \
  --output /path/out/checkpoint.pt
```

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
  --region haidian --output-root /path/embeddings --device npu:0
```

## 下游数据与协议

下游 NPZ 必须包含：`embeddings[N,D,H,W]`、`labels[N,H,W]`、唯一 `patch_ids[N]`。空间 fold
JSON 的每项包含互斥 `train / val / test`。

标准头只有 `linear`、`mlp`、`wide_mlp`、`deep_wide_mlp`、`conv3x3`、`unet` 和
`deeplab_lite`。few-shot 的 N 指 N 个正样本 patch，并确定性配对 N 个负样本 patch；不是任意
抽 N 个 patch。checkpoint 封存数据、标签、fold 和实际训练 patch 清单摘要。

阈值只用 validation 选择，test 报告 F1、AP、AUC。full-label 与 5/10/50-shot 必须分开报告；
更换标签、fold、shot、seed 或数据文件后，身份摘要不一致会直接拒绝评测。

## 验证集遮挡重建诊断

`xuannv experiment reconstruct` 从已登记的实验 checkpoint 和同一缓存加载模型，
在编码前清零指定源的目标月输入和掩码；`--aliases` 指定需要同时屏蔽的重复表示。
`--context prefix` 还会清零所有源的未来月份和无法确定日期的静态输入。
该模式只检查推理输入依赖，不证明训练权重从未见过未来月份。

```bash
xuannv experiment reconstruct --config /path/config.yaml --cache /path/cache \
  --checkpoint /path/run/epoch_0200.pt --output /path/new-reconstruction \
  --device npu:0 --target s2_recon --months 0 1 2 3 4 5 --context offline
```

评分只读训练与验证记录，均值基线只拟合训练像元。逐月、逐波段报告归档归一化单位的
RMSE、MAE、偏差和有效数量；模型全域与时序插值共有域分开，空域记 null。
输出逐图块预测 NPZ、充分统计量及身份 JSON，放在仓库外。原生解码头结果是模型诊断，
不能替代所有嵌入使用相同预算解码器的公平比较。开发测试覆盖遮挡、共享张量保护、
无历史域、无效像元及训练/验证/测试读取边界；真实模型结果须另外运行并登记。

## 多任务结果归档

已完成的跟进计划可生成独立 TeX、JSON、逐条件 CSV，供审阅后放入论文实验文件夹。
入口要求跟进程序处于 `ready_to_publish`，核对 checkpoint、导出、评价、预测数组和独立
复算记录；支持样本必须与 B0 一致。输出目录必须为新目录，不覆盖已有实验记录。

```bash
xuannv experiment report-multitask --plan /path/T0_followup_plan.json \
  --baseline-verification /path/P0_verification.json --output /path/new-report-staging
```

后续调参组还必须传入 `--reference-followup /path/T0_followup`，比较同更新预算的 T0。
表中保留候选相对 B0、T0 的原始指标差值；两组分数都以 B0 误差归一化，随后相减，
不更换分母。未定义的 R² 保留为空并记录定义条件数。负结果不会被转换成改善结论。
命令只生成待审阅制品；审阅、编译、提交 Git、同步 Overleaf 并核对远端提交之后，
才能启动下一训练组。失败或未完成的运行仍需单独记录状态及原因，不能生成完成报告。

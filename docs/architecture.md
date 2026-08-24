# 架构与模块边界

## 不变量

一个样本代表 1,280 m × 1,280 m patch。时序源先按月聚合到共同的 `YYYYMM` 序列，模型输出
`[B, M, 64, 128, 128]`。vMF bottleneck 是唯一允许的 L2 归一化瓶颈；项目不使用
`einops`。

```text
physical sources --source_map--> canonical slots
       | temporal                    | highres + availability
       v                             v
per-source stem -> gated temporal STP -> monthly embedding -> highres fusion
                                                   -> vMF -> 64-D map
                                                   -> P10C reconstruction heads
```

## 包内模块

- `config.py`：严格、自包含的 schema v1；拒绝未知字段、重复键和 `_base_`。
- `data/`：区域 manifest 到规范 source 的映射、真实 GeoTIFF batch 和区域 loader 轮转。
- `models/`：sensor stem、STP、月度模块、高分融合、vMF 和 decoder。
- `training/`：P10C 损失、困难 masking、DDP 训练运行时、严格 checkpoint 和旧权重兼容。
- `export/`：登记 artifact 与逐 patch 原子 NPZ 导出。
- `downstream/`：唯一的下游包，包含标准头、空间 fold、balanced few-shot 与 F1/AP/AUC。
- `data_process/`：全国父网格、采样、catalog、物化、预处理、十等分与审计。
- `cli/`：唯一 `xuannv` 命令入口；可选依赖延迟导入。

`scripts/` 只能放 NPU 启动和发布门禁包装，生产 Python 逻辑必须位于包内。

## 区域边界

核心代码不得检查 `haidian` 或 `harbin` 来决定模型行为。每个区域只提供：

- 物理 source 到规范槽位的 `source_map`；
- 本区域 manifest、patch grid、统计量和监督标签根；
- 训练 sampling weight。

缺失模态保留零 availability；其 frame 只是形状载体，不作为有效输入，重建 mask 同时为零。
多区域训练使用同一 `TrainingSystem`，按 `sampling_weight` 轮转区域 loader，避免不同原生高分
尺寸被错误堆叠到同一 batch。

## 状态与制品

新 checkpoint 包含格式版本、配置 SHA、Git SHA、source schema、区域清单、epoch、模型、
semantic probe、优化器、调度器和指标。写入采用临时文件 + `fsync` + 原子替换。

旧海淀权重不转换、不覆盖。兼容 profile 先核验登记 SHA，再一对一映射旧高分 encoder/decoder
键，并要求全部 431 个键恰好消费。

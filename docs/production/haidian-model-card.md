# 海淀 P10C V1 模型卡

## 模型

- 名称：`haidian-embedding-v1` / P10C epoch 800。
- 时段：2025-12 至 2026-05。
- 训练范围：海淀 320 个 1,280 m patch。
- 输出：每月 `64 × 128 × 128`，10 m 等效网格，vMF 单位球面。
- checkpoint SHA-256：
  `69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790`。

输入是 Sentinel-2、Sentinel-1、Landsat、高分光学和高分 SAR。目标包括多源/高分重建、OSM
区域无关弱语义、modality/month/spatial masking、semantic hard negatives、uniformity 和 vMF。
最终下游人工标签不进入上游训练。

## 预期用途

适合在海淀及合同兼容数据上导出月度 embedding、检索和训练轻量下游制图头。跨区域使用应先做
输入/统计量/缺模态审计，再按固定空间 fold 报告下游指标。

不适合把 OSM 未标注像素直接解释为确定背景，也不适合在没有目标区验证时声称全国泛化。完整
监督的原始影像 + UNet/DeepLab-lite 仍是强基线；该 embedding 的主要价值是少量标注和跨任务
复用。

## 风险与限制

- 月度影像对云、雾、阴影和观测稀疏敏感；必须使用像素质量 mask。
- OSM 弱语义存在漏标、错标和时间滞后。
- 高分物理产品可能因区域不同而不同；`source_map` 仅是 schema 映射，不代表传感器等价。
- 海淀单城证据不能外推为全国性能。

制品位置与摘要见 [artifact manifest](artifacts.json)，评测见 [海淀证据摘要](haidian-evidence.md)。

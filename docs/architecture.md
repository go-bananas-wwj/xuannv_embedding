# 架构与模块边界

## 不变量

一个年度样本代表 1,280 m × 1,280 m patch。S2/S1 进入年度 STP，5m 多光谱和 2m PAN
先在原生网格编码，再以面积交叠权重表达为 5m；模型输出 `[B, 1, 64, 256, 256]`。
月度兼容模型仍输出 `[B, M, 64, 128, 128]`。vMF bottleneck 是唯一允许的 L2 归一化瓶颈；
项目不使用 `einops`。

```text
physical sources --source_map--> canonical slots
       | annual temporal             | native highres observations
       v                             v
 source stems -> annual STP -> F10 -> 5m grid   native encoder -> area alignment
                                      \             /
                                       gated joint 5m decoder -> vMF -> 64-D map
                                       -> held-out date/source reconstruction
```

## 包内模块

- `config.py`：严格、自包含的 schema v1；拒绝未知字段、重复键和 `_base_`。
- `data/`：区域 manifest 到规范 source 的映射、真实 GeoTIFF batch 和区域 loader 轮转。
- `models/`：sensor stem、STP、月度模块、高分融合、vMF 和 decoder。
- `models/annual.py`：年度 STP、原生尺度高分编码、2m→5m面积聚合、质量/日期池化和联合解码。
- `training/annual.py`、`training/annual_data.py`：年度留出遮挡、原生观测批处理和重建目标。
- `training/`：P10C 损失、困难 masking、按月跨卡均匀性、冻结教师潜变量预测、DDP 训练运行时、
  严格 checkpoint 和旧权重兼容。
- `export/`：登记 artifact 与逐 patch 原子 NPZ 导出。
- `downstream/`：唯一的下游包，包含标准头、空间 fold、balanced few-shot 与 F1/AP/AUC。
- `data_process/`：全国父网格、采样、catalog、物化、预处理、十等分与审计。
- `cli/`：唯一 `xuannv` 命令入口；可选依赖延迟导入。

`scripts/` 只能放加速器启动和发布门禁包装，生产 Python 逻辑必须位于包内。

## 区域边界

核心代码不得检查 `haidian` 或 `harbin` 来决定模型行为。每个区域只提供：

- 物理 source 到规范槽位的 `source_map`；
- 本区域 manifest、patch grid、统计量和监督标签根；
- 训练 sampling weight。

缺失模态保留零 availability；其 frame 只是形状载体，不作为有效输入，重建 mask 同时为零。
多区域训练使用同一 `TrainingSystem`，按 `sampling_weight` 轮转区域 loader，避免不同原生高分
尺寸被错误堆叠到同一 batch。

高分融合是低分特征的残差门控更新；高分 mask 全零时逐位返回低分特征。年度输出携带月级
`validity_mask`，场景级 embedding 和均匀性目标都只使用有效月份，导出时一并保留该 mask。

年度模型的 PAN 和 5m 分支均先编码后对齐；面积算子只负责地理网格转换，不是可学习的
语义层。无高分观测时联合解码退化为低分特征；无效像元在编码前被置零并携带支持比例。

年度训练可启用 `model.annual_context_pooling: true`。输入掩膜继续在全部编码器前隐藏目标值，
独立的 `source_validity_masks` 只保存原始有效性，供 STP 之后的年度池化使用。这使人工遮挡块
内部的上下文预测保留重建梯度，同时不把真实 nodata 标为有效。该开关默认关闭，以保留已训练
版本的数值路径；完整输入下两条路径的推理结果一致。回归测试隔离块内部，验证主干与瓶颈梯度、
目标值不可见及真实缺测回退。

新训练还应启用 `model.annual_exact_mask_coverage: true`。地理可见性从 10m 投影到 2m 时，
历史 Float32 面积权重的行和会偏离 1，配合 `1-1e-6` 阈值会误删完整可见 PAN 网格中
约 48.56% 的像元。该开关用精确的整数网格包含关系投影可见性；非整数网格使用归一化的
Float64 几何权重。它不改变特征面积聚合算子，也不增加模型参数。开关默认关闭以保留旧制品
的输入路径；新旧输入策略须显式登记，不能静默混用。测试覆盖 128→640 完整支持、
非整数网格面积守恒、真实 nodata 和遮挡值不可见性。

`training.highres_visible_weight` 为显式的可见区域像素权重；留出区域的像素权重固定为 1。
高分损失按两类区域的加权有效支持度归一化，再按观测与 source 聚合，最后乘既有的
`highres_loss_weight`。训练同时报告 `highres_heldout_recon` 与 `highres_visible_recon`。
原生 PAN 聚合单元可同时覆盖两类区域，因此分别聚合目标与支持，再计算各部分损失。
该字段缺省为 `null`，保留旧 `highres_visible_targets` 行为；显式设置后，以实际编码器可见性
划分两类区域。固定留出评测须统一使用相同遮挡、目标划分和权重，不能沿用各模型训练目标比较。

年度语义蒸馏允许教师使用不同的 STP 宽度和深度，但输入 source、PAN 分支、64 维输出和
月份合同必须一致。`training.latent_prediction_pool_factors: [2, 4, 8]` 在 5m 输出上生成
10/20/40m 的有效支持加权目标，通过学生专用 1×1 predictor 比较余弦距离；教师固定且不运行
重建 decoder。损失内的特征归一化不改变模型瓶颈合同。未指定尺度时保留旧的 8×8 目标。
冻结教师视图按教师自己的配置准备，保留其输入预处理策略；修正学生 PAN 掩膜不会悄悄改变
既有教师函数。两者仍使用同一年度、位置和训练统计量。

可选的 `training.annual_feature_anchor_weight` 直接保持原生 5m 特征函数。主学生前向仍使用
原训练遮挡和 vMF 噪声；额外学生前向与冻结参考读取同一未遮挡视图，均使用 vMF 确定性方向，
关闭重建 decoder。仅在双方有效像元上平均 `0.5 * sum_channels((mu - mu_ref)^2)`；
参考停止梯度，不使用预测器、池化或额外归一化。学生的额外前向保留梯度与 checkpointing，
因此须单独核验显存和吞吐。该约束不增加导出模型参数，默认关闭；它是待评估的训练目标，
工程实现不代表细节或语义精度提升。

`training.view_consistency_weight` 默认为 0；开启后，同一年度观测生成第二个独立遮挡视图，
以学生模型的 EMA 提供停止梯度的多尺度目标。两视图使用独立随机数流，增加第二视图不会改变
主视图的遮挡序列。EMA 只在成功的优化器更新后推进，累计步尾部也遵守这一规则；目标模型始终
处于 eval 模式。`view_consistency_pool_factors` 默认 `[2, 4, 8]`，
`view_consistency_ema_decay` 默认 `0.999`；不对不同年份施加不变性。

## 状态与制品

新 checkpoint 包含格式版本、配置 SHA、Git SHA、source schema、区域清单、epoch、模型、
semantic probe、优化器、调度器和指标。写入采用临时文件 + `fsync` + 原子替换。
开启 EMA 时，额外的 `auxiliary_state` 保存 EMA 权重、更新次数和 decay；恢复时严格核验。
旧 checkpoint 不要求该可选字段。冻结外部教师不写入学生权重，身份摘要写入指标。

旧海淀权重不转换、不覆盖。兼容 profile 先核验登记 SHA，再一对一映射旧高分 encoder/decoder
键，并要求全部 431 个键恰好消费。

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

## 状态与制品

新 checkpoint 包含格式版本、配置 SHA、Git SHA、source schema、区域清单、epoch、模型、
semantic probe、优化器、调度器和指标。写入采用临时文件 + `fsync` + 原子替换。

旧海淀权重不转换、不覆盖。兼容 profile 先核验登记 SHA，再一对一映射旧高分 encoder/decoder
键，并要求全部 431 个键恰好消费。

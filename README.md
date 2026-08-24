# 玄女 XuanNv Embedding

玄女把多源月度遥感观测编码为可复用的密集地理特征。每个 1,280 m × 1,280 m
patch 输出 `64 × 128 × 128` embedding；建筑、道路、水体等任务在冻结 embedding 上训练
标准下游头，不需要为每个任务重新训练上游模型。

本仓是从旧仓经审计白名单重建的生产仓。代码不按城市分支执行：区域差异只通过配置、
manifest、统计量和 `source_map` 表达。权重、影像、embedding、日志和大型报告不进入 Git。

## 生产基线

- 模型：P10C，月度多源 stem → STP → 高分融合 → vMF → 重建 decoder。
- 输入：Sentinel-2（12）、Sentinel-1（2）、Landsat（7）、高分光学（3）和可选高分 SAR（1）。
- 输出：6 个月、64 维、128 × 128；vMF 输出逐像素单位范数。
- 已登记制品：海淀 P10C epoch 800，SHA-256
  `69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790`。
- 边界：海淀是已验证生产基线；哈尔滨是冻结跨区评测，不代表全国泛化；China 配置是数据
  pilot，不是全国训练完成声明。

## 安装

需要 Python 3.11。

```bash
pip install .
pip install ".[data-process]"   # 网格、STAC、栅格和 manifest
pip install ".[downstream]"    # 标准下游头与指标
pip install ".[npu,data-process]" # Ascend 真实栅格训练/导出（已配置 CANN）
```

唯一入口是 `xuannv`：

```bash
xuannv --help
xuannv data --help
xuannv train --help
xuannv export --help
xuannv downstream --help
```

## 常用流程

```bash
# 严格解析自包含配置并训练；默认使用配置中的真实区域 manifest
xuannv train \
  --config configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt

# 原海淀 checkpoint 不改写，按登记 SHA 和 431 键兼容合同加载
xuannv export \
  --config configs/production/haidian_p10c_v1.yaml \
  --checkpoint /path/to/haidian_embedding_v1_p10c_epoch800.pt \
  --compatibility-profile haidian_p10c_v1 \
  --region haidian \
  --output-root /path/out/embeddings \
  --device npu:0

# 下游阈值只在 validation 上选择，test 只做一次固定评测
xuannv downstream train --dataset task.npz --folds folds.json --fold 0 \
  --head conv3x3 --shot 5 --output head.pt
xuannv downstream evaluate --dataset task.npz --folds folds.json --fold 0 \
  --checkpoint head.pt --output metrics.json
```

数据命令统一为 `xuannv data grid|registry|partition|materialize|preprocess|manifest|validate`。
每个命令的必填输入可用 `--help` 查看；生产写入拒绝覆盖完成目录，并保留可恢复的 partial
状态或原子发布结果。

## 文档

- [架构与模块边界](docs/architecture.md)
- [配置和 manifest 合同](docs/configuration-and-manifest.md)
- [全国数据处理](docs/data-processing.md)
- [训练、导出与下游评测](docs/training-export-downstream.md)
- [海淀模型卡](docs/production/haidian-model-card.md)
- [海淀评测证据](docs/production/haidian-evidence.md)
- [哈尔滨迁移证据与限制](docs/production/harbin-transfer.md)
- [迁移与来源](MIGRATION.md)
- [NPU 发布清单](docs/release/npu-checklist.md)

大制品发布在
[ModelScope `WeijieWu/xuannv_haidian_embdding`](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding)，
登记路径与摘要见 [artifact manifest](docs/production/artifacts.json)。

## 许可证与安全

代码采用 [Apache-2.0](LICENSE)。贡献前请阅读 [CONTRIBUTING](CONTRIBUTING.md)；安全问题按
[SECURITY](SECURITY.md) 私密报告，不要在公开 issue 中粘贴令牌或敏感数据。

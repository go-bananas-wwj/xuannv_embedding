# XuanNv Embedding 项目指南

本文件面向 AI 编程代理。阅读前请假设你对本项目一无所知；以下信息均来自仓库实际内容，而非通用假设。

## 项目概述

玄女（XuanNv Embedding）是一个区域无关的月度地理空间 embedding 生产框架。它将多源月度遥感观测（Sentinel-2、Sentinel-1、Landsat、高分光学/高分 SAR）编码为可复用的密集地理特征。每个 1,280 m × 1,280 m 的 patch 输出 `64 × 128 × 128` 的 embedding；建筑、道路、水体等下游任务只需在冻结 embedding 上训练标准头，无需重新训练上游模型。

本仓库是经审计白名单重建的生产仓库，不继承旧仓库历史。代码不按城市分支：区域差异仅通过配置、manifest、统计量和 `source_map` 表达。权重、影像、embedding、日志和大型报告不进入 Git。

- 包名：`xuannv-embedding`
- 入口命令：`xuannv`
- Python 版本：`>=3.11`
- 主框架：PyTorch 2.6.0
- 许可证：Apache-2.0

## 技术栈

- **构建后端**：setuptools + wheel（`pyproject.toml`）
- **核心依赖**：`numpy>=1.24,<2`、`PyYAML>=6.0`、`torch==2.6.0`、`tqdm>=4.66`
- **可选依赖**：
  - `data-process`：geopandas、pandas、planetary-computer、pyarrow、pyogrio、pyproj、pystac-client、rasterio、requests、shapely、stackstac、xarray、zarr（`<3`）
  - `downstream`：matplotlib、scikit-image、scikit-learn、seaborn
  - `dev`：black、build、pip-audit、pytest、pytest-cov、ruff、twine
  - `npu`：`torch-npu==2.6.0.post5`
- **模型约束**：vMF bottleneck 是唯一允许的 L2 归一化瓶颈；项目**不使用 `einops`**。

## 代码组织

所有生产 Python 代码位于 `src/xuannv_embedding/`。`scripts/` 只放加速器启动和发布门禁包装：
`launch_ddp.sh` 是共用 torchrun 启动器，`cuda/launch.sh`（默认 8 卡）和 `npu/launch_6card.sh`
（默认 6 卡）只设置默认卡数后转发；`release/` 放仓库策略与模型门禁的薄包装。

```text
src/xuannv_embedding/
├── cli/              # 唯一 `xuannv` 命令入口；可选依赖延迟导入
│   ├── __init__.py   # 顶层 argparse 与子命令分发
│   └── __main__.py   # 支持 `python -m xuannv_embedding.cli`
├── config.py         # 严格自包含的 schema v1 配置解析
├── data/             # 区域 manifest → 规范 source 映射、真实 GeoTIFF batch、区域 loader
├── data_process/     # 全国父网格、采样、catalog、物化、预处理、十等分、审计
├── downstream/       # 标准下游头、空间 fold、balanced few-shot、F1/AP/AUC 评测
├── export/           # 登记 artifact 与逐 patch 原子 NPZ 导出
├── models/           # sensor stem、STP、月度模块、高分融合、vMF、decoder
├── training/         # P10C 损失、masking、DDP 运行时、严格 checkpoint、旧权重兼容
└── utils/            # geo、manifest、仓库策略等通用工具
```

### 主要模块边界

- **config.py**：拒绝未知字段、重复 YAML 键、`_base_` 继承；所有数值/布尔/路径都严格校验。
- **models/model.py**：`AEFModel` 是主模型，流程为：多源时序 stem → STP 编码器 → 月度嵌入 → 上采样 → 可选高分融合 → vMF bottleneck → 解码器。
- **training/cli.py**：训练入口，支持单卡、`torchrun` CUDA/CPU/NPU DDP、`--synthetic` smoke。
- **export/cli.py**：从严格 checkpoint 或旧海淀兼容 profile 导出 embedding。
- **data_process/cli.py**：`xuannv data <grid|registry|partition|materialize|preprocess|manifest|validate>` 的实现。

## 构建与安装

```bash
# 1. 安装 PyTorch CUDA 12.4 构建
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 安装基础包
pip install .

# 3. 按需安装可选依赖
pip install ".[data-process]"   # 真实栅格训练/导出所需
pip install ".[downstream]"    # 下游评测
pip install ".[dev]"           # 开发、格式化与测试
pip install ".[npu]"           # 昇腾 NPU
```

## 常用命令

唯一入口是 `xuannv`：

```bash
xuannv --help
xuannv data --help
xuannv train --help
xuannv export --help
xuannv downstream --help
```

### 训练

```bash
# 单卡真实训练
xuannv train \
  --config configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt \
  --device cuda:0

# 单机 8 卡 DDP
scripts/cuda/launch.sh configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt

# 三节点 24 卡 DDP（每节点各跑一次，只有 NODE_RANK 不同；见 docs/multi-node/）
NODE_RANK=0 scripts/cuda/launch_24card.sh configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt

# CPU smoke（仅验证前向/反向/保存，不读真实数据）
xuannv train \
  --config configs/production/haidian_p10c_v1.yaml \
  --output /tmp/synthetic.pt \
  --synthetic --steps 1 --epochs 1 --batch-size 1 --spatial-size 16 \
  --device cpu --no-amp
```

### 导出

```bash
# 新格式 checkpoint
xuannv export \
  --config configs/production/haidian_p10c_v1.yaml \
  --checkpoint /path/checkpoint.pt \
  --region haidian \
  --output-root /path/embeddings \
  --device cuda:0

# 旧海淀 checkpoint 必须提供兼容 profile
xuannv export \
  --config configs/production/haidian_p10c_v1.yaml \
  --checkpoint /path/haidian_embedding_v1_p10c_epoch800.pt \
  --compatibility-profile haidian_p10c_v1 \
  --region haidian \
  --output-root /path/embeddings \
  --device cuda:0
```

### 下游评测

```bash
# 训练下游头
xuannv downstream train \
  --dataset task.npz --folds folds.json --fold 0 \
  --head conv3x3 --shot 5 --output head.pt

# 评测（阈值只用 validation 选择，test 只做一次固定评测）
xuannv downstream evaluate \
  --dataset task.npz --folds folds.json --fold 0 \
  --checkpoint head.pt --output metrics.json
```

### 数据处理

```bash
xuannv data grid         # 构建 1,280 m 父网格
xuannv data registry     # 生成采样 registry
xuannv data partition    # 生成确定性十等分
xuannv data materialize  # 从冻结 catalog 物化多源栅格
xuannv data preprocess   # 对齐并切 patch
xuannv data manifest     # 生成 manifest v1 与 sidecar
xuannv data validate     # 审计 manifest 或父网格包
```

## 配置与 Manifest 合同

### 配置 schema v1

运行配置必须自包含，**不得使用 `_base_`**。顶层字段固定为：

```yaml
schema_version: "1"
paths:
  data_root: /data/xuannv_embedding
  output_root: /data/xuannv_embedding/outputs
  artifact_root: /data/xuannv_embedding/modelscope_upload
experiment:
  name: ...
  seed: 42
model:
  embed_dim: 64
  input_sources:
    s2: {channels: 12, role: temporal}
    highres_optical: {channels: 3, role: highres}
  target_heads:
    s2_recon: {source: s2, loss_type: continuous, channels: 12, weight: 0.8}
training:
  epochs: 800
  lr: 2.0e-6
  ...
data:
  months: [2025-12, 2026-01, ...]
  datasets:
    - region: haidian
      manifest_path: ...
      statistics_dir: ...
      patch_grid_path: ...
      source_map:
        physical_s2_name: s2
      supervised_label_roots:
        osm_building: ...
      sampling_weight: 1.0
```

- `model.input_sources` 中的 `role` 只能是 `temporal` 或 `highres`。
- `data.datasets[].source_map` 把物理 source 名映射到规范 source；**核心代码不检查城市名**。
- `data.months` 必须与 `model.num_months`、`model.ref_year`、`model.ref_month` 一致。

### Manifest v1

- 支持 `.json` 列表和全国规模 `.jsonl`。
- 每条记录必填 `patch_id`、`region`、`sources`；路径必须是数据根下的 POSIX 相对路径，禁止绝对路径、URL 和 `..` 逃逸。
- `(region, patch_id)` 在 manifest 内必须唯一。
- 同名 `<manifest>.meta.json` sidecar 保存 schema、月份、记录数、生成器版本和内容 SHA-256；读取时核验摘要与记录数。
- `null` 或空列表表示真实缺模态，dataset 会生成 `availability=0` 并将重建监督 mask 置零。

## 代码风格

- **格式化**：`black`，行宽 100，目标 Python 3.11。
- **Lint**：`ruff`，行宽 100，启用 `E`、`F`、`I`、`W`。
- **约束**：
  - 不得引入城市名驱动的核心分支。
  - 不得引入 `einops`。
  - 不得将数据、权重、日志或未核实指标提交到 Git。
  - 配置必须自包含，不能用 `_base_`。
  - 生产 Python 逻辑必须位于包内，`scripts/` 只放启动/门禁包装。

## 测试

测试使用 `pytest`，配置在 `pyproject.toml`：

```toml
[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

### 测试文件映射

测试基本与包结构一一对应：

- `test_config.py`：配置 schema、校验、交叉合同
- `test_model.py`：AEFModel 前向/反向、vMF、多分辨率、高分融合
- `test_training_runtime.py`：`TrainingSystem`、DDP 模拟、保存/恢复
- `test_p10c_training.py`：端到端 smoke 训练
- `test_checkpoint_compatibility.py`：旧海淀 431 键兼容加载
- `test_data_*.py`：父网格、registry、partition、materialize、preprocess、manifest、STAC、OSM 等
- `test_downstream.py`：下游头与评测协议
- `test_release_validation.py`、`test_repository_policy.py`：发布门禁与仓库策略

### 本地运行

```bash
python -m black --check src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
python scripts/release/check_repository.py
```

部分测试需要 `data-process` 和 `downstream` extra；基础包安装后即可运行 `test_cli_bootstrap.py` 等核心测试。

## CI / 发布

GitHub Actions 工作流位于 `.github/workflows/ci.yml`：

1. **quality**：安装 `dev` extra，运行 `black --check`、`ruff check`、`scripts/release/check_repository.py`。
2. **test-and-package**：
   - 安装 `data-process,downstream,dev`。
   - 运行 `pytest -q`。
   - 运行 `python -m build` 构建 wheel。
   - 在干净虚拟环境中安装 wheel 并验证 `xuannv --help` 及各子命令 help。
   - 运行 synthetic CPU smoke 训练。
   - 上传 `dist/` artifact。
3. **secret-scan**：`gitleaks-action` 扫描全历史。

### 发布清单

- 发布前必须执行 `docs/release/release-checklist.md` 中的全部门禁，包括单卡/多卡 DDP、旧 checkpoint 兼容、全国父网格审计、十等分核验等。清单按加速器平台分节：CUDA 与 NPU 各自独立验收，一个平台的通过记录不能替代另一个平台。
- 失败时不得打标签；修复必须单独提交并重新运行受影响门禁。
- 大制品发布在 ModelScope `WeijieWu/xuannv_haidian_embdding`，登记路径与摘要见 `docs/production/artifacts.json`。

## 安全与合规

- **Git SHA 来源证明**：从 wheel（不含 `.git`）训练时，必须设置环境变量 `XUANNV_GIT_SHA` 为合法 commit SHA；无法证明来源时训练会被拒绝，避免产出无来源 checkpoint。
- **Checkpoint 身份校验**：新 checkpoint 包含配置 SHA、Git SHA、source schema、区域清单；恢复/导出时逐项核对，不一致则拒绝加载。
- **旧权重保护**：原海淀 checkpoint 不转换、不覆盖；兼容 profile 先核验登记 SHA，再一对一映射 431 个旧键，并要求全部键恰好消费。
- **私密报告**：安全问题通过 GitHub Security → Report a vulnerability 私密报告；不要在公开 issue、讨论、日志或测试中粘贴令牌、私有下载链接、凭据、个人数据或未公开影像路径。
- **仓库策略**：`scripts/release/check_repository.py` 会运行 `xuannv_embedding.utils.repository_policy.validate_repository()`，检查不应进入 Git 的大文件/路径/模式。

## 关键文档

- `docs/architecture.md`：架构与模块边界
- `docs/configuration-and-manifest.md`：配置和 manifest 合同
- `docs/data-processing.md`：全国数据处理流程
- `docs/training-export-downstream.md`：训练、导出与下游评测
- `docs/multi-node/`：多节点 24 卡分层验收、故障速查与 ACP 迁移
- `docs/production/haidian-model-card.md`：海淀模型卡
- `docs/production/haidian-evidence.md`：海淀评测证据
- `docs/production/harbin-transfer.md`：哈尔滨迁移证据与限制
- `MIGRATION.md`：旧仓迁移与来源映射
- `CONTRIBUTING.md`：贡献指南
- `SECURITY.md`：安全政策

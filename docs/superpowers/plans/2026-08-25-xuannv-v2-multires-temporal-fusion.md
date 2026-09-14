# 玄女 V2 本地全国数据与多分辨率模型升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 完全复用本地 2020—2021 全国 1% S2/S1/Landsat 数据，完成严格数据预检、
多产品独立时间线、多分辨率高分融合、约 100M 参数生产模型、mini/smoke/NPU 验证及
分片导出。

**Architecture:** 本地月度 ZIP 是不可变输入，通过成员索引直接读取，缺景写显式掩码，
禁止自动下载。每个产品在自身分辨率编码，按显式 `[start,end)` 查询时间上下文；高分影像
先在原生网格学习特征，再经结构/外观双时间记忆融合到 10m 网格，最后只由 vMF 瓶颈执行
空间 embedding 的 L2 归一化。

**Tech Stack:** Python 3.11、PyTorch、torch-npu、Rasterio、PyArrow、Zarr、pytest。

## 实施状态（2026-08-25）

Stage 01—06 已完成并分别推送 annotated tag。最终门禁为 `309 passed`，Black、Ruff、
仓库内容检查和发行包构建全部通过。真实 8×Ascend 910B smoke 使用两个独立 `torchrun`
进程完成 100 optimizer step，在第 50 step 跨进程恢复；8 个 rank 同步，恢复首批 loss
差值为 0，峰值显存约 3.35GB，吞吐约 21.08 sample/s。

本轮对数据事实作了两项必要修正：

- 本地可验证高分数据只有海淀 PlanetScope 3m。0.5m/2m/5m 由合同和合成单元测试覆盖，
  不宣称完成了不存在的真实产品验证。PlanetScope 日期为 2025—2026，不能因果接入
  2020—2021 全国时序，因此使用独立 `highres_real.parquet` 和真实高分 mini 验证。
- 现有全国 dense 波段统计是显式的 smoke 抽样统计，不冒充生产全量统计。生产数据集会
  拒绝不完整统计；启动全国 62,000 样本长训前，必须运行无
  `--max-observations` 限制的全量统计任务。高分 PlanetScope 训练划分统计已完整生成。

由于 ZIP 直接读取的数据等待比例超过 10%，已按计划生成并校验本地 Zarr cache 后重跑；
最终数据等待比例约 19.9%，因此后续全国长训仍需继续做 I/O profiling，但不影响本轮
正确性 smoke 的通过结论。全程没有下载像元。

最终复审又补齐三项 provenance/causal 门禁：高分 TIFF、UDM2 和监督标签都记录并在
训练前复验 size/SHA256；高分候选按每个输出区间先做时间过滤再取双记忆选择结果的并集；
导出使用模型实际 observation selection，逐 interval 写入贡献观测，不再把未来候选景
列入较早 causal 输出的 lineage。

## Global Constraints

- 工作分支固定为 `wwj`，每阶段测试、原子提交、push、annotated tag；禁止 force-push。
- 禁止 `einops`；V2 配置 `schema_version: "2"`，V2 runtime/checkpoint 不接受 V1。
- 卫星像元只读自 `/data2/china_xuannv_embedding/data`，缺测不触发网络下载。
- 数据、影像、权重、checkpoint、embedding、日志和大型报告不得提交 Git。
- 正式输出固定为 10m、64 维；生产模型参数量门禁为 90M—120M。

---

## 1. 数据源合同

### 全国本地输入

```text
/data2/china_xuannv_embedding/data/
├── pc-s2/{2020,2021}/{01..12}/pc-s2_YYYY_MM.zip
├── pc-s1/{2020,2021}/{01..12}/pc-s1_YYYY_MM.zip
├── pc-ls/{2020,2021}/{01..12}/pc-ls_YYYY_MM.zip
├── aef-labels/aef-labels.zip
└── static/{esa_worldcover_2020,esa_worldcover_2021,osm_china,clcd_china,
            copernicus_dem_glo30}/
```

- S2：10 通道 `[B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12]`，存储 10m；
  其中 B05/B06/B07/B8A/B11/B12 原生 20m。B09 是 945nm、60m 水汽校正波段，
  本地不存在，因此不补、不虚构。
- S1：2 通道 `[vh,vv]`，存储 10m，原生约 20m，已经重采样。
- Landsat：6 通道 `[red,green,blue,nir08,swir16,swir22]`，43×43、约 30m。
- TIFF 不含精确采集时间、QA、scale/offset 和场景 ID：记录 `acquired_at=null`、
  `time_precision=month`、`available_at=next_month_start`，不伪造信息。

网格来自：

```text
/data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805
/data/xuannv_embedding/outputs/
china_full_grid_tenfold_delivery_v7_精简Shape_对齐修复_20260825/
原始1%采样Patch_62000/原始1%采样_完整1280米Patch.shp
```

海淀 PlanetScope 正式高分源为
`/data/xuannv_embedding/raw/haidian/highres_optical/_unzipped`。哈尔滨现有 0.5m patch
缺少原始影像和产品来源，固定为 `legacy_unverified`，只做兼容测试。

### V2 输出根目录

```text
/data2/xuannv_embedding/v2/
├── locks/local_archive_sha256.jsonl
├── registry/{candidate_62000,split_80_10_10,mini_16,smoke_620_plus_32}.parquet
├── registry/local_archive_inventory.parquet
├── observations/index/{local_zip_members,availability}.parquet
├── observations/{dense_2020_2021,highres}/
├── labels/{worldcover,clcd,osm,aef}/
├── statistics/{product_id}.json
├── reports/preflight/{run_id}.json
├── runs/{run_id}/checkpoints/
└── products/{version}/interval={interval}/utm={epsg}/part-{shard}.zarr
```

网络策略固定为 `allow_remote_metadata=false`、`allow_remote_pixels=false`，本地缺测写 mask。

## 2. 公共接口与数据处理

新增不可变 `ProductSpec` 和 `ObservationRef`。产品合同包含波段、角色、原生/存储 GSD、
dtype、时间精度、是否重采样及 QA 可用性。观测引用包含 patch/product、观测区间、
`acquired_at`、`available_at`、ZIP 路径、成员名、存在性和质量状态。

V2 batch：

```text
source_frames[product]       [B,N,C,H,W]
source_pixel_masks[product]  [B,N,1,H,W]
source_observation_masks     [B,N]
source_time_bounds           [B,N,2]
source_available_at          [B,N]
output_intervals             [B,M,2]
highres_frames[product]      [B,N,C,H_native,W_native]
highres_masks[product]       [B,N,1,H_native,W_native]
highres_acquired_at          [B,N]
highres_geotransforms        [B,N,6]
```

- 扫描 72 个 ZIP，记录大小、成员数、成员哈希和 SHA256；逐源逐月与 62,000 网格比对。
- ZIP 中多余 ID 阻断；缺失 ID 写 `not_present_in_local_archive`，不补零、不复制、不下载。
- 仅按有限值、全零、全常量、CRS、transform 和 footprint 检查像元，不虚构 QA。
- 只用训练 split 统计各产品波段；S2/Landsat 保持 `stored_dn`。
- 以 `macro_id` 分组确定性划分 49,600/6,200/6,200。
- mini 固定 16 个：UTM 43N—53N 各一份三源完整样本，加 S2/S1/Landsat 缺失、
  UTM 接缝和高分覆盖各一份。
- smoke 固定 496 train、62 val、62 test 加 32 个哨兵，共 652 个。
- smoke 直接读 ZIP；只有 NPU 数据等待超过 10% 才离线重打包 Zarr。

## 3. 模型升级

### 稠密产品

- S2 10→64、S1 2→64，保持 128×128；Landsat 先在 43×43 卷积，再把特征对齐
  到 128×128，原始像元不预先放大。
- 每产品使用独立观测时间、product/GSD/time-precision embedding。
- 支持 `within_period`、默认 365 天 `causal_window` 和 `centered_window`。
- 因果模式仅允许 `available_at <= output_interval.end`；产品内时间注意力后再跨产品门控。

### 高分产品

- 原生网格先执行 `3×3 Conv + GroupNorm + GELU`，然后抗混叠 stride-2：0.5m 三次
  到约 4m、2m 一次到约 4m、3m/5m 不预缩放；最后按仿射变换聚合到 10m。
- 不使用固定 5×5 子格；每个 `product_id` 独立 adapter，完整保留景维和时间。
- 结构记忆最多 8 景/730 天，外观记忆最多 4 景/90 天；超限按质量、时间和 ID
  确定性选择。causal 禁止未来景，centered 才允许双侧观测。

### 主干和损失

```yaml
embedding_dim: 64
stem_dim: 64
spatial_dim: 768
temporal_dim: 384
precision_dim: 192
num_blocks: 8
num_heads: 12
gradient_checkpointing: true
```

保留重建、uniformity、semantic probe 和 vMF；增加训练期高分细节统计头，预测每个
10m 单元的高分特征均值、局部方差和梯度能量。推理丢弃 probe/detail head。

## 4. 执行阶段和验证

### Stage 01 — `wwj-stage-01-contract`

- [x] 建立隔离 `wwj` 工作区并提交本计划。
- [x] 以失败测试驱动 V2 配置、产品、时间、网络和 profile 合同。
- [x] 配置门禁通过后 commit、push、创建 Draft PR、打 annotated tag。

### Stage 02 — `wwj-stage-02-data`

- [x] 实现 ZIP inventory、成员索引、preflight、split/mini/smoke registry。
- [x] 运行 16 patch、2020-01/2021-01 的真实 data mini，网络请求为零。

### Stage 03 — `wwj-stage-03-model`

- [x] 实现独立时间线、显式输出区间、稠密 adapter 和高分双记忆。
- [x] synthetic mini 使用不等长 S2/S1/Landsat、多景 2m/5m 和两个输出区间；
  输出 `[2,2,64,32,32]`，causal 无未来泄漏，高分原生卷积梯度非零。
- [x] 生产 profile 参数量位于 90M—120M。

### Stage 04 — `wwj-stage-04-training`

- [x] real mini：16 patch、两个月、batch 2、训练 2 step、保存恢复后再训练 1 step。
- [x] 单 batch 过拟合 20 step 总损失至少下降 5%；checkpoint 记录数据/config/Git SHA。

### Stage 05 — `wwj-stage-05-export`

- [x] 652 个真实样本、全 24 月、365 天上下文训练 50 step，验证恢复和缺失模态。
- [x] 输出 sharded Zarr 与 `catalog.parquet`，完整记录来源和输出区间。

### Stage 06 — `wwj-stage-06-npu-smoke`

- [x] 8×910B、每卡 microbatch 1、梯度累积 8、AMP/checkpointing，运行 100 step；
  第 50 step 保存恢复，记录吞吐、数据等待、显存和 loss。
- [x] 实际检测并使用 8 张 NPU；保存各 rank RNG/sampler 状态并验证跨进程恢复。

每个阶段执行：

```bash
python -m pytest -q
python -m black --check src tests scripts
python -m ruff check src tests scripts
python scripts/release/check_repository.py
python -m build
```

全国 62,000 样本长时间训练不自动启动；本轮止于真实 smoke、8 卡合同和可回退制品。

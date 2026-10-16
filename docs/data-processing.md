# 全国数据处理

生产能力来自旧仓已有测试证据的白名单，现全部位于 `xuannv_embedding.data_process`：

- 1,280 m UTM owner-zone 父网格、macrocell inventory 和成员关系审计；
- 确定性分层采样 registry 与唯一性检查；
- Planetary Computer / CDSE STAC catalog 冻结；
- Sentinel-2、Sentinel-1、Landsat 的场景优先或 patch 优先物化；
- 多源 NetCDF/GeoTIFF 对齐、分类 QA 最近邻重采样和 Welford 统计量；
- OSM 类别审计、全国十等分、重排和 QGIS 精确边界导出；
- 哈希、重复覆盖、缺测、原子写入和 partial 恢复检查。

## 命令

```text
xuannv data grid                  构建 owner-zone 父网格
xuannv data registry              从质量 atlas 生成采样 registry
xuannv data partition             生成确定性十等分
xuannv data materialize           从冻结 catalog 物化多源栅格
xuannv data preprocess            对齐并切 patch
xuannv data manifest              生成 manifest v1 与 sidecar
xuannv data validate              审计 manifest 或父网格包
xuannv data catalog               从本地高分归档建立观测目录与低分配对缓存
xuannv data finalize-catalog      抽样验收并发布已物化目录
xuannv data prepare-observations  校验观测、生成掩膜和年度索引
xuannv data pair-observations     配对同期高低分观测并验证读取
xuannv data quality-annual        执行年度观测初筛并重算统计量
xuannv data cloud-annual          生成可恢复的 S2 云影筛选候选
xuannv data cloud-highres         按波长识别 5m 光学波段并生成云影候选
xuannv data pan-annual            审计 2m 全色观测并接入年度候选清单
xuannv data release-quality       完成云影、配准、重复和放行门禁
xuannv data check-annual          抽样读取年度原生网格数据
xuannv data targets-audit         核验独立目标包、年度关联、覆盖及候选训练统计量
xuannv data targets-prepare       冻结空间实验划分并生成原生DEM的40m监督缓存
xuannv data p0                    从观测 pilot 准备真实 P0 训练数据
```

每个命令使用 `--help` 查看精确参数。`grid` 和 `materialize` 拒绝覆盖完成目录；批次先写入临时
位置，再原子发布。物化失败保留可识别的 partial 状态和 fingerprint，重跑必须匹配同一输入。
fingerprint 覆盖 patch 身份与两套坐标边界、月份、物化策略、像元/尺寸合同、source/QA schema，
以及每个冻结 STAC catalog JSONL 的大小和 SHA-256；任一项变化都拒绝复用旧 partial。

`targets-audit` 对已发布的独立监督包执行只读验收；输出必须为数据根以外的新目录：

```bash
xuannv data targets-audit \
  --target-root /path/to/targets \
  --annual-root /path/to/annual_manifests \
  --data-root /path/to/image_root \
  --output /path/to/new_audit_directory --workers 2
```

它重算所有归档的 SHA-256，核对数组合同及静态行号顺序，全量读取静态目标和掩膜，
按当前年度划分、高分支持及 macro 地理块统计覆盖，抽检影像头几何，并检查跨年位置与地理块交叉。
OSM 只检查全部归档哈希和数组元数据，暂不全量扫描其语义状态。
产物包含 `target_contract.json`、`coverage_by_split.json`、`coverage_by_macro.parquet`、
`geometry_audit.json`、`spatial_split_audit.json` 及 `train_statistics.json`。
静态训练统计量按唯一 train 位置计数，排除所有 validation/test 位置；在最终评测排除名单冻结前
仅为候选值。退出码 0 表示本命令的数据检查通过；仍须查看 `training_ready` 和 `open_gates`，
它不认证原生标签尺度、物理标定、地物配准或训练效果。失败保留已生成证据，重新执行需使用新目录。

`targets-prepare` 在上述审计后生成独立实验数据，输出同样必须为新目录：

```bash
xuannv data targets-prepare \
  --audit-root /path/to/audit \
  --source-integrity /path/to/native_dem_integrity.json \
  --initial-config /path/to/original_pretraining_config.yaml \
  --seen-patches /path/to/previously_seen_positions.json \
  --output /path/to/new_prepared_directory --workers 4
```

`--source-integrity` 是原生 DEM 分卷按顺序记录的本地完整性凭据，包含 `files` 中每个文件的
`path/bytes/mtime_ns/sha256`，以及完整 ZIP 成员 CRC 检查的 `all_zip_member_crc_passed: true`。
本命令核对凭据与当前文件身份，拒绝源文件在处理期间变化；它不将本地摘要当作供应商签名。
`--seen-patches` 可以重复，须包含所有曾查看的候选及评测位置；原预训练 manifest 则从
`--initial-config` 读取并与审计时的完整摘要绑定。

输出冻结 development/validation/confirmation 三个角色及嵌套的 screen/expanded 训练位置。
选样只依据当前年度划分、坐标、macro/component、高分可用性与确定性哈希；所有新评测位置
均避开原预训练和既往查看的地理块，按完整父格多边形核验缓冲，包括相邻 UTM 分区。
DEM 从原生 GLO-30 DSM 直接聚合为每父格 `32×32` 的 40m 网格，保留有效 mask、米制值、
原生文件指纹、逐行摘要和处理记录。40m 是监督网格，不能解释为新的测量精度。
WorldCover 使用已验收的 2020/2021 年度 10m 数组。三套 `train_statistics.<scope>.json`
分别只用该训练清单的有效目标，静态 DSM 每位置计一次。`preparation.json` 只放行工程联调，
物理标定及独立定位/真值要求仍需单独完成。

`xuannv experiment prepare-multisource` 在这套冻结位置上追加本地监督缓存，复用原生 DSM，
按 40m/640m/5m 的监督网格聚合坡度、CLCD、夜光及 OSM 结构，并核验实际使用的七个新增
年度归档。它不重新选样或读取确认位置；输出与原监督目录分开保存，细节见
[训练、导出与下游评测](training-export-downstream.md)。

## 全国不变量

- `parent_key = EPSG:grid_col:grid_row`，同一父格只能有一个 owner UTM zone；
- sampled 与 unsampled 是完整且互斥的父网格分区；
- registry `patch_id` 唯一，macrocell 声明计数必须与 atlas 实际计数一致；
- 十片成员总数等于父网格总数，每个 shape 只属于一片；
- GeoParquet 几何、存储 bounds、canonical footprint 和 identity hash 必须一致；
- 缺测以 mask 和质量记录表达，不生成 synthetic 遥感观测。

数据、catalog cache、Zarr、Shape、统计量与审计输出都放在仓库之外。Git 只保存代码、配置、
schema 和小型证据摘要。

当前正式年度数据入口是
`/data/heyuhang/processed/national_annual5m_quality_v6`。其合同和保守训练质检已经通过：六份
manifest 共抽读600个真实样本，父格、文件、原始产品和高分像元跨集合冲突均为零，训练统计量从
最终train元数据重算。Landsat因缺少云影QA暂不进入该版本；无法可靠确认相对配准的高分观测已隔离。
这表示数据可用于模型开发和8卡训练联调，不表示云影准确率或绝对定位精度已经独立认证。

正式方案、质量限制与训练顺序见[年度5m技术报告](xuannv_annual5m_report.tex)，机器证据索引见
[report-evidence.json](report-evidence.json)。

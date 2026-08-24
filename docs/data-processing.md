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
xuannv data grid         构建 owner-zone 父网格
xuannv data registry     从质量 atlas 生成采样 registry
xuannv data partition    生成确定性十等分
xuannv data materialize  从冻结 catalog 物化多源栅格
xuannv data preprocess   对齐并切 patch
xuannv data manifest     生成 manifest v1 与 sidecar
xuannv data validate     审计 manifest 或父网格包
```

每个命令使用 `--help` 查看精确参数。`grid` 和 `materialize` 拒绝覆盖完成目录；批次先写入临时
位置，再原子发布。物化失败保留可识别的 partial 状态和 fingerprint，重跑必须匹配同一输入。
fingerprint 覆盖 patch 身份与两套坐标边界、月份、物化策略、像元/尺寸合同、source/QA schema，
以及每个冻结 STAC catalog JSONL 的大小和 SHA-256；任一项变化都拒绝复用旧 partial。

## 全国不变量

- `parent_key = EPSG:grid_col:grid_row`，同一父格只能有一个 owner UTM zone；
- sampled 与 unsampled 是完整且互斥的父网格分区；
- registry `patch_id` 唯一，macrocell 声明计数必须与 atlas 实际计数一致；
- 十片成员总数等于父网格总数，每个 shape 只属于一片；
- GeoParquet 几何、存储 bounds、canonical footprint 和 identity hash 必须一致；
- 缺测以 mask 和质量记录表达，不生成 synthetic 遥感观测。

数据、catalog cache、Zarr、Shape、统计量与审计输出都放在仓库之外。Git 只保存代码、配置、
schema 和小型证据摘要。

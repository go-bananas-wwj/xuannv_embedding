# 配置与 manifest 合同

## 配置 schema v1

运行配置必须包含 `schema_version / paths / experiment / model / training / data`，必须自包含，
不得使用 `_base_`。解析器拒绝未知字段和 YAML 重复键，也不把字符串强制转换成布尔值或数值；
学习率、权重、概率、温度和区域采样权重必须满足各自的有限值及正/非负范围。

P0 可通过 `data.target_months` 指定 `data.months` 的有序无重复子集；完整时间轴与
`num_months/ref_year/ref_month` 保持一致，loader 只读取目标月，模型按明确的月度索引输出。
默认空列表保持原有全部月份行为。

`data.highres_mode` 默认 `legacy`，保留旧高分单帧路径；设为 `observations` 时，loader
保留高分帧并输出 `highres_months`，模型进行按月、无序集合残差融合，空支持严格回退到同权重低分输出。
`data.highres_max_observations` 默认为 4，是每个 source、每个父格的上限，超过时必须在 manifest
生成阶段明确选帧。每条高分路径的月份表示分配的目标月，实际采集日和时间差另存观测目录；
不要直接把采集日期文件名当作跨月候选的目标月。高分帧先在存储网格编码，再对齐特征。

`model.input_sources` 的每个规范槽位显式声明：

```yaml
input_sources:
  s2: {channels: 12, role: temporal}
  highres_optical: {channels: 3, role: highres}
  highres_sar: {channels: 1, role: highres}
```

`training.latent_prediction_weight` 默认是 `0`。两阶段训练时先用该字段为 0 的配置产出教师，
再将权重设为正值并通过 `xuannv train --teacher-checkpoint ... --teacher-config ...` 启动学生。教师加载仍严格核对
`source_schema` 与区域清单；只有训练目标权重变化导致的配置摘要差异被允许。

年度 v6 manifest 通过 `provenance.observations` 保存变长原生高分观测：每条记录带有目标月份、
源文件、通道数、原生高宽、有效掩膜、质量状态和处理版本。PAN 保留 `1×640×640`，吉林一号多光谱
保留 `6×256×256`；loader 不在离线阶段把二者强行缩放成同一数组。缺失高分分支用空列表和
`availability=0` 表达，模型必须显式处理缺失，不能把零占位解释为零值观测。

每个 `data.datasets` 项声明 `region`、`manifest_path`、`statistics_dir`、`patch_grid_path`、
`source_map`、`supervised_label_roots` 和 `sampling_weight`。例如物理源
`highres_optical_harbin` 可以映射到规范槽位 `highres_optical`；物理产品名称不会进入模型
分支判断。

完整配置：

- [海淀 P10C V1](../configs/production/haidian_p10c_v1.yaml)
- [哈尔滨 P10C](../configs/production/harbin_p10c.yaml)
- [海淀 + 哈尔滨](../configs/production/mixed_haidian_harbin_p10c.yaml)
- [China P10C pilot](../configs/examples/china_p10c_pilot.yaml)
- [全国年度 5m v1](../configs/production/annual5m_v1.yaml)

年度模型可通过 `training.independent_targets` 接入监督；默认 `null` 保持旧训练行为。
严格字段为 `schema: annual-dem40m-worldcover-v1`、`target_root`、`prepared_root`、
`preparation_sha256`、`scope`（`screen` 或 `expanded`）、`weights`、`dem_mean`、`dem_std`，
以及可选的 `head_lr_multiplier`（默认 10）、`ramp_batches`（默认 512）、
`engineering_only`（默认 true）。`weights` 只接受 `dem_elevation`、`worldcover` 的正权重，
DEM 标准化必须逐值匹配该 scope 的训练统计量。配置加载本身不读取监督数据，导出也不依赖目标文件。

监督读取器核对准备凭据、原目标 manifest/index、冻结划分、训练统计量及原生 DEM 凭据的摘要。
WorldCover 归档在训练初始化时完整核验，再由各 worker 独立打开；DEM 每次读取核验行摘要。
WorldCover 原码 `10,20,30,40,50,60,70,80,90,95,100` 映射到 `0..10`，缺测为 `-100` 且 mask=false。
2020/2021 只读取匹配年份，评测角色默认封存。监督值只交给损失，不能进入编码器或推理输入。
两个新头仅从公开 64 维嵌入按面积池化到 DSM 40m / WorldCover 10m 后读出，不增加归一化瓶颈。

`schema: annual-local-multisource-v2` 扩展支持 `dem_slope`、`clcd`、`nightlights`、
`osm_semantic` 和 `osm_structure`，其余配置字段及旧 schema 保持原定义。v2 必须指向经过准备的
训练缓存；当前缓存覆盖冻结的 screen 位置，包含两个年份，评测目标继续封存。
坡度使用 40m DSM 的中心差分，单位为度，损失除以 90；CLCD 为 40m 上的 9 类面积比例；
夜光在 640m 上按有效面积平均，保留已发布数值；OSM 的建筑、道路、水面、水道在 10m 上监督，
建筑、道路、水道的结构在 5m 上监督。已知支持不足一半的 CLCD/夜光粗单元忽略；
其余按已知覆盖率及置信度加权。OSM 未知和仅有当前来源位的像元不自动成为历史标签，
历史正例优先于规则负例。二元损失按跨 rank 的每通道正负软目标量分别归一化，再平均有效通道。

## manifest v1

支持 `.json` 列表和全国规模 `.jsonl`。每条记录必填：

```json
{"patch_id":"p1","region":"region-a","sources":{"physical_s2":["region-a/s2.tif"],"physical_sar":null}}
```

可选字段是 `source_patch_id`、`grid`、`geometry`、`quality`、`provenance`。路径必须是数据根下
的 POSIX 相对路径，禁止绝对路径、URL 和 `..` 逃逸；`(region, patch_id)` 在整个 manifest
内必须唯一，写入、读取和 legacy 适配都会拒绝重复身份。

同名 `<manifest>.meta.json` sidecar 保存 schema、月份、记录数、生成器版本和内容 SHA-256。
读取时同时核验摘要与记录数。旧海淀/哈尔滨平铺 JSON 仅通过只读 adapter 读取；adapter 不会
回写旧文件或伪造 sidecar。

转换和验证：

```bash
xuannv data manifest --legacy old.json --region region-a \
  --output manifest.jsonl --months 2025-12 2026-01
xuannv data validate --manifest manifest.jsonl
```

## 缺模态

`null` 或空列表表示真实缺失。dataset 会生成 availability=0，并将对应重建监督 mask 置零。
零 tensor 只用于维度占位，不能被解释为零值观测。配置中的通道、月份或重复规范映射冲突会在
读数据前失败。只要某区域 manifest 中实际提供了某个连续 source，该区域就必须提供通道数匹配、
mean/std 全部有限且 std 严格为正的 `<canonical_source>_stats.json`；只有全区域真实缺失的 source
可以不提供统计量。

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

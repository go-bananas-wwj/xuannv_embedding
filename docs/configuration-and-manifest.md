# 配置与 manifest 合同

## 配置 schema v1

运行配置必须包含 `schema_version / paths / experiment / model / training / data`，必须自包含，
不得使用 `_base_`。解析器拒绝未知字段和 YAML 重复键，也不把字符串强制转换成布尔值或数值；
学习率、权重、概率、温度和区域采样权重必须满足各自的有限值及正/非负范围。

`model.input_sources` 的每个规范槽位显式声明：

```yaml
input_sources:
  s2: {channels: 12, role: temporal}
  highres_optical: {channels: 3, role: highres}
  highres_sar: {channels: 1, role: highres}
```

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

# 贡献指南

1. 从 `main` 创建短期分支；一个提交聚焦一个可验证改动。
2. 功能和修复先写失败测试，再实现；配置必须自包含且不能使用 `_base_`。
3. 不得引入城市名驱动的核心分支、`einops`、数据、权重、日志或未核实指标。
4. 运行：

```bash
python -m black --check src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
python scripts/release/check_repository.py
```

涉及 NPU、checkpoint 或全国网格的变更还必须执行对应的
[发布清单](docs/release/release-checklist.md)。代码采用 [Apache-2.0](LICENSE)，提交即表示贡献内容可
按该许可证发布。

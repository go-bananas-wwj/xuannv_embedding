# 年度 5m 技术报告维护

全国年度方案只维护一份正文：[xuannv_annual5m_report.tex](xuannv_annual5m_report.tex)。运行
`make -C docs report` 生成本地 PDF；TeX 是可编辑主文件，PDF 和编译缓存不进入 Git。

数据结论必须来自 `/data/heyuhang/processed/national_annual5m_quality_v6` 的机器产物，摘要和文件
SHA-256 登记在 [report-evidence.json](report-evidence.json)。修改数量、状态或科学结论时，同时更新
报告版本、核验日期、证据索引和参考文献。必须区分已实现、拟实施与待验证；数据合同通过不等于
模型已训练，也不等于云影或定位精度得到独立认证。

当前正式训练统一按单节点 8 卡设计。历史调研、阶段性运行记录和重复方案不在仓库维护；需要追溯时
使用服务器外部归档及 v6 数据目录中的日志、摘要和数据库。

评审中发现的实现缺陷在修复后直接并入本报告或 [architecture.md](architecture.md)，不再维护并行的临时方案文档。

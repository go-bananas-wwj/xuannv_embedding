# 2026-10-10 区域论文代码整合验证

整合现有区域训练分支、全区读取与随机森林批量预测分支，以及生产仓的实验归档入口。
CLI同时保留`experiment`（训练/评测）和`experiments archive`（历史目录整理）。
最终模型及配对结果见[论文复现入口](../experiments/independent-regions-20261010/README.md)。
原训练快照、原始数据、导出和配方登记均未修改，未重新启动论文实验。

本地Python 3.11验证：

| 检查 | 结果 |
| --- | --- |
| `PYTHONPATH=src python -m pytest -q` | 749 passed；两项既有依赖警告 |
| CLI、归档、全区读取的组合检查 | 12 passed |
| Black（src/tests/scripts） | 212文件通过 |
| Ruff（src/tests/scripts） | 通过 |
| 仓库内容、链接和严格配置门禁 | 276文件、约2.17 MB、6配置通过（此文加入前） |
| sdist与wheel构建 | 通过 |
| 独立虚拟环境wheel安装及CLI help | 通过；复用主机已安装依赖 |
| Gitleaks，既有main到整合分支的77个提交 | 无泄露发现 |

区域核心模型、原网格高分模块、区域训练、静态目标和损失模块与已执行的训练快照
逐字节相同。既有哈尔滨真实2样本NPU门禁记录了有限损失2.111591、538个有限梯度张量，
初始化来自当地父权重且静态监督系数为0.50；该门禁是已执行快照的记录，不称为此次整合
重新运行的NPU门禁。新增读取与CLI代码通过以上CPU回归。

没有创建新发布标签；完整发布、旧权重兼容和多卡NPU复验仍按
[NPU发布清单](npu-checklist.md)对应要求执行，不能以本次整理代替新的硬件发布门禁。
仓库只保存源码、配置模板、小型评分CSV和说明，所有大制品与完整日志均留在仓库外。

## 全新安装的依赖兼容修订

GitHub全新环境解析到PyArrow 26，导入时拒绝当前NumPy 1.26，导致数据测试无法收集。
在独立环境中使用相同组合复现失败后，将数据extra约束为`pyarrow>=14,<26`，保留
当前NumPy/NPU版本合同。PyArrow 25的官方安装文档列出NumPy最低版本1.21.2，
见[Apache Arrow安装说明](https://arrow.apache.org/docs/python/install.html)。
新增NumPy数值列（含NaN）到Parquet往返的回归，验证真实数据接口，未改动模型或评分算法。

修订后独立环境使用PyArrow 25.0.1与NumPy 1.26.4，完整CPU回归750 passed。
新测试包含真实NumPy列转换、NaN保持及Parquet往返。Black检查213文件、Ruff、
严格配置/链接/内容门禁和重新构建sdist/wheel均通过。

GitHub的两核运行器另暴露一条测试夹具写死8个Numba线程的问题。生产报告器正确拒绝
超出线程池容量的请求，因此仅调整夹具，使用最多8个实际可用线程比较串行与并行抽样。
以`NUMBA_NUM_THREADS=2`复现原失败；保持生产报告器、抽样种子、重复次数与评分算法不变。
两线程环境下对应任务与两类报告器的40项回归通过，格式、内容门禁和重新打包通过。

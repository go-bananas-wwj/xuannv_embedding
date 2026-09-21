# Changelog

## 未发布

- 接入年度 `annual5m` 正式路径：原生 5m/PAN 编码、精确面积支持聚合、日期/质量池化、留出遮挡、
  64 维导出和 8 卡真实数据 smoke 门禁。
- 新月度时间编码与 batch 无关，同时为历史 checkpoint 保留历史数值路径；教师 checkpoint 改为严格
  核对配置摘要、source schema 和区域身份。

### CUDA 与多加速器运行时

- 年度模型新增冻结教师潜变量预测路径；教师 checkpoint 仅核对 source schema 与区域，学生
  checkpoint 只保存可训练学生权重。
- 修复连续月份时间编码、缺测高分残差恒等回退、月度有效性池化与跨卡按月均匀性负样本。
- 导出 NPZ 增加 `validity_mask`，并停止用最后一个时间戳伪造补齐观测。

- 自动选择 CUDA 设备与 NCCL 后端；分布式后端按设备类型解析（CUDA nccl、NPU hccl、CPU gloo）。
- CUDA AMP 在支持 bf16 的硬件上使用 bfloat16 且不启用 GradScaler，仅在不支持时回退 fp16。
- 修复 `precision_scale: 1` 时 `upsample_head` 被跳过导致的 DDP 反向失败；6 个未用参数仍保留在
  431 键登记合同内。
- 统一 CUDA 与 NPU 到共用的 `scripts/launch_ddp.sh`，平台入口只设置默认卡数。
- 发布模型门禁改为加速器无关，包装脚本更名为 `scripts/release/run_model_gates.py`。
- DDP 启动器固定训练解释器，避免各节点镜像解析到另一套 torch。
- 修复 fp16/NPU AMP 首次梯度溢出即终止训练：梯度范数改为只观测，溢出交由 GradScaler 跳过，
  并在训练摘要中以 `nonfinite_gradient_steps` 计数。
- 逐 batch 的 `torch.cuda.synchronize` 改为仅在提供 `step_callback` 时执行，恢复
  H2D 与计算的重叠。
- checkpoint 的 `source_code_sha256` 覆盖包内全部随包分发文件，不再漏掉 `export/*.json`
  这类登记 431 键兼容契约的 package-data。
- `xuannv diagnose --output` 等 JSON 报告写入会自动创建父目录，不再因目录缺失失败。

### 文档

- 补齐 `xuannv data` 子命令清单与 `xuannv diagnose` 的用途、前置条件和不保证事项。
- 正式 CUDA 训练统一为单节点 8 卡，删除不再维护的历史多节点路线。
- 全国年度 5m 方案合并为一份可编辑技术报告，并登记 v6 数据证据。
- 发布清单按加速器平台分节；说明空间重采样由登记权重冻结。
- 新增 `AGENTS.md` 项目指南。
- 新增 `docs/training-acceleration.md`，记录年度训练吞吐瓶颈定位、根因与实测加速证据，
  以及保持全局批量不变的超参改动与随机种子对照实验。

### 测试

- 覆盖 CUDA 设备解析与 bf16 AMP 策略。
- 新增 fp16 GradScaler 溢出回归：非有限梯度必须被计数而不是抛错终止训练。
- 新增 `xuannv diagnose` 的真实栅格端到端测试，覆盖月份绑定、排列不变性、缺测回退、
  边界反传与 CLI 报告写入。
- 新增全部 `xuannv data` 子命令与顶层子命令的 help 门禁，延迟导入破坏不再无人捕获。
- 覆盖 `source_code_sha256` 对 package-data 变更的敏感性。

## 1.0.0 - 2026-08-24

- 从旧仓白名单重建区域无关的 P10C 生产包。
- 保留海淀 epoch 800 严格 SHA/431-key 兼容。
- 合并模型、训练、导出、下游和全国数据处理到单一 `xuannv_embedding` 包。
- 增加 manifest v1、区域 source 合同、balanced few-shot 和版本化原子 checkpoint。
- 增加公开仓体积、文件类型、配置、链接、测试、构建和秘密扫描门禁。

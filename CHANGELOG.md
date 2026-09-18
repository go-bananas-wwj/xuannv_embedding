# Changelog

## 未发布

### CUDA 与多加速器运行时

- 自动选择 CUDA 设备与 NCCL 后端；分布式后端按设备类型解析（CUDA nccl、NPU hccl、CPU gloo）。
- CUDA AMP 在支持 bf16 的硬件上使用 bfloat16 且不启用 GradScaler，仅在不支持时回退 fp16。
- 修复 `precision_scale: 1` 时 `upsample_head` 被跳过导致的 DDP 反向失败；6 个未用参数仍保留在
  431 键登记合同内。
- 统一 CUDA 与 NPU 到共用的 `scripts/launch_ddp.sh`，平台入口只设置默认卡数。
- 发布模型门禁改为加速器无关，包装脚本更名为 `scripts/release/run_model_gates.py`。
- DDP 启动器固定训练解释器，避免各节点镜像解析到另一套 torch。

### 文档

- 正式 CUDA 训练统一为单节点 8 卡，删除不再维护的历史多节点路线。
- 全国年度 5m 方案合并为一份可编辑技术报告，并登记 v6 数据证据。
- 发布清单按加速器平台分节；说明空间重采样由登记权重冻结。
- 新增 `AGENTS.md` 项目指南。

### 测试

- 覆盖 CUDA 设备解析与 bf16 AMP 策略。

## 1.0.0 - 2026-08-24

- 从旧仓白名单重建区域无关的 P10C 生产包。
- 保留海淀 epoch 800 严格 SHA/431-key 兼容。
- 合并模型、训练、导出、下游和全国数据处理到单一 `xuannv_embedding` 包。
- 增加 manifest v1、区域 source 合同、balanced few-shot 和版本化原子 checkpoint。
- 增加公开仓体积、文件类型、配置、链接、测试、构建和秘密扫描门禁。

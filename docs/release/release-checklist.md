# 发布清单

以下门禁全部通过后才允许创建发布标签。机器输出保存在仓库外，Git 只记录最终摘要。
每个加速器平台独立验收：一个平台的通过记录不能替代另一个平台。

## 平台无关门禁

- [x] GitHub `quality`、`test-and-package`、`secret-scan` 全绿。
- [x] 全新目录 clone、默认 wheel 安装、CLI help 和 CPU 测试通过。
- [x] 全国父网格只读审计通过；sampled/unsampled、owner zone、哈希和重复覆盖均通过。
- [x] 十片 manifest 只读核验：1–10 齐全、总量等于父网格总量、成员唯一。
- [x] 原海淀 checkpoint SHA 未变化，431 键全部消费。

## Ascend NPU（v1.0.0 已验收）

验收环境 Ascend910B4-1、PyTorch `2.6.0`、torch-npu `2.6.0.post5`；证据见
[验收摘要](validation-summary.md)。

- [x] 加载 `/usr/local/Ascend/cann-9.0.0/set_env.sh`，核验 PyTorch `2.6.0`、torch_npu
  `2.6.0.post5` 和设备型号。
- [x] 单卡 synthetic 128×128 mini-e2e：前向、反向、有限 loss、原子 checkpoint。
- [x] 同一真实海淀 2-patch batch 的旧/新模型 embedding `torch.equal`；同时与由归档旧源码
  `149f33b` 独立生成并按输入/输出 SHA 冻结的 NPU reference 精确一致。
- [x] 哈尔滨真实缺高分 SAR：availability=0、重建 mask=0、输出有限且形状正确。
- [x] 6×NPU 最小 DDP：每 rank 参与、checkpoint 仅 rank 0 原子发布、严格恢复再训练一步。
- [x] 输出 `[B,6,64,128,128]`，无 NaN/Inf，vMF 像素范数在容差内为 1。

## NVIDIA CUDA（待验收）

目标环境 PyTorch `2.6.0` 的 CUDA 12.4 构建；不要混装不同主次版本。CUDA AMP 在支持 bf16 的
硬件上使用 bfloat16 且不启用 GradScaler，因此 loss 数值不会与 NPU fp16 记录逐位一致。

- [ ] 核验 `torch.version.cuda`、`torch.cuda.is_available()`、驱动版本与设备型号。
- [ ] 单卡 synthetic 128×128 mini-e2e：前向、反向、有限 loss、原子 checkpoint。
- [ ] 同一真实海淀 2-patch batch 的旧/新模型 embedding `torch.equal`；并与冻结的独立旧运行时
  reference 比对（跨平台数值差异需在门禁容差内记录，不得静默放宽）。
- [ ] 哈尔滨真实缺高分 SAR：availability=0、重建 mask=0、输出有限且形状正确。
- [ ] 8×CUDA 最小 DDP（nccl）：每 rank 参与、checkpoint 仅 rank 0 原子发布、严格恢复再训练一步。
- [ ] 输出 `[B,6,64,128,128]`，无 NaN/Inf，vMF 像素范数在容差内为 1。

失败时不得打标签。修复必须单独提交、推送并重新运行受影响门禁。

## 模型门禁命令

模型门禁要求仓库外 reference 根目录同时包含已登记的 2-patch 输入和旧运行时输出。
`--device` 省略时优先选择 CUDA，其次 NPU，最后 CPU：

```bash
python scripts/release/run_model_gates.py \
  --haidian-config configs/production/haidian_p10c_v1.yaml \
  --harbin-config configs/production/harbin_p10c.yaml \
  --checkpoint /path/haidian_embedding_v1_p10c_epoch800.pt \
  --legacy-reference-root /path/legacy_reference_haidian_p10c_v1 \
  --output /path/model_gates.json --device cuda:0
```

## DDP 启动

```bash
# 单机 8 卡
scripts/cuda/launch.sh configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt

# 三节点 24 卡：每节点相同 MASTER_ADDR/MASTER_PORT，NODE_RANK 依次为 0/1/2
NNODES=3 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  scripts/cuda/launch.sh configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt
```

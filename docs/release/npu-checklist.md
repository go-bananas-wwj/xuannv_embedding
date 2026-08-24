# NPU 发布清单

以下门禁全部通过后才允许创建 `v1.0.0`。机器输出保存在仓库外，Git 只记录最终摘要。

- [ ] 加载 `/usr/local/Ascend/cann-9.0.0/set_env.sh`，核验 PyTorch `2.6.0`、torch_npu
  `2.6.0.post5` 和设备型号。
- [ ] 单卡 synthetic 128×128 mini-e2e：前向、反向、有限 loss、原子 checkpoint。
- [ ] 原海淀 checkpoint SHA 未变化，431 键全部消费。
- [ ] 同一真实海淀 2-patch batch 的旧/新模型 embedding `torch.equal`；同时与由归档旧源码
  `149f33b` 独立生成并按输入/输出 SHA 冻结的 NPU reference 精确一致。
- [ ] 哈尔滨真实缺高分 SAR：availability=0、重建 mask=0、输出有限且形状正确。
- [ ] 6×NPU 最小 DDP：每 rank 参与、checkpoint 仅 rank 0 原子发布、严格恢复再训练一步。
- [ ] 输出 `[B,6,64,128,128]`，无 NaN/Inf，vMF 像素范数在容差内为 1。
- [ ] 全国父网格只读审计通过；sampled/unsampled、owner zone、哈希和重复覆盖均通过。
- [ ] 十片 manifest 只读核验：1–10 齐全、总量等于父网格总量、成员唯一。
- [ ] GitHub `quality`、`test-and-package`、`secret-scan` 全绿。
- [ ] 全新目录 clone、默认 wheel 安装、CLI help 和 CPU 测试通过。

失败时不得打标签。修复必须单独提交、推送并重新运行受影响门禁。

模型门禁要求仓库外 reference 根目录同时包含已登记的 2-patch 输入和旧运行时输出：

```bash
python scripts/release/run_npu_model_gates.py \
  --haidian-config configs/production/haidian_p10c_v1.yaml \
  --harbin-config configs/production/harbin_p10c.yaml \
  --checkpoint /path/haidian_embedding_v1_p10c_epoch800.pt \
  --legacy-reference-root /path/legacy_reference_haidian_p10c_v1 \
  --output /path/model_gates.json --device npu:0
```

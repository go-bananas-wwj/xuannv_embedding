# v1.0.0 发布验收摘要

验收日期：2026-08-24。机器输出、checkpoint 和数据均保存在仓库外；本页只记录可复核摘要和
SHA-256。模型/NPU 门禁运行于 `eb1796b`；其后的生产提交仅修改全国数据审计及其测试，不改变
P10C 模型、checkpoint 映射、训练目标或 embedding 数值路径。

## 结论

- CPU：全新目录 clone 后，248 项完整测试通过；Black、Ruff、sdist/wheel 构建全部成功。
- 干净安装：在独立虚拟环境安装本地 wheel，依赖检查与全部 CLI help 通过；安装包报告版本
  `1.0.0`，并从 wheel 的来源元数据复核到精确 Git SHA
  `596e5431f8ea90a11041d87e9268c9a2cd479fbc`。
- GitHub：`quality`、`test-and-package`、`secret-scan` 在候选提交 `596e543` 全部通过；CI 包含
  wheel/sdist、干净虚拟环境安装、全部 CLI help 和 CPU synthetic smoke。
- 仓库策略：该全新 clone 含 108 个 tracked 文件、824,494 bytes；无禁止文件和超限文件。
- 安全：新仓完整历史 gitleaks 零命中；GitHub secret scanning 与 push protection 已启用。

## NPU 与兼容性

- 环境：Ascend910B4-1，PyTorch 2.6.0，torch-npu 2.6.0.post5。
- 单卡 128×128 synthetic mini-e2e：1 batch、有限 loss `3.2825193405151367`、原子 checkpoint，
  scheduler `last_epoch=1`。
- 海淀原 checkpoint SHA-256 仍为
  `69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790`；431 个模型键全部消费。
- 真实海淀 2-patch 输出形状 `[2,6,64,128,128]`，全部有限，vMF 最大范数误差
  `3.5762786865234375e-7`。当前规范模型、同进程旧命名模型和归档旧源码独立 NPU reference
  三者逐元素精确一致；输出 tensor SHA-256 为
  `872b6409409a1681042f5f5bf87d4b56fb91de56f3663c3b30caf2fcd845198b`。
- 哈尔滨真实 patch 输出 `[1,6,64,128,128]` 且全部有限；缺失高分 SAR 的 availability 与
  supervision 非零计数均为 0。
- 6×NPU fresh/resume 各完成 1 个 optimizer step：fresh epoch 0、scheduler 1、loss
  `3.254943370819092`；resume 从 epoch 1 恢复、scheduler 2、loss `3.257829427719116`。

独立模型门禁报告 SHA-256：
`737b1cce90607a8ae72111105965b59beb7d603857585c60b00d75448414e194`。

## 全国数据门禁

- 父网格：5,785,781；sampled 62,000；unsampled 5,723,781；62,000 个 sampled key 全部精确匹配。
- 缺失、重复 parent key、sampled/unsampled 交集、错误分区、未知子记录、元数据/geometry/hash
  不一致、owner zone 错误和同区正面积重叠全部为 0。
- 旧 1% 单对诊断的实时/冻结计数均为 6,701，最大跨区比例均为
  `0.7418952088702487`。
- 正式 UTM seam policy 审计 10,411 对；总重叠面积 `2,842,239,086.4277167 m²`；全局重复面积
  比例 `0.0002998324802476501`，低于上限 `0.001`；非相邻、owner 顺序和离缝计数均为 0。
- package manifest 外部信任锚 SHA-256：
  `5e4316095664e4c4da3a4171bb49dd914f736d14f7bf20eca76f871bbae6ac41`。最终只读审计报告
  SHA-256：`26da517a6e4d3e70a7241514508dc77dcd9ec687386172b461a80f84e06a5c2a`。
- 全国十片 1–10 齐全，总数及唯一 parent 均为 5,785,781；1–9 各 578,578，10 为 578,579；
  重复、错片、缺失和未知计数均为 0。报告 SHA-256：
  `8246242ec9075f7e4d512aa5e31faa70f74cd84f864be41d4d1d573258011601`。

## 剩余发布动作

全新目录 clone、安装和 CPU 复验已经通过。候选发布提交的 GitHub CI 全绿后创建 `main`
保护、`v1.0.0` 标签和 GitHub Release。旧仓不可达泄露对象的 GitHub 缓存清理由仓库管理员
继续按 `MIGRATION.md` 提交 Support 请求；原令牌已撤销，全部公开 refs 与新仓历史均为零命中。

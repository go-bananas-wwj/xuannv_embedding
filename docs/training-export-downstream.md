# 训练、导出与下游评测

## 训练

CUDA 环境可直接运行单卡或 `torchrun`。训练命令默认优先使用 CUDA；`--synthetic` 只用于发布
smoke，必须显式给出 `--steps`。
真实栅格训练与导出需要安装 `data-process` extra：
`pip install ".[data-process]"`；基础安装仍可查看全部命令帮助并运行 synthetic CPU smoke。

CUDA 发布环境锁定为 `torch==2.6.0` 的 CUDA 12.4 构建；不要混装不同主次版本。
从不含 `.git` 的已安装 wheel 训练时，启动前必须把发布提交写入 `XUANNV_GIT_SHA`；非法或无法
证明的 SHA 会在设备初始化前被拒绝，避免产出无来源 checkpoint。

```bash
xuannv train --config configs/production/haidian_p10c_v1.yaml \
  --output /path/out/checkpoint.pt --device cuda:0

torchrun --standalone --nproc-per-node=8 -m xuannv_embedding.cli train \
  --config configs/production/mixed_haidian_harbin_p10c.yaml \
  --output /path/out/checkpoint.pt
```

仓内 `scripts/cuda/launch.sh` 默认启动单机 8 卡。两个平台入口（`scripts/cuda/launch.sh` 8 卡、
`scripts/npu/launch_6card.sh` 6 卡）都只设置默认卡数并转发给共用的 `scripts/launch_ddp.sh`；
`NPROC_PER_NODE` 可覆盖卡数。

### 8 卡训练

当前正式训练统一使用单节点 8 卡。年度 5 米方案与阶段门禁见
[年度技术报告](xuannv_annual5m_report.tex)。启动器固定为：

```bash
scripts/cuda/launch.sh CONFIG --output /path/out/checkpoint.pt
```

年度训练的吞吐瓶颈、必需的 GDAL 环境变量与实测加速倍数见
[训练加速](training-acceleration.md)。在网络文件系统上不设
`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` 会让训练慢 5 倍以上。

年度模型直接使用 `configs/production/annual5m_v1.yaml`：原生年度 manifest、共同地理遮挡、
日期/质量感知集合聚合、5m/PAN面积对齐和统一 64 维瓶颈均在同一前向中训练。真实训练必须
读取 `AnnualObservationDataset`；不提供 annual synthetic 替代路径。潜变量教师是可选第二阶段，
必须同时提供 `--teacher-checkpoint` 和 `--teacher-config`，两者通过配置摘要、source schema 和区域清单严格核验。
配置中的 `training.latent_prediction_weight` 默认是 0，只有显式打开该权重并提供教师 checkpoint
才会增加潜变量损失。教师权重不写入学生 checkpoint，避免恢复时把两套模型混在一起。

```bash
scripts/cuda/launch.sh CONFIG --output /path/out/student.pt \
  --teacher-checkpoint /path/out/teacher.pt \
  --teacher-config /path/configs/teacher.yaml
```

月度兼容模型可先用 synthetic smoke 验证运行时；年度模型直接对正式年度数据执行短跑。短跑至少记录每卡峰值显存、
有效样本与目标计数、数据等待时间、吞吐、有限 loss、梯度有限性、checkpoint 原子写入和严格恢复。
非有限 loss 直接终止训练；非有限梯度不终止，而是记入摘要的 `nonfinite_gradient_steps`
（每 rank 一份，也在 `rank_metrics` 中）。fp16/NPU 路径出现少量跳步属于 GradScaler 的正常
行为，但该计数接近 `optimizer_steps` 说明这一步实际没有学到东西，必须先查清再继续长训。
正式长训只有在缺模态 batch、PAN 原生网格和吉林一号原生网格都能稳定前反向后才启动。

全局 batch = 8 × `data.batch_size` × `training.gradient_accumulation_steps`。例如
`batch_size: 3`、`gradient_accumulation_steps: 2` 时为48。改变单卡 batch 或累积步数后，
必须重新核对学习率、warmup和有效目标数，不能只保持名义 epoch 数。

CUDA 使用 nccl。CUDA AMP 在支持 bf16 的硬件上使用 bfloat16 并且不启用 GradScaler（bf16 不需要
loss scaling），仅在硬件不支持 bf16 时回退到 fp16 + GradScaler。

多区域使用相同模型和损失。区域 loader 按 `sampling_weight` 确定性轮转；每个 batch 保持同一区域，
以保留不同物理产品的原生高分尺寸。`--resume` 严格恢复模型、semantic probe、optimizer、scheduler
和 epoch。学习率按绝对 epoch 执行线性 warmup 与 cosine 退火；`save_every` 每隔指定 epoch 原子
写入带 `.epoch-NNNN` 后缀的恢复点，最终 checkpoint 始终写到 `--output`，两者不会互相覆盖。
恢复或导出新格式 checkpoint 时，还会逐项核对当前配置文件 SHA-256、source schema 和 region
清单；任一身份不一致都会在加载权重前失败，不能把别的运行状态静默套到当前数据合同。

DDP 包装前，各进程还会核对参数和 buffer 的名称、注册顺序、形状、类型与训练标志；
同形状但不同传感器的层也必须一致。年度 PAN 投影层按配置列表顺序注册。新 checkpoint
逐优化器参数组保存参数名，恢复前检查槽位身份；顺序改变时直接拒绝。历史多 PAN checkpoint
未保存这项身份，其优化器状态无法安全续训；仍可按名称加载模型进行推理或用
`--initialize-from` 开始新的优化器。已有单模型评测数值保留，修复前多卡运行的配方比较需要重跑。

修改训练目标后继续优化既有模型，使用成对的 `--initialize-from`、`--initialize-config`。
它们先按原配置严格核验 checkpoint，只加载模型参数，重新创建 optimizer、scheduler 和损失
predictor；原 checkpoint 的摘要、epoch、Git SHA 会记入新制品。该操作与 `--resume` 互斥，
输出不得覆盖初始化权重或教师权重。年度潜变量预测的教师可以比学生小，接口合同仍严格匹配；
原生特征保持的参考须保留模型结构与观测合同。

```bash
xuannv train --config /path/ablation.yaml --output /path/new/student.pt \
  --initialize-from /path/original/model.pt --initialize-config /path/original/config.yaml \
  --teacher-checkpoint /path/small/model.pt --teacher-config /path/small/config.yaml
```

## 年度消融实验

独立目标实验先使用 `xuannv data targets-prepare` 完成数据准备，再物化已冻结的实验：

```bash
xuannv experiment prepare-targets \
  --prepared-root /path/to/prepared --target-root /path/to/targets \
  --initial-config /path/to/original/config.yaml --output /path/to/new_experiments
```

它生成 2020/2021 配对清单、一个 D3 工程配置，以及种子 17/42 的 R0/R1/D1/D2/D3
五组筛选配置；扩大阶段仅固定数据，不提前选择配方。旧模型的输入统计量用于全部继续训练组，
新增 DSM 监督使用 screen 训练位置的统计量。具体预算和评测约束见年度技术报告第14节。
所有新组统一开启梯度检查点；模型参数合同不变，为不同原生高分组合及 NCCL 收尾保留显存余量。
工程配置只允许显式 `--epochs 1 --steps 1..128`，建议 8 卡完成 128 步后另存恢复检查：

```bash
scripts/cuda/launch.sh /path/to/engineering/D3.yaml \
  --initialize-from /path/original/model.pt --initialize-config /path/original/config.yaml \
  --epochs 1 --steps 128 --output /path/new_engineering/model.pt
scripts/cuda/launch.sh /path/to/engineering/D3.yaml \
  --resume /path/new_engineering/model.pt --epochs 1 --steps 2 \
  --output /path/new_engineering/resumed.pt
```

这次短跑不使用教师或 EMA，也不构成精度比较。监督头和渐入批次计数保存在 criterion 状态中，
恢复同时核对原配置身份及 optimizer 两组状态。损失按跨 rank 有效目标总量归一化，
全缺测保留有限零项及参数梯度图；人工遮挡不删除独立目标监督。准备凭据未放行时，
正式的 D1/D2/D3 训练会被拒绝。监督头、单位/类别映射及准备摘要会写入新 checkpoint 元数据。

本地多源监督的工程接入使用已有的冻结划分与工程 manifest：

```bash
xuannv experiment prepare-multisource \
  --prepared-root /path/to/prepared \
  --target-root /path/to/targets \
  --experiment-plan /path/to/frozen_target_experiments/plan.json \
  --output /path/to/new_local_supervision --workers 4
```

命令只生成新的监督缓存、`multisource.yaml`、同数据的 `control.yaml` 和计划凭据，不启动训练。
原始标签与已冻结清单只读；缓存逐行校验，派生缓存和源码均登记摘要。多源工程配置使用
地形/覆盖/夜光/OSM 各 0.05，总辅助权重 0.2，前 32 批渐入；这是一套未调参的起始设置。
用前述 8 卡启动器执行 `multisource.yaml --epochs 1 --steps 128`，再另存恢复 2 步，
即可覆盖同一批 2,048 个位置的两个年份。控制组留待后续效果对照，不在接入检查中自动排队。
原五组方案保留为历史计划；当前快速路线为修复基线与多监督联合版两个组。
CLCD、OSM 接入上游后，相同产品的下游评分只能作产品相关的开发/回归检查。

年度 ABC 实验使用 `xuannv experiment prepare/run/reconstruction/export/probe/compare`。`prepare` 冻结
按 UTM 区域与 MS/PAN 覆盖率分层的训练/验证样本，排除已有下游开发集，记录源 manifest、
训练统计量、标签、初始化权重及教师的摘要；生成相同学习率、数据、epoch 的八个对照配置。
`run` 在指定 GPU 上依次执行，按严格 checkpoint 自动恢复；日志与状态保存在输出目录。
`reconstruction` 对各模型使用同一逐 patch 遮挡种子和纯留出高分目标。
新 `prepare` 自动显式启用原生 PAN 精确可见性，并在计划中设置
`exact_mask_coverage_targets: true`；所有模型均用
修正后的共同目标评分，模型输入继续遵守各自配置。`experiment adapt-inputs` 可以登记只修改
输入预处理的对照：严格加载原权重、保留来源与 epoch，验证全部模型张量摘要不变，并记录
本次优化器更新数为 0；它不能冒充重新训练的模型。
`export` 导出同一组语义与 5m 城市开发样本，保留真实 split 和像素有效性。
`probe` 使用训练 fold 的统计量拟合标准化；语义线性头按 20.48km 空间块分五折，5m 的
线性头与卷积头按城市分五折。城市协议继续使用已有 OSM 映射范围筛选，边界容差固定 5m。
`compare` 在相同 patch、fold 上比较，用整座城市或空间块配对重采样 2,000 次计算置信区间。
空间组少于 5 时只报告点差，不给出可能误导的退化 bootstrap 区间。下游头的随机性可用
`experiment probe --suite city --seed-offset 100 --save-heads` 单独复核；结果写入独立种子目录，
保留主实验结果，保存的头可用于同一固定样本的预测可视化。
`experiment probe --suite city --deterministic` 在后缀为 `_deterministic` 的独立目录中
验收读出的数值复现性，已存在的该策略结果目录会被拒绝，避免覆盖历史读出。
该选项强制确定性算法（不允许仅警告）、确定性 cuDNN 卷积并关闭算法测速；TF32 设置保持原值。
CUDA 启动前应设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（也接受 `:16:8`）；未初始化 CUDA 时
命令可设置默认值，已初始化但没有合法设置时直接拒绝。结果同时记录硬件、PyTorch/CUDA/cuDNN、
后端开关及原特征导出的源码 SHA，读出源码身份单独保存，不能重写原特征身份。
退出时恢复 Torch 后端开关，已安装的 CUDA workspace 环境设置保留在该进程中。
默认不启用该策略，以保留历史实验路径；不同数值策略的结果应分别归档，不能择优挑选。
启用前应以同一特征、头种子和设备重复训练，核验头张量与逐 patch 预测计数；固定种子本身不足以
保证 CUDA 卷积可复现。这不承诺跨 PyTorch 版本、平台或未验收的 NPU 复现。
附加 `--threshold-free` 可记录 AP 与 ROC-AUC，辅助判断固定阈值下的变化是否来自分类分数
的排序或校准；这些诊断不参与已锁定的候选选择，也不在测试集上调阈值。
`--suite city --feature-mode pooled10m` 将同一冻结嵌入先按 2×2 平均池化至 10m，再双线性
插值回 5m，并重新训练相同的下游头；输出单独目录，用于检查原生 5m 是否真正提供细节收益。
`lock-selection` 在候选评测前登记语义与细节各占一半的选择规则，并设置分项回退边界；
`select` 只允许执行一次。`freeze-confirmation` 可以从既有标签包冻结额外位置，排除开发集，
按当前年度 manifest 的真实 split 分配下游训练与确认样本。确认模式只用年度 train 位置拟合
下游头，年度 validation/test 位置只评分；标签仍是同源弱标签，不构成独立真值认证。

本地有带日期的 OSM PBF 时，安装 `.[data-process,osm,downstream]`，运行
`experiment freeze-confirmation-osm --plan CONFIRMATION_PLAN --snapshot FILE --snapshot-date YYYY-MM-DD`。
它生成独立的 `plan_fine.json`，保留原确认方案不变，并登记快照摘要、解析器版本与跳过的几何数。
细粒度确认只在当前年度 train 位置训练头，validation/test 位置只评分；缺失标注不能解释为已验证负样本。
这些已有弱标签与历史测试结果均属于开发证据，不能宣称新的独立盲测结果。

```bash
xuannv experiment --help
xuannv experiment run --plan /path/experiment/plan.json --gpus 0,1,2,3,4,5,6,7
xuannv experiment reconstruction --plan /path/experiment/plan.json --arm A_context
xuannv experiment export --plan /path/experiment/plan.json --arm A_context
```

## P0 诊断

`xuannv diagnose` 在真实观测上核验月份绑定的高分融合合同，是启动正式长训前的门禁，不是精度
评测。它要求配置里 `data.highres_mode: observations` 且 `data.target_months` 非空，并逐项检查：

- 高分观测重排后 embedding 不变，重复同一观测后重排同样不变；
- 掩码全零的观测与完全不传该 source 逐位等价，且掩码外像元数值不影响结果；
- 未被任何观测覆盖的目标月份与无高分基线逐位相同；
- 缺测 source 与全部无效输入的重建损失为 0；
- 时序 stem、高分编码器和融合模块的梯度有限且非零，三类边界 batch 反传梯度有限。

```bash
xuannv diagnose --config /path/observations.yaml \
  --checkpoint /path/checkpoint.pt --output /path/reports/p0.json --device cuda:0
```

任一不变量失败直接抛错，不会写出 `passed: true` 的报告。报告显式记录
`landmark_registration` 和 `radiometric_and_cloud_QA` 为 `not_verified`、`p1_ready` 为
`false`：通过诊断只说明训练合同自洽，不代表绝对定位精度或云影准确率已经认证。

## 导出

新格式 checkpoint 直接严格加载；原海淀 checkpoint 必须提供兼容 profile。导出文件为每 patch
一个 NPZ，含 `embedding[M,D,H,W]`、`timestamps[M]`，以及可用时的
`validity_mask[M,1,H,W]`，经过有限值检查后原子发布。

```bash
xuannv export --config configs/production/haidian_p10c_v1.yaml \
  --checkpoint /path/epoch_800.pt --compatibility-profile haidian_p10c_v1 \
  --region haidian --output-root /path/embeddings --device cuda:0
```

## 下游数据与协议

下游 NPZ 必须包含：`embeddings[N,D,H,W]`、`labels[N,H,W]`、唯一 `patch_ids[N]`。空间 fold
JSON 的每项包含互斥 `train / val / test`。

标准头只有 `linear`、`mlp`、`wide_mlp`、`deep_wide_mlp`、`conv3x3`、`unet` 和
`deeplab_lite`。few-shot 的 N 指 N 个正样本 patch，并确定性配对 N 个负样本 patch；不是任意
抽 N 个 patch。checkpoint 封存数据、标签、fold 和实际训练 patch 清单摘要。

阈值只用 validation 选择，test 报告 F1、AP、AUC。full-label 与 5/10/50-shot 必须分开报告；
更换标签、fold、shot、seed 或数据文件后，身份摘要不一致会直接拒绝评测。

七目标对照可用 `xuannv experiment prepare-target-comparison --local-plan /path/local/plan.json
--evaluation-plan /path/frozen_development/plan.json --output /path/new_comparison` 冻结。
它复用已核验的 screen 缓存和 8,192 位置双年清单，生成两组、种子 42/17 的四次研究配置；
每次 2 轮、每轮 512 步、全局 batch 32，从同一原100M制品重新初始化。
七目标总权重沿用 0.2，渐入为 128 批；不使用教师、EMA，不按中途分数选 checkpoint。
联合版显式设置 `research_max_updates: 1024`，不会把准备凭据改成正式放行。
执行 `experiment run --plan ... --arms control_s42 multisource_s42 multisource_s17 control_s17`；
运行器将已冻结的 `steps_per_epoch` 传给训练入口。重建评测统一关闭独立目标及可见重建，
冻结嵌入探针沿用开发标签、fold、训练步数及阈值。WorldCover/OSM 的同产品评分仅作开发诊断；
新确认集保持封存，研究对照结果不会自动覆盖原生产模型。

独立监督头可先在冻结的年度编码器上预热：

```bash
torchrun --standalone --nproc_per_node=8 -m xuannv_embedding.cli experiment warmup-target-heads \
  --config /path/warmup.yaml --initialize-config /path/original/config.yaml \
  --checkpoint /path/original/model.pt --output /path/new/heads.pt --steps 1024
```

该入口要求显式 `research_max_updates`，冻结全部编码器、融合层及 vMF 参数和状态，
只优化七个训练头。头学习率为 `training.lr * head_lr_multiplier`，按完整权重训练，
不对冻结编码器应用辅助渐入。默认种子 42/17 是两套头初始化，二者共享同一次冻结特征、
样本顺序、输入掩膜和vMF噪声；不能将其记为两个独立的主干训练种子。
冻结主干采用eval模式，vMF单独保留训练噪声；读出头使用FP32。默认每512步保存头状态，
最终逐rank核对编码器与头状态摘要。制品只含监督头、头优化器及来源身份，不是新编码器。
`target_warmup.restore_warmed_heads` 在核对原编码器、目标准备、归一化和输入策略后严格载入头，
不会把冻结阶段的优化器状态带入主干微调。
若头仍在学习，可用 `--readout-state /path/heads.pt --steps TOTAL` 延长冻结预热；
它严格核对头、原模型、目标与输入策略、batch及卡数，恢复头优化器，只接受完整数据遍历边界。
`TOTAL` 是累计头步数，仍受新配置的显式研究预算约束。输入掩膜沿用下一遍历的确定性种子，
vMF噪声RNG重新初始化，不声明逐位接续；这不会恢复或更新主干优化器。

`experiment diagnose-target-heads --config ... --initialize-config ... --checkpoint ...
--warmup ... --midpoint ... --positions TRAIN_POSITIONS.json --output gradients.json`
在既定训练位置比较初始/预热头，以及原始/修复后的上下文梯度路径。诊断使用FP32、训练模式和固定vMF噪声、
不构造优化器；记录实际加权损失的范数与夹角、高分/融合/vMF完整小参数组和一个明确命名的STP参数。
它不读取新的开发或确认角色，不把局部梯度冲突解释为下游退化的因果证明。

主干微调可在 `train` 的 `--initialize-from` / `--initialize-config` 之外，增加
`--initialize-target-heads /path/heads.pt --target-head-seed 42`。这会校验头文件摘要凭据、
原模型张量摘要与完整目标/输入合同，只载入头权重；联合训练的 Adam 状态重新开始，并在
checkpoint 的 `warmed_head_identity` 中登记。`--resume` 则恢复完整联合训练状态，不能再次
传头初始化参数。输出路径不得覆盖原主干或预热头。

`training.annual_vmf_lr_multiplier` 默认为 `1.0`，保留历史优化器参数顺序；设为 `0`
只冻结 `bottleneck.proj`，仍允许损失经该投影反传到上游。大于零的其他值为投影独立学习率
倍率，不改变 `log_kappa`、其他模型参数或目标头倍率。该设置仅适用于年度模型。
`training.annual_feature_anchor_weight` 默认为 `0`。研究原特征保持时，可以提供结构和观测合同
一致的冻结参考，并显式设置正权重；参考只用于约束未遮挡观测上的原生 5m 确定性特征。
它不替代 DEM、WorldCover 等独立目标。新训练同时提供 `--teacher-checkpoint` 和
`--teacher-config`，恢复时继续提供同一对参数；严格身份检查与其他教师目标相同。
参考保持 eval 和停止梯度，额外学生前向保持梯度、关闭采样和 decoder，不改变主训练的 vMF
随机数序列。只记录 `feature_anchor` 及其加权值；关闭时不会增加这些损失键。
`annual_feature_anchor_mode` 默认 `native`，比较整向量；`highpass_10m` 只比较同一共同有效
掩膜下、每个 10m 区块内部的原生 5m 特征残差。它允许区块均值继续变化，沿用相同的全局
有效像元归一化、参考身份和恢复合同；没有额外归一化、可训练预测器或多尺度参数。
此模式增加训练前向和激活内存，必须实测多卡恢复、梯度和高分丰富样本的显存；
导出仍是同一份学生模型，不增加推理参数，也不据该损失下降宣布下游效果合格。
统一留出重建评测关闭该训练约束；通用对照配置生成器也重置该字段，避免继承参考依赖。

`training.update_diagnostics_interval` 默认 `0`；正整数在首次及相应优化器步上记录实际参数
差值、相对权重范数及与负原始梯度的夹角。它只支持无 GradScaler 的 FP32/BF16，
不裁剪梯度、不重写 Adam 状态；原始梯度范数与实际 Adam 更新量必须分别解释。

新研究 checkpoint 记录保守的累计批次预算账本。保持原配置与每轮显式 `--steps`，可用
`--resume /path/periodic.pt` 接续剩余轮数；恢复入口验证原初始化身份、累计轮数和总预算，
不能通过多次恢复超出 `research_max_updates`。缺少账本的历史研究 checkpoint 仍拒绝恢复。
账本不改变正式数据放行状态；采样按下一轮接续，vMF RNG 重启，不声明逐位一致。

空间开发读出诊断使用 `experiment development-prepare` 冻结协议和既有 development 位置，
再物化本角色的 WorldCover/DSM 标签。该入口校验审计与划分身份，拒绝原预训练或旧候选位置；
validation/confirmation 不进入此流程。macro 与 component 的并集保持同 fold，两个年度按父格
绑定，训练 fold 与留出 fold 额外保留 1,290m 父格边界缓冲。协议先保存后读取开发标签。
`development-export --plan ... --arm ...` 从观测独立导出 10m/40m 聚合特征，
`development-probe --plan ... --arm ...` 为每个制品重新拟合相同的线性 ridge 读出；
固定正则 0.001，每年度记录最多抽样 1024 个覆盖像素、256 个DSM单元。
归一化及分类权重只用训练 fold，留出像元全量评分，阈值不调参。候选必须保留与原制品相同
的观测有效支持，不得通过删困难位置改善指标。它衡量新空间位置上的同产品可读性，
不代表独立人工5m精度，也不能覆盖既定道路保留条件。

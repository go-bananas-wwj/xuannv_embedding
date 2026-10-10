# 中国 1% 数据存储与实验归档

本仓库只保存布局、清单和可重复的迁移工具；影像、权重、缓存和日志仍保留在
`/data2`。目标根目录是 `/data2/xuannv_embedding`：原始来源位于
`raw/china_1pct/{s2,s1,landsat,gaofen,jilin1,static}`，处理结果位于
`processed/china_1pct/{assets,versions}`，训练记录位于 `experiments`。

## 操作顺序

先创建布局并生成清单。清单不跟随软链接，哈希只在需要冻结版本时启用：

```bash
xuannv data storage init --root /data2/xuannv_embedding
xuannv data storage inventory \
  --root /data2/china_xuannv_embedding/data \
  --output /data2/xuannv_embedding/catalog/storage-20260914/source-inventory.json
xuannv data storage plan \
  --root /data2/xuannv_embedding \
  --output /data2/xuannv_embedding/catalog/storage-20260914/migration-plan.json
```

迁移计划是审阅边界。工具使用同一文件系统上的目录重命名，不复制整套影像；目标
非空、目标是软链接、源不是目录或源被标记为活动写入时会停止。完成迁移后默认建立
旧路径到新路径的相对软链接，旧代码可以继续读取。吉林一号下载任务结束并完成清单、
备份和抽样校验前，不得使用 `--allow-active`。

```bash
xuannv data storage migrate --plan /data2/xuannv_embedding/catalog/storage-20260914/migration-plan.json
xuannv data storage verify --journal /data2/xuannv_embedding/catalog/storage-20260914/migration.json
xuannv data storage rollback --journal /data2/xuannv_embedding/catalog/storage-20260914/migration.json
```

`migration.partial.json` 是逐项写入的中断恢复线索；回退只接受仍指向目标的兼容软链接，
不会覆盖迁移后新增或改写的源路径。`cleanup` 只生成待审核清单，永远不自动删除：

```bash
xuannv data storage cleanup --root /data2/xuannv_embedding \
  --output /data2/xuannv_embedding/catalog/storage-20260914/cleanup-candidates.json
```

## processed 的容量口径

`processed` 是共享资产和不可变版本清单，不是每个实验一份 600 GiB 的副本。影像、云掩膜、
标签和修正层在 `assets` 中只保留一份；`versions/<version>` 保存输入摘要、规则、样本
范围、划分和统计量的引用。只有确实产生的重采样、裁剪、缓存或修正文件才占用额外空间，
训练缓存另行计量并记录重建方法。因此 processed 的大小应由文件清单统计，不能按 raw
大小直接估算。

## 实验归档

历史训练目录通过以下命令导入：

```bash
xuannv experiments archive \
  --source-root /data2/xuannv_embedding/v2/experiments \
  --archive-root /data2/xuannv_embedding/experiments \
  --mode symlink
```

每个运行保存 `run.json`、`config.yaml`、`dataset.json`、`events.jsonl` 和 README；
`artifacts` 默认是完整原目录的相对软链接，`--mode copy` 才会物理复制。归档会记录
配置轮次、指标实际记录轮次、续训起点、心跳观测时间范围和累计 `wall_seconds`，不会把
单次循环的累计时间逐轮相加，也不会根据目录名猜测设计或成功状态。缺失配置、精确启动
时间和评测口径均明确写成未记录。`index.parquet` 和 `catalog/experiment_index.json`
用于查询，实验和运行 README 的人工说明区可继续补充。

## 备份边界

正式备份目标为另一块盘 `/data/backups/xuannv_embedding/restic`。备份前重新核算容量，
保持 `/data` 至少 200 GiB 空闲；优先覆盖高分来源、冻结标签、完成产物、实验和吉林一号
完成原包，并保留覆盖表。备份必须保存真实文件并做首次恢复演练；软链接只用于兼容旧路径，
不能作为备份内容。基础月度原包、可重建解压副本和缓存若未纳入长期备份，必须在覆盖表中
明确列出。迁移、备份、300 个位置的 train/val/test 读取验证和恢复校验完成前不清理旧数据。

备份命令默认只在显式给出 `--dry-run` 时不触碰仓库；它先检查来源、备份盘剩余空间和批次
预算，再通过 `RESTIC_PASSWORD_FILE` 调用 restic，并把不含口令的回执写入清单：

```bash
xuannv data storage backup \
  --source /data2/xuannv_embedding/experiments \
  --repository /data/backups/xuannv_embedding/restic \
  --password-file /root/.config/xuannv/backup-password \
  --output /data2/xuannv_embedding/catalog/storage-20260914/backup-receipt.json \
  --max-gib 1024 --min-free-gib 200
```

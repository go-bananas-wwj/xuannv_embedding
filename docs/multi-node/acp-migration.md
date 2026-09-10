# 迁移到大装置 ACP

正式训练在 ACP 上跑。本文说明为什么必须迁、迁之前要准备什么、迁之后怎么验收。

## 为什么调试环境拿不到 RDMA

CCI（容器实例）是 **Serverless** 产品，定位是免运维与推理/在线服务。Serverless 的隔离模型
本身就不开放 capabilities 与特权容器——因此 `CapEff` 为 0、`ulimit -l` 的 hard limit 无法
提升，控制台里也找不到相应开关。这是产品定位的必然结果，不是配置遗漏。

补充一点容易走弯路的事实：**Kubernetes 没有设置 memlock ulimit 的 Pod 字段**（它属于容器
运行时/宿主层面的设置），而 capabilities 是标准 Pod 字段。所以即便平台愿意放开，可行的做法
也是授予 `CAP_IPC_LOCK`（它会完全绕过 `RLIMIT_MEMLOCK`），而不是去调 ulimit。

结论：在 CCI 上凑 24 卡 RDMA 无解，且没有调参空间（判定过程见
`nccl-troubleshooting.md` 中 `ibv_reg_mr_iova2` 一条）。ACP 才是面向分布式训练的产品，
RDMA（IB / RoCE）是其原生能力。

## ACP 的两个关键事实

- **平台会自动注入 NCCL 环境变量**，按集群属性给出推荐值，官方建议优先使用平台默认值。
- 训练网分 IB、RoCE v2 200G、RoCE v2 400G 三类，类型在 AEC2 集群详情页的「训练网类型」
  可见。调优前先确认自己在哪一类上。

本仓库的 `scripts/cuda/env_roce.sh` 中所有 NCCL 变量都写成 `${VAR:-默认值}` 形式，
**平台注入的值会自动覆盖仓内默认值，无需为 ACP 改代码**。同理，该文件刻意不给
`NCCL_NET` 设默认值——留空平台才能选中自己的网络插件。

> 以上来自官方文档检索结果，未逐页核对。提交任务前请向平台确认镜像规范与网络类型。

参考：
[ACP 环境变量](https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpUserGuide/acpEnvironmentVariable)、
[ACP 产品页](https://www.sensecore.cn/product/acp)、
[CCI 文档](https://www.sensecore.cn/help/docs/cloud-foundation/compute/cci)

## 镜像清单

需向平台确认的前置项：**基础镜像规范**（CUDA/驱动版本对齐要求）。以下是与本仓库相关的部分。

**1. `XUANNV_GIT_SHA` 必须在构建期固化。** 镜像内没有 `.git`，且容器内通常以 root 运行
（仓库属主不匹配会让 `git rev-parse` 因 dubious ownership 失败）。两种情况都会让训练拒绝
启动。这不是可选项。

**2. 换成正式安装。** 调试期是 editable 安装、指向共享盘上的源码目录；镜像里必须是真正
安装进 site-packages 的版本，否则镜像与源码耦合。

**3. 数据仍走持久卷。** 不要把数据打进镜像，ACP 任务挂载同一个卷即可。

**4. `export/*.json` 随包分发。** `pyproject.toml` 已声明
`[tool.setuptools.package-data]` → `xuannv_embedding = ["export/*.json"]`，wheel 会带上
`artifacts.json` 与 `legacy_reference.json`，无需额外处理。

骨架（基础镜像待平台确认后填入）：

```dockerfile
ARG BASE_IMAGE=<平台指定的 CUDA 基础镜像>
FROM ${BASE_IMAGE}

ARG GIT_SHA
ENV XUANNV_GIT_SHA=${GIT_SHA}

WORKDIR /opt/xuannv
COPY . .
RUN pip install --no-cache-dir ".[data-process]"    # 正式安装，非 editable

# 数据与产出走挂载卷，不进镜像
```

构建时传入：`docker build --build-arg GIT_SHA=$(git rev-parse HEAD) .`

## 提交任务

`NNODES` / `NODE_RANK` / `MASTER_ADDR` 由 ACP 注入，不需要像自有集群那样在三台上手工分别
启动，也不需要 `scripts/cuda/cluster.env`。

## 迁移后的验收

**不要直接上全量训练。** 按 `bring-up-checklist.md` 从 L2 起重跑一遍：环境换了，下层结论
不自动成立。重点确认两项：

- **L3 自检的带宽符合 RDMA 量级**（远高于 TCP 的 ~3.7 GB/s）。若仍是 TCP 量级，说明平台的
  RDMA 没生效，查 `NCCL_DEBUG=INFO` 里实际选中的后端。
- **L4 的 checkpoint 仍是 431 键、`upsample_head` 的 6 个键仍在。** 换环境不应改变这一点，
  它变了说明有更深的问题。

调试阶段已经排除的坑（DDP 未用参数、梯度累积、checkpoint 合同）不需要在 ACP 上重新排查
——它们与网络后端无关，这正是先用 TCP 把训练侧跑通的价值。

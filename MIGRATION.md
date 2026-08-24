# 旧仓迁移与来源

新仓 `go-bananas-wwj/xuannv_embedding` 采用白名单重建，不继承旧仓
`go-bananas-wwj/xuannv_embdding` 的提交历史。旧仓先做私有完整 mirror + bundle，再对全部 refs
执行精确令牌 replacement；没有借历史重写顺便删除其他内容。新仓历史经独立秘密扫描。

## 主要来源映射

| 来源 | 原 SHA | 脱敏 SHA | 公开 archive tag | 新仓落点 |
| --- | --- | --- | --- | --- |
| legacy main | `313faefc6bc251c9ba3cb3a202b5629b9c181576` | `f3262040592d64f3f5b14ac738f76d611c38e411` | `archive/legacy-main-20260824` | 仅溯源，不迁代码 |
| 海淀 P10C | `149f33bbf6232cc45345b550ec0da0bb0915d898` | `585210e683b31c58632ff68174c5de754e51a22e` | `archive/haidian-v1-production-20260710` | `8940cae` 起的模型核心 |
| v3 修复线 | `c6b1169a1e799115e98fa754f21e7bc9f350c7e6` | `8eb009b5cf42dc83adc5a44ed4a3a091c9ff52b1` | `archive/v3-semantic-64d-20260821` | 仅白名单修复 |
| 全国父网格 | `38addd71ea28346b8cac0b0bb8bcce3db6c5b813` | `080fc62e97841d237a5e640dd0093f8404f2cee4` | `archive/china-full-grid-20260806` | `4e0f5f4` |
| 全国十等分最终冻结 | `309510e9816b179623265527fa3be9018994c628` | `115777f189b8ec52ef7470e34d97111142e674dd` | `archive/china-tenfold-20260824` | `4e0f5f4` |
| 哈尔滨冻结迁移 | `d014ca29295098d3bd9535907a5ba57ee6a8cdc9` | `de6a73c745959fdcd51e15288b291eb3ad840bc1` | `archive/p10c-harbin-transfer-20260730` | 合同与限制摘要 |

十等分最初计划快照 `245977a8a311a47776fbca11c86241e47d2c79ce` 对应脱敏 SHA
`25c2b7fe45394303c59ad5053817a612d9c7617b`；最终冻结还包含之后已验证的 QGIS 导出测试。

## 明确排除

未迁入 synthetic fusion、P11–V5 实验目标、论文矩阵、BP/周报/申请材料、失败实验、展示代码和
大图。数据、checkpoint、embedding 与日志仍在外部数据盘或 ModelScope。

旧仓 refs 已扫描为零秘密命中。令牌已撤销；旧的不可达 Git 对象缓存清理属于旧仓管理员的
GitHub Support 后续事项，不改变新仓独立历史的扫描结果。

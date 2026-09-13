# V5 数据操作细则

## 路径与执行合同

部署时显式指定 `--source-root`（吉林一号目录）、`--dataset-root`（新质量版本）、
`--report-root`、`--base-root`（旧全国数据集）。绝不覆盖旧版本。
外部根目录绑定和实际绝对路径保存在数据盘运行记录，不提交私有路径。
统一CLI入口为 `xuannv data prepare-v5`；支持的阶段以实际 `--help` 为准，
未实现阶段必须拒绝执行，不能生成成功标记。

来源固定为 ModelScope `ptyzjr/Jilin1_Aligned_Train`。仅此来源允许远程像素；
其他源本地优先。认证从进程环境读取，禁止打印或持久化Token、请求认证头及签名URL。
预留约600GB空间，涵盖约208GB原包、210GB原生影像及质量产物；监测实际空间。

## A0—A3：落盘与索引

1. **source-lock**：读取发布revision、ARCHIVES.tsv、ARCHIVE_INDEX.tsv、SHA256SUMS；
   交叉核对包名、字节数、校验值。锁定全国registry内容与split计数、旧质量来源。
   输出 `manifests/source.lock.json`、`locks/input.lock.json`。变更已有锁必须拒绝。
2. **download**：首批两个包，最多并发2；`.partial`续传严格校验Content-Range起点、
   总大小；无有效续传重新下载。网络失败最多重试3次，认证错误不重试。
   大小和SHA256全部匹配才原子改名。输出 `manifests/download_status.parquet`。
3. **extract**：包先校验；拒绝路径越界、软/硬链接、设备及重复非等价成员。
   临时目录完成后发布，按包记录完成标记，崩溃可恢复但不能暴露不完整包为成功。
   TIFF数对照manifest；文件流式SHA256，逐块解码验证。
   输出 `raw_file_inventory.parquet`、`archive_integrity.json`、`rejected_files.parquet`。
4. **catalog**：读取实际CRS/仿射变换/1280m bounds，与全国网格唯一匹配；目录仅辅助。
   原采集时间和精度保留，不能推断不存在的时区或使用下载时间。同景不同分辨率
   共享scene_group_id；继承原split，全部年份登记、2020/2021另建视图。
   输出 `observations/highres/jilin1/{files,scene_groups}.parquet`、网格审核和覆盖表。
   不完整组按分支标记；同标识不同内容隔离，等价重复去重并保留来源。

## B1：辐射合同与统一读取

接口返回原生数组、逐波段有效掩膜、变换、CRS及元数据；QA、统计、加载共用。
先read_masks/NoData/非有限值，再逐波段scale/offset，最后QA和标准化。
零值不默认缺失，不统一裁剪反射率至0—1，不重复大气校正。
JL抽样合同为int16、NoData=-28672、scale=.0001、offset=0，逐文件验证。
5m按描述取B1—B6；B0独立；10m只取B7—B12；20m只取B13—B19；
缺失标识不得按通道位置猜测。记录JL1GP01/02及实际波长变体。
S1核查单位及VH/VV，不重复RTC；S2核查处理基线与偏移；Landsat核查RGB及缩放。
未知合同隔离。输出product_contracts、radiometry_audit、band_contract_failures。
测试包括有意义的零/负值、负NoData、每波段不同缩放、打乱波段及缺失分支。

## B2：质量掩膜

JL以B5/B4/B6（红/绿/近红外）调用冻结OmniCloudMask；GF以MS红/绿/近红外，
不得单独将PAN/B0送入多光谱模型。锁定模型权重SHA256、版本及预处理。
保留classes 0晴空/1厚云/2薄云/3云影；仅云/影膨胀约30m（ceil(30/gsd)）。
NoData单独保存；最终掩膜为数据有效与非云缓冲交集。传递掩膜使用真实CRS和
transform，细到粗保守汇聚，不能仅resize或将投影写为固定EPSG。
保留缓冲前后掩膜，即便晴空率<60%也不清零；单独保存strict_scene_qualified。
输出每产品classes.zarr、valid_masks.zarr、observation_quality.parquet。
旧sidecar仅在来源、合同及掩膜语义可验证时复用；仅有已清零掩膜不能恢复为新版本。
测试全云、全空、局部云、边界、不同分辨率传递及严格筛选不修改原掩膜。

## B3：配准

同年清晰基础参考仅供检查；多个有效窗口做边缘/平移匹配，记录偏移、置信度、
有效窗口数及一致性。算法参数用固定样例和已知人工平移测试冻结；未经标定不得
声明亚像元通过。可靠匹配残差≤5m可进入像元监督；超限或不确定隔离并分别统计。
保留原始影像，不自动扭曲修复。输出quality/alignment、alignment_summary和样例。

## B4—B7：可读取的训练数据

1. **targets**：检查标签网格、类别和年度；2020样本不取2021标签。DEM声明静态；
   OSM保留正例/可靠负例/未知及置信度。只建引用manifest，不复制旧大数据。
2. **statistics**：仅train有效像元，按产品及必要变体，流式均值/方差，分位数
   标明精确或抽样方法。每观测计一次，不随年度四季度复用重复计数；报告区域/年份
   贡献。空通道、异常方差失败；保存来源与质量合同指纹，用小数组对照直接计算。
3. **sample-index**：键patch/year/quarter；基础只取当季3月，高分取同年全年。
   保存全部候选；默认每年GF4组、JL4组，先每季度最高有效面积一组，再补剩余。
   按有效面积降序、日期及场景ID稳定排序；训练/验证/test继承原split。
   输出quarter_samples、annual_highres_candidates、annual_highres_default_selection。
4. **verify**：各split100位置、两年四季，不调用嵌入模型；检查原生shape、缩放、
   掩膜、缺月/仅雷达/无高分/部分分支/全云；原始与缓存一致，重复运行有效跳过、
   源指纹改变拒用旧产物。输出loader_verification、reproducibility_checks和dataset.lock。

## C1：验收产物与停止条件

`acceptance_report.md`：文件对账、按年/区域/split覆盖、云/配准前后保留量、
GF/JL覆盖四组合、异常影响、逐项检查及源/代码/数据版本。
`visual_review/index.html`及samples.csv：96组，基础光学/GF/JL各32，覆盖晴空、
多云、薄云、云影、边界和配准异常；不足如实说明，原图/分类/掩膜/网格对照。
`locks/acceptance.json`：未通过硬检查为incomplete；全部检查通过才ready_for_review，
自动程序无权写accepted。实际用户验收绑定数据锁hash，质量变化使旧验收失效。
发布验收材料后停止；模型计划保留在主计划，不在本阶段执行。

## 每步记录与提交

状态须含step、input_fingerprint、parameters、outputs、counts、excluded_reasons、
checks、code_commit、status、started_at、finished_at。运行中/失败不能改为成功。
阶段输出临时写入后发布；进程互斥，禁止两个进程同时修改同一运行。
每个行为改动先失败测试后实现，执行针对性pytest、Black/Ruff及仓库/发行门禁，
每个改进单独提交并push至wwj。外部完整清单不提交Git。

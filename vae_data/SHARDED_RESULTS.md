INCONCLUSIVE — 2026-11-16 Dynamic Scene Sampling Revision

本次独立一轮252.49秒、10次尝试；D1/D2/D4-A/D4-C成功，D3/D4-B未通过。真实Surface/Cave与Air65/稳定池19筛选已接入；没有正式Shard或训练。[完整实测/六型预览/性能](reports/scene_sampling_20261116/README.md)。

---

以下历史报告原文保留，其中“本轮/唯一一轮”仅指其当次授权。

# 最新：2026-11-16 全点预计算 / 六型 GT

**INCONCLUSIVE**：唯一一轮202.49秒，D1/D2/D3/D4-C成功，D4-A/B未通过。[完整实测、六型PNG/GIF、命令及限制](reports/exhaustive_20261116/README.md)。本次没有正式Shard或训练。

---

以下为2026-11-15历史报告，原文保留；其中“本轮”指当日旧测试。

INCONCLUSIVE — VAE 100K / 1K Shard Generator

数据生成器已实现；有限Native测试只得到D1合格样例，不能宣称四类物理产出、正式1K或100K已验证。只改数据系统，无模型、训练器、DataLoader或GPU任务。旧版代码、数据、报告及日志保留。

## 本轮唯一原生测试

命令：

```bash
.venv/bin/python -m vae_data native-smoke \
  --dataset outputs/vae_sharded_smoke_20261115 \
  --wall-seconds 300 --samples-per-category 2 --attempts-per-category 2
```

实际墙钟140.02秒，总计8次尝试，每类2次；没有额外扫描或重跑，成功总数2，未触及每类2/合计8的上限。选点使用公开记录过的天然地图位置提示，不构造地形；原生世界为seed191664964、Mapgen v7。

| 类别 | 尝试 | 成功五状态链 | 实际结果 |
|---|---:|---:|---|
| D1 Water | 2 | 2 | 候选7/10状态，中心目标累计变化28/211 |
| D2 Lava | 2 | 0 | 中心无目标变化，2.166/2.119个模拟秒后按规则拒绝 |
| D3 Fire | 2 | 0 | 中心无目标变化，2.216/2.218个模拟秒后按规则拒绝 |
| D4-A | 1 | 0 | 当前有限候选中没有预计中心接触的几何方案 |
| D4-B | 1 | 0 | 响应JSON读取异常，未产出可用链 |
| D4-C | 0 | 0 | 仅Unit覆盖，无原生结果 |

D1场景观察7.291/8.020个模拟秒，分别因中心2秒不变和10候选上限停止。初始化不计入原生变化。D2/D3的失败只反映这次场景/观察规则下的产出，不能推断其从不演化；没有为过测延长2秒阈值，也没有把Lava黏度7当作7帧周期。

## 成功样例与存储回读

每条保存 `uint16[5,48,48,48]` tokens及独立uint8 light；中心五状态互异，同一个固定世界16³、时间/step严格递增。全48³ Combined Token解码/重编码回读一致。数据位于：

- `outputs/vae_sharded_smoke_20261115/examples/D1_0/`
- `outputs/vae_sharded_smoke_20261115/examples/D1_1/`

[Water 0 视图](reports/sharded_20261115/D1_0.png) · [GIF](reports/sharded_20261115/D1_0.gif) · [Water 1 视图](reports/sharded_20261115/D1_1.png) · [GIF](reports/sharded_20261115/D1_1.gif)。图显示中心16³，实际数组完整保存48³；不代表所有微观中间step都被记录。

成功样例的Native证据：非技术实体0、Dropped Items0、Entity-Induced Voxel Changes0、unsupported falling attempts0，保留真实engine/game版本。拦截了附属实体和物品创建请求；没有导出过滤冒充零实体。

Craftium `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`，VoxeLibre `178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`。原生运行时为提交`dd43d619d303c92fe55655bbd80ebb8b340a2bf1`基础上的未提交新实现，不将最终提交冒充当时clean revision。

## 已处理及仍未验证

- 发现响应可能在Python读取时尚未完整写入。新Lua写临时文件再原子rename，boot先于ready字典发布；静态/Unit检查通过，**没有追加Native验证修复效果**。
- 最终交付还补齐floodable节点几何、天然静态池连通性、地下Lava池查询、合法窗口复核、候选序列索引及独立QA。此次成功小例在最后元数据补充前产生，未回写其科学数据或伪造额外成功。
- D4原生Pre–Reaction–Post未验证，D4-C没有真实尝试；A/B/C几何、重叠帧角色和合法五窗口只有Unit证据，不能称原生固化成功。
- 完整25群落、每片精确8/4/4/4、正式1K READY、多worker真实游戏吞吐、100K成本均未实测。本机已验证的并行是两个独立fixture小片与共享词表，不是正式Native多片结果。

## 检查

47项Python Unit Tests通过：新分片21项、历史Unified12项、P1原14项；快速检查为秒级，不安装新依赖。覆盖中心/邻接隔离、2秒计时、10候选、连续五状态、重叠D4物理帧、旧固体/邻域反应拒绝、三种几何、配额、词表并发、完整链提交、未提交尾部恢复、READY原子发布及只读保护。Unit的临时数据标为`unit_fixture`，正式生成器拒绝该模式。

Python/Lua语法和CLI通过；Native两条链另完成全48³无损回读。没有正式生成1K Shard、100K或其他原生动态，没有训练或模型评价。

原始失败/成功统计：[native_results.json](reports/sharded_20261115/native_results.json)，小例运行摘要：[native_run.txt](reports/sharded_20261115/native_run.txt)。原始Zarr留在忽略的outputs，不提交大数据。

## 使用

[服务器手册](README.md)提供单片、连续多片、多worker、Resume、check、status命令。所有正式采集由用户启动；未达配额的片保持partial，绝不READY，也不替换缺失类别。恢复不恢复原生物理时间，开启新的rollout并保留此前已提交样本。

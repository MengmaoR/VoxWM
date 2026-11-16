INCONCLUSIVE — VAE Dynamic Scene Sampling & Intervention Planning Revision

本轮是真实Craftium/VoxeLibre CPU采集，只有这一次新授权的六型验证；没有正式Shard、100K、GPU或模型训练。前轮全部报告与GT保留，本轮未用前轮成功补充当前失败。

## 场景与算法

- 删除通用`center_hint`，按合格、世界对齐16³ Chunk均匀选取。D1/D2/D3/D4-B不先找天然流体/可燃体素，D3燃料仅用于干预规划；A/C只找各自天然静态池。
- D1/D3为Surface；D2/D4正式采样各50% Surface/Cave。Surface取真实地形固体—Air界面，忽略液面和树冠；Cave由真实地形之下的连通空气选择，不使用固定减深。有限验证随机首高度、失败后交替，成功即停，因此不保证小例两个高度都有成功。
- 过滤先于全点规划：高度证据→中心最大六邻域Air≥65→A/C短时原生稳定观察后最大对应流体分量≥19。Source/稳定Flowing均可，不跨中心、不加二维面积或体积比例。稳定仅指两次实际观测的Combined Token相同，非无限期稳定保证。
- B拒绝预先存在的天然Water/Lava接触；两套完整源点候选独立采样，并排除预计接触已有天然流体的路线。组合仍需联合中心产物/时间阶段潜力。
- 复用共享CSR图、乐观反向不可达剪枝、64-source稀疏bitset和10 scheduled-wave秒传播；合法点全部评估，不预随机裁点。至少四个不同中心潜在时间阶段，首阶段≤2秒、前四阶段间隙≤2秒。一个时刻多voxel不算多阶段。
- `p=1/N`逐点独立Bernoulli，零命中均匀保底，最多4源；B两套采样后查询稀疏兼容关系，必要的一次条件保底明确记录分布。全部source一次放置，后续只观察原生演化。
- 预计算是空间/时间潜力，不模拟原生GT；Lava黏度、queue、竞争、Fire ABM机会仍不确定。实际五状态与D4中心Pre–Reaction–Post继续独立验收。

共享图构建O(V+E)，传播按64个source共享bitset frontier，避免对每个source独立扫48³；最坏仍随全部source数、10个时间波和range层数增长。仅保存逐source汇总和选中source的详细到达网格，不构造N×48³长期缓存。D4使用接触因子与有界条件查询，不建全Water×Lava笛卡尔矩阵。

## 本轮原生实测

总墙钟 **252.49秒**；10次尝试，六类全部覆盖，每类最多3次、成功最多1条，900秒硬上限未触发。固定诊断世界seed=191664964、有限站点；不是随机世界的大规模产率测试。中心沿用原来的16³网格，Frame/27块Assembly与数据配方不变。

| 类型 | Attempt | Height | 结果 | 候选状态 | 目标体素累计变化 | 原生观察数 | 原因 |
|---|---:|---|---|---:|---:|---:|---|
| D1 | 1 | surface | success | 10 | 82 | 34 | accepted five-state chain |
| D2 | 1 | cave | success | 10 | 69 | 35 | accepted five-state chain |
| D3 | 1 | surface | rejected | 1 | 0 | 10 | no_structural_fire_propagation |
| D3 | 2 | surface | rejected | 1 | 0 | 10 | no_structural_fire_propagation |
| D3 | 3 | surface | rejected | 1 | 0 | 10 | no_structural_fire_propagation |
| D4-A | 1 | surface | success | 8 | 61 | 60 | accepted five-state chain |
| D4-B | 1 | cave | rejected | 0 | 4 | 24 | solidification_only_in_neighbor_region |
| D4-B | 2 | surface | rejected | 3 | 756 | 110 | complete_event_but_no_valid_five_state_window |
| D4-B | 3 | cave | rejected | — | — | 1 | center_probe_no_height |
| D4-C | 1 | cave | success | 9 | 54 | 42 | accepted five-state chain |

D4-A稳定中心水池471格、D4-C稳定中心岩浆池189格。A/C本次分别记录25/27次原生reaction写入；写入数不同于中心合法事件或候选状态数。D3三次没有结构传播，未以Fire年龄变化补数。D4-B第一次固化仅在邻块，第二次有中心完整事件但只有3个不同候选、不能形成五状态，第三次无合格Cave高度；全部如实拒绝。未改变2秒超时、物理、地形或配额追成功。

| 类型 | 高度 | Attempts | 空间合格 | 可行干预 | Accepted |
|---|---|---:|---:|---:|---:|
| D1 | surface | 1 | 1 | 1 | 1 |
| D1 | cave | 0 | 0 | 0 | 0 |
| D2 | surface | 0 | 0 | 0 | 0 |
| D2 | cave | 1 | 1 | 1 | 1 |
| D3 | surface | 3 | 3 | 3 | 0 |
| D3 | cave | 0 | 0 | 0 | 0 |
| D4-A | surface | 1 | 1 | 1 | 1 |
| D4-A | cave | 0 | 0 | 0 | 0 |
| D4-B | surface | 1 | 1 | 1 | 0 |
| D4-B | cave | 2 | 1 | 1 | 0 |
| D4-C | surface | 0 | 0 | 0 | 0 |
| D4-C | cave | 1 | 1 | 1 | 1 |

空间合格包含高度/Air及A/C稳定池筛选，可行干预表示选到规划合格源点，Accepted必须是真实五状态链。Cave本轮已有D2/D4-C成功，但有限数据不能证明高度分布无偏。

| 成功类型 | 接收状态的实际sim_time（秒） | 原生step |
|---|---|---|
| D1 | 13.613821, 14.187781, 15.251979, 16.287809, 17.087511 | 267, 278, 297, 316, 331 |
| D2 | 14.065488, 15.197857, 16.068529, 17.128127, 18.248685 | 276, 297, 313, 333, 354 |
| D4-A | 15.997955, 16.284558, 17.064203, 18.145850, 19.293929 | 315, 321, 335, 355, 377 |
| D4-C | 17.162717, 18.261395, 19.034782, 19.283454, 20.108273 | 336, 356, 370, 374, 390 |

## 离线预计算性能（先测量，再进行上述原生验证）

使用已有真实GT及固定synthetic geometry，不开启新引擎，不对每个source做native simulation。表中的峰值是单Python进程RSS，包含import、输入和共享图；不是GPU显存，也不是C++增量内存。compiled kernel已加载，编译/加载时间单独保存在JSON。

| 输入 | 任务 | 全部合法点 | 时间（秒） | 峰值RSS（MB） | 筛候选（秒） | B配对（秒） |
|---|---|---|---:|---:|---:|---:|
| synthetic fixture | water | water:1072 | 0.0199 | 92.0 | 0.00159 | 0.0000 |
| synthetic fixture | lava | lava:1072 | 0.0154 | 81.5 | 0.00433 | 0.0000 |
| synthetic fixture | fire | fire:1072 | 0.0253 | 72.9 | 0.00299 | 0.0000 |
| synthetic fixture | pair | water:1072, lava:1072 | 0.0206 | 103.1 | 0.00348 | 0.0035 |
| existing native GT | water | water:33852 | 0.2756 | 99.4 | 0.02531 | 0.0000 |
| existing native GT | lava | lava:33852 | 0.2724 | 87.0 | 0.02545 | 0.0000 |
| existing native GT | fire | fire:53275 | 0.4520 | 76.2 | 0.03943 | 0.0000 |
| existing native GT | pair | water:56756, lava:56756 | 0.6395 | 119.3 | 0.13189 | 0.0980 |

旧GT未带新Surface/Cave probe metadata，这里的全点测量只衡量算法，不能声称这些历史场景通过新筛选。中心Air计数耗时另见JSON。相同24源点的Python参考BFS与C++输出逐项相等；相对加速分别为：water 6.56×, lava 3.44×, fire 2.87×, pair 2.95×。这是小固定fixture对照，不外推整个原生采集加速。

本轮实际原生场景全点规划耗时范围0.0151–0.4465秒；Python采集进程峰值RSS 132.2 MB（不含子引擎）。不是整个252秒的耗时归因；引擎启动、地图生成、探测与真实演化也消耗时间。

## Trace、Viewer与预览

完整原始GT：`outputs/vae_scene_sampling_20261116/traces/<type>/attempt_N/`，每次真实观察的48³ token/light按VWT1首帧+无损稀疏增量保存，保留真实时间/step、原始场景、干预前后、reaction和最终五帧引用。失败也保留；第三次B无干预，只有真实拒绝场景，不制造后续帧。

`planning/*.npz`保留全部合法点及命名统计列；`summary.json`记录数量/时间/RSS/随机概率/选择位置，`selected_N.bin`为选中源最早到达估计，不是GT。正式Shard仍只写五个接收状态及成功D4独立Anchor，不保存完整debug轨迹。

沿用已有Native-State Viewer；只补充相对初始化时间。六型均可切换attempt，查看48³/16³、实际source、估计可达域、真实变化、原生时间/step、接收五帧与D4 Pre/Reaction/Post。没有重做渲染、修改词表或原始GT。

```bash
.venv/bin/python -m vae_data export-traces --dataset outputs/vae_scene_sampling_20261116
.venv/bin/python -m vae_data view --dataset outputs/vae_scene_sampling_20261116 --port 8766
# http://127.0.0.1:8766/?type=D4-A&region=16
```

- [D1预览/诊断](D1/report.md)：[PNG](D1/contact_sheet.png) / [真实过程GIF](D1/evolution.gif) / [JSON](D1/report.json)。全部attempt可在Viewer查看。
- [D2预览/诊断](D2/report.md)：[PNG](D2/contact_sheet.png) / [真实过程GIF](D2/evolution.gif) / [JSON](D2/report.json)。全部attempt可在Viewer查看。
- [D3预览/诊断](D3/report.md)：[PNG](D3/contact_sheet.png) / [真实过程GIF](D3/evolution.gif) / [JSON](D3/report.json)。全部attempt可在Viewer查看。
- [D4-A预览/诊断](D4-A/report.md)：[PNG](D4-A/contact_sheet.png) / [真实过程GIF](D4-A/evolution.gif) / [JSON](D4-A/report.json)。全部attempt可在Viewer查看。
- [D4-B预览/诊断](D4-B/report.md)：[PNG](D4-B/contact_sheet.png) / [真实过程GIF](D4-B/evolution.gif) / [JSON](D4-B/report.json)。全部attempt可在Viewer查看。
- [D4-C预览/诊断](D4-C/report.md)：[PNG](D4-C/contact_sheet.png) / [真实过程GIF](D4-C/evolution.gif) / [JSON](D4-C/report.json)。全部attempt可在Viewer查看。

完整Trace/Viewer导出留在忽略的本机outputs，不随Git传输；Git包含轻量预览、实际统计和报告。新机器要查看这些**已发生的**过程，需要复制该outputs目录；不自动再跑引擎冒充同一数据。

## 检查与未验证项

50项快速CPU Unit Tests通过（1.86秒）：65/19精确阈值、Source/稳定Flowing、六邻域不越中心、真实Surface/Cave、取消天然目标定位、拒绝在规划前、27块对齐、全合法源无随机截断、时间阶段而非voxel计数、Bernoulli保底、B兼容、D4中心事件/去重及既有Trace/Viewer等。Python/Lua/JS语法及CLI检查通过。Fixtures仅验证实现，绝不列为Native成功。另对本次导出的真实Trace/Viewer输入做逐体素一致性回读，见`viewer_verification.json`。

仍未验证：D3在新中心策略下的可靠产率；D4-B五状态稳定产出；更多世界/群落/高度的通过率；探测上界191之外的覆盖；正式1K配额和READY、多worker真实吞吐、100K成本。现有单seed有限站点不够估计无偏全局产率。没有额外native retry、正式Shard、VoxEvolve实验、GPU、训练、原生物理修改或docs展示站改动。

本轮基于git `52e3f0d1359e1b6a89871f789b2423df14f88770` 加未提交修订运行，metadata保留dirty=true；最终提交不冒充当时clean revision。Craftium `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`，VoxeLibre `178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`。

正式生成/恢复/检查均由用户决定并手动执行，见[服务器命令](../../README.md)。Codex本轮验证已停止。

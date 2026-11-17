# VoxWM — VAE 100K / 1K Shard 数据采集协议

版本：2026-11-17 / `vae-region48-fixed24-v2`，Static策略`static-height-integer-v1`。本次Static以用户《Static Height-Adaptive Continuous Sampling Revision》为最高权威；Dynamic保留此前《Unified Revision: Interior Intervention, Fixed 24Hz Simulation & High-Throughput GT Collection》。本协议只修改数据生成，不改变 VAE 架构、不实现模型、训练器或 DataLoader。旧 200K 协议保存在 `vae_data/history/UNIFIED_200K_2026-11-15.md`，旧代码、结果和日志保留，不混入当前数据。

## 1. 数量和空间单位

总量100,000个Samples，100个独立Shard。每片严格1,000：Static500，D1 Water125，D2 Lava125，D3 Fire125，D4 Water–Lava Solidification125。每类动态每片25条完整链，每链恰好5个中心不同状态，链不能跨片。D4完全替代旧Vegetation。

每个Sample保存同一原生时刻、真实连续的48³ region，等于3³个相邻16³采样窗口。Dynamic沿用世界16格网格；Static水平坐标保持网格，高度允许逐整数偏移，采样窗口不等同于引擎MapBlock。轴为ENU(east,north,up)，region origin = center chunk origin − (16,16,16)，中心目标为数组 `[16:32,16:32,16:32]`。完整48³是上下文，目标仍是中心16³，不改成邻域VAE。

复用单一Combined Token `VocabularyID(Node Name, raw param2)`，uint16；raw light独立uint8，仅辅助。全数据集词表append-only，多个worker在短临界区内分配稳定ID，禁止局部词表ID未经映射就写入正式样本。按World Seed固定80/10/10 split，所有该seed场景归同一split；不要求每片精确80/10/10。

## 2. Static

25个主要Biome Groups沿用既有VAE/VoxGen映射，每片每组20：Surface8、Elevated/Complex4、Shallow Underground4、Deep/Cave4，共500。每群落尽量4个真实Biome Transition，作为已有样本附加属性，不另加数量；不足时报告soft-target缺口。

中心Chunk高度以中心点C计，起点O=C−8；中心为[C−8,C+7]，完整48³为[C−24,C+23]。主世界包含端点[-62,193]，所以C必须在[-38,170]，不能仅检查中心16³。S来自实际Native地表定位，不能把S与C混为一谈。

| Static stratum | 合法整数中心C（含端点） | 真实中心条件 |
|---|---|---|
| Surface | [S−7,S+8]∩[-38,170] | 必须包含S |
| Elevated Complex | [S+9,170]∩[-38,170] | C−8>S；0.10≤Air<0.98 |
| Shallow Underground | [S−24,S−8]∩[-38,170] | 允许全Stone/Dirt/Ore，拒绝全Void |
| Deep Underground (`deep_cave`) | [-38,S−32]∩[-38,170] | 不要求Cave、不设Air/连通性门槛；拒绝全Void |

Deep上界由S−24再减8得到，不依赖本次随机Shallow选择。Air仅指`air`，Void仅指`mcl_core:void`；完整Region出现`ignore`/未加载数据属于完整性错误，单独拒绝。禁止人工改地形以满足比例。

先按未完成“群落×高度”缺额加权选目标，再用Craftium原生`get_biome_data`按64格间隔、最多16,384个水平点螺旋定位；噪声查询不加载候选地形，命中后才加载并复核真实地表群落和捕获归属。原生变体沿用25组映射，不把目标标签冒充实际标签。有限查找不保证稀有群落或某种结构一定存在。

高度初次在完整合法区间上用`rng.integers(lower,upper+1)`均匀取整，失败后在剩余整数中均匀抽取，不按16格取整、吸附或从固定深度列表选择。每地点最多16次；每个高度只尝试一次；Elevated发现Air≥98%时把同一地点的上界降到C−1，此后只能向下。范围为空、候选耗尽或到次数上限即换地点/seed，仍受原有总attempt/墙钟限制，不修改配额。

Static唯一位置键为`(world_seed,center_e,center_n,center_u)`，中心等于起点+(8,8,8)。共享SQLite位置表与分片提交同事务，跨片也拒绝相同位置，不因内容变化重复收录；允许不同位置的48³窗口部分重叠。保留原有同源状态去重、seed split及每片单seed≤100。

Metadata保留surface_reference_y、selected_chunk_center_y、center_origin、region_origin、region_height_min/max、height_candidate_lower/upper、height_attempt_count、air_fraction、void_fraction、height_rejection_reason、static_height_policy_version及简短失败高度记录。Validator从实际坐标重算边界/区间，并从完整Token数组重新识别Air/Void/ignore；不会只相信报告比例。

Capture按每个实际世界体素确定底层MapBlock及局部坐标，复用最多4³个引擎块指针，保留27个采样窗口及ENU二进制顺序。Prepare/Forceload覆盖完整Region实际相交块；水平对齐、竖向偏移时通常为36块，不能假定只有27块。每次Prepare使用独立P0/VoxelManip同Tick原始字节核对全部48³=110,592个节点，字段为Node Content ID、param1/light、raw param2，任一不符立即失败。

48³ uint16 Token / uint8 light与Fixed24布局不变；Static新策略写入dataset/shard manifest。旧策略目录拒绝继续写入新Static，请使用新目录；旧文件/词表/结果不改写，历史Dynamic Trace仍可读取。Dynamic继续原有全局Chunk网格、时序、干预与五状态规则。VAE采样窗口偏移不改变持久World State的全局Chunk Grid。

每次只保持当前世界/区域，采集后直接提交或丢弃。不写Discovery Cache、全时间轨迹或不断增长的场景索引。

## 3a. 动态中心选择（本次修订）

保留现有世界网格16³对齐及27块48³ Assembly。删除通用`center_hint`：D1/D2/D3/D4-B不再先查找天然流体或可燃体素定位中心。D3的燃料检查仅在后续干预规划中进行；D2记录已有水作为干扰风险。

| 类型 | 高度机会 | 天然流体筛选 |
|---|---|---|
| D1 / D3 | Surface | 无 |
| D2 | Surface / Cave各50% | 无 |
| D4-A | Surface / Cave各50% | 仅稳定Water池 |
| D4-B | Surface / Cave各50% | 不检索；拒绝已有Water/Lava接触 |
| D4-C | Surface / Cave各50% | 仅稳定Lava池 |

在有限原生区域探测中按合格**对齐Chunk**均匀选中心，不按池中体素数加权。每站只读48×48水平范围、overworld最低高度所在16格层至191的竖向列；不保证覆盖世界全部高度或所有天然池，探测范围及拒绝原因写入metadata。Surface用真实地形固体—Air边界，排除树冠/木结构和液面；被湖水覆盖的湖底不当作Surface。Cave必须有地形屋顶之下的真实六邻域连通空气，不用Surface减固定深度、不挖洞。

过滤顺序：候选高度证据 → 中心最大六邻域Air分量≥65 → A/C短时原生观察（至少1.1模拟秒）后中心最大稳定对应流体分量≥19 → 全部合法干预点规划。稳定是同世界坐标的Combined Token在两次真实观察中相等，允许Source和稳定Flowing；不额外要求二维面积或体积比例。连通计数严格限制中心16³，邻块不能拼接小分量。B不对散落天然流体套用旧池面积门槛；两种新源的预计路径避免接触已有天然流体，仍需联合中心固化潜力。

每类/高度分别记录attempts、spatial_qualified（完整空间筛选通过）、qualified（已选中可行干预）、accepted（真实五状态链成功）。有限验证若没有合格中心，可只捕获一个真实拒绝区域作debug诊断，禁止在该区域放源；正式Shard直接拒绝并换Chunk。

## 3. D1–D3

从天然世界选择区域，可进行合法source/原生点火初始化；后续只运行Craftium/VoxeLibre原生物理。初始化放置及点火本身不计native变化。初始化后的baseline可作D1–D3首个状态，必须还有真实中心动态才能形成五状态链。

只统计中心16³的对应变化。D1=Water状态；D2=Lava流动状态；D3=Fire及与火邻接的可燃方块改变。邻接Chunk的改变不能增加候选或重置超时，也不能跨Chunk拼成一条链。

每场景最多10个中心不同候选状态；D1/D2/D4中心连续48 Tick、D3连续120 Tick无有效对应变化、或达到10候选、或达到最大观察时长，即停止。所有超时用原生整数Tick差；相对时间=frame_index/24，不能用墙钟或浮点轮询调度。Fire年龄变化不能重置超时。当前最大模拟观察D1=30s、D2=40s、D3=40s、D4=60s；命令另有明确总墙钟限制。默认liquid_update=1.0s，Lava viscosity=7是level调整阻力，不是7帧更新周期，不改其规则/参数；计算可快于实时，原生dt仍固定为1/24。

候选按真实时间排序，随机选一个相邻五状态窗口；不足5则拒绝，不补齐。重复中心状态不重复候选，48³邻域变化不能虚增中心状态数。

## 4. D4 固化

- A：天然静态水池旁/上方放Lava。
- B：无现成流体池的天然地形分别放Water/Lava。
- C：天然静态Lava池旁/上方放Water，支持地下查询。

每片25条，按A/B/C中一个9条、另两个8条轮换（shard0为8/8/9，shard1为9/8/8，shard2为8/9/8）。缺失不替代。

人工放置域为中心local[1..14]^3、region[17:31]^3，最多2744点；在此域内先检查合法性/保护/障碍，再对全部合法源点预计算，不使用预随机32/64子集。天然池不受此人工源点域限制；Python调用前后和Lua放置前复核全部source，越界整次拒绝。传播图/原生后续动态仍为完整48³，不能裁到14³。共享原生Node合法性/障碍图，反向乐观不可达判定、10秒时间分层和64-source稀疏bitset批处理；所有合法点有结果或明确的不可达依据。仅选中点保留详细可达域，不保存N×48³密集矩阵。编译小型C++17 CPU内核，不引入新的Python依赖。

Water range7、Lava range3，考虑向下流动、水平level衰减、source与flowing差异、floodable及另一流体阻挡。Lava黏度产生的level调整、Fire ABM机会与湿度均存在不确定性：10个scheduled-wave秒是可解释的传播潜力，不是原生确定性预报；随机过程最低实际产出可为0，不能声称10秒必达。可能存在未建模的可燃/冰雪移除时，禁用据此不安全的反向剪枝。

区分完整可达、中心可达、到达时间及时间阶段；至少四个不同的中心潜在变化时间带，D1/D2/D4首带不晚于2秒、前四带间隔不超过2秒；D3对应5秒。一个时间带影响多个体素仍只算一次。D4只统计预计Stone/Cobble/Obsidian实际产物位于中心的接触，不把邻接触点当产物。初始化后基准可作D1–D3首帧，D4 Anchor仍独立。

合格集合N上逐点独立Bernoulli p=1/N；零命中均匀保底、最多4源点、超限无放回随机裁剪。D4-B完整评估两类源点：先过滤10秒内能进入中心/产物接触halo的单源潜力集合，分别p=1/N，再查询隐式水–岩浆接触因子图。组合必须同时满足真实产物位置和联合四时间阶段潜力；一次有界条件保底评估全部lava并均匀选择兼容点，明确记录条件概率/分布变化；不枚举完整笛卡尔积，不保证每个场景一定找到配对。A/C按上节中心连通≥19及短时稳定性筛池，Source/稳定Flowing均可；B使用两套独立候选集合和联合兼容性。不新建池、不改地形。

Fire枚举全部合法air点火位置（无保护、下方非水），再评估原生传播邻域、中心可燃结构连通性及falling风险。仅相同Fire节点的raw param2年龄变化不计结构传播；不修改原生ABM、随机规则或falling排除策略。

用原生cooling写入的被动观测确认：真实Water/Lava接触、原状态以及Stone/Cobblestone/Obsidian新转换。观察器转发原始set_node，不改变规则、参数或结果。已有固体不是事件证据；未被可靠捕获的转换不伪造为成功。

Initial Anchor为全部source放置完的原始48³初态，独立保存且不进入VAE候选。Pre为reaction前最后一个已捕获中心状态；Reaction为首次观察到对应真实转换的帧；Post为其后第一次中心相关动态改变；后续reaction同样记录。初始化完成后的每个原生Tick均捕获；原生写入Tick与post-step观察Tick分别记录。一个Tick内部未暴露的多个顺序写入不等于多个可用状态。

统一Frame Registry以 `(rollout_id,simulation_step)`标识物理帧，重叠角色/Reaction IDs合并，Snapshot不重复存储。不同时间相同中心状态不能补数；角色可引用已有相同中心状态，但保留该引用的原始物理时间，并另存实际最后Pre观察的frame ID/time/step/us及中心相等证据，不伪造时间。无法满足真实Pre–Reaction–Post时间关系的事件仍拒绝。候选最多10，从中随机选连续五状态窗口，必须中心各不同，至少有一组完整Pre–Reaction–Post，时间严格递增、反应在中心、实际contact/write证据有效。没有合法窗口就拒绝。只有最终成功D4场景保留Anchor。

## 4a. Fixed24原生时序与接口

- `simulation_hz=24`，物理每step用相同的float32最近可表示`1/24`；计时用整数native tick，绝不累计墙钟dt。相对Tick0是全部人工干预完成后的状态。
- 固定版本Craftium采用opt-in补丁：Server等待Client请求→执行原生step/液体/ABM/Node Timers/packet处理→post-step捕获→完成信号→Client同dt完成→发送Python观测并阻塞等待下一动作。Python处理、规划和写盘时不额外推进世界。
- 移除固定模式的FpsControl sleep与Server Receive等待；跳过Client的墙钟lag扣减，禁vsync。ABM扫描不因墙钟预算提前中断；液体queue purge时钟用模拟时间。原生传播范围、黏度、interval/chance及RNG算法不改。
- 采集用只读C++世界坐标完整48³快照（27个采样窗口，底层引擎块数可不同）；与同Tick原P0 VoxelManip结果比较一次后缓存静态几何metadata。每Tick复用native buffer与Combined Token LUT，保留Node/raw param2/light，不重建27个VoxelManip，也不重复跑逐voxel名字回解审计。
- 保存`frame_index`、`native_simulation_step`、`relative_sim_time`、`absolute_sim_time`、seed、center、rollout ID及Server/Client/Lua时钟。原来的`simulation_step/sim_time`作为绝对Tick/时间别名保留。准备阶段无相对Tick；演化阶段只调用`Native.tick()`，严格+1。
- 当前VAE模式不渲染RGB，只发小型空白兼容packet；不是实际RGB数据。未来VoxEvolve可启用RGB，但原生截图/Camera在Client渲染循环中的相位及逐像素等价要另行验证。
- 时序正确不等于完整随机世界可复现：全局RNG、emergence、异步加载及初态/队列未快照；不同吞吐下严格voxel-state等价尚未证明，不将视频24FPS当证据。当前实测见`vae_data/reports/fixed24_20261116/README.md`。

## 5. 存储、恢复和发布

```
dataset/
  vocabulary.json
  dataset_manifest.json
  shards/
    shard_0000/
      snapshots.zarr/       # tokens/light [1000,48,48,48]
      anchors.zarr/         # 仅成功D4的25个独立anchor
      metadata.parquet
      journal.sqlite        # 样本/完整链的提交记录
      statistics.json
      manifest.json
      READY
```

未完成片暂存为 `.shard_XXXX.partial/`。每片一个worker独占锁，各片独立并行；全局仅词表分配短时加锁。Zarr写完后一次事务提交完整5状态链，恢复去除未提交数组尾部，保留已提交数据。原生模拟时间不恢复；中断场景在新世界/新rollout重新尝试，不伪装时间回放。

只保存选中五状态、成功D4 Anchor及必要事件引用；未选候选释放，不保留Raw Trajectory或发现缓存。记录Seed、坐标、Biome/地形、真实时间/步、source初始化、事件标签、engine/game版本、原生接口与Voxel-only证据。QA检查类别、Biome/地形硬配额、5状态完整性、中心变化、D4完整事件、同源重复、词表、Parquet与journal一致以及Native Provenance。

只在全部1K数据验证通过后，将目录和READY原子发布。READY片后续生成一律跳过，不能意外更新。Transition软目标不足单独报告，不能伪造标签。未满配额/失败片永不READY，不补其他类别、不跨片拆链。

## 6. 纯Voxel与本次权限

继续从引擎入口禁止非技术实体、掉落和实体导致的Voxel变化；技术player不可见无交互。当前不支持falling-entity物理，发生该请求则排除当前rollout，不发布被冻结的物理。保留原生液体、cooling、fire及其概率/参数；只用固定dt推进原生时钟，不按墙钟限速。

六型D1/D2/D3/D4-A/D4-B/D4-C仍为每型最多3次Attempt/1成功、总墙钟900秒；此次不重开六型、不增加动态样例。本次单独有限Clock检查总预算120秒：准备阶段失败13.98秒，修复后240 Tick复核14.98秒，合计28.96秒。真实只读自然场景，不计正式样本；物理跨吞吐/RNG等价仍未完成。历史报告/输出保留。

正式Shard只保存选中五状态和成功D4独立Anchor；每Tick都检查中心。Debug Trace保留所有实际Tick，包括零变化Tick，以首帧+无损稀疏delta表示，不能复制或插值。

GT-only Viewer独立于docs展示站，复用其只读Three/OrbitControls资源与ABV1格式；按真实sim_time播放实际观察帧，支持48³/中心16³、切片、27边界、类型高亮、source及估计/实际变化叠加、相对干预时间、事件/接收帧标记。每型轻量PNG/GIF/JSON/Markdown入Git，全部Trace留outputs并忽略。所有预计算、Actual Native、Accepted VAE States和Synthetic Tests明确区分。

不运行正式1K片、100K、广域扫描、其他原生动态任务或训练。服务器正式生成全部由用户手动启动。本轮验收是代码完成、检查通过、有限native测试完成或如实失败，而非虚构数据集已就绪。

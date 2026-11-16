# VoxWM — VAE 100K / 1K Shard 数据采集协议

版本：2026-11-15 / `vae-region48-shards-v1`。最新用户《VAE 数据生成系统重构》指令为本次最高权威。本协议只修改数据生成，不改变 VAE 架构、不实现模型、训练器或 DataLoader。旧 200K 协议保存在 `vae_data/history/UNIFIED_200K_2026-11-15.md`，旧代码、结果和日志保留，不混入当前数据。

## 1. 数量和空间单位

总量100,000个Samples，100个独立Shard。每片严格1,000：Static500，D1 Water125，D2 Lava125，D3 Fire125，D4 Water–Lava Solidification125。每类动态每片25条完整链，每链恰好5个中心不同状态，链不能跨片。D4完全替代旧Vegetation。

每个Sample保存同一原生时刻、真实连续的48³ region，等于3³个世界对齐16³ Chunk。轴为ENU(east,north,up)，region origin = center chunk origin − (16,16,16)，中心目标为数组 `[16:32,16:32,16:32]`。完整48³是上下文，目标仍是中心16³，不改成邻域VAE。

复用单一Combined Token `VocabularyID(Node Name, raw param2)`，uint16；raw light独立uint8，仅辅助。全数据集词表append-only，多个worker在短临界区内分配稳定ID，禁止局部词表ID未经映射就写入正式样本。按World Seed固定80/10/10 split，所有该seed场景归同一split；不要求每片精确80/10/10。

## 2. Static

25个主要Biome Groups沿用既有VAE/VoxGen映射，每片每组20：Surface8、Elevated/Complex4、Shallow Underground4、Deep/Cave4，共500。每群落尽量4个真实Biome Transition，作为已有样本附加属性，不另加数量；不足时报告soft-target缺口。

VoxGen原始空间协议要求真实连续27个Chunk，细分配额原为建议；本次按最新用户明确的8/4/4/4执行。不得拼接群落、不拿简单群落顶替缺失群落。Static表示一次真实采样，允许自然动态内容；保留种子、高度、地形差异，每片单seed最多100个样本。Elevated/Complex不接受纯空气或简单全实心块来凑数。

每次只保持当前世界/区域，采集后直接提交或丢弃。不写Discovery Cache、全时间轨迹或不断增长的场景索引。

## 3. D1–D3

从天然世界选择区域，可进行合法source/原生点火初始化；后续只运行Craftium/VoxeLibre原生物理。初始化放置及点火本身不计native变化。初始化后的baseline可作D1–D3首个状态，必须还有真实中心动态才能形成五状态链。

只统计中心16³的对应变化。D1=Water状态；D2=Lava流动状态；D3=Fire及与火邻接的可燃方块改变。邻接Chunk的改变不能增加候选或重置超时，也不能跨Chunk拼成一条链。

每场景最多10个中心不同候选状态；中心连续2个原生模拟秒无对应变化、或达到10候选、或达到最大观察时长，即停止。所有超时用实际sim_time，不用墙钟/帧数推测。当前最大模拟观察D1=30s、D2=40s、D3=40s、D4=60s；命令另有明确总墙钟限制。默认liquid_update=1.0s，Lava viscosity=7是level调整阻力，不是7帧更新周期，禁止时钟加速。

候选按真实时间排序，随机选一个相邻五状态窗口；不足5则拒绝，不补齐。重复中心状态不重复候选，48³邻域变化不能虚增中心状态数。

## 4. D4 固化

- A：天然静态水池旁/上方放Lava。
- B：无现成流体池的天然地形分别放Water/Lava。
- C：天然静态Lava池旁/上方放Water，支持地下查询。

每片25条，按A/B/C中一个9条、另两个8条轮换（shard0为8/8/9，shard1为9/8/8，shard2为8/9/8）。缺失不替代。

源点先做当前48³几何可达性筛选：Water range7、Lava range3，考虑重力优先、向下流动、水平level衰减、可floodable节点及固体/另一流体障碍，优先中心接触。筛选是有界空间潜势包络，不预测准确时间，不模拟source renewal、两流竞争或全部黏度过程；实际接触与固化只能由native确认。A/C先短时观测已有自然source池的稳定性；不新建池或改地形。

用原生cooling写入的被动观测确认：真实Water/Lava接触、原状态以及Stone/Cobblestone/Obsidian新转换。观察器转发原始set_node，不改变规则、参数或结果。已有固体不是事件证据；未被可靠捕获的转换不伪造为成功。

Initial Anchor为全部source放置完的原始48³初态，独立保存且不进入VAE候选。Pre为reaction前最后一个已捕获中心状态；Reaction为首次观察到对应真实转换的帧；Post为其后第一次中心相关动态改变；后续reaction同样记录。采样不是每引擎step全量轨迹，原生写入step/us与包含该写入的观察帧step分别记录，不能声称捕获了所有未采样中间状态。

统一Frame Registry以 `(rollout_id,simulation_step)`标识物理帧，重叠角色/Reaction IDs合并，Snapshot不重复存储。不同时间相同中心状态不能补数；若重复状态无法同时满足最后pre与唯一性，相关事件不算完整。候选最多10，从中随机选连续五状态窗口，必须中心各不同，至少有一组完整Pre–Reaction–Post，时间严格递增、反应在中心、实际contact/write证据有效。没有合法窗口就拒绝。只有最终成功D4场景保留Anchor。

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

继续从引擎入口禁止非技术实体、掉落和实体导致的Voxel变化；技术player不可见无交互。当前不支持falling-entity物理，发生该请求则排除当前rollout，不发布被冻结的物理。保留原生液体、cooling、fire，不加速ABM或模拟时钟。

允许单元/静态和有界native样例。每类最多2条成功五状态48³链，总计最多8；本轮采用每类最多2次尝试、总墙钟300秒。未成功实报，不扩测试规模追成功。D4第三子类型用单元测试覆盖。

不运行正式1K片、100K、广域扫描、其他原生动态任务或训练。服务器正式生成全部由用户手动启动。本轮验收是代码完成、检查通过、有限native测试完成或如实失败，而非虚构数据集已就绪。

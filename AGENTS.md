# VoxWM 执行守则

当前唯一活动数据生成入口是 `python -m vae_data` → `vae_data/sharded/`。科学合同为 [VAE_DATA_COLLECTION_PROTOCOL.md](VAE_DATA_COLLECTION_PROTOCOL.md) / [当前Drive Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view)，最新用户指令优先。11/12 Manifest的模型架构不在本轮修改范围。

- 本轮只实现数据生成，不做模型、训练器、DataLoader、模型评价或新算法研发。
- 当前规模100K，100个严格1K片；每片Static500，D1/D2/D3/D4各125，动态每类25条完整五状态链。D4是Water–Lava Solidification，完全替代历史Vegetation，不自动恢复旧200K配方。
- 正式样本保存真实同一时刻48³(27个相邻16³采样窗口)，目标与动态判定仅中心 `[16:32]^3`。Dynamic保持全局16格网格；Static仅高度可逐整数偏移，与引擎MapBlock明确区分，完整Region高度必须在[-62,193]。邻块变化不能加候选、拼链或重置超时；D1/D2/D4为48 Tick，D3为120 Tick。
- Dynamic最多10个中心不同候选，随机连续五状态窗口，不足则拒绝。D4必须完整中心Pre–Reaction–Post，物理帧统一去重，成功场景的Anchor独立保存且不计入1K。
- Static每片25主要Biome各20，Surface8/ElevatedComplex4/Shallow4/DeepCave4，真实Transition尽量每组4且为附加属性。不拼接、不拿其他群落或类别补缺。先按缺额选目标群落/高度，再用原生Biome噪声接口有限定位，命中后加载并复核真实地表群落；失败保留缺额，不退回随机地点凑数，不建Discovery Cache。Static按新高度协议：中心C∈[-38,170]，Surface包含真实S，Elevated从S+9到170且Air≥10%/<98%，Shallow为[S−24,S−8]，Deep为[-38,S−32]（均裁到允许域）；每地点最多16次不重复高度，空旷Elevated只向下，地下允许全实心但拒绝全Void，ignore另作完整性错误。
- 全局uint16 Combined Token=(Node Name,raw param2)，词表只追加；uint8 light辅助。48³无损存储，中心16³仍是VAE目标。World Seed固定80/10/10 split，不跨split。
- 纯Voxel：从原生入口禁非技术实体、掉落及实体引起的Voxel变化；技术player不可见无交互。当前falling-entity动态不支持，出现即排除该rollout，不默默冻结物理后发布数据。
- 使用Craftium/VoxeLibre原生液体、固化与火。不写流体轨迹，不改规则/ABM概率/黏度；固定24Hz物理时间、整数Tick为权威，无墙钟FPS限速。允许维护固定版本的最小Timing/Scheduler补丁，默认旧GT会话不启用。几何可达性只筛源点，不能冒充native事件证据；初始化操作不能计native演化。
- 低缓存：Static立即提交或丢弃；Dynamic保留至多10候选和必要Anchor/事件引用；不写完整raw轨迹、不积累Discovery Cache。分片独占写入，全局词表短锁，完成QA后原子READY；READY片只读。
- 动态中心先按真实固体—Air Surface或地下连通Cave选择；D1/D3仅Surface，D2/D4两高度各有50%机会。D1/D2/D3/D4-B不天然目标体素定位，A/C仅对应稳定天然池。全点规划前中心最大六连通Air≥65；A/C最大连通稳定对应流体≥19（Source/Flowing可，不跨中心、不加面积/比例门槛）。保留现有16³对齐与48³ Assembly。按类/高度记录attempt/空间合格/规划合格/接受数。
- 所有人工source仅中心local[1..14]^3，即region[17:31]^3，先域限制再合法枚举；Python调用前后及Lua放置前复核，任何越界整次拒绝。传播仍在完整48³原生世界，不裁剪到14³；天然池不受人工source域约束。
- 全部合法source参与共享图/反向乐观边界/10sim秒bitset批处理；禁止先随机截取256/32/64点。至少四个不同时间阶段才算潜力，不能用四个同刻voxel凑数；D3潜力筛选同样采用5秒间隙，其他2秒；几何估计与实际native证据分开。p=1/N独立采样、零命中均匀保底、有限source cap；D4-B必须联合兼容，记录条件采样偏移。
- 六型Native协议上限仍为每型3次/1成功、整轮900秒，不因时钟修订自动重开。历史两轮结果保留；固定时钟轮只做过约29秒Clock检查；随后用户在11/17明确授权重跑D1/D3/D4-A/B/C并跳过D2，且批准修复坐标接口后重新给五类各3次额度。该轮已完成：修复段7次尝试，五类各1成功，两段累计约254.52秒；D2沿用旧结果并明确标记。完成后不自动追加。更多Water/Lava/Fire受控随机跨吞吐验证须先说明预算并取得授权。11/17用户高度修订另允许有限只读Static核验；`static-verify`单次最多180秒，不创建正式Shard。正式1K/100K、广域扫描及训练由用户手动启动。
- Static使用按实际世界坐标读取的Capture，每次Prepare以同Tick P0/VoxelManip逐体素核对Node/param1/param2；Forceload覆盖实际相交的全部引擎块。Static按(seed,真实中心)跨片去重，新高度策略不混入旧目录。固定24Hz正式采集每Tick推进恰好一次、post-server-update捕获，不能用float时间轮询、丢帧或复制状态。干预完成为相对Tick0，绝对/相对时间由整数Tick/24计算。VAE格式v2不混入历史variable-step Shard。
- 六型debug及明确有限Clock验证允许完整原生观察Trace（首帧+稀疏增量），包括失败；正式Shard仍只保存选中五帧/必要Anchor。GT-only Viewer独立于docs，禁止改动或部署现有展示网站。
- 原始GT与词表只读；不改已有Viewer材质/方块渲染。Viewer几何/材质映射独立。检查同一世界坐标的真实Node Name/raw param2，Source丢失或固体变液体不可无写入证据当普通流动。Fire年龄更新不计结构动态；Ghost必须与当前GT明确区分。
- 科研与效率优先：简单可读，不建平台、复杂服务、冗余预检、hash审计或无关重构。只做相关快速Unit/语法/CLI检查，通常累计30秒内；测试Fixture绝不能当Native数据或物理证据。
- 保留 `vae_p0/`、`vae_p1/`、`vae_pilot/`、旧Unified源码/README/协议快照、所有旧结果/日志与docs。旧入口 `vae_data.legacy_entry` 仅历史参考，不混入当前格式。旧CoDL只读，不修改或推送。
- 唯一开发远端 `MengmaoR/VoxWM` / master。每次编码前检查并保留用户改动、`git pull --ff-only`。正常commit/push，不强推、不重写历史，不触碰展示站部署/权限。
- 交付说明实际Native、Unit及未验证功能，写明真实缺口。实现完成即停止，等待用户手动运行。

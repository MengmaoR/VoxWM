# VoxWM 执行守则

当前活动任务是 Voxel-Only Unified VAE Data Collection，代码 `vae_data/`。本文件定义实施边界，科研数据细则由 [当前协议](VAE_DATA_COLLECTION_PROTOCOL.md) 定义。

- 权威顺序：最新用户指令 > [Drive 当前 Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view) / 仓库同步协议 > 未被覆盖的11/12 VAE Manifest。11/13及旧P1规则是历史，不自动恢复固定20火点、人工种植或旧raw Node ID正式采集。
- 科研与效率优先：实现简单可读，不建平台、复杂服务、冗余预检或无关重构。仅执行与改动相关的一轮快速检查；通常不超过30秒，不为检查安装大依赖。当前用户明确授权有墙钟上限的真实CPU原生集成验证，此例外不授权200K或训练。
- 全项目研究对象仅为Voxel Nodes。禁止Mobs/NPC/Animals、Dropped Items及实体引起的Voxel变化。技术采集Player不可见、静止且无环境交互，不进入数据。新采集必须从原生入口阻止实体，不靠导出过滤。
- `vae_data`在临时builtin overlay中提前封闭实体/物品API和刷怪笼实体辅助路径，实际检查实体、掉落与实体变更数。Falling Node依赖临时实体，明确排除：出现该请求即隔离整个rollout，不能把被阻止的物理转移当作正常稳定数据。原始引擎源码不修改。
- 原生物理交给Craftium/Luanti/VoxeLibre，不写逐帧流体/生长轨迹，不调用grow、不改ABM概率、不改植物高度metadata。D4仅发现天然竹/甘蔗/藤蔓后原生挖掘修剪，保留根/附着源；纯Decay不是Growth；不能按成熟度/隐含上限预筛。
- 使用Discovery Index、全新独立世界重建和早期原生回调。区分Spatial Reconstruction与Temporal Replay，不称Python就绪后的状态为mapgen t=0；初态不同必须实报。每个rollout保留种子、版本/配置、固定坐标、初态hash、干预RNG/phase和实际模拟时间/步。
- 固定世界对齐16³ ENU Chunk，不从Camera估计原点。单一uint16 Combined Token=(Node Name,raw param2)，全局append-only词表且精确roundtrip。uint8 light辅助保存，不输入VAE。同seed全部场景属于同一split。
- 配额保持Ordinary100K、D1–D4各25K。Ordinary均衡25主要biome与地形/种子，不用容易场景填缺额。动态标准5–8不同状态、变化进度采样、初终态仅一次；真实2–4状态独立aux，禁止重复补齐或Synthetic Fixture冒充原生。
- 保留原始lossless增量轨迹，干预前/后、settling、native演化分相。植物观察600/1200/1800秒可延长，但有显式墙钟限额；不因单株成熟或得到8状态而停止整个群落。流体/火可在实测不变窗口后停止，不声称此后永无变化。
- 当前CPU采集无需Torch/CUDA/权重。GPU模型推理/训练与正式200K只由用户明确启动。未实测不写成功，失败或产出不足写INCONCLUSIVE/NO-GO，不虚构速度或成本，不无限追求绿色检查。
- `vae_p0/`、`vae_p1/`、`vae_pilot/`、既有GT工具、所有历史结果/日志与`docs/`保留。旧工具不是新合同的正式采集入口。旧CoDL仓库只读，不修改、不提交、不推送；不从旧PERSIST baseline推导既定新模型架构。
- 唯一开发远端为 `MengmaoR/VoxWM`，master。每次编码前先保留用户未提交工作并 `git pull --ff-only`。正常commit/push，不强推、不重写历史，不修改展示站部署/共享权限。
- 完成后交付实际检查、真实测量和未验证项；停止，等待用户决定正式采集或模型研发。

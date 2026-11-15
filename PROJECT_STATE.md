# VoxWM 当前状态

2026-11-15：**Unified VAE Collection / IMPLEMENTED — NATIVE INCONCLUSIVE**。

- 活动入口：`vae_data/`；[当前协议](VAE_DATA_COLLECTION_PROTOCOL.md) / [Drive Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view)。旧P0/P1/pilot代码、日志、结果与docs完整保留，不再是生产入口。
- 已实现低成本多区域Discovery、持久索引/缓存、三次独立重建/早期采集、可选多源/可变火点/天然植物群落修剪、完整增量轨迹、Combined Token标准/辅助池、seed split、恢复与显式200K入口。
- 真实CPU验证：索引60个Chunk；标准样本Ordinary12 / D1 Water62 / D2 Lava0 / D3 Fire21 / D4 Vegetation0，Lava另有2个辅助状态。天然水流与原生火焰传播可用，熔岩未满足5状态；4株天然竹子截断/保根/元数据保持正确，约601秒未生长。详见[实测报告](vae_data/NATIVE_VALIDATION_RESULTS.md)。
- Voxel-only实测：非技术实体0、掉落实体0、实体引起的Voxel改变0。阻止了植物挖掘掉落请求，原生水/火仍有变化；依赖临时实体的falling场景明确排除，非替代物理实现。
- 三次重建未实现全区域初态一致；精确时间轨迹也不同。逐Chunk匹配和变化见报告，不能声称可确定性回放。尚未验证25 biome/多种子规模产出和200K成本。
- 本轮真实数据：`outputs/vae_unified_verified_20261115/`；轻量报告/可视化已提交，原始Zarr和轨迹留本地。运行时基于提交f886530的未提交实现，后续修复记录/采样/导出问题，报告不把最终提交冒充实测时的clean源码。
- 当前实现提交见 `git log -1`；未启动200K、GPU推理或训练；未修改旧CoDL或docs。

唯一下一步：用户依据实测缺口决定是否给天然植物更长观察/更多天然场景及熔岩更多有效空间，并处理初态重建差异；在这些证据不足时不启动正式200K。

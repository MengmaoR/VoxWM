# VoxWM 执行守则

当前唯一活动数据生成入口是 `python -m vae_data` → `vae_data/sharded/`。科学合同为 [VAE_DATA_COLLECTION_PROTOCOL.md](VAE_DATA_COLLECTION_PROTOCOL.md) / [当前Drive Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view)，最新用户指令优先。11/12 Manifest的模型架构不在本轮修改范围。

- 本轮只实现数据生成，不做模型、训练器、DataLoader、模型评价或新算法研发。
- 当前规模100K，100个严格1K片；每片Static500，D1/D2/D3/D4各125，动态每类25条完整五状态链。D4是Water–Lava Solidification，完全替代历史Vegetation，不自动恢复旧200K配方。
- 正式样本保存真实同一时刻48³(27个世界对齐16³)，目标与动态判定仅中心 `[16:32]^3`。邻块变化不能加候选、拼链或重置2个sim秒无变化超时。
- Dynamic最多10个中心不同候选，随机连续五状态窗口，不足则拒绝。D4必须完整中心Pre–Reaction–Post，物理帧统一去重，成功场景的Anchor独立保存且不计入1K。
- Static每片25主要Biome各20，Surface8/ElevatedComplex4/Shallow4/DeepCave4，真实Transition尽量每组4且为附加属性。不拼接、不拿其他群落或类别补缺。
- 全局uint16 Combined Token=(Node Name,raw param2)，词表只追加；uint8 light辅助。48³无损存储，中心16³仍是VAE目标。World Seed固定80/10/10 split，不跨split。
- 纯Voxel：从原生入口禁非技术实体、掉落及实体引起的Voxel变化；技术player不可见无交互。当前falling-entity动态不支持，出现即排除该rollout，不默默冻结物理后发布数据。
- 使用Craftium/VoxeLibre原生液体、固化与火。不写流体轨迹，不改规则/ABM/黏度/采样时钟，不加速时间。几何可达性只筛源点，不能冒充native事件证据；初始化操作不能计native演化。
- 低缓存：Static立即提交或丢弃；Dynamic保留至多10候选和必要Anchor/事件引用；不写完整raw轨迹、不积累Discovery Cache。分片独占写入，全局词表短锁，完成QA后原子READY；READY片只读。
- 本轮仅授权有限原生小例：每类最多2条成功链，总计最多8。已执行一轮每类2次尝试、总墙钟300秒上限；结果实报，不追加原生动态追成功。正式1K/100K、广域扫描及训练一律由用户在服务器手动启动。
- 科研与效率优先：简单可读，不建平台、复杂服务、冗余预检、hash审计或无关重构。只做相关快速Unit/语法/CLI检查，通常累计30秒内；测试Fixture绝不能当Native数据或物理证据。
- 保留 `vae_p0/`、`vae_p1/`、`vae_pilot/`、旧Unified源码/README/协议快照、所有旧结果/日志与docs。旧入口 `vae_data.legacy_entry` 仅历史参考，不混入当前格式。旧CoDL只读，不修改或推送。
- 唯一开发远端 `MengmaoR/VoxWM` / master。每次编码前检查并保留用户改动、`git pull --ff-only`。正常commit/push，不强推、不重写历史，不触碰展示站部署/权限。
- 交付说明实际Native、Unit及未验证功能，写明真实缺口。实现完成即停止，等待用户手动运行。

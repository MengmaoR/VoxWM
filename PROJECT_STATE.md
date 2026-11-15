# VoxWM 当前状态

**VAE 100K / 1K Shard Generator — IMPLEMENTED / NATIVE INCONCLUSIVE**。

- 当前入口 `vae_data/sharded/`；[当前协议](VAE_DATA_COLLECTION_PROTOCOL.md)、[手动服务器命令](vae_data/README.md)、[本轮实测](vae_data/SHARDED_RESULTS.md)。100片×1K，每片Static500+四动态各125，D4为水岩浆固化；旧200K/植物规则仅历史。
- 已实现完整48³上下文、中心16³动态、最多10候选/连续五状态、D4重叠事件注册与独立Anchor、Static群落地形配额、全局append-only词表、多CPU分片、恢复、独立QA和原子READY。
- 本轮唯一原生测试：总墙钟上限300秒，实际约140秒；每类2次尝试，D1成功2条，D2/D3中心2秒无变化而拒绝，D4-A几何筛选无解，D4-B响应JSON错误。未把初始化、邻块变化或Fixture凑成成功。
- JSON响应改为原子交接并补充离线检查，未追加native验证。D4-C、事件重叠、配额与READY等有Unit覆盖；正式分片并行生成的物理吞吐、完整25群落与1K/100K未实跑。
- 原生小例 `outputs/vae_sharded_smoke_20261115/`，轻量报告在 `vae_data/reports/sharded_20261115/`。47项Unit通过；真实成功样例为两条完整五状态48³，原生版本/零实体证据在报告中。
- 旧P0/P1/pilot、旧Unified源码/报告、所有数据日志及docs保留；只切换默认入口，旧 `__main__` 保存在 `legacy_entry.py`。没有模型/训练器/DataLoader或GPU运行。

唯一下一步：用户先查看本轮失败与2秒停止规则带来的实际产出缺口，决定在服务器手动验证/生成；Codex不自动启动正式1K或100K。

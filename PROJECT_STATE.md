# VoxWM 当前状态

**Fixed24 Clock VERIFIED / Full Native Physics Equivalence INCONCLUSIVE**。

- 已实现所有六型人工干预仅中心14³，放置前后复核，后续传播仍完整48³。D3超时120 Tick，其他48 Tick；原16³网格、100K/1K配方、五状态合同不变。
- 固定Craftium的opt-in时序补丁已编译；Server/Client/Lua均固定1/24、post-update捕获。只读Native验证经历一次准备阶段错位及明确修复，复核连续Tick0..240通过，合计28.96秒，未重开六型。
- 含完整Trace实测74.26 Tick/s、3.09 sim s/s；Python/child峰值RSS分别88.5/627.4 MB。241个真实状态，230个零delta Tick；邻域水流每24 Tick更新，中心没有有效动态，不计成功链。[报告/性能/命令](vae_data/reports/fixed24_20261116/README.md)。
- 62项快速CPU测试、Python/Lua/Shell语法与CLI通过。完整数据在`outputs/vae_fixed24_clock_20261116_fixed/`；旧六型GT/Viewer/报告不改。新格式v2不混合历史variable-step数据。
- 完整原生初态/RNG尚未快照；Lava/Fire新时序、跨吞吐严格voxel等价及RGB渲染相位未验证。未运行新六型、正式Shard、100K或训练。

唯一下一步：用户决定是否授权固定初态/受控RNG的Water/Lava/Fire跨吞吐有限验证；Codex停止，不自动追加实验。

# VoxWM 当前状态

**INCONCLUSIVE — VAE Dynamic Scene Sampling & Intervention Planning Revision**。

- 已取消通用天然目标体素定位；真实Surface/Cave、中心六连通Air≥65、A/C稳定流体≥19后才执行完整合法源点规划。现有16³对齐、48³ Assembly、100K/1K配方、五状态/2秒规则保持不变。
- 本次新授权的一轮原生CPU验证已结束：252.49秒、10次尝试；D1、D2、D4-A、D4-C各成功一条，D3和D4-B仍有缺口。包含真实Cave成功，不以合成结果或前轮成功补缺。[实际结果、预览与性能](vae_data/reports/scene_sampling_20261116/README.md)。
- 50项快速CPU测试及语法/CLI通过。完整GT在`outputs/vae_scene_sampling_20261116/`（不入Git），六型可复用已有Native-State Viewer；仅补相对干预时间。历史报告与输出不覆盖。
- 已有GT全点离线测量约0.27–0.64秒，峰值单进程RSS119 MB；本轮采集进程RSS峰值132 MB，不含引擎。正式1K、100K、完整群落/高度产率及多worker真实吞吐尚未验证。

唯一下一步：用户查看本轮六型GT，决定如何处理D3结构传播与D4-B五状态缺口；Codex不再追加原生验证或正式采集。[手动生成/恢复/检查](vae_data/README.md)。

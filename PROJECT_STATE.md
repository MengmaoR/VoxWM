# VoxWM 当前状态

2026-11-14：**VAE P0 / NO-GO**。真实固定世界坐标 16³、Combined Token 和 raw light 采集闭环已实现；尚未打通全部四类动态，不进入 200K。

- 权威：Drive `VoxWM_VAE_PILOT_DATA_PROTOCOL_2026-11-13.md` 数据协议优先于 11 月 12 日 Manifest；实现 `vae_p0/`，结果 `VAE_P0_RESULTS.md`。
- 实际原生 CPU 测试：seed 191664964，墙钟 262.43 秒；最终 26 snapshots：Ordinary 10、D1 Water 8（1 clip）、D2 Lava 0、D3 Fire 8（1 clip，单次随机点火后原生传播）、D4 Vegetation 0。
- 327,680 节点与原生世界逐项核对；72,712,192 voxel 编码/解码比对通过；观察到的 Combined Vocabulary K=128。Ordinary 覆盖 3/25 biome 分组；实际地层 surface4、air2、shallow2、deep2，未取得 elevated 样本。
- D2/D4 各执行一次有界修复后停止。D4 的 8 状态衰退片段不算生长；另一 Chunk 只观察到一次藤蔓新增、2 状态，未达最低5状态。失败诊断完整保留，不能用它们补配额。
- 本地最终数据 `outputs/vae_p0_20261114_final/`；原始试采和启动失败副本保留于 outputs/，日志在 `runs/vae_p0/`。仓库仅提交轻量统计和 PNG/GIF/JSON；旧 `vae_pilot/` 报告保留但不代表新协议 200K 入口。
- 独立原生 GT/Test A/seed42/游玩入口保留；既有 docs/ 和旧 CoDL 未修改。未运行 200K 或 GPU 训练。

唯一下一步：用户审阅 NO-GO 报告，决定是否另行授权针对自然岩浆流和竹/藤生长的有限采集研究。本轮停止，不自动追加实验或更改配额。

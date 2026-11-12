# VoxWM 当前状态

2026-11-13：VAE Pilot 16³原生采集流程已实现，并完成有界真实CPU测试；未启动200K生成或训练。

- 已保留独立GT环境/Test A/原生seed42/游玩入口；原有docs/不改动。
- `vae_pilot/`实现随机世界、25主要biome分组均衡、Water/Lava/Fire/Vegetation原生变化发现、5–8不同状态进度采样、去重与seed级split，以及实际产出/缺口/成本统计。固定总配额200K。
- 已编译安装固定版本Craftium/VoxeLibre，不含Torch/CUDA或模型权重。
- 真实小测试：seed191664964，4个位置、96.357秒观察模拟时间、133.830秒世界墙钟时间，137独立snapshots：Ordinary89、Water48（6个8状态标准clip）；Lava/Fire/Vegetation有效snapshot均0。25组中覆盖5组；完整200K成本与世界数无法可靠估算。
- 首次试跑因换位置后的C++裁剪延迟停止；修复后只接受metadata和观测共同覆盖的同一世界chunk，不把窗口移动记为动态。后续修正火烧植物不算生长，最终实测如上。所有本地原始输出留在outputs/与runs/；仅轻量统计报告随源码发布。
- `MengmaoR/VoxWM`为唯一开发远端；旧CoDL仓库未改动。

下一步：用户检查实测报告，决定是否进行更长的有界自然事件采集；三类零有效产出的缺口不自动补造或改配额。200K入口仅供用户在确认后手动运行。

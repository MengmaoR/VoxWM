# VoxWM 当前状态

**当前只推进 E00-Masked。Plain E00-A/B 归入历史，保留已有产物。**

- `outputs/vae_100k_clip1` 与 `outputs/vae_100k_clip2` 各有 100,000 个已完成样本。VAE 在读取时按 `(Node Name, raw param2)` 对齐词表，并使用中心 16³ 和原 world-seed split。
- E00-Masked 使用固定的 160 状态训练子集和独立的 160 状态验证子集。重建目标为 `0.50 Static CE + 0.25 Dynamic-Unchanged CE + 0.25 Dynamic-Changed CE`，另加 `1e-6 × raw KL`。
- 现有 Masked 小样本结果只用于 sanity check。新训练流程保存分项 loss 曲线、最佳 checkpoint 和早停状态，独立网页读取日志。训练配置、运行和监控命令见 [Masked 使用说明](voxel_vae/README.md)。
- 本次更新验证范围为本地合成日志、曲线和 CLI。真实 GPU 下的早停与续跑待用户验证。

下一步由用户手动启动或续跑 Masked，并查看独立验证曲线和最佳 checkpoint 的重建。

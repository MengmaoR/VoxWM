# VoxWM 当前状态

**当前阶段为 E01-Masked 正式数据训练。E00 的小样本实验保留作历史。**

- 两个 `vae_100k_clip` 各有 100,000 个已完成状态。E01 用两库完整 `train` split，原 world-seed 划分和 canonical 词表映射沿用。
- E01 复用固定 3D ResNet VAE、类别平衡采样和 Masked loss。默认 bf16、最多 20,000 步，每 1,000 步在固定 1,280 状态的独立验证集上评价和选 checkpoint。160 条训练诊断状态只用于评估。
- 独立 `e01` 入口复用 loss 曲线、网页监控和早停。输出写入 `outputs/voxel_vae/e01/phase_masked/`。
- E01 支持在 GPU 2、3 上用 DDP 合跑一份训练，每步全局 batch=64。单卡命令仍可用，输出记录实际进程数。
- 本地用 CPU/Gloo 合成测试验证双进程的参数更新、早停与 checkpoint。真实 GPU/NCCL 运行及提速待用户在服务器验证。

下一步由用户复用已准备的 E01 目录，或首次准备后手动启动双卡训练。[运行命令](voxel_vae/README.md)

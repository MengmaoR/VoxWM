# VoxWM 当前状态

**当前阶段为 E01-Masked 正式数据训练。E00 的小样本实验保留作历史。**

- 两个 `vae_100k_clip` 各有 100,000 个已完成状态。E01 用两库完整 `train` split，原 world-seed 划分和 canonical 词表映射沿用。
- E01 复用固定 3D ResNet VAE、类别平衡采样和 Masked loss。默认 bf16、最多 20,000 步，每 1,000 步在固定 1,280 状态的独立验证集上评价和选 checkpoint。160 条训练诊断状态只用于评估。
- 独立 `e01` 入口复用 loss 曲线、网页监控和早停。输出写入 `outputs/voxel_vae/e01/phase_masked/`。
- 当前完成本地实现。真实数据准备、GPU 训练和最佳权重评估由用户在服务器执行。

下一步由用户执行 E01 数据准备并核对实际训练池数量，然后手动启动训练。[运行命令](voxel_vae/README.md)

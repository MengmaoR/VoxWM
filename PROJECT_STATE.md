# VoxWM 当前状态

**当前阶段为 E01-Masked 正式数据训练。E00 的小样本实验保留作历史。**

- 两个 `vae_100k_clip` 各有 100,000 个已完成状态。服务器索引记录 159,726 条 train、19,620 条 validation、20,654 条 test。E01 沿用原 world-seed 划分和 canonical 词表映射。
- E01 复用固定 3D ResNet VAE、类别平衡采样和 Masked loss。默认 bf16、最多 20,000 步，每 1,000 步在固定 1,280 状态的独立验证集上评价和选 checkpoint。160 条训练诊断状态只用于评估。
- 独立 `e01` 入口复用 loss 曲线、网页监控和早停。输出写入 `outputs/voxel_vae/e01/phase_masked/`。
- E01 支持在 GPU 2、3 上用 DDP 合跑一份训练，每步全局 batch=64。单卡命令仍可用，输出记录实际进程数。
- GPU 2、3 的 E01 已跑完 20,000 步，耗时约 4 小时 32 分。最低验证总 loss 为 0.287724，出现在第 19,000 步。第 20,000 步的验证总 loss 为 0.288398。
- 续训支持 `--resume --epochs 2`，从最后 checkpoint 启动全覆盖采样。每个完整 epoch 有 2,496 步，每条训练状态恰好参与一次。新采样方式的真实训练由用户手动执行。

下一步由用户复用现有 E01 目录和 checkpoint，手动启动双卡续训。[运行命令](voxel_vae/README.md)

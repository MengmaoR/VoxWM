# VoxWM 当前状态

**E01-Masked 已在 30,000 步结束续训。E00 的小样本实验保留作历史。**

- 两个 `vae_100k_clip` 各有 100,000 个已完成状态。E01 的新划分为 160,000 条 train、20,000 条 validation 和 20,000 条 test。原 train 和 validation 保留，从旧 test 按完整 world seed 分别补入 274 条和 380 条。新划分通过 `split_160k.json` 启用，沿用原数据和 canonical 词表映射。
- E01 复用固定 3D ResNet VAE、类别平衡采样和 Masked loss。默认 bf16、最多 20,000 步，每 1,000 步在固定 1,280 状态的独立验证集上评价和选 checkpoint。160 条训练诊断状态只用于评估。
- 独立 `e01` 入口复用 loss 曲线、网页监控和早停。输出写入 `outputs/voxel_vae/e01/phase_masked/`。
- E01 支持在 GPU 2、3 上用 DDP 合跑一份训练，每步全局 batch=64。单卡命令仍可用，输出记录实际进程数。
- 新划分下每个完整 epoch 有 2,500 步，每条训练状态恰好参与一次。E01 从 20,000 步续到 25,000 步，随后在 27,500 步触发早停。最后一轮关闭早停，在 GPU 4、5、6、7 上续到 30,000 步。
- 验证总 loss 最低点在 27,000 步（0.253373），Changed CE 最低点在 25,000 步（0.587967），D3 Changed CE 最低点在 29,000 步（1.431195）。30,000 步验证总 loss 为 0.259818，Changed CE 为 0.620825。
- 服务器留有第 27,000 步的 best checkpoint 和第 30,000 步的 last checkpoint。[E01 结果、曲线与版本记录](voxel_vae/reports/e01_masked_20261122/README.md)

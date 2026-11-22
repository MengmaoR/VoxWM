# VoxWM 当前状态

**当前阶段为 E01-Masked 正式数据训练。E00 的小样本实验保留作历史。**

- 两个 `vae_100k_clip` 各有 100,000 个已完成状态。E01 的新划分为 160,000 条 train、20,000 条 validation 和 20,000 条 test。原 train 和 validation 保留，从旧 test 按完整 world seed 分别补入 274 条和 380 条。新划分通过 `split_160k.json` 启用，沿用原数据和 canonical 词表映射。
- E01 复用固定 3D ResNet VAE、类别平衡采样和 Masked loss。默认 bf16、最多 20,000 步，每 1,000 步在固定 1,280 状态的独立验证集上评价和选 checkpoint。160 条训练诊断状态只用于评估。
- 独立 `e01` 入口复用 loss 曲线、网页监控和早停。输出写入 `outputs/voxel_vae/e01/phase_masked/`。
- E01 支持在 GPU 2、3 上用 DDP 合跑一份训练，每步全局 batch=64。单卡命令仍可用，输出记录实际进程数。
- 新划分已启用。每个完整 epoch 有 2,500 步，每条训练状态恰好参与一次。E01 从 20,000 步续到 25,000 步，随后在 GPU 4、5、6、7 上续训并于 27,500 步触发早停。
- 最低验证总 loss 为 0.253373，出现在第 27,000 步。验证 Changed loss 在 25,000 步为 0.587967，在 27,500 步为 0.604236。

下一步由用户从第 27,500 步的 checkpoint 手动执行 `--resume --epochs 1 --no-early-stop`，续训到 30,000 步后检查验证 Changed 和 D3。[运行命令](voxel_vae/README.md)

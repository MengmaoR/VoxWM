# VoxWM 当前状态

**Vox2Pix 正式 100K 数据已采齐。** THU 的 `vox2pix_100k_rgb32` 包含 20,000 个 48³ 场景、100,000 个视角。Surface / Cave / Underwater 分别为 12,000 / 6,000 / 2,000 个场景。地图实测 1,514 种状态，其中 1,424 种出现在替换前的 VAE 完整数据中。[状态覆盖报告](vox2pix_data/reports/vae_coverage_20261126/README.md) · [部署与采集命令](vox2pix_data/THU_SERVER.md)

**VAE 待选项：替换 24 个 Static 场景。** 若后续协议要求不同场景的 48³ 在同一 seed 内严格零重叠，可在 23 对 Static 中各替换较不稀有的一侧，再替换 1 对 Static/D3 中的 Static。当前保留原数据，待用户决定。[候选样本与覆盖模拟](voxel_vae/reports/scene_overlap_20261126/resampling_impact.json)

**VAE 三块训练清单已冻结。** `chunk_selection.jsonl.gz` 保存 200,000 个快照的划分和选块。仅三个 Static Train 快照采用外围三块，其余快照包含中心块。Train 的 480,000 块包含 1,668 种状态，覆盖 Valid 的 1,368 种、Test 的 1,304 种、Vox2Pix 地图的 1,514 种和白模的 1,075 种。20 张导入图保留原始节点名和 `param2`，光照用 `uint8(0)` 占位。替换包仍有 18 对 Static 48³ 重叠。`scene_7680` 与 `scene_18165` 的原始 Vox2Pix Test 地图进入了 VAE Train。[固定清单](voxel_vae/README.md#published-chunk-selection-for-training)

**Vox2Pix RGB32 / 独立 Fog20–30 / Material Ambient0.35：VERIFIED WITH KNOWN NATIVE VISUAL LIMITATION。** 25 项 Python Unit、C++ 策略测试、原生 build、12 距离固定相机校准和原三个场景的 15 帧回放完成。实测 draw32、Fog20/30、Far Plane2000，Server send/generate 均为8 MapBlocks（128格）。48³ Voxel/Camera 与原记录匹配，White/Water/Face/BACKGROUND 数组不变。31–32格和34格标记实际提交；30–34格标记纹理被雾完全遮蔽，但天空/Fog 色差可留微弱均匀色块，水下原生表面边界仍可见。版本 `vox2pix-v3-local48-rgb32-absolute-fog20-30`。[本轮报告](vox2pix_data/reports/final_rendering_32/README.md) · [命令](vox2pix_data/README.md#final-native-rendering-rgb32)。该轮记录的是正式采集前的渲染验证。

**Vox2Pix final32 1K采集及Mac并行短测已完成。** 2 workers完成200场景/1000视角，Surface/Cave/Underwater=120/60/20，有效采集约40分54秒（1467视角/小时）。数据：`outputs/vox2pix_pilot_1k_rgb32/`。用户授权的4/6/8短测每档同30场景/150视角，全部成功完成，端到端约1847/1675/1268视角/小时；该Mac当时负载下推荐4 workers，100K按短测线性约54.2小时，不能套用到THU。各档均有换页。100K逻辑payload约11.62GB，APFS小文件布局外推约35.8GB。原始数据和日志仅本地保留，[测量报告及资源序列](vox2pix_data/reports/parallel_mac_20261124/README.md)已记录。

**E01-Masked 27,000 步验证最优 checkpoint 已完成冻结审计，结论为 FREEZE WITH KNOWN D3 LIMITATION。**

- 两个 `vae_100k_clip` 各有 100,000 个已完成状态。E01 的新划分为 160,000 条 train、20,000 条 validation 和 20,000 条 test。原 train 和 validation 保留，从旧 test 按完整 world seed 分别补入 274 条和 380 条。新划分通过 `split_160k.json` 启用，沿用原数据和 canonical 词表映射。
- E01 复用固定 3D ResNet VAE、类别平衡采样和 Masked loss。默认 bf16、最多 20,000 步，每 1,000 步在固定 1,280 状态的独立验证集上评价和选 checkpoint。160 条训练诊断状态只用于评估。
- 独立 `e01` 入口复用 loss 曲线、网页监控和早停。输出写入 `outputs/voxel_vae/e01/phase_masked/`。
- E01 支持在 GPU 2、3 上用 DDP 合跑一份训练，每步全局 batch=64。单卡命令仍可用，输出记录实际进程数。
- 新划分下每个完整 epoch 有 2,500 步，每条训练状态恰好参与一次。E01 从 20,000 步续到 25,000 步，随后在 27,500 步触发早停。最后一轮关闭早停，在 GPU 4、5、6、7 上续到 30,000 步。
- 验证总 loss 最低点在 27,000 步（0.253373），Changed CE 最低点在 25,000 步（0.587967），D3 Changed CE 最低点在 29,000 步（1.431195）。30,000 步验证总 loss 为 0.259818，Changed CE 为 0.620825。
- 服务器留有第 27,000 步的 best checkpoint 和第 30,000 步的 last checkpoint。[E01 结果、曲线与版本记录](voxel_vae/reports/e01_masked_20261122/README.md)
- 27,000 步 checkpoint 的完整 held-out test 中，Static exact 为 96.72%，Static non-air 为 95.21%，Changed exact 为 85.90%，变化召回为 94.17%，未变化位置稳定率为 98.27%。D3 Changed exact 为 53.56%，包括 3,038 个节点类型错误和 2,876 个仅 `param2` 错误。
- 1,306 个训练中出现的 token 均有可测 embedding drift。160,000 条训练状态的 latent 统计覆盖 16 个活跃通道，7,960 对 test 中相邻动态状态均有非零 latent L2。[冻结报告](voxel_vae/reports/e01_freeze_audit_20261122/report.md) · [统一 train-set normalization](voxel_vae/reports/e01_freeze_audit_20261122/latent_normalization.json)

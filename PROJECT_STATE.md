# VoxWM 当前状态

**E00-A真实GPU sanity overfit已通过；E00-B结果尚待完成检查。E01未授权。**

- 正式数据为`outputs/vae_100k_clip1`和`outputs/vae_100k_clip2`：各100个READY、100,000 samples；每库Static 50,000，D1/D2/D3/D4各12,500，10,000条完整动态链，world-seed split由原metadata固定。
- 两库本地词表分别1,538/1,537项，绝大多数共同语义的local ID不同。`voxel_vae/vocabulary.py`按`(Node Name,raw param2)`排序构建训练canonical vocabulary，并在读取时remap；不改正式Zarr。
- `voxel_vae/dataset.py`只信任`journal.sqlite` committed samples，只读`snapshots.zarr/tokens`中心16³；忽略Zarr未提交尾部、light和D4 anchors，并跨root检查world seed不跨split。
- E00模型为32D embedding、96/192/192 3D ResNet Encoder、`[16,4,4,4]` posterior与对称nearest-upsample Decoder；无skip/attention。loss仅CE + `1e-6` raw KL。
- Trainer支持fp32/bf16、AdamW、effective batch64的32/8/8/8/8平衡采样、microbatch accumulation、clip1.0、step日志、真实validation、严格checkpoint/resume、reconstruction/dynamic/latent diagnostics。
- 固定E00-A 8样本与E00-B 160 train + 160 validation subset由`prepare`生成；A未达到overall和non-Air均99%时，程序拒绝启动B。
- E00-A在真实GPU上完成3000步：overall/non-Air exact均为1.0，CE从7.323916降至0.0000552，16/16 latent channels active，无NaN/Inf；这只证明8样本可过拟合，不是generalization结果。
- `voxel_vae.recon_viewer`可从E00-B最佳checkpoint导出固定validation gallery（8 Static + D1–D4各2条完整五状态链），并以只读3D页面并排显示GT与重建；不使用test split。
- 本地已通过数据/采样6项、模型/反向/checkpoint3项及viewer2项轻量Unit；Codex没有启动GPU训练。

唯一下一步：完成并检查E00-B，然后用最佳checkpoint导出validation reconstruction gallery；未获得新指令前不进入E01。详见[voxel_vae/README.md](voxel_vae/README.md)。

# VoxWM 当前状态

**E00 Voxel VAE代码已实现，真实E00-A/B尚未运行。E01未授权。**

- 正式数据为`outputs/vae_100k_clip1`和`outputs/vae_100k_clip2`：各100个READY、100,000 samples；每库Static 50,000，D1/D2/D3/D4各12,500，10,000条完整动态链，world-seed split由原metadata固定。
- 两库本地词表分别1,538/1,537项，绝大多数共同语义的local ID不同。`voxel_vae/vocabulary.py`按`(Node Name,raw param2)`排序构建训练canonical vocabulary，并在读取时remap；不改正式Zarr。
- `voxel_vae/dataset.py`只信任`journal.sqlite` committed samples，只读`snapshots.zarr/tokens`中心16³；忽略Zarr未提交尾部、light和D4 anchors，并跨root检查world seed不跨split。
- E00模型为32D embedding、96/192/192 3D ResNet Encoder、`[16,4,4,4]` posterior与对称nearest-upsample Decoder；无skip/attention。loss仅CE + `1e-6` raw KL。
- Trainer支持fp32/bf16、AdamW、effective batch64的32/8/8/8/8平衡采样、microbatch accumulation、clip1.0、step日志、真实validation、严格checkpoint/resume、reconstruction/dynamic/latent diagnostics。
- 固定E00-A 8样本与E00-B 160 train + 160 validation subset由`prepare`生成；A未达到overall和non-Air均99%时，程序拒绝启动B。
- 本地已通过数据/采样6项及模型/反向/checkpoint3项轻量Unit；没有读取正式服务器样本、没有启动GPU训练，也没有产生实验数值。

唯一下一步：在训练服务器安装/确认CUDA PyTorch，运行`python -m voxel_vae.e00 prepare`，再由用户手动启动E00-A；只有A通过后运行E00-B。详见[voxel_vae/README.md](voxel_vae/README.md)。

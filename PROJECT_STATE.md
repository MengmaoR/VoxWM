# VoxWM 当前状态

**E00-A/B真实GPU sanity overfit已完成；实现闭环通过，但validation泛化较弱。E01未授权。**

- 正式数据为`outputs/vae_100k_clip1`和`outputs/vae_100k_clip2`：各100个READY、100,000 samples；每库Static 50,000，D1/D2/D3/D4各12,500，10,000条完整动态链，world-seed split由原metadata固定。
- 两库本地词表分别1,538/1,537项，绝大多数共同语义的local ID不同。`voxel_vae/vocabulary.py`按`(Node Name,raw param2)`排序构建训练canonical vocabulary，并在读取时remap；不改正式Zarr。
- `voxel_vae/dataset.py`只信任`journal.sqlite` committed samples，只读`snapshots.zarr/tokens`中心16³；忽略Zarr未提交尾部、light和D4 anchors，并跨root检查world seed不跨split。
- E00模型为32D embedding、96/192/192 3D ResNet Encoder、`[16,4,4,4]` posterior与对称nearest-upsample Decoder；无skip/attention。loss仅CE + `1e-6` raw KL。
- Trainer支持fp32/bf16、AdamW、effective batch64的32/8/8/8/8平衡采样、microbatch accumulation、clip1.0、step日志、真实validation、严格checkpoint/resume、reconstruction/dynamic/latent diagnostics。
- 固定E00-A 8样本与E00-B 160 train + 160 validation subset由`prepare`生成；A未达到overall和non-Air均99%时，程序拒绝启动B。
- E00-A在真实GPU上完成3000步：overall/non-Air exact均为1.0，CE从7.323916降至0.0000552，16/16 latent channels active，无NaN/Inf；这只证明8样本可过拟合，不是generalization结果。
- E00-B完成5000步。训练集overall/non-Air为99.9992%/99.9994%；最佳checkpoint在step500，完整validation为80.4147%/68.6464%，到step5000退化为77.4339%/64.7001%，属于明显过拟合。最佳validation dynamic change recall为9.11%，changed-pair exact为0；E00证明实现可学习，但没有证明160样本上的状态泛化或动态细节保持。
- 词表覆盖不是主要原因：160-train已覆盖validation中99.55%的voxel token实例；validation gallery的94.46%错误发生在训练见过的token上。主要限制是160状态相对23.6M参数过小、训练过久，以及等权逐voxel CE对稀疏动态/非Air细节没有特殊保护（均为E00既定协议）。
- 动态GT变化在相邻pair中仅约占训练voxel的0.23%、validation的0.29%；32/8/8/8/8只平衡sample类别，不会提高变化voxel在CE中的权重。E00没有时间/顺序输入，也没有changed-voxel或temporal loss，因此动态重建尤其差符合当前目标的直接预期，不能把它解释成Dynamics学习结果。
- 已修复模型（含Embedding）在应用seed之前构造的可复现性问题。后续新run会保存`embedding_initial.npz`、报告初始化到最终的row drift，并把dynamic metrics按D1–D4拆分；不改变已有checkpoint或E00 loss。已完成旧run没有step-0 embedding，无法事后精确恢复其初始化漂移。
- `voxel_vae.recon_viewer`可从E00-B最佳checkpoint导出固定validation gallery（8 Static + D1–D4各2条完整五状态链），并以只读3D页面并排显示GT与重建；不使用test split。
- 本地已通过数据/采样6项、模型/反向/checkpoint4项及viewer2项轻量Unit；Codex没有启动GPU训练。

唯一下一步：由用户决定是否基于E00证据设计正式E01数据规模与训练协议；未获得新指令前不启动E01或修改既定VAE。详见[voxel_vae/README.md](voxel_vae/README.md)。

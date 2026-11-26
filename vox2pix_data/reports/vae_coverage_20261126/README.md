# Vox2Pix 100K 与 VAE 的 voxel 状态覆盖

比较单位是精确的 (node_name, raw param2)。分母为在实际数据中出现过的状态。一个 48³ 地图由五个视角共享，只计一次。白模按像素计数，排除 BACKGROUND。面朝向和水层数是独立的渲染字段。

THU 的正式目录为 `/data/disk2/txj/vox2pix/vox2pix_100k_rgb32`。全部 20,000 个场景和 100,000 个 640×360 视角均已读取，Surface / Cave / Underwater 分别为 12,000 / 6,000 / 2,000 个场景。存储报告记录压缩后的逻辑数据量为 11,888,826,264 字节。原生注册节点名称有 2,517 个。地图实际使用 422 个名称，可见白模使用 337 个名称。

| 状态集合 | 48³ 输入地图 | 白模可见表面 |
|---|---:|---:|
| 实际 token 种数 | 1,514 | 1,075 |
| 在 VAE 完整 48³ 数据中出现 | 1,424 | 1,051 |
| 有冻结 checkpoint 已训练输入 embedding | 1,241 | 1,002 |
| VAE 数据中出现、旧输入 embedding 未训练 | 183 | 49 |
| Vox2Pix 有、VAE 数据中缺失 | 90 | 24 |
| VAE 数据有、Vox2Pix 此层缺失 | 154 | 527 |

VAE 共有 1,580 行 32 维输入 embedding。全量 200K 个普通 48³ 快照中出现 1,578 种，原 E01 的 27k 最佳 checkpoint 中有 1,306 行满足训练中心频率大于零且相对初始化的 L2 变化大于 1e-6。VAE 全量与 Vox2Pix 地图的并集为 1,668 种。两行全库零频的历史 ID（1115、1161）在 Vox2Pix 也没有出现。

- 地图共有 2,211,840,000 个体素位置。90 种 VAE 词表外状态共出现 481 次。其中 40 种为 VAE 已有节点名的其他 param2，另外 50 种来自 38 个 VAE 从未见过的节点名。
- 100,000 张白模共 23,040,000,000 个像素。BACKGROUND 占 7,240,786,344 个，有效表面像素为 15,799,213,656 个。24 种 VAE 词表外状态覆盖 219,358 个有效像素。例如 `mcl_deepslate:deepslate_with_redstone_lit`、param2=0 在地图中出现 3 次，在白模中对应 102,331 个像素。
- [逐状态计数](token_counts.csv) 给出两层 Vox2Pix 次数、VAE 体素次数、canonical ID 及其训练状态。[汇总](coverage.json) 给出各集合种数与出现次数。

## 复现

在可读取正式数据的主机上，从 VoxWM 仓库根目录导出两个计数文件。第一次读取 3D 地图，第二次读取白模。输出仅包含词频和数据规模。

```bash
DATASET=/data/disk2/txj/vox2pix/vox2pix_100k_rgb32
OUT=outputs/vox2pix_vae_coverage
mkdir -p "$OUT"
.venv/bin/python -m vox2pix_data.token_inventory "$DATASET" --layer scene --workers 4 > "$OUT/scene.json"
.venv/bin/python -m vox2pix_data.token_inventory "$DATASET" --layer surface --workers 4 > "$OUT/surface.json"
.venv/bin/python -m vox2pix_data.vae_token_overlap --scene "$OUT/scene.json" --surface "$OUT/surface.json" --output "$OUT"
```

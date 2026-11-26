# Dynamic 保留一个稀有度名额后的词表覆盖

合并全部划分。200,000 个完整 48³ 普通快照包含 100,000 帧 Static 及来自 20,000 条五帧链的 100,000 帧 Dynamic。每帧保留中心 16³。Static 取稀有度前 K 个外围块。Dynamic 先取动态排序前 K−1 个，再取当前帧剩余外围块中稀有度最高的一个。块位置重复时，稀有度顺延，确保每帧总共 K+1 个不同块。动态分数为五帧间发生 combined token 变化的相邻帧对数，取值 0–4。

下表的遗漏分组依据 token 在完整 48³ 中的 Static / Dynamic 归属。ID 1115 和 1161 在全部普通快照和已审计 D4 anchors 中均未出现，单独计入“全库零频”。

| 外围 K | 每帧块数 | 总块数 | Static 种数 | Dynamic 种数 | 合并种数 | 漏仅 Static | 漏仅 Dynamic | 漏共有 | 全库零频 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 200,000 | 1,176 | 1,053 | 1,333 | 136 | 47 | 62 | 2 |
| 2 | 3 | 600,000 | 1,407 | 1,388 | 1,578 | 0 | 0 | 0 | 2 |
| 3 | 4 | 800,000 | 1,407 | 1,390 | 1,578 | 0 | 0 | 0 | 2 |
| 4 | 5 | 1,000,000 | 1,407 | 1,393 | 1,578 | 0 | 0 | 0 | 2 |

三个 K 均覆盖全部 1,578 种实际出现的 token，包括原先仅在外围出现的 245 种。ID 6 的三次体素出现均入选。以各类别自身的完整快照为参照，Static 在三个 K 下均覆盖全部 1,407 种。Dynamic 的 1,394 种中分别有 6、4、1 种仅由选中的 Static 块补足，均属共有类别。

## 复现

从 VoxWM 根目录运行。

```bash
PACKS=/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category
.venv/bin/python -m voxel_vae.outer_chunk_audit --mixed \
  "$PACKS/vae_100k_static.tar.zst" "$PACKS/vae_100k_dynamic.tar.zst" \
  --output outputs/voxel_vae/outer_chunk_mixed_selection_20261126_rerun/coverage.json
```

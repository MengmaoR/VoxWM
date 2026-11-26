# Vox2Pix 缺失状态与 VAE Static 替换模拟

在 THU 正式 100K 数据的 **20,000 张独立输入地图**里，90 种 VAE 未见的 `(node_name, raw param2)` 共出现 **481 次，集中在 93 张地图、93 个世界种子**。每张地图的五个 RGB 视角只计一次。93 张的完整语义频数分布也各不相同。其中 18 张在中心 16³ 命中，合计 32 个体素、15 种状态。[原始全库词频](../vae_coverage_20261126/token_counts.csv) · [93 张逐图明细](maps.csv)

## 高度与群落

地图中心取实际 ENU 原点加 (24,24,24)，高度范围为 −58 至 47。原始地表测量值为 −25 至 72。以下分配使用原生记录的地表群落和参考高度。表内先检查整个 48³ 的绝对高度，再检查相对高度，所以各列互斥。

| Vox2Pix 场景 | 地图 | 目标状态种数 | 出现次数 | 地表 | 高空复杂 | 浅层地下 | 深层洞穴 | 高度间隙 | 低于下限 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Surface | 11 | 69 | 271 | 11 | 0 | 0 | 0 | 0 | 0 |
| Cave | 72 | 36 | 191 | 1 | 0 | 12 | 19 | 7 | 33 |
| Underwater | 10 | 8 | 19 | 1 | 0 | 2 | 0 | 0 | 7 |
| **总计** | **93** | **90（去重）** | **481** | **13** | **0** | **14** | **19** | **7** | **40** |

VAE 允许中心高度 C 为 −38 至 170，整个窗口在 [−62,193] 内。若地表为 S，地表层要求 C−S 在 [−7,8]，高空复杂要求 ≥9，浅层要求 [−24,−8]，深层要求 ≤−32。7 张在绝对范围内且 C−S 为 −31 至 −25。40 张越界的中心高度为 −58 至 −39。地表原生 biome 名按 VAE 的别名表归组，脚下群落另存。使用中心真实 Air/Void 占比和全图 Ignore 筛选。水下按下沉后的真实中心计算，逐图核对了检索和最终水平坐标。

93 张覆盖 25 个群落组中的 23 个，以 Jungle（8）、Plains（7）、FlowerForest 和 Mesa（各 6）最多。MangroveSwamp、MesaBryce 各 0 张。仅三张 Surface 地图就含 50 种互异目标状态，分别为 Forest 的 scene_7680（26 种）、BambooJungle 的 scene_18602（23 种）、Swampland 的 scene_15656（20 种）。[25 群落 × 各高度完整计数](biome_height.csv)

## 一换一配对

在 100,000 个原 Static、200 个重打包分片中查找同一群落组和高度层的候选，每个旧样本最多用一次。先在全部可匹配的历史重叠对与配额间求最大匹配。同样能消除最多重叠时优先移除较不稀有的端点。剩余位置从各自配额格按整张 48³ 稀有度由低到高补入。稀有度使用替换前 Static＋Dynamic、所有划分的全图总词频，依次比较完整的全局词频向量、图内次数、canonical ID。表中两列是分别从同一原始数据出发的独立模拟。

| 结果 | 使用原生记录的群落与参考高度 | 附加中心列对地表的实物佐证 |
|---|---:|---:|
| 可配对地图 | 46 | 19 |
| 配对涉及的群落 × 高度格 | 34 | 17 |
| 移除重叠场景 / 同格最低稀有度场景 | 8 / 38 | 5 / 14 |
| 剩余旧重叠对（原有 24 对） | 16 | 19 |
| 新产生的同 seed 空间重叠 | 0 | 0 |
| 48³ 新增全库状态 / 体素出现次数 | 87 / 398 | 78 / 325 |
| Static 实际覆盖（原 1,407） | 1,527 | 1,511 |
| Static＋Dynamic 实际覆盖（原 1,578） | 1,665 | 1,656 |
| 丢失的原有 Static / 全库状态 | 0 / 0 | 0 / 0 |
| 中心 16³ 实际纳入的新增状态 | 13 | 8 |
| 中心＋两个最稀有外围块实际纳入的新增状态 | 83 | 74 |

46 张里，19 张能在原图中心柱找到与记录吻合的地表（13 地表、6 浅层）。22 张的参考地表在保存的 48³ 外。另 5 张的中心柱在记录地表以上仍有水。Vox2Pix Cave 的调查以地形顶面为参考，VAE 的原生地表查询还计入液体。对这 27 张，原记录方案是条件模拟，不能将其全部解释为独立核实过原 VAE 地表和群落。即使地表不在窗口内，测得的地形高度仍可为洞穴深度提供下界。水面与记录的海底地表不同时，表中保留原始测量值和实测下界。[逐图证据](maps.csv) · [一对一方案与原样本 ID](replacement_plan.csv)

两种方案的每个 Static 配额格数量都保持，Dynamic 原样保留。93 个新 seed 与原 Static/Dynamic 场景 seed 的交集为空。两个方案都消除了原有的 Static/D3 跨类重叠。群落交界属于软目标，Vox2Pix 未保存 VAE 的全区域多点群落记录。两个方案分别移出 33 和 15 个已标记交界的 Static。即使将新样本的交界贡献计为零，原本达到每分片每群落 4 个交界目标的格子仍全部达标。

外围选块一行按各自替换后的全库词频重排入选的 Vox2Pix 地图。原样本未出现的 90 种只能由这些新图贡献。临时将新增语义按名称和 param2 排序附在既有 ID 后，仅用于确定排序并列项。上述种数反映数据覆盖，实际 embedding 权重没有更新。[90 种状态的逐项覆盖](token_coverage.csv)

## 三种受绝对高度限制的状态

| 状态 | param2 | 出现次数 | 所在地图中心高度 |
|---|---:|---:|---|
| mcl_deepslate:deepslate_with_redstone_lit | 0 | 3 | −51（3 张） |
| mcl_core:stone_with_redstone_lit | 0 | 4 | −57、−49、−46、−41 |
| mcl_fire:fire | 90 | 1 | −47 |

这三种的所有原图都在现行高度范围外。使用现存完整 48³ 地图并保持高度规则时，87 是可新增种数的上限。

## 原分片格式兼容性

**直接按原采集协议原样写入的数量为 0。** 原 VAE Static 要求中心 16³ 的水平原点在全局 16 格网格上，93 张候选均未同时满足东、北两轴对齐。Vox2Pix 原始地图保存 Node ID 和 raw param2，缺少原 VAE 快照的逐体素 light 通道。这次只模拟 token 训练输入及同类别替换，保留实际位置、原词表、权重和数据包。[Static 原点约束](../../../vae_data/sharded/native.py) · [Vox2Pix 原始字段](../../storage.py) · [可机读汇总](summary.json)

## 复现

从 VoxWM 仓库根目录执行。第一条 Python 命令在可只读访问正式 Vox2Pix 数据的主机运行，后两条使用本地 VAE Static 压缩包。中间文件仅含计数和元数据。

```bash
OUT=outputs/vox2pix_substitution_20261126
mkdir -p "$OUT"
.venv/bin/python -m vox2pix_data.rare_map_inventory \
  /data/disk2/txj/vox2pix/vox2pix_100k_rgb32 \
  --coverage-csv vox2pix_data/reports/vae_coverage_20261126/token_counts.csv > "$OUT/maps.json"
.venv/bin/python -m vox2pix_data.static_replacement catalog \
  /Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category/vae_100k_static.tar.zst \
  "$OUT/static.jsonl.gz"
.venv/bin/python -m vox2pix_data.static_replacement simulate \
  "$OUT/maps.json" "$OUT/static.jsonl.gz" "$OUT/report"
```

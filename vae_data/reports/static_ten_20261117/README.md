GO — 用户要求的10个不同群落Static场景采齐，已接入原预览。

# Ten-biome Static / 2026-11-17

真实Native采集107.29秒，15次位置尝试、10个成功。Surface/Elevated/Shallow/Deep为4/2/2/2；10个Biome Group不同、10个中心高度C也互不重复。本次不执行“每个群落独立8/4/4/4”；不创建正式Shard，不重跑动态。使用现有主动群落定位、连续高度、Native Capture与验证，采样RNG seed=42。

| Biome Group | 实际Native Biome | Stratum | 实际S | 中心C | 中心Air |
|---|---|---|---:|---:|---:|
| Plains | Plains | surface | 15 | 9 | 54.956% |
| Desert | Desert | surface | 21 | 28 | 88.037% |
| Swampland | Swampland | surface | 6 | 11 | 71.948% |
| MushroomIsland | MushroomIslandShore | surface | 1 | 1 | 43.750% |
| MegaTaiga | MegaTaiga | elevated_complex | 44 | 61 | 96.143% |
| Forest | Forest | elevated_complex | 8 | 18 | 90.991% |
| Taiga | Taiga | shallow_underground | 5 | -10 | 3.076% |
| Jungle | Jungle | shallow_underground | 1 | -12 | 35.400% |
| Mesa | Mesa_sandlevel | deep_cave | 1 | -36 | 3.711% |
| IcePlains | IcePlains | deep_cave | 19 | -30 | 0.000% |

所有样例完整48³在世界高度边界内，中心起点均不是16的倍数；完整Node/light/param2按同Tick独立P0/VoxelManip逐体素对照。每个样例仅一个真实时刻，48³上下文及中心16³可切换，Native实际坐标、时间、种子和词表保留。

5次未接受候选均保留原因：两次引擎连接关闭、两次RoofedForest高处搜索耗尽、一次Plains定位后实际为SunflowerPlains。未改Voxel或降低条件。Tree-biome优先，仅为本次小样例保留两个Elevated名额；不足时换未使用群落，不修改正式采集配额。

## 预览

[原8766预览，已打开Static](http://127.0.0.1:8766/?type=Static&attempt=scene_03_Plains_surface&region=48&run=static_ten_20261117)。选择Static后，Scene下拉切换10个群落；左侧显示S、C、Stratum和Air/Void。可使用16³/48³、剖切、Inspector及自由视角。原D1–D4和旧D2保留，未覆盖原始GT。

新原始数据：`outputs/vae_static_ten_20261117`，每个`traces/Static/scene_*`保留无损单帧Trace、light、metadata及局部Combined Token词表。预览导出在该目录的`viewer/Static/`，并将派生数据追加到原`outputs/vae_fixed24_native_20261117_fixed_no_d2/viewer/Static/`；原动态Trace和词表不改写。outputs不入Git，本报告仅发布轻量实测摘要。

## 检查

- 两项针对性Unit（Static导出/追加、旧D2保留）通过，Python/JS语法与CLI解析通过。
- 全部10个Viewer体素与各自原始Trace逐体素完全相等，坐标与C一致。
- Playwright桌面1440×1000实测十场景切换、地下中心切片/Inspector以及旧D2历史标签通过；应用console error为0。Browser plugin不可用，使用已安装Playwright；仅有截图ReadPixels驱动性能提示。本轮未做移动端矩阵。
- 桌面截图保存在`/tmp/voxwm_static_ten_surface.png`、`/tmp/voxwm_static_ten_deep.png`；浏览器检查详情`/tmp/voxwm_static_ten_preview_qa.json`。

## 用户重复运行（不自动执行）

```bash
# 新目录；上限40次位置尝试/600秒，成功10条即停。
.venv/bin/python -m vae_data static-ten \
  --dataset outputs/vae_static_ten_new --wall-seconds 600 --renderer software
.venv/bin/python -m vae_data view --dataset outputs/vae_static_ten_new --port 8767
# 浏览器 http://127.0.0.1:8767/?type=Static
```

macOS可改为`--renderer auto`。`--preview-dataset EXISTING`可追加至尚无Static分类的旧预览，已有Static时明确拒绝覆盖。正式1K/100K、25群落完整配额和大规模吞吐未在本轮运行。

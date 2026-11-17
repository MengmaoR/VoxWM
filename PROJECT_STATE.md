# VoxWM 当前状态

**10个不同群落的真实Static试采完成，已接入原预览。**

- Surface/Elevated/Shallow/Deep为4/2/2/2；10个Biome Group、10个不同中心高度，107.29秒、15次位置尝试。本轮不要求每群落独立高度配比，不创建正式Shard。[结果表](vae_data/reports/static_ten_20261117/README.md)。
- 新数据`outputs/vae_static_ten_20261117/`；原8766预览增加Static分类和十场景下拉，展示Native S/C、群落与Air/Void。原动态GT及旧D2保留。
- 10/10导出体素与原始GT完全一致；浏览器十场景切换、地下中心切片/Inspector、旧D2标签检查通过，应用console error为0。没有重跑动态或训练。
- 原连续高度/Capture修订保留：完整Region在[-62,193]，Static逐整数高度，Dynamic网格与Fixed24时序不变。[前轮核验](vae_data/reports/static_height_20261117/README.md)。
- 正式1K/100K及25群落全配额吞吐尚未运行；旧数据只读，正式新策略需新输出目录。

唯一下一步：用户在现有预览检查这10个Static样例，再决定是否启动首个正式1K分片；Codex不自动追加采集。

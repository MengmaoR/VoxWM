# GT来源与外部依赖

本目录未复制旧项目的Git历史、已生成数据、模型或Viewer。

- GT场景/采集/稀疏存储迁自 `MengmaoR/LTWM-CoDL`，本次读取的源版本为 `9dbb2ffb1f244209807f7c604cfa98e0c66abc32`。Test A保留原树木、地层、九格悬空水源、河道/池塘及实际鼠标旋转策略。
- 引擎直接依赖 [francelico/craftium](https://github.com/francelico/craftium/tree/ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49)，固定版本 `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`。这是此前GT使用的同一引擎版本。VoxeLibre及原生依赖按该仓库的submodule版本安装，保留它们自己的许可证。
- `voxwm_gt/actions.py` 的纯CPU动作映射和 `assets/voxel_classes.json` 的类别表来自 [固定上游中的数据工具](https://github.com/francelico/PERSIST/tree/b37a211667b9c2fb27174790f517cc7019cf103b)。MIT版权声明保存在 `licenses/action_and_palette_MIT.txt`。这两份本地数据工具不导入或要求原模型工程，不含神经网络、权重或推理代码。
- 既有类别id保留，原生地图遇到类别表之外的node/param2时附加新类别，并写入每次输出的classes.json/classdict.json；没有把未知节点替换为空气。读取不同运行数据时使用各自类别表。

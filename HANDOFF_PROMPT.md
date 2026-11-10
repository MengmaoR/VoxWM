# VoxWM — 新架构研发交接

你将在一个独立的新GitHub仓库研发我们自己的Voxel World Model。

工作目录：`/Users/tianxj/Desktop/VoxWM`。
先读这里的AGENTS.md、PROJECT_STATE.md和README.md。
旧项目`/Users/tianxj/Desktop/CoDL`的baseline测试已完成，旧对话只是暂停工作，没有要求归档；不要修改旧仓库，不继承旧Git历史。

本目录唯一从旧项目迁入的代码是GT相关内容：原生Craftium/Luanti/VoxeLibre环境、真实同步采集、稀疏存储、原Test A合成场景和必要的CPU动作/类别工具。没有迁入PERSIST模型/推理、Viewer、compare、旧结果、权重或论文草稿。

GitHub已连接：`https://github.com/MengmaoR/VoxWM.git`，当前分支master。先检查本地改动，再执行`git pull --ff-only`，保留双方工作。远端原有`docs/`展示页及数据已完整保留，不属于本次GT迁入内容，不修改或重新部署。不要自行创建仓库、连接旧CoDL远端、强推，或用clone/reset覆盖本目录。只向VoxWM正常commit/push。

当前数据能力：
1. `python scripts/test_a.py`：生成原Test A的真实GT与视频。
2. `python scripts/test_seed42.py`：直接由原生v7 mapgen、seed42生成世界，再录制GT与视频，不套用Test A地形。
3. `python -m voxwm_gt generate`：支持合成场景、原生地图、自定义voxel NPZ及外部动作序列。
4. `python -m voxwm_gt play`：实际键鼠游玩，可同步录制。

以上真实测试尚未由Codex运行，不得声称已复现。安装和手动运行命令见README。不要自动启动完整GT、GPU推理或训练。保持C++同帧Voxel/Center、官方ENU及中心Camera约定，不重引入玩家位置估算Center的错误。

采用科研第一、最小实现的工作模式：清晰的Python/PyTorch代码；不做复杂预检、哈希审计、冗余测试、工程平台或无关重构；默认只做一轮不超过30秒的必要本地检查。用户手动运行真实实验，Codex负责实现、简短文档与新GitHub交付。

新架构按我随后提供的方案和实验要求实施。当前独立仓库已接入，请完成GT环境交接，不自行设计模型、不把旧GRU/LTWM/PERSIST直接包装为新架构，也不自动开展新训练。

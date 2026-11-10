# VoxWM 当前状态

2026-11-10：独立GT源码已整理，真实引擎测试尚未执行。

- 从旧项目仅抽取GT环境、Test A场景、同步Voxel/Camera/RGB采集和稀疏存储。
- 已去除旧仓库、PERSIST运行时、GPU依赖、Viewer和compare耦合。
- 提供原Test A和原生v7 seed42两项手动生成测试；支持自定义voxel世界、交互游玩及录制。
- 固定Craftium源码版本，安装脚本直接获取Craftium及VoxeLibre，不克隆模型工程，不下载权重。
- 已通过一轮Python/Lua/Shell语法、CLI和极小稀疏数据读写/类别编码/同Center Camera检查；没有导入或启动Craftium/Torch。
- 未安装/编译引擎，未运行真实GT、视频生成、GPU推理或训练。已连接新远端`MengmaoR/VoxWM`（master），先pull后整合，远端既有`docs/`保持原样。

下一步：用户手动安装GT环境并运行两个测试。新对话在本目录pull新GitHub后，按用户提供的架构方案研发。

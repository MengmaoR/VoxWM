# VoxWM 执行守则

本目录是独立的新架构研发工作区。目前仅迁入原生GT环境，尚无新的世界模型实现或训练结果。

- 科研和效率优先：最小实现、简单可读，不增加哈希审计、预检平台、复杂服务、重复测试或无关重构。
- `/Users/tianxj/Desktop/CoDL` 是已完成baseline的旧项目，只读参考；不修改、不提交、不推送那个仓库，不把旧执行守则当作新科学任务。
- 仅继承GT环境、真实采集、Test A合成场景和必要CPU动作/类别工具；不带入旧模型、推理、Viewer、compare、结果、权重、历史实验及Git历史。
- 支持Test A、原生seed地图和用户自定义voxel场景。真实物理交给Craftium/Luanti/VoxeLibre，禁止用脚本伪造逐帧物理或绝对玩家轨迹。
- 保持C++ `get_voxel_data_cpp` 同帧Voxel/param2/Center，经原wrapper转ENU，49³ right-crop至48³。Camera中心来自真实观测Center，不从玩家位置推测。
- GT可本地CPU运行，RGB需要本地图形栈或Linux Mesa；不依赖Torch/CUDA/模型权重。模型推理/训练由用户手动在服务器执行。
- Codex默认仅实现代码、语法/CLI及必要极小合成检查，不自动安装大型环境、运行真实游戏GT或训练。真实两项测试脚本由用户手动运行。
- 当前 VAE 数据权威为 Drive `VoxWM_VAE_PILOT_DATA_PROTOCOL_2026-11-13.md`，数据规则优先于 11 月 12 日 Manifest。活动最小闭环在 `vae_p0/`；`vae_pilot/` 是旧表示试采历史，仅复用原生接口和 biome/split 工具，禁止把其旧 `--full` 当作新协议 200K 入口。
- 固定世界坐标 16³；单一 Combined Token=VocabularyID(Node Name,raw param2)，uint16；raw light 单独保存且不作模型输入。P0 通过原生 VoxelManip 同帧读取节点/light/param2，不能用 Camera 推断 Chunk 原点。
- 当前任务授权有上限的 CPU 原生 P0：每个失败类别最多一次明确有界修复，不自动启动 200K。Ordinary100K、D1–D4 各25K的配额不改；不足5不同状态不算动态clip，不靠重复帧、人工流体或植物补齐；Fire单次随机点火和点火前状态需记录。全部报告只写实测值。
- 新架构依据用户后续提供的方案实现，不从旧baseline或展示动效推导既定架构，不自动启动模型开发。
- 当前唯一开发远端为 `https://github.com/MengmaoR/VoxWM.git`，分支master。远端既有`docs/`展示页及数据完整保留，不主动修改或部署。每次编码前先安全保留本地改动并`git pull --ff-only`，只向这个远端正常提交/推送；不向旧CoDL仓库或其他展示站仓库推送，不强推。
- 本地检查只做一轮，原则上总计不超过30秒，不为了测试安装大型依赖。交付说明实际执行和未执行的内容。

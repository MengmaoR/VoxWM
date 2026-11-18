# VoxWM 当前状态

**动态观察批传输及SSH无关的后台采集控制已实现，待用户在服务器实测。**

- `advance_batch`默认24拍，可选48；引擎内部每拍保留fixed24同步/post-step真实48³捕获。首帧+RAW稀疏delta与实际Client时钟经现有socket返回；不逐拍写RAW文件。
- Python仍逐拍验证时钟并调用原Collector。48/120拍无变化、10候选、最大观察时长、D4真实事件/五帧与Anchor不变。首次stop后的尾巴不进入候选，有尾巴的世界不复用。
- 不调整并行数；服务器运行示例保持用户实测的64世界。此前Xvfb/词表锁修复保留，已有数据只读，原格式续跑检查不放宽。
- 新监控器通过`start`脱离SSH后台启动；`dashboard`是无采集子进程的独立只读界面，`status`输出JSON，`stop`精确结束本次标记的完整树并核验所有采集锁已释放。Collector同时启用parent-death清理兜底；后台监控器意外消失时，仍可按继承标记清理被PID 1收养的后代，绝不向PID 1发信号。每个活跃Shard单独执行600秒停滞检测，健康Shard不能掩盖卡死Shard。
- 已针对现场mapgen错误修复纯Voxel隔离层：`add_entity`仍不创建真实实体，但返回无副作用代理，避免Witch Hut/Village回调对`nil:get_luaentity()`崩溃。
- 支持`only_category`缺额补采；当前`collection_64x1_static.json`只向新目录`outputs/vae_100k_heightadaptive`采每片Static 500，完成后停止并显示独立Static进度。旧`vae_100k_fixed24`动态目录保持只读，二者尚未合并。
- 本轮后台生命周期、独立Shard停滞、锁释放、回调代理及既有运行时共30项针对性Unit通过（1项Linux专属检查在macOS跳过，约2秒）；另做Python语法和CLI只读检查。未运行Native、完整引擎构建或吞吐测试。既有Static/动态样例、Viewer和结果保留。

唯一下一步：服务器拉取最新版后，用`start`恢复Static采集、另开`dashboard`观察；[命令](vae_data/README.md#仅补齐static缺额)。不自动运行Native、合并旧动态或启动GPU。

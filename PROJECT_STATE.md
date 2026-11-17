# VoxWM 当前状态

**服务器高并行采集停滞的代码修复完成，待用户续跑验证。**

- 用户服务器报告`outputs/vae_100k_fixed24`约49,411样本后停滞；本机未访问该服务器/数据。源码确认词表每次提交全量fsync且持全局锁、等锁无超时、worker受片数限制、Linux没有显示分配及无进度检测。[修复与检查](vae_data/reports/parallel_runtime_20261117/README.md)。
- 当前默认按可用CPU启动世界进程，可手动指定任意正整数；每片唯一Writer不变。Linux每会话管理独立Xvfb，显示失败停止重试；已知Token免锁免写，新Token写盘在锁外，锁等待有上限；无进度时停止本命令进程树。
- 本地18项针对性合成/进程测试通过（约2秒）。没有运行Native采集、Linux多显示压力测试、1K/100K或GPU。数据配额、去重、词表ID、Static高度和Fixed24原生物理不变。
- 已有[10群落Static样例](vae_data/reports/static_ten_20261117/README.md)、[固定时钟动态样例及旧D2](vae_data/reports/fixed24_rerun_20261117/README.md)和预览完整保留。

唯一下一步：用户在服务器停止旧采集命令，pull后以兼容的同一数据目录`--resume --workers auto`续跑，观察正式提交是否恢复；[完整命令](vae_data/README.md#中断恢复--当前服务器续跑)。Codex不自动启动。

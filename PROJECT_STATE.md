# VoxWM 当前状态

**Static连续整数高度与任意坐标Capture已完成并通过有限Native核验。**

- 中心C∈[-38,170]，完整48³不越[-62,193]；Surface/Elevated/Shallow/Deep按Native S选连续整数。Static配额仍25×20、8/4/4/4，目标群落主动定位保留。
- 13个合法高度的C++/独立VoxelManip Node/light/param2逐体素完全一致，-39/171采集前拒绝；新Forceload实际覆盖27或36引擎块。
- 初地点Elevated不合格如实保留；相邻已有GT树木地点补一例C=30。Shallow C=-21、Deep C=-37均为全实心。两段Native合计34.45秒，20项针对性Unit及语法/CLI检查通过。[报告与预览](vae_data/reports/static_height_20261117/README.md)。
- Static按seed+真实中心跨片去重；允许窗口重叠。48³ Token/light布局、Dynamic全局网格与Fixed24时序保持；旧数据不改写，新策略需新输出目录及增量编译Capture补丁。
- 前轮D1/D3/D4-A/B/C fixed24样例已通过，D2保留旧结果，本轮未重跑动态。[前轮报告](vae_data/reports/fixed24_rerun_20261117/README.md)。未启动正式1K/100K、GPU或训练；正式群落覆盖及吞吐尚未测量。

唯一下一步：用户在服务器手动采集一个有时间上限的1K分片，检查实际配额缺口和吞吐；Codex不自动启动。

# VoxWM 当前状态

**用户指定的五类fixed24 Native重跑通过；D2保留旧结果。**

- D1、D4-A、D4-B首次成功；D3、D4-C第二次成功，各保留一条真实五状态链。修复段7次尝试127.47秒；含接口错误初段总计约254.52秒，低于用户批准的900秒。
- 已修复Lua把整数坐标返回为浮点时的误拒绝；位置必须为有限整数值且中心偏移1..14，不取整或改数据。源点域与连续24Hz Tick检查通过。
- 当前Viewer 8766已服务`outputs/vae_fixed24_native_20261117_fixed_no_d2/`；D2明确标为保留旧结果。Playwright验证新数据目录、六型切换、逐帧交互和无console error通过。[本轮结果/预览](vae_data/reports/fixed24_rerun_20261117/README.md)。
- 原始GT/失败记录保存在outputs，不入Git；旧数据不覆盖。未运行正式Shard、100K、GPU或训练；受控RNG跨吞吐严格等价仍未证明。

唯一下一步：用户在已更新Viewer中检查五类新GT，并决定是否进入正式数据采集。Codex不自动追加实验。

# VoxWM 当前状态

**动态观察批传输实现完成，待用户构建并在服务器实测。**

- `advance_batch`默认24拍，可选48；引擎内部每拍保留fixed24同步/post-step真实48³捕获。首帧+RAW稀疏delta与实际Client时钟经现有socket返回；不逐拍写RAW文件。
- Python仍逐拍验证时钟并调用原Collector。48/120拍无变化、10候选、最大观察时长、D4真实事件/五帧与Anchor不变。首次stop后的尾巴不进入候选，有尾巴的世界不复用。
- 不调整并行数；服务器运行示例保持用户实测的64世界。此前Xvfb/词表锁修复保留，已有数据只读，原格式续跑检查不放宽。
- 本地16项针对性检查通过（3.317秒），包括CPU编码/解码、时钟/停止点合成测试；另做补丁静态应用、语法/CLI检查；未运行Native、完整引擎构建或吞吐测试。[实现与检查报告](vae_data/reports/advance_batch_20261117/README.md)。既有Static/动态样例、Viewer和结果保留。

唯一下一步：用户停止旧采集，pull并执行`bash scripts/build_fixed24.sh`，随后以`--workers 64 --tick-batch-size 24 --resume`手动续跑，观察完整循环的实测吞吐；[命令](vae_data/README.md#中断恢复--当前服务器续跑)。不自动启动1K/100K或GPU。

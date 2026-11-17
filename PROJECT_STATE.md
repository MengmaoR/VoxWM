# VoxWM 当前状态

**动态观察批传输及外部停滞自动恢复已实现，待用户在服务器实测。**

- `advance_batch`默认24拍，可选48；引擎内部每拍保留fixed24同步/post-step真实48³捕获。首帧+RAW稀疏delta与实际Client时钟经现有socket返回；不逐拍写RAW文件。
- Python仍逐拍验证时钟并调用原Collector。48/120拍无变化、10候选、最大观察时长、D4真实事件/五帧与Anchor不变。首次stop后的尾巴不进入候选，有尾巴的世界不复用。
- 不调整并行数；服务器运行示例保持用户实测的64世界。此前Xvfb/词表锁修复保留，已有数据只读，原格式续跑检查不放宽。
- 新监控器按JSON配置启动采集，默认100片目标、同时最多64片、每片1个世界；终端Dashboard原位显示Shard/样本进度、attempt、速率/ETA、进程数和600秒停滞倒计时。停滞时仅停止带本次随机运行标记的完整进程集合，确认无残留后按原配置恢复。状态与每次stdout/stderr保存到独立supervisor日志目录。
- 本地16项针对性检查通过（3.317秒），包括CPU编码/解码、时钟/停止点合成测试；另做补丁静态应用、语法/CLI检查；未运行Native、完整引擎构建或吞吐测试。[实现与检查报告](vae_data/reports/advance_batch_20261117/README.md)。既有Static/动态样例、Viewer和结果保留。

唯一下一步：用户停止旧采集，pull并执行`bash scripts/build_fixed24.sh`，检查JSON中的dataset后手动启动supervisor，观察真实卡死时能否精确清理并恢复；[命令](vae_data/README.md#外部监控与自动重启)。不自动启动1K/100K或GPU。

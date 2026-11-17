# Dynamic advance_batch — 2026-11-17

实现完成；未运行Native、完整引擎构建或服务器吞吐测试。用户报告的70/120世界与空载性能是外部现场数据，不是本轮测量，不推算2–3小时100K或承诺提速倍数。

## 改动

- 引擎增量补丁`voxwm_gt/patches/advance_batch.patch`与`batch_transport.h`：仍使用原fixed24 Server/Client信号量，每拍`1/24`且post-step捕获。批内Client不调用Python握手，不写每拍clock stamp；最后一拍才发普通观测，并在同一socket附长度前缀的内存批包。
- `native.lua`：新`advance_batch(n)`，每次1..48拍；逐拍调用原C++ `voxwm_region`，记录整数Tick、时钟、engine_us、真实reaction和audit；JSON仅批末发布，chunks元数据共享一次。旧单拍capture/advance保留。
- `batch.py`/`clock.py`：有界内存传输，首拍完整RAW，后续每拍递增index+RAW差异（node:uint16/light:uint8/param2:uint8）；无变化拍仍有记录。实际Client完成Tick/dt由Client写入批包，不由Server数值伪造。检查1..n拍、全批连续时钟、固定dt、载荷完整。socket临时超时模式结束后恢复原fd阻塞标志。
- `Native.tick_batch`/`generate.observation_ticks`：逐拍解码并交给原Collector；首次stop即关闭迭代，尾巴不编码Token、不进候选/事件/链。最大模拟时长通过末批裁到剩余Tick保持原值。`Collector.observe`与配额/Registry/采样器本身未修改。
- 原代码实际会跨地点复用Native世界（sites_per_world），与用户报告中的假设不同；因此只有出现未消费批尾时标记世界退休，单世界和多世界入口都会在本次尝试结束后关闭，杜绝额外物理时间影响后续场景。物理不回滚、已保存链不改写。
- Trace失败恢复优先用最后已解码帧，防止拿初始化时的native48.bin冒充后续批帧。正式Shard仍只保存选中五帧与成功D4 Anchor；已有debug Trace规则不变，批包本身不写盘。

CLI新增`--tick-batch-size {1,24,48}`：generate默认24，clock-verify默认1；64世界由用户命令保留，不修改并行调度策略。`1`可对照旧路径。保留低缓存：一个原始状态+一个最多48拍的稀疏批包；worst case高变化数据仍有有界内存成本，不承诺所有场景压缩率。

## 检查

16项针对性检查通过，3.317秒。覆盖：C++ RAW/delta编码到Python逐体素重建（含light-only/high-param2）、全48个无变化Tick保留、socket一次传输与阻塞标志恢复、48/120拍超时、中途变化重置、尾巴不增加候选、10候选停止及相同RNG五帧选择、D4 Pre/Reaction/Post来自相同fixture事件、最大时长、全批时钟错位/跳拍/dt错误拒绝、旧单拍一次一Tick、多世界单Writer接口。Fixture仅验证传输/采样语义，不是原生物理证据。

固定上游源码在临时目录叠加base fixed24、integer-origin capture、batch三层补丁通过，最新层反向检测通过；不修改本机third_party源码。独立C++头fixture完成编译；Python/Lua语法、CLI及Shell语法通过。完整Luanti构建与批内真实时钟/液体/RNG行为仍须服务器验证。

## 用户运行

```bash
git pull --ff-only origin master
bash scripts/build_fixed24.sh

# 用户保留实测的64世界，数据目录必须与现有协议兼容；先停止旧命令。
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k_fixed24 --shard 0 --count 100 --workers 64 --resume \
  --tick-batch-size 24 --max-attempts 5000 --wall-seconds 86400 \
  --stall-seconds 300 --renderer software
```

可选只读批时钟命令及输出见[操作手册](../../README.md#动态逐拍批传输)。performance的`batch_transfer`含captured/consumed/discarded Tick、往返数、wire_bytes、传输墙钟；完整`loop_wall_seconds`包含Python消费/规划以外的观察工作与尾巴开销。不要用通信次数降到1/24替代真实端到端测量。

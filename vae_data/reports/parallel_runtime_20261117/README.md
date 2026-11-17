# 高并行采集停滞修复 — 2026-11-17

代码问题已确认并修复；服务器真实高并行尚未验证。依据用户提供的同日《采集停滞报告》，未访问服务器、未修改已有数据或运行引擎。

## 核对结果

| 问题 | 本地源码证据与范围 |
|---|---|
| 单X服务器满员 | 正式`renderer_config`原来用SDL offscreen，缺少显示分配；`:99`和满员日志属于服务器临时wrapper/现场证据，本地没有伪造复现 |
| 全局词表堵塞 | 原`storage.globalize`每次提交均持`.vocabulary.lock`序列化、fsync全词表，即使无新Token；flock无等待上限 |
| 失败反复启动 | 原`generate_shard`把显示启动错误与普通场景失败混为一类，继续尝试 |
| 清理缺口 | 原`close_env`仅在channel已连接时清理进程/管道；首次连接失败时可能残留 |
| 并行上限 | 原CLI用`min(workers,count)`，100片最多100世界；同进程加线程还会遇Craftium C扩展阻塞recv持有GIL |
| 无提交停滞 | 原外层只有总墙钟/中断清理，失败重试可以掩盖正式提交已经停止 |

## 实现

- `voxwm_gt/display.py`：每会话使用Xvfb `-displayfd`自动领取独立显示号；没有固定显示列表或世界数上限，结束时关闭。Software模式使用SDL x11+Mesa，`LP_NUM_THREADS`默认1，可由环境覆盖。
- `voxwm_gt/native.py`/`session.py`：优先使用隔离运行目录中真实`bin/luanti`，绕过服务器手改minetest wrapper；启动时短轮询发现引擎退出；SDL/X错误归为InfrastructureError且保留stderr原因。首次连接失败也关闭socket/管道/进程组。
- `storage.py`：无新Token不锁不写；有新Token先在锁外写临时词表并fsync，再短锁复核已有词表并原子替换；遇并发发布重做合并，保持已发布ID。锁等待/发布竞争有30秒预算，不靠强行解别人的锁。
- `parallel.py`：多个独立进程采场景，每片只保留唯一Writer，结果经有界一任务/世界的Pipe提交。每世界独立RNG/词表/原生环境；cursor保留各RNG，可恢复；配额/源去重/seed上限不变。
- `__main__.py`：已提交样本标记或新词表才算进度，默认300秒无进度FAIL（可调整/关闭）；显示错误快速失败。清理仅属于本命令的worker/引擎/Xvfb，保留已提交样本。

`--workers auto`默认采用当前CPU affinity核数（否则CPU数），手动任意正整数不截断到CPU或片数。100片×200世界为每片2世界/1 Writer；它是世界并行数，不是保证CPU百分百占用。READY片跳过，末尾任务结束时活跃世界会自然减少。

内核/网络盘卡住的fsync不能由Python锁超时强制中断；现在它不会持有词表全局锁，其他进程仍能发布。若整体文件系统或不可中断IO失去响应，不能承诺SIGKILL立即结束；停滞日志和本次PID用于现场判断，不自动修数据。

## 实际检查

- 18项定向Unit/进程fixture通过，约2秒：竞态发布保留双方新增Token/原ID、无新Token免锁免fsync、fsync时全局锁空闲、锁等待有界、两世界进程共用单Writer（含外层ProcessPool嵌套）、512并行参数/自动192核解析、显示分配/清理mock、启动错误不重试、停滞清理、原提交恢复及Ctrl-C。
- Python语法、`generate --help`、diff空白检查通过。
- Fixture不是Native数据；没有启动Xvfb/真实游戏/大规模并行，没有访问服务器、采1K/100K、重建引擎或GPU。
- 没改Lua、Craftium C++、原生物理、Fixed24、配额、采样/去重规则、已有结果/Viewer。

[服务器pull与续跑命令](../../README.md#中断恢复--当前服务器续跑)。本修复无需编译，原有当前格式/Static策略检查保留；不要将不兼容数据强行恢复。

接口依据：[Xserver官方`-displayfd`手册源码](https://cgit.freedesktop.org/xorg/xserver/tree/man/Xserver.man?id=b3b81c8c2090cd49410960a021baf0d27fdd2ab3)、[SDL驱动配置](https://wiki.libsdl.org/SDL2/SDL_HINT_VIDEODRIVER)、[Mesa环境变量](https://docs.mesa3d.org/envvars.html)。这些接口资料不是本项目服务器实测证据。

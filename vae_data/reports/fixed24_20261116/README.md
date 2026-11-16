INCONCLUSIVE — Fixed24 clock verified; full native physics equivalence pending

本次实现中心14³干预域、D3 120-Tick超时、opt-in Craftium fixed24时序和逐Tick GT。没有修改VoxeLibre物理/RNG算法、模型/训练、既有GT或Viewer渲染；没有重跑六型样例、正式1K/100K或VoxEvolve数据。

## 已实现与证据

| 项目 | 状态 | 证据/限制 |
|---|---|---|
| 全部六型人工source位于中心14³ | CPU VERIFIED | 先mask再合法枚举，最多2744点；全点规划/两个D4-B sampler/保底共用此域。Python调用前后、Lua全部放置前检查；后续传播图仍为完整48³ |
| D3无变化120 Tick，其余48 Tick | CPU VERIFIED | 62项相关测试；年龄/邻块变化不能重置；结构变化可以重置；最大观察30/40/40/60秒保留 |
| Server/Client/Lua固定1/24 | NATIVE CLOCK VERIFIED | 240连续Tick的实际时钟；绝对/相对时间由整数Tick计算；原生float32 dt使用最近可表示1/24，不按墙钟累计 |
| post-update原生捕获 | NATIVE CLOCK VERIFIED | Server AsyncRunStep全部执行、Receive后才调用Lua post-step，再释放完成信号；首次同Tick C++/原P0完整payload相等 |
| 无实时FPS上限 | VERIFIED IN THIS RUN | FPS limiter/Server等待/Client wall-lag扣减均在fixed24模式停用；vsync关闭；实测含Trace吞吐大于24 Tick/墙钟秒 |
| Tick0..240与无变化Tick保留 | VERIFIED | 241个真实观察；230个真实零delta Tick；无插值、补帧或构造静止帧 |
| Water原生更新节奏 | OBSERVED | 完整48³有1900次Water token更新，发生在相对21,45,...237 Tick，每次相隔24 Tick；中心更新0，不算D1成功链 |
| Lava/Fire新时序动态 | NOT RUN | 本次只读clock场景未观察到Lava/Fire变化，不能用旧六型结果代替 |
| 相同完整初态/RNG的跨吞吐voxel等价 | NOT VERIFIED | 原生全局RNG、emergence、完整队列未快照；提供两run比较接口但本轮未再运行对照，不能声称严格确定性 |
| Pixel-level equivalence | NOT VERIFIED | VAE模式跳过RGB，兼容packet中的空白图像不是GT RGB；原生Camera/截图与渲染相位须另行验证 |

## 有限原生检查

总原生工作墙钟 **28.96秒**（120秒总预算内）。首次准备阶段失败13.98秒：异步prepare回调仍在globalstep中捕获，Server/Client/Lua一致性检查拒绝。将该捕获也排到post-step后，一次明确修复，复核14.98秒通过，未追加动态样例。首次失败原文单独保留，不覆盖。

复核命令：

```bash
.venv/bin/python -m vae_data clock-verify \
  --dataset outputs/vae_fixed24_clock_20261116_fixed --ticks 240 \
  --wall-seconds 100 --seed 191664964
```

原生绝对Tick 1155..1395，相对0..240，24/48/120/240 Tick对应1/2/5/10模拟秒。完整Trace位于`outputs/vae_fixed24_clock_20261116_fixed/trace/`，VWT1可恢复Node Name/raw param2和原始light；不入Git。

## 性能（实测CPU；非GPU）

| 项目 | 实测 |
|---|---:|
| Native update / Tick | 0.467 ms |
| C++ capture+payload write / Tick | 0.486 ms |
| Python/Lua文件通信 / Tick | 0.480 ms |
| Token编码/解码与Assembly / Tick | 1.559 ms |
| 总tick调用 / Tick（不含Trace） | 8.436 ms |
| Trace delta与周期性metadata写入 | 1.199 s / 240 Tick |
| 逐Tick循环总墙钟（含Trace） | 3.232 s |
| 吞吐（不含Trace） | 118.54 Tick/s，4.94 sim s/s |
| 吞吐（含Trace） | 74.26 Tick/s，3.09 sim s/s |
| Capture / Native update | 1.040 |
| Python峰值RSS | 88.5 MB |
| Child峰值RSS（引擎，含mapgen） | 627.4 MB |

各分项并不覆盖总调用的全部成本：Client同步/空白packet、Gym、JSON与其他Python处理仍占余量。Trace时间含稀疏delta计算/写入及周期性metadata checkpoint，非纯压缩算法耗时；最终metadata关闭写入未计该分项。Python与child峰值分别报告，不能当同时峰值相加。主瓶颈是Python/通信/metadata及debug Trace，不是原生物理；正式Shard不写完整Trace。不外推为100K产能。

优化包括：C++固定27块直接读取并复用buffer；只在准备时与P0逐字节比较/生成Biome等静态metadata；Combined Token LUT复用，保留精确pair；无逐Tick 27次VoxelManip重建/逐voxel名字回解；VAE不截图/渲染RGB；Trace不再每Tick重写整个JSON索引。保留现有Frame/27块Assembly，没有滑动窗口或新采样算法。

## 关键时序

`Python action → Client请求 → Server固定step → 原生ABM/Timer/globalstep/liquid/Receive完成 → post-step capture → completion barrier → Client同dt完成 → observation → 等待下一Python action`。

`core.registered_voxwm_poststeps`是补丁提供的尾部只读采集钩子。`voxwm_clock()`暴露整数tick/dt，`voxwm_region(low)`输出原生27个对齐16³的Node/light/param2字节。干预在该尾部一次放完、设相对Tick0、立即capture；随后`Native.tick()`每次只允许+1。异步prepare的capture同样延迟到尾部。演化期间不能调用setup-only `capture48`绕过tick连续性；移除结束后多走一个未捕获Tick的主动entity probe，零实体证据继续来自每Tick原生guard。

固定模式停用ABM的墙钟扫描早停，避免机器负载改变哪些block有机会执行；保留原来的shuffle/chance/interval。液体queue purge原有策略仍在，但其timeout使用模拟时间。没有改VoxelLibre任何节点物理定义、ABM函数、传播范围或黏度。

## 服务器有限验证（用户决定后手动执行）

从VoxWM仓库根目录：

```bash
git pull --ff-only origin master
# 已有独立环境：只做增量build，不安装系统软件
bash scripts/build_fixed24.sh
# 新机器应先运行 scripts/setup_gt.sh 和安装 vae_p0/requirements.txt，见vae_data/README。
.venv/bin/python -m vae_data clock-verify \
  --dataset outputs/clock24_fast --ticks 240 --wall-seconds 60 --seed 191664964
# 额外的CPU consumer负载对照：必须由用户另行决定启动，不自动接续
.venv/bin/python -m vae_data clock-verify \
  --dataset outputs/clock24_loaded --ticks 240 --wall-seconds 60 --seed 191664964 \
  --consumer-work-ms 5 --compare-to outputs/clock24_fast
```

两条验证各最多60墙钟秒（合计≤120），没有六型新样例。consumer负载只在该诊断命令中做CPU工作，不sleep、不进入正式generator。比较器报告初始区域是否相同、整个实际voxel序列是否相同及首个不匹配Tick，**不控制/快照全局RNG，所以不能把恰好相同或不同直接当严格physics等价结论**。本轮没有执行第二个吞吐条件。

新六型验证仍为原来的每型3次/1成功、整轮900秒；未经新授权不重开。若后续要证明受控随机跨速度物理等价，需要完整原生初态/队列/RNG可重放，再另行授权Water/Lava/Fire各两次240 Tick、总墙钟≤900秒；当前代码没有伪造这些条件，也不宣称该验收完成。

## 后续VoxEvolve接口

`GTSession(..., fixed_step=True)`启用时钟补丁；现有默认会话仍用旧模式，不重写旧GT。当前`Native`明确使用`voxel_observation=False, render_rgb=False`，通过post-step完整世界坐标快照取GT，省掉重复的49³观察和RGB。未来可接相同tick/capture钩子及`RegionEncoder`，保持整数Tick和原始Node/raw param2/light。若加入RGB/Camera，应单独处理Client render相位；目前不保证同Tick图像或逐像素等价。

数据格式为`vae-region48-fixed24-v2`；旧variable-step Shard不能混合/恢复到v2，必须新目录。100K数量、Static配额、5状态/10候选、D4 Anchor/Pre–Reaction–Post及split不变，旧报告/词表/数据全部保留。五个正式VAE状态允许不连续Tick，但每个原始Tick都会在Native中被检查。

完整验收目前仍INCONCLUSIVE：时钟与只读快照已验证；原生Lava/Fire、六型新source实际运行、受控RNG跨吞吐和像素层面尚待授权验证。62项Unit与语法/CLI/build通过。本轮基于git `879dad8af38ae9b55b232097f4626d461d6e27b1` 的未提交实现运行，metadata保留dirty状态；最终提交不冒充当时clean revision。

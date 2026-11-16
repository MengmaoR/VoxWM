GO — requested five-type fixed24 rerun (D2 intentionally retained)

用户授权重跑D1、D3、D4-A/B/C，D2不重跑。修复Lua整数坐标浮点表示的误拒绝后，用户明确批准重新给这五类各最多3次/1成功，且两段原生运行累计不超过900秒。

修复后 **127.47秒、7次尝试、五类各成功一条**。初始失败/中断段约127.05秒，两段合计约 **254.52秒**，未超过900秒。修复段命令硬上限760秒，加初段约127秒也低于900。D2没有启动Native采集，没有把其旧结果计作本轮新通过。

| 类型 | Attempt | Height | 结果 | 实际观察数 | 演化Tick | 原因 |
|---|---:|---|---|---:|---:|---|
| D1 | 1 | surface | success | 241 | 239 | accepted five-state chain |
| D3 | 1 | surface | rejected | 279 | 277 | fewer_than_four_distinct_center_transitions |
| D3 | 2 | surface | success | 252 | 250 | accepted five-state chain |
| D4-A | 1 | surface | success | 780 | 777 | accepted five-state chain |
| D4-B | 1 | cave | success | 336 | 334 | accepted five-state chain |
| D4-C | 1 | surface | rejected | 1 | — | center_probe_no_pool |
| D4-C | 2 | cave | success | 205 | 202 | accepted five-state chain |

D3成功样例记录新火焰5、可燃物→Fire14、Fire→Air4，以及单独的Fire Age更新16。年龄更新不重置120-Tick超时、不用于补五状态。所有实际人工源点均为中心local[1..14]^3；演化轨迹从相对Tick0连续，按24Hz记录。

初始段包含7次坐标类型误拒绝、3次其他已完成的失败/拒绝记录及1个中断尝试；原样保留，不改成成功，也不把接口错误当作原生物理失败。[接口修复说明](interface_fix.md) / [初始结果](initial_interface_failed_results.json)。Boot connection错误的底层原因本轮未额外追查。

## Viewer 已切换到新目录

`http://127.0.0.1:8766/?type=D4-A&region=16&run=fixed24_20261117`

新GT：`outputs/vae_fixed24_native_20261117_fixed_no_d2/`。当前8766只读服务器已从旧目录切到这里。D2的Viewer-only导出复用旧accepted trace：`outputs/vae_scene_sampling_20261116/traces/D2/attempt_1`；保留原始Voxel/时间/词表/来源，并在页面明确标“保留的旧结果（本轮未重跑）”。未把旧D2伪装成fixed24新结果。

复现本次五型命令（仅供用户另行决定运行，不自动启动）：

```bash
.venv/bin/python -m vae_data native-six \
  --dataset outputs/vae_fixed24_native_new_no_d2 \
  --skip-d2 --attempts-per-type 3 --wall-seconds 900
```

本机查看现成结果，无新模拟：

```bash
.venv/bin/python -m vae_data view \
  --dataset outputs/vae_fixed24_native_20261117_fixed_no_d2 --port 8766
```

8766已经启动，直接打开链接即可；重新启动命令前先停止旧服务。完整原始Trace在outputs中不入Git，保留各失败attempt；正式Shard未生成。轻量预览见下表，D2沿用历史Viewer项。

| 新类型 | 预览 | 过程 | 诊断 |
|---|---|---|---|
| D1 | [PNG](D1/contact_sheet.png) | [GIF](D1/evolution.gif) | [报告](D1/report.md) |
| D3 | [PNG](D3/contact_sheet.png) | [GIF](D3/evolution.gif) | [报告](D3/report.md) |
| D4-A | [PNG](D4-A/contact_sheet.png) | [GIF](D4-A/evolution.gif) | [报告](D4-A/report.md) |
| D4-B | [PNG](D4-B/contact_sheet.png) | [GIF](D4-B/evolution.gif) | [报告](D4-B/report.md) |
| D4-C | [PNG](D4-C/contact_sheet.png) | [GIF](D4-C/evolution.gif) | [报告](D4-C/report.md) |

## 最小检查

- 仅运行1项finite-dispatch、2项坐标针对性及1项保留D2重导出测试，全部通过。CLI解析、JS语法通过；未重跑整套测试矩阵。
- 实际新轨迹源点位置及相对Tick连续性通过，见`rerun_summary.json`。Native保存成功五状态链前已执行原有validator。
- 前端检查使用已有Playwright（Browser插件不可用），桌面1440×1000。新URL/标题/非空画面/无错误overlay/console error均通过；逐帧按钮更新真实状态，六个类型均加载成功，D2旧结果标记可见。截图留`/tmp/voxwm_fixed24_rerun_viewer.png`，机器记录`viewer_qa.json`；未另做移动端/其他浏览器或渲染回归矩阵。

本轮运行基于git `44d7911110b1e716838c826ecb2f44f94b176967`加未提交的skip选项/接口修复，metadata保留dirty标记。没有修改模型或原生物理，没有GPU、正式1K/100K或训练。此次五型成功不等于受控RNG跨吞吐物理等价证明，后者仍未完成。

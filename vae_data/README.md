# VAE 100K — 1K Shard Generator

当前入口：`python -m vae_data` → `vae_data/sharded/`。仅生成数据，无模型/训练器/DataLoader。依据[当前协议](../VAE_DATA_COLLECTION_PROTOCOL.md) / [Drive Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view)。旧200K代码、结果与日志保留；[旧README](LEGACY_UNIFIED_README.md)中的命令需使用 `python -m vae_data.legacy_entry`，不是当前生产格式。

**当前请求的五类fixed24 Native样例重跑通过**：D1/D3/D4-A/B/C各一条成功，D2保留已通过的旧结果并在Viewer标注未重跑。[本轮新数据/预览/Viewer检查](reports/fixed24_rerun_20261117/README.md)。未采集正式1K/100K，受控RNG跨吞吐严格等价仍未证明。

## 服务器环境（用户手动执行）

从VoxWM仓库根目录操作。已有环境无需重新安装；CPU仿真，无Torch/CUDA/模型权重。

```bash
git pull --ff-only origin master
# 仅首次新服务器安装；系统依赖见根README。脚本不自动sudo。
bash scripts/setup_gt.sh
uv pip install --python .venv/bin/python -r vae_p0/requirements.txt
# 已有环境：增量应用/编译固定版本的opt-in时序补丁
bash scripts/build_fixed24.sh
```

Linux示例使用 `--renderer software`，macOS可改为 `--renderer auto`。不改变原生液体/火规则，不做时钟加速。

## 生成指定一个1K片

```bash
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k_heightadaptive --shard 0 --count 1 --workers 1 \
  --max-attempts 5000 --wall-seconds 7200 --renderer software
```

`shard`索引0..99。每片硬配额：Static500，四动态各125；每类25条五状态链。Static每群落20条，8/4/4/4。D4是水岩浆固化，不是植物；子类型A/B/C每片8/8/9轮换。动态每Tick采集：D1/D2/D4中心48 Tick无目标变化停止，D3为120 Tick；不能为了过测延长。未达到配额只保留partial和缺口，不READY、不替代类别。

Static现在先按剩余“群落×高度”配额选目标，再在当前seed内主动查询群落；不再随机生成地点后碰运气匹配配额。使用原生Biome噪声接口，64格间隔螺旋、每次最多16,384点；只在候选命中后加载该处地形，并检查真实地表群落。原生变体沿用既有25组映射。未命中或实际群落不符会记录原因并换位置/seed，受原有时间/尝试上限约束，不能拿其他群落补齐。搜索不会创建地形缓存；目标/命中/实际群落保存在现有attempt和样本`static_selection`中。

Static中心高度C逐整数随机，不吸附到16格引擎网格；完整Region=[C−24,C+23]必须位于[-62,193]，即−38≤C≤170。Native地表S与采样中心C分别记录。Surface选包含S的中心；Elevated取[S+9,170]且0.10≤Air<0.98；Shallow取[S−24,S−8]；Deep(`deep_cave`)取[−38,S−32]，均裁到允许域。地下允许全实心、全Air及混合结构，只排除全`mcl_core:void`；完整Region的`ignore`另作完整性错误。每地点最多16个不重复高度，过于空旷的Elevated仅向下重试。

需先运行`bash scripts/build_fixed24.sh`增量编译世界坐标Capture补丁；不改变原生物理或Fixed24时间。每次Prepare与同Tick独立P0/VoxelManip逐体素比较Node/light/param2，实际相交引擎块全部Forceload。Static按seed+真实中心跨片去重，允许窗口部分重叠。Token/light布局和Dynamic网格不变；旧目录不混入新Static策略，请使用上面的新输出目录，原数据不修改。稀有群落命中率和正式1K吞吐尚未实测。

## 有限Native Static核验（不生成正式Shard）

```bash
.venv/bin/python -m vae_data static-verify \
  --dataset outputs/vae_static_height_check_new --wall-seconds 180 \
  --seed 191664964 --site -272 1456 --renderer software
```

单世界、最多180秒，检查13个合法C及两个非法边界；C++/VoxelManip同Tick逐体素比较，另尝试Elevated/Shallow/Deep各一个真实样例。输出`static_verification.json`、`examples.png`、小型`examples/*/state.npz`及metadata/vocabulary；无法找到结构时明确保留NO_CANDIDATE，不追加无界搜索。macOS可用`--renderer auto`。正式采集仍需用户手动启动。

## 已完成的10群落Static试采

[实测表及原预览入口](reports/static_ten_20261117/README.md)：10个不同群落组、不同C，总高度分配4/2/2/2；107.29秒采齐，未创建正式Shard。原8766预览选择`Static`，通过`Scene / native attempt`切换；原动态数据保留。

需要用户自行重复时使用新目录：

```bash
.venv/bin/python -m vae_data static-ten \
  --dataset outputs/vae_static_ten_new --wall-seconds 600 --renderer software
.venv/bin/python -m vae_data view --dataset outputs/vae_static_ten_new --port 8767
# http://127.0.0.1:8767/?type=Static
```

`--preview-dataset EXISTING`仅向尚无Static分类的既有预览追加派生数据，不覆盖已有Static。正式配额、动态采集和材质不变。

## 连续多片 / 多CPU workers

```bash
# 生成 shard_0010 .. shard_0013，最多4个独立CPU世界并行
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k_heightadaptive --shard 10 --count 4 --workers 4 \
  --max-attempts 5000 --wall-seconds 14400 --renderer software
```

Worker按片独占目录与锁；Native世界/临时引擎独立；全局词表用短锁追加ID。只需要一个命令调度同一批片；相同片被另一个进程占用时明确失败，避免并发写入。

## 中断恢复

```bash
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k_heightadaptive --shard 10 --count 4 --workers 4 --resume \
  --max-attempts 5000 --wall-seconds 14400 --renderer software
```

Ctrl-C会停止本命令拥有的worker和引擎进程；已提交数据可恢复。已READY片直接跳过，保持原文件；partial片恢复已提交样本/词表/RNG cursor，去除未提交Zarr尾部。未完成场景开启新的native rollout，不恢复旧模拟物理时间。`max-attempts`是本次每片尝试上限，墙钟为整条命令总上限；到限可不满配额，需要用户决定是否继续。

## 检查某片、查看全局进度

```bash
.venv/bin/python -m vae_data check --dataset outputs/vae_100k_heightadaptive --shard 10
# 检查partial中已提交数据，允许数量尚未齐；complete=false不会发布READY
.venv/bin/python -m vae_data check --dataset outputs/vae_100k_heightadaptive --shard 10 --allow-incomplete
.venv/bin/python -m vae_data status --dataset outputs/vae_100k_heightadaptive
```

Check为只读，不自动修数据。错误返回非零；status只报告READY与partial统计，100个READY才是正式100K完成。检查结果包含Transition软目标缺口；不能把不足4的真实区域伪装成transition。

## 格式和无损读取

每个正式样本：`tokens:uint16[48,48,48]`、`light:uint8[48,48,48]`，ENU。目标中心为 `[16:32]*3`，region origin = center origin −16。Static高度起点可以不是16的倍数；27个采样窗口仍相对16格连续。上下文保存完整，动态判定与VAE目标都限定中心。

```python
import zarr, pyarrow.parquet as pq
from vae_p0.codec import Vocabulary
root='outputs/vae_100k_heightadaptive'
path=root+'/shards/shard_0010'
a=zarr.open_group(path+'/snapshots.zarr',mode='r')
tokens=a['tokens'][0]                 # 完整48³原生状态
light=a['light'][0]
center=tokens[16:32,16:32,16:32]       # 目标16³
names,param2=Vocabulary(root+'/vocabulary.json').decode(tokens)
rows=pq.read_table(path+'/metadata.parquet').to_pylist()
```

`metadata_json`保留seed/固定split、坐标、时间、sim step、physical frame ID、中心Biome/地形、初始化、事件角色及native版本/零实体证据。`journal.sqlite`原子提交样本和完整链；`chains.data`包含选中帧、D4事件、独立Anchor引用与实际source操作。Anchor不计1K，只有成功D4才保存。全局词表只追加不重排，多个shard共用。

每场景最多10个中心不同候选，另保留必要当前帧和D4 anchor；未选状态释放。没有discovery cache、完整raw trajectory或下游latent缓存。同源重复中心状态拒绝；不同seed的真实重复结构不被全局删除。

## 动态场景筛选

D1/D3仅真实Surface；D2和D4按50/50给Surface/Cave机会。Surface必须含地形固体—Air边界，不把液面/湖底当地表；Cave必须有真实地下连通空气。保留16³世界网格，不滑动中心。所有人工source先限制中心local[1..14]^3（region[17:31]^3），最多2744点；后续原生传播不受此域限制。所有中心最大六邻域Air分量≥65；仅D4-A/C寻找天然对应水/岩浆池，短时观察后中心稳定最大分量≥19（Source或Flowing皆可）。取消通用天然目标体素`center_hint`及旧二维池/体积门槛。通过筛选后才做全点规划。

D4-B两种新源独立采样并检查联合固化潜力；拒绝已有天然水岩浆接触及预计源路径接触已有流体。每类/高度的attempt、空间合格、规划合格、accepted写入statistics；没有合格场景就报告缺口，不改配额。

## 全点预计算与独立GT Viewer

小型C++17 CPU内核首次按源文件mtime编译到系统临时目录，需`c++`（安装GT环境已需要编译器）；不需要Torch/CUDA。共享CSR图+乐观反向不可达界+64源稀疏bitset传播，不预删合法点。流体与Fire的时间估计仅为不确定潜力，实际合格仍由native五状态/事件判定。选点p=1/N；D4-B查询隐式联合兼容，不建Water×Lava密集矩阵。源码说明及复杂度见[报告](reports/exhaustive_20261116/README.md)。

```bash
# 已有本轮完整Trace时：仅CPU导出/查看，无引擎运行
.venv/bin/python -m vae_data export-traces --dataset outputs/vae_scene_sampling_20261116
.venv/bin/python -m vae_data view --dataset outputs/vae_scene_sampling_20261116 --port 8765
# 浏览器 http://127.0.0.1:8765/?type=D4-C&region=16
```

Viewer支持六型和各attempt、真实模拟时间及相对干预时间播放/逐观察帧/拖动、48³/16³、XYZ剖切、自由轨道/缩放、27边界、块类高亮、源点、估计时间可达域/实际变化叠加、Pre/Reaction/Post/接收五帧。静态预览含真实中心切片；没有神经网络预测。`docs/vendor`仅提供只读Three/OrbitControls资源，不修改/部署网站。服务器可手动`--host 0.0.0.0`，默认仅localhost。

```bash
# 使用已存在的真实GT，只做CPU测量；不会创建原生世界
.venv/bin/python -m vae_data profile-planner \
  --real outputs/vae_sharded_smoke_20261115/examples/D1_0 \
  --fire outputs/vae_native_six_20261116/traces/D3/attempt_2 \
  --pair outputs/vae_native_six_20261116/traces/D4-B/attempt_1 \
  --output outputs/planning_profile.json
```

本次完整Trace在`outputs/vae_scene_sampling_20261116/traces/<type>/attempt_N/`，VWT1无损保存tokens/light和真实metadata。`planning/*.npz`为全部合法源点的region-local ENU坐标/统计；`selected_N.bin`是选中源uint8[48³]最早估计时间，255为估计未达。summary含世界坐标、p/保底/条件分布及不确定性。`planning_followup`明确是运行后只读原场景的CPU复算，不能当作已执行干预。

正式Shard不写这些完整Trace。本次轻量过程图在`vae_data/reports/scene_sampling_20261116/`（历史exhaustive报告保留），完整Trace不入Git；到新服务器需用户自行复制现有输出，或明确决定另做一次有界测试。历史`native-smoke`仅保留复现旧四大类测试，当前协议使用六Unique Types。

## 固定时钟有限检查

```bash
.venv/bin/python -m vae_data clock-verify \
  --dataset outputs/clock24_check --ticks 240 --wall-seconds 120 --seed 191664964
```

只读原生场景，不生成链/正式Shard。检查Server/Client/Lua、Tick连续性与同Tick快照；输出clock_results.json和完整Trace。无自动后续实验。不同吞吐对照用`--consumer-work-ms 5 --compare-to <先前clock目录>`，需用户单独启动；不保证已控制全局RNG。

## 开发检查及明确授权的小例

```bash
.venv/bin/python -m unittest discover -s vae_data/sharded/tests -q
# 以下只供用户明确决定重复小例时手动启动；本轮Codex没有重开六型，需新的明确授权。
.venv/bin/python -m vae_data native-six \
  --dataset outputs/vae_native_six_new --wall-seconds 900 \
  --attempts-per-type 3 --renderer software
```

`native-six --skip-d2`仅重跑D1/D3/D4-A/B/C，D2明确列入skipped_types，不能计为本轮新通过。Viewer可以保留已有D2导出，但必须显示“旧结果，本轮未重跑”。

不创建正式Shard/READY。每型最多3次尝试，成功1条即停止，整轮最多900秒；输出目录必须全新。失败保留Trace及原因，不能通过重复命令追加Codex测试预算。Synthetic Fixture只在临时Unit目录，不能当Native物理证据。

## 原生节点状态与异常检查

[本轮D2逐体素审计及Viewer说明](reports/transition_audit_20261116/README.md)。Viewer直接读取原Combined Token vocabulary，支持Semantic、Native-State、Change Inspection三模式；液体解码0x07/0x08及真实liquid_range，火焰年龄不作方向，未知Node Definition标为简化fallback。

点击体素或输入世界ENU坐标查看Token/Node/param2/液位/Fire Age；Freeze comparison pair固定两次真实观察，Before/After/Difference始终查询同一世界坐标。Previous leaves ghost只显示旧状态的线框。

```bash
.venv/bin/python -m vae_data.sharded.audit_transitions \
  --trace outputs/vae_native_six_20261116/traces/D2/attempt_2 \
  --start 18.9 --end 19.2 --output outputs/d2_transition_audit
```

以上只读已有GT；不会补造中间状态、运行引擎或改变原始词表。完整逐节点写入journal未保存，无法从采样间隔唯一确定每次内部callback。

# VAE P1 — Dynamic Data Collection

P1 独立于 P0；P0 源码、数据与报告不修改。新协议允许 **一次受控初始化**，之后全由原生 VoxeLibre 演化。不实现 VAE 或其它模型。

权威：本次用户 P1 Implementation prompt。已读取 [11/12 Manifest](https://drive.google.com/file/d/1hYmOg6SHIEPYcjvUmKgjP-Gr01U1i_DE/view) 与 [11/13 数据协议](https://drive.google.com/file/d/1TxzLSq71INFWbKi5hFO9fxvtucb6RphA/view)；在可访问 Drive 中未找到指定 `VoxWM_VAE_P1_DATA_PROTOCOL_2026-11-14.md`，按 prompt 明确允许的回退规则实施。P1 的 Controlled Initialization / Sugar Cane 规则优先于 P0 的限制。

## 本地无游戏检查

复用现有 `.venv`，无新依赖；新机器先按根目录 `scripts/setup_gt.sh` 安装原生环境，再安装 `vae_p0/requirements.txt`。以下命令均从 VoxWM 根目录执行：

```bash
.venv/bin/python -m unittest vae_p1.test_p1
.venv/bin/python -m vae_p1 --dry-run --output outputs/p1_dry --seeds 42
```

默认模式也是 Dry Run。它使用明确标记的数组 fixture 验证编排、编码、存储、配对与可视化，**不启动 Craftium，不模拟流体物理，不是原生实验结果**。输出目录必须不存在。

## 用户手动执行原生 P1

先少量尝试，以下是手动命令，Codex 未执行：

```bash
# 一个种子的一组完整 A/B/C；三种植物分别有同等搜索预算
.venv/bin/python -m vae_p1 --native --output outputs/p1_seed42 \
  --seeds 42 --settings A B C --categories D1 D2 D3 D4 \
  --sites-per-category 1 --delay-seconds 120 --wall-seconds 21600

# D2 Controlled：10 次场景尝试精确对应 7 普通 / 2 条2–4格路径 / 1 紧邻水体
.venv/bin/python -m vae_p1 --native --output outputs/p1_lava_controlled \
  --seeds 42 --settings C --categories D2 --sites-per-category 10 --wall-seconds 7200

# D1 Controlled：5 次尝试分配池边、落差、窄通道、障碍、Chunk边界
.venv/bin/python -m vae_p1 --native --output outputs/p1_water_contexts \
  --seeds 42 --settings C --categories D1 --sites-per-category 5 --wall-seconds 3600

# 植物区域：每种植物均尝试一个区域，并行观察最多20个有效候选
.venv/bin/python -m vae_p1 --native --output outputs/p1_plants \
  --seeds 42 --settings C --categories D4 --sites-per-category 1 --wall-seconds 7200
```

Linux 无显示设备时加 `--renderer software`；macOS 默认窗口。`--native` 必须指定墙钟预算。预算耗尽会保留已捕获状态并明确记录 censored / failure，不自动延长或重试。

A/B 默认在同一命令中配对执行：每个场景开启并销毁独立 `GTSession`，A 选定的世界坐标冻结给 B；B 不在 A 的世界里等时间。若分开执行：

```bash
.venv/bin/python -m vae_p1 --native --output outputs/p1_A --settings A --categories D1 --seeds 42 --wall-seconds 900
.venv/bin/python -m vae_p1 --native --output outputs/p1_B --settings B --categories D1 --seeds 42 \
  --paired-candidates outputs/p1_A/pairs.json --wall-seconds 1200
```

分开执行时须保持 seeds/categories/sites 参数一致。B 保留原坐标即使候选已经成熟或消失，记录复核结果，不另找更好的点替换。Native Mod 的随机结果不保证逐位复现；地形/候选原始状态保存在原始轨迹中，不能只凭相同 seed 断言所有状态完全相同。

## 采集机制

- **A Fresh Natural**：新区域 emerge / activation 后立即捕获，再筛选候选；筛选期间发生的变化仍包含在 A 的监测窗口内。保留 Mod 初始化、首个 globalstep、on_generated 回调、emerge、activation、first capture 与 observation start 的引擎时间和 step。First Capture 不被称作 Mapgen t=0。
- **B Delayed Natural**：同 seed/site/候选的独立世界；默认先自然演化120秒（可配置），期间仍保存原始观察，但不把这些状态计入 B 的正式窗口。只有确实观察到目标变化且后续窗口无变化，才标记已提前稳定。
- **C Controlled**：先记录原态，单次放置 source / 原生幼株或点火；同一 Lua callback 立即捕获初始化后状态。进度从最终 post-intervention 状态归零，不把 Intervention 变化算作 Native Evolution。

候选流程：天然节点发现 → 物理资格过滤 → biome/地层/可流动空间/落差/障碍/水距/边界/局部多样性分层 → seeded random selection。Controlled 从天然表面、可燃物的邻接空气位置找点，而非随机抽整块空中位置。保存源节点数、位置数、抽查数、拒绝原因及最终选择。

D1 首次60秒；D2 180秒；D3 120秒。D4 为累计600 / 1200 / 1800秒，仍有原生合法生长候选才延长。任何类别已有充分的8状态原生事件即可提前结束当前场景。所有秒数是**真实引擎模拟时间**，不改 ABM、概率、dt 或生长机制。

- D2 的70/20/10作用于 **Controlled Scene Attempts**。小于10时采用最大余数整数分配，实际分配写入 plan；不因某类没候选就偷换到另一类。水距2–4要求天然水体及水平/向下的空气路径；未来是否真的流到水由原生流体决定。固化仅在水/岩浆原态变为新产物且有接触几何证据时确认；预先存在的石头不算反应。记录七种节点计数、接触、产物及观察时间区间。捕获间隔内瞬时接触若无证据，不强行归因。
- Fire Controlled 要求20个物理有效且相距至少3 voxel的点，每点独立p=0.1；全未选中才随机保底1点；实际全点火失败时最多再尝试剩余点中的1点。点数不足直接报物理环境无效，不伪造20点。保留全部随机结果、点火位置/结果/时间及保守的连通火势合并证据。
- D4 竹/甘蔗/藤蔓搜索预算相等，不强制成功数相等。竹通过原生 `on_place` 创建幼株，由官方函数随机分配高度，采集器不设最终高度、不调用 grow。甘蔗使用原生 soil/water/frosted-ice 规则；藤蔓核对朝向、支撑和扩展空间。仅衰退不能成为标准生长 clip。

## 格式与结果

```text
output/
  plan.json, pairs.json              # 随机计划、冻结配对位置
  vocabulary.json                    # P0 Combined Token, append-only uint16
  snapshots.zarr/{tokens,light}       # 仅标准5–8状态clips
  metadata.parquet                    # seed/instance/固定ENU原点/事件/时间/step/split
  raw_trajectory.bin, raw_trajectory.jsonl  # 初态+稀疏 token/light 增量及每次捕获索引
  aux_short_event/                    # 2–4状态真实短事件，独立Zarr/Parquet，不进标准池
  events.json, statistics.json, VAE_P1_RESULTS.md
  visualizations/<scene_id>/         # 状态、目标新增/消失/状态替换投影及GIF
```

所有数据固定16³，raw light仅辅助保存，不是 VAE 输入。多个块共享一个字典；不全局删除独立上下文的重复局部结构。全 World Seed 的 split 保持一致。原始轨迹保存所有捕获，包括未改变和 light-only 状态；可无损重建的是这些捕获时刻，不是观测间隔中从未读取的引擎状态。

标准 clip 沿累计目标相关变化进度选5–8个不同状态，含一次初始化后基线；干预本身不贡献变化量。短片段保存至 aux，衰退/灭火等会被明确标记，不自动当作成功生长/传播。对每个场景一次计算观测时间分母，不把同一事件的多个 snapshots 当成多个独立试验。

```python
from vae_p1.storage import Reader
from vae_p1.tracking import Vocabulary
root='outputs/p1_seed42'
s=Reader(root).frame(0)  # raw_record_id
node_names, raw_param2=Vocabulary(root+'/vocabulary.json').decode(s.tokens)
```

从原始轨迹补看干预前/刚完成/原生演化，输出单独的 `_raw_phases` 图片，不改数据：

```bash
.venv/bin/python -m vae_p1.visualize --root outputs/p1_seed42 --scene-id 42_D1_all_0_C
```

主指标按 setting/category/plant 记录标准动态 snapshots / simulation-hour 和 /wall-hour。wall分母含该场景引擎启动、发现、初始化、观察和关闭；simulation分母为正式观察窗口，B 的等待时间另记。另报候选率、原生事件率、短片段、首次变化延迟、增长/衰退、变化位置、初始化成功、空间/biome与水岩浆产出。没有观测不写成零成功的科学结论；Dry Run 的原生指标为 null。

## 时间预算依据与尚未测量项

一个seed、每类一个site、ABC全开，共18个独立引擎实例（D4三种植物）。名义首次观察6480模拟秒，加 B 等待720秒，共 **7200秒 = 2模拟小时**；若植物均延长至1800秒，最多 **18000秒 = 5模拟小时**。提前得到8状态可更短。上述不含启动、emerge、搜索、编码、写盘开销；不能把模拟小时直接当成墙钟小时。

P0 只有短窗口、较少块的吞吐，不能当作P1实测速度。P1新增候选筛选、12个固定Chunk和原始轨迹IO；真实墙钟吞吐、有效率、内存/磁盘量及200K成本均待用户实跑。初始运行须检查7/2/1分层的真实候选可得性、天然竹的隐藏cap、原生幼株放置与长期生长、火势合并和快速水岩浆接触漏检。若失败，报告真实失败；本入口无200K或GPU训练开关。

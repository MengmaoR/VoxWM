# VoxWM — Voxel VAE E01-Masked

Vox2Pix v3 已通过三类场景共 15 帧的原生验证。实验使用单层白模、Water Count 和统一的原生亮度下限。[v3 采集与报告命令](vox2pix_data/README.md#v3-native-smoke) · [实测照明与沉底结果](vox2pix_data/reports/v3_lightfloor_smoke/README.md)。

历史 Vox2Pix v2 1K 共 200 个场景、1,000 个视角。[三层白模 Viewer](vox2pix_data/README.md#inspect-an-existing-pilot) · [实测速率与采集日志](vox2pix_data/reports/pilot_1k/README.md)。

两个正式100K Clip已完成。E01-Masked 用两个 Clip 的完整训练集训练 Combined-Token 3D ResNet VAE。E00 小样本结果保留为历史。VAE 从每个 48³ 样本读取中心 16³，训练保存分项 loss 曲线并支持独立验证早停。[E01 数据准备、训练与监控命令](voxel_vae/README.md)

E01-Masked 在 30,000 步结束续训。[服务器结果、曲线与配置](voxel_vae/reports/e01_masked_20261122/README.md)

27,000 步验证最优 checkpoint 已完成 20,000 条 held-out test 和全训练集 latent 审计，结论为 **FREEZE WITH KNOWN D3 LIMITATION**。[冻结报告与下游归一化](voxel_vae/reports/e01_freeze_audit_20261122/report.md)

[当前数据协议](VAE_DATA_COLLECTION_PROTOCOL.md) · [历史生成/检查命令](vae_data/README.md)。模型代码独立位于 `voxel_vae/`，复用已有数据格式。

新增[动态观察批传输](vae_data/reports/advance_batch_20261117/README.md)：默认24拍/可选48拍，物理仍逐拍24Hz、每拍捕获；首帧+稀疏delta在内存中经socket返回，原Collector决定停止并丢弃批尾。启用前由用户运行`bash scripts/build_fixed24.sh`增量编译；本次不调整并行数，续跑示例固定`--workers 64`。没有运行Native或实测吞吐。

新增外部[采集监控与自动恢复](vae_data/README.md#外部监控与自动重启)：`start`把监控器及采集树放入脱离SSH的后台会话，`dashboard`单独只读显示进度，`stop`精确停止该后台运行并确认锁已释放。默认同时最多64个Shard、每片1个世界；每个活跃Shard独立执行10分钟停滞检测，只终止本次唯一随机标记继承出的Python/Craftium/Xvfb进程，确认清理完成后按同一配置`--resume`。它不修改配额、24Hz或采样协议。

已有旧动态、另采新策略Static时使用[Static-only supervisor配置](vae_data/README.md#仅补齐static缺额)：Static写入新的`vae_100k_heightadaptive`目录，每片仍以500为目标；旧动态目录不重采、不覆盖，也不与不兼容manifest混写。

两批完成后使用[受控合并脚本](vae_data/README.md#合并完整100k)：保留旧目录的50K D1-D4、删除其旧Static，并导入新目录的50K height-adaptive Static。脚本会合并并重映射不同的Combined Token词表，在独立staging中重建/验证100个Shard，再保留备份后切换活动路径；Codex不自动执行真实合并。

已修复[服务器高并行采集停滞](vae_data/reports/parallel_runtime_20261117/README.md)：独立Xvfb、无新增Token免写词表、新词表写盘移出全局锁、锁等待/无提交停滞检测。`--workers auto`按可用CPU分配世界进程，也可显式指定200等任意正整数，不受100片限制；每片仍只有一个写入器。仅完成本地快速进程测试，真实高并行续跑由用户执行。

已完成[10群落Static真实试采与旧预览集成](vae_data/reports/static_ten_20261117/README.md)，高度分配4/2/2/2，数据位于本地outputs，不入Git。

Static已支持逐整数中心高度与完整48³世界边界；任意高度Capture通过13个合法高度的C++/VoxelManip逐体素核对，并取得Elevated/Shallow/Deep真实示例。[高度修订报告](vae_data/reports/static_height_20261117/README.md)。开始新策略采集前运行`bash scripts/build_fixed24.sh`，使用新目录，不能把新Static混入旧策略数据；Dynamic网格及Fixed24规则不变。

当前用户指定的 **D1/D3/D4-A/B/C fixed24重跑全部通过**，D2沿用已通过的旧结果、不重跑。源点中心14³、D3超时120 Tick、逐Tick GT已在新样例中运行；Viewer 8766已切到新目录并标记D2历史来源。[本轮结果/预览](vae_data/reports/fixed24_rerun_20261117/README.md)。完整受控RNG跨吞吐等价仍未证明；未运行正式1K/100K或训练。

[Node-state 审计与 Viewer](vae_data/reports/transition_audit_20261116/README.md)：已核对D2 attempt_2的18.9–19.2秒；消失的是Flowing Lava，被替换的是红蘑菇。支持原生液面/Firelike、固定世界坐标Before/After和点击Inspector。Viewer展示已有GT，不参与采集。

```bash
# 全部真实生成由用户在CPU服务器手动启动，下面只展示入口
.venv/bin/python -m vae_data generate --dataset outputs/vae_100k_heightadaptive --shard 0 --count 1 --workers 1 --wall-seconds 7200
.venv/bin/python -m vae_data check --dataset outputs/vae_100k_heightadaptive --shard 0
.venv/bin/python -m vae_data status --dataset outputs/vae_100k
```

纯Voxel边界复用隔离引擎overlay，禁止实体、掉落与当前不支持的falling-entity动态。旧P0/P1/pilot、Unified代码/结果/日志与docs保留；旧README及协议快照在`vae_data/LEGACY_UNIFIED_README.md`、`vae_data/history/`。旧格式不是当前生产入口。

以下保留独立GT环境的安装和历史使用方法；P0/P1/pilot及 `docs/` 不覆盖、不删除。

供新世界模型研发使用的干净数据入口。只有Craftium/Luanti/VoxeLibre、GT采集与真实视频；没有神经网络、权重、推理、旧Viewer或compare依赖，不需连接旧CoDL目录。

已安装/编译独立 CPU 原生环境，并已执行有界 VAE 数据试采；Test A/seed42 的专用完整视频测试仍由用户手动启动。开发仓库为 `https://github.com/MengmaoR/VoxWM.git`（master）。远端既有`docs/`展示页及数据完整保留，不属于GT模块的运行依赖。

```text
voxwm_gt/             引擎session、生成、游玩、录制、稀疏存储
  scenes/test_a.lua   原Test A合成场景
  scenes/native_world.lua  原生地图/用户voxel世界与同帧C++采集
  assets/             本地类别表，不是模型资源
scripts/setup_gt.sh   独立CPU环境安装（用户执行）
scripts/test_a.py     原Test A真实GT和视频测试
scripts/test_seed42.py 原生seed42真实GT和视频测试
outputs/             用户运行后的数据，不入Git
HANDOFF_PROMPT.md     给新对话的交接prompt
```

## 1. 安装（用户执行）

从本目录执行，要求git、uv、CMake、make/C++编译器和原生图形/音视频库。脚本不执行sudo/brew、不修改旧环境。CPU仿真不要求CUDA；RGB仍需要图形栈，macOS使用原生窗口，Linux默认Mesa软件渲染，每会话自动启动并回收独立Xvfb，不需要预先设置DISPLAY。

macOS系统依赖参考（已安装则无需重复）：

```bash
brew install cmake pkg-config sdl2 jpeg-turbo libpng freetype sqlite zstd luajit libogg libvorbis openal-soft curl gettext gmp
```

Ubuntu系统依赖参考（由用户手动执行）：

```bash
sudo apt-get install build-essential cmake pkg-config libsdl2-dev libgl1-mesa-dev libx11-dev libxi-dev libjpeg-dev libpng-dev libfreetype6-dev libsqlite3-dev libzstd-dev libluajit-5.1-dev libogg-dev libvorbis-dev libopenal-dev libcurl4-openssl-dev gettext libgmp-dev zlib1g-dev xvfb
```

```bash
cd /Users/tianxj/Desktop/VoxWM
bash scripts/setup_gt.sh
```

脚本直接clone固定版本Craftium及其原生submodules，在本目录创建Python3.11 `.venv`。无需PERSIST仓库或模型资源；原macOS PNG/OpenGL/OpenAL处理及Linux软件渲染保留。不要安装Craftium的`examples`依赖组。

## 2. 两项真实测试（用户分别启动）

原Test A：玩家不平移，以真实鼠标动作转约2.5圈，原树木/地下石层、3×3悬空水源、河道与池塘。保留seed42、48³、24FPS、640×360、FOV72、原停止规则；不固定为580帧，原生时序可能影响结束帧数。

```bash
.venv/bin/python scripts/test_a.py
```

原生seed42：不覆写地形、不建人工湖；原生v7地图和自然出生点，默认No-op录240帧，保留实际环境演化。

```bash
.venv/bin/python scripts/test_seed42.py
```

测试脚本真实调用引擎，生成后只检查voxel/track/video的长度、48³形状及Camera可用性。需要极短人工smoke时加`--max-frames 16 --output outputs/test_a_short`，短数据不等于完整Test A。所有输出目录必须不存在，不会覆盖旧输出。

默认输出`outputs/test_a/`与`outputs/seed42/`，各自包含：

- `voxels.bin`：uint16类别，首帧完整+后续稀疏增量，DeltaReader可无损重建。
- `track.npz`：Camera、Action、native_action、真实Voxel Center/Origin、Player/Camera位置、方向、时间和帧号。
- `rgb.mp4`、`initial_rgb.npy`：真实引擎RGB，24FPS；非AI合成视频。
- `classes.json`、`classdict.json`、`nodes.json`、`record.json`及引擎日志。

```python
from voxwm_gt.storage import DeltaReader
r = DeltaReader('outputs/test_a/voxels.bin')
voxel_t = r.frame(10)
```

## 3. 通用生成与实际游玩

```bash
# 原生地图，可换seed、mapgen、帧数和真实动作策略
.venv/bin/python -m voxwm_gt generate --scene procedural --seed 42 --max-frames 240 --policy rotate --output outputs/native_rotate
# 用本地NPY给定动作：整数[T,4]，每行是一个真实引擎transition；最多保存T+1帧
.venv/bin/python -m voxwm_gt generate --scene procedural --seed 42 --actions actions.npy --max-frames 241 --output outputs/native_actions
# 交互游玩；同一录制器同步保存观测，不指定output则只玩
.venv/bin/python -m voxwm_gt play --scene procedural --seed 42 --output outputs/play_seed42
.venv/bin/python -m voxwm_gt play --scene test_a
```

在“VoxWM GT”窗口用WASD移动、空格跳跃、Shift潜行、Ctrl加速、方向键转头；Tab切换鼠标捕获，左键挖掘、右键放置，按住1–9并右键选择物品槽，Esc退出。操作均通过实际引擎动作执行；没有逐帧绝对轨迹。引擎临时世界在退出后清理，若要保留本次数据需指定`--output`。

可用`GTSession`接自己的控制策略：

```python
import numpy as np
from voxwm_gt.session import GTSession
with GTSession(scene='procedural', seed=42) as session:
    rgb, info = session.rgb, session.info
    rgb, reward, terminated, truncated, info = session.step(np.zeros(4, dtype=np.int64))
```

## 4. 指定合成voxel数据

`--scene custom --world scene.npz`将用户给定初态写入真实游戏，再由原生物理演化；不模拟自己的流体。NPZ格式：

- `voxels`：整数[X,Y,Z]，轴按Luanti引擎(x,y,z)，值索引node_names。
- `node_names`：一维Unicode节点名，例如air、mcl_core:dirt、mcl_core:water_source。
- `node_param2`：可选，一维且与node_names等长，默认0。
- `origin_engine`：3个整数，voxels[0,0,0]的引擎节点坐标。
- `spawn_engine`：3个数，合法玩家初始位置；场景应提供支撑/空间。
- `look_degrees`：可选[yaw,pitch]，默认[0,0]，引擎角度。

```bash
.venv/bin/python -m voxwm_gt generate --scene custom --world scene.npz --seed 42 --output outputs/custom_gt
.venv/bin/python -m voxwm_gt play --scene custom --world scene.npz --output outputs/custom_play
```

原生地图和自定义世界保留真实物理；只对原Test A使用其水流稳定+旋转结束的停止条件。自定义世界不自动裁掉模型未知类别，遇到新node/param2会追加类别并记录。

## 5. 坐标与迁移边界

每份RGB、Voxel、Center、Camera来自同一引擎观测包。使用`get_voxel_data_cpp`，官方wrapper转ENU(x,z,y)，49³按right-crop变为48³。Camera仍保存10D表示：位置`(cam_pos-(obs_voxel_center-.5))/48`，真实crop origin独立为`obs_voxel_center-24-.5`，不从玩家位置推测Center。

来源和许可见THIRD_PARTY.md。没有迁移B/C场景、baseline模型、推理/评价、网页、旧结果、旧虚拟环境或Git历史。本目录已接入VoxWM GitHub，保留其原有历史和docs内容，不继承旧CoDL的Git历史。新架构与训练由用户后续独立定义。


## 历史 VAE P1

[历史运行手册](vae_p1/README.md) · [原实现报告](vae_p1/IMPLEMENTATION_REPORT.md)。旧 P1 的14项 Unit Tests 和18场景 Synthetic Dry Run不是本轮Native结果；其固定20点火、人工幼株初始化等规则已被当前协议替代。原代码、报告和输出全部保留，不作为正式200K入口。

## 历史 VAE P0 数据闭环

以 [2026-11-13 数据协议](https://drive.google.com/file/d/1TxzLSq71INFWbKi5hFO9fxvtucb6RphA/view) 为准；实现和格式见 [vae_p0/README.md](vae_p0/README.md)，真实结果见 [VAE_P0_RESULTS.md](VAE_P0_RESULTS.md)：**NO-GO**，26 个合格 snapshots（Ordinary10 / Water8 / Fire8），Lava0 / Vegetation0；不能启动正式200K。固定世界 16³、Combined Token `(Node Name,raw param2)`、独立 raw light、真实时间/模拟步、5–8 个 distinct-state 变化进度采样，Zarr+Zstd/Parquet。没有神经网络或 200K 自动任务。

```bash
uv pip install --python .venv/bin/python -r vae_p0/requirements.txt
.venv/bin/python -m vae_p0 --output outputs/vae_p0_new --wall-seconds 720
```

## 旧采集试跑（新数据协议前）

历史[原生16³试采](vae_pilot/README.md)使用独立 node/param2，尚未满足当前 Combined-Token 协议；其 `--full` 不可作为当前正式生成入口。旧目标为固定Ordinary100K、Water/Lava/Fire/Vegetation各25K。支持随机世界、25类biome均衡、原生动态发现、5–8状态change-progress clips、按world seed拆分及实际产出统计；不包含VAE训练。

2026-11-13已安装独立CPU环境并完成有界真实小测试：1世界/4位置，137 snapshots（Ordinary89、Water48，6个8状态水流clip）；其余三类有效产出为0。真实统计见[本次报告](vae_pilot/reports/2026-11-13/report.md)，不能据此承诺完整200K的可达性或成本。未运行200K或GPU。

```bash
.venv/bin/python -m vae_pilot --output outputs/vae_pilot --max-worlds 1 --sites-per-world 4 --site-seconds 24 --sample-seconds 0.25 --fire-ignite-after-seconds 3 --wall-seconds 210 --snapshot-cap 256
```

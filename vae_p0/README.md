# VoxWM VAE P0 — 原生数据最小闭环

当前唯一采集协议是 [2026-11-13 Pilot Data Protocol](https://drive.google.com/file/d/1TxzLSq71INFWbKi5hFO9fxvtucb6RphA/view)，优先于 [2026-11-12 VAE Manifest](https://drive.google.com/file/d/1hYmOg6SHIEPYcjvUmKgjP-Gr01U1i_DE/view) 的旧数据表示与逐变化采样约定。这里不实现或修改 VAE 网络。

## 运行

在 VoxWM 根目录，使用已经由 `scripts/setup_gt.sh` 配置的独立 CPU 原生环境：

```bash
uv pip install --python .venv/bin/python -r vae_p0/requirements.txt
# 一个极小表示/采样/存储检查，不启动游戏
.venv/bin/python -m vae_p0.check
# 新建输出目录；不覆盖旧结果。真实 P0 默认总墙钟预算 720 秒
.venv/bin/python -m vae_p0 --output outputs/vae_p0_new --wall-seconds 720
```

macOS 使用本地图形栈；Linux 可使用 `--renderer software`。只运行一个随机 v7 世界，默认随机种子发生器 seed=42，第一个世界 seed=191664964；`--seed` 可显式指定世界种子。之前随机世界发现的水/熔岩/森林/竹藤位置只作为候选线索，因此本次不是无偏的事件发现率调查。换世界仍按真实节点筛选，不保证这些位置存在目标。

先采少量 Ordinary，再分别观察 D1 水、D2 岩浆、D3 火、D4 竹/藤。失败类别最多做一次已写明原因的有界修复；不延长原生 ABM、修改生长元数据、布置流体或构造植物。火先找自然火，必要时在自然可燃环境随机点燃一次；保留点燃前状态。不会启动 200K 或训练。

## 格式

- `snapshots.zarr/tokens`：`uint16[N,16,16,16]`，一个 Combined Token，无独立 type/state embedding。
- `snapshots.zarr/light`：原始 `param1`，`uint8[N,16,16,16]`，仅辅助保存，不是模型输入。
- `vocabulary.json`：`id <-> (Node Name, raw param2)`，追加 ID、不重排，最大 65536；每次采集对全部 voxel 进行精确编码/解码比对。
- `metadata.parquet`：`snapshot_id, category, event_id, ignition_mode, change_progress, world_seed, chunk_origin, chunk_index, axes, biome, surface_biome, biome_group, biome_transition, stratum, surface_height, height, timestamp, simulation_step, engine_us, wall_time_unix, split, craftium_commit, voxelibre_commit`。
- `statistics.json`、`VAE_P0_RESULTS.md`：实测数量、各尝试的 distinct states/变化量、产出率、失败与有条件线性估算。零产出类别的成本估算是 `null`。
- `visualizations/`：真实候选状态的 PNG/GIF 与可重建的 NPZ/JSON；不足 5 状态明确标为非标准 clip，不计训练样本。

数组轴为 ENU `(east,north,up)`；原点是首个节点的整数世界坐标，且每轴为 16 的倍数。Chunk 始终固定，不依赖移动 Camera 或裁剪估计。所有原始状态读取来自同一次 VoxelManip `get_data/get_light_data/get_param2_data`；每个新观测区域的 4096 节点逐项与原生 `get_node` 的 name/param1/param2 比对。既有 Craftium Python C++ channel 只有 node/param2 两通道，故此处用官方 VoxelManip 接口补全 raw light，不伪造 light；原有 48³ GT 的 C++ 采集不变。

动态监测逐真实命令响应采样，但最终状态按**累计目标相关 Token 变化进度**选取，目标 8、最低 5，不按时间等间隔凑帧，不补重复。仅 light 变化和无关节点变化不能增加目标进度。同一事件内去重，不对不同世界/独立上下文做全局状态清除。藤蔓/竹子仅消失不能记作 D4 生长；火焰只有初始点火或年龄改变、没有原生传播时也不算 D3 成功。split 完全由 world seed 决定；一个 P0 世界不能验证整个 80/10/10 分布。

Ordinary 的地形标签相对于实测地表和结构；纯空气/石块保留少量，不删除整类。P0 记录实际 biome/地层覆盖，不把 25-biome 目标写成已覆盖。

## 检视数据

```python
import zarr, pyarrow.parquet as pq
from vae_p0.codec import Vocabulary
root = 'outputs/vae_p0_new'
a = zarr.open_group(root + '/snapshots.zarr', mode='r')
meta = pq.read_table(root + '/metadata.parquet').to_pylist()
names, raw_param2 = Vocabulary(root + '/vocabulary.json').decode(a['tokens'][0])
raw_light = a['light'][0]
```

本入口仅为 P0，没有 200K 开关。旧 `vae_pilot/` 的 raw-node/param2 数据流程是 11 月 13 日新协议前的试采历史，不可直接当作当前 Combined-Token 200K 生成器。真实 P0 报告保存在根目录 `VAE_P0_RESULTS.md`，原始大数组留在本地 `outputs/`。完成后由用户决定下一步。

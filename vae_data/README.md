# VAE 100K — 1K Shard Generator

当前入口：`python -m vae_data` → `vae_data/sharded/`。仅生成数据，无模型/训练器/DataLoader。依据[当前协议](../VAE_DATA_COLLECTION_PROTOCOL.md) / [Drive Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view)。旧200K代码、结果与日志保留；[旧README](LEGACY_UNIFIED_README.md)中的命令需使用 `python -m vae_data.legacy_entry`，不是当前生产格式。

**当前原生结果为INCONCLUSIVE**：本轮8次有限尝试，D1成功2条，D2/D3未达到五状态，D4未成功。JSON响应交接错误已做原子写修复，但没有追加native运行；详见[实测与检查报告](SHARDED_RESULTS.md)。1K/100K正式采集、完整25群落吞吐均未实跑，不把CLI存在当作已验证产能。

## 服务器环境（用户手动执行）

从VoxWM仓库根目录操作。已有环境无需重新安装；CPU仿真，无Torch/CUDA/模型权重。

```bash
git pull --ff-only origin master
# 仅首次新服务器安装；系统依赖见根README。脚本不自动sudo。
bash scripts/setup_gt.sh
uv pip install --python .venv/bin/python -r vae_p0/requirements.txt
```

Linux示例使用 `--renderer software`，macOS可改为 `--renderer auto`。不改变原生液体/火规则，不做时钟加速。

## 生成指定一个1K片

```bash
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k --shard 0 --count 1 --workers 1 \
  --max-attempts 5000 --wall-seconds 7200 --renderer software
```

`shard`索引0..99。每片硬配额：Static500，四动态各125；每类25条五状态链。Static每群落20条，8/4/4/4。D4是水岩浆固化，不是植物；子类型A/B/C每片8/8/9轮换。动态中心连续2个模拟秒无目标变化即停止，不能为了过测延长。未达到配额只保留partial和缺口，不READY、不替代类别。

## 连续多片 / 多CPU workers

```bash
# 生成 shard_0010 .. shard_0013，最多4个独立CPU世界并行
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k --shard 10 --count 4 --workers 4 \
  --max-attempts 5000 --wall-seconds 14400 --renderer software
```

Worker按片独占目录与锁；Native世界/临时引擎独立；全局词表用短锁追加ID。只需要一个命令调度同一批片；相同片被另一个进程占用时明确失败，避免并发写入。

## 中断恢复

```bash
.venv/bin/python -m vae_data generate \
  --dataset outputs/vae_100k --shard 10 --count 4 --workers 4 --resume \
  --max-attempts 5000 --wall-seconds 14400 --renderer software
```

Ctrl-C会停止本命令拥有的worker和引擎进程；已提交数据可恢复。已READY片直接跳过，保持原文件；partial片恢复已提交样本/词表/RNG cursor，去除未提交Zarr尾部。未完成场景开启新的native rollout，不恢复旧模拟物理时间。`max-attempts`是本次每片尝试上限，墙钟为整条命令总上限；到限可不满配额，需要用户决定是否继续。

## 检查某片、查看全局进度

```bash
.venv/bin/python -m vae_data check --dataset outputs/vae_100k --shard 10
# 检查partial中已提交数据，允许数量尚未齐；complete=false不会发布READY
.venv/bin/python -m vae_data check --dataset outputs/vae_100k --shard 10 --allow-incomplete
.venv/bin/python -m vae_data status --dataset outputs/vae_100k
```

Check为只读，不自动修数据。错误返回非零；status只报告READY与partial统计，100个READY才是正式100K完成。检查结果包含Transition软目标缺口；不能把不足4的真实区域伪装成transition。

## 格式和无损读取

每个正式样本：`tokens:uint16[48,48,48]`、`light:uint8[48,48,48]`，ENU。目标中心为 `[16:32]*3`，region origin = center origin −16。上下文保存完整，动态判定与VAE目标都限定中心。

```python
import zarr, pyarrow.parquet as pq
from vae_p0.codec import Vocabulary
root='outputs/vae_100k'
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

## 开发检查与授权原生小例

```bash
.venv/bin/python -m unittest vae_data.sharded.tests.test_sharded
# 仅用户明确需要时：新输出目录；每类成功上限2，总计最多8，禁止自行循环扩测
.venv/bin/python -m vae_data native-smoke \
  --dataset outputs/vae_native_examples_new --wall-seconds 300 \
  --samples-per-category 2 --attempts-per-category 2 --renderer software
```

小例不创建正式Shard，不写READY。Synthetic Fixture写到独立unit模式临时目录，正式生成器拒绝该格式；不能拿单元测试证明物理成功。

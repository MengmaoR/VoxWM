PASS

# VoxWM E01 多 Clip Token 对齐快检

在服务器 `/user-data/txj/VoxWM` 上只读检查两个原始 Clip、既有 E01 索引、发布词表和 27,000 步最佳 checkpoint。

## 词表及映射

| 统计 | Clip 1 | Clip 2 |
| --- | ---: | ---: |
| 本地词表 K | 1,538 | 1,537 |
| 独有语义条目 | 43 | 42 |
| 重复 Local ID 或重复语义 | 0 | 0 |
| 逐条映射通过 | 1,538 / 1,538 | 1,537 / 1,537 |

共有 **1,495** 个 `(node_name, raw param2)`，其中 **1,490** 个在两份 Clip 中的 Local ID 不同。Canonical 词表是按该二元组排序的精确并集，**K = 1,580**。两个 root 的映射全部覆盖、索引在范围内、各自一一对应。跨 root 的相同语义落到同一 Canonical ID，语义碰撞为 0。

| Node | raw param2 | Clip 1 Local ID | Clip 2 Local ID | Canonical ID |
| --- | ---: | ---: | ---: | ---: |
| `air` | 0 | 2 | 0 | 0 |
| `mcl_mushrooms:mushroom_brown` | 12 | 0 | 191 | 1086 |

## 实际训练输入

从既有索引覆盖每个 root 的 100 个 shard，每 shard 抽 1 个 Static 和 1 个 Dynamic，轮换 D1–D4 并覆盖 train、validation、test。比对原始 Zarr 中心 16³ 与 `VoxelSnapshotDataset.read_numpy()` 的实际返回值。两份 Clip 合计 **400 个快照、1,638,400 个体素**，Local ID 越界 0，逐体素语义偏差 0。两边样本共有 289 种语义，均得到相同 Canonical ID。

- [Dataset](../dataset.py#L262-L284) 根据 `record.root_index` 查对应 LUT，将中心体素映射为 canonical `int64`。Batch 采样器只生成样本索引，训练从该 Dataset 取张量后才调用模型。[Sampler](../sampler.py#L21-L43) · [Train](../train.py#L460-L487)
- 验证和 held-out test 复用相同 Dataset。Change Mask 也先读取 canonical 五状态，再做逐位置差分。沿 E01 的训练入口核对了所有模型调用，均经过上述重映射。[Validation](../evaluate.py#L42-L46) · [Test](../freeze_audit.py#L157-L175) · [ChangeMaskCache](../change_masks.py#L88-L98)
- 模型只创建一份 `nn.Embedding`，用 canonical IDs 查表后进入同一个 encoder。[Embedding 定义](../model.py#L28) · [Lookup](../model.py#L68-L78)

## 既有索引与 checkpoint

实际索引和 checkpoint 中的 root 顺序均为 **clip1 → clip2**，与两个 LUT 一致。按已保存的 E01 split 读取现有索引，训练池为 Clip 1 **79,980** 条、Clip 2 **80,020** 条，总计 **160,000** 条。Validation 和 test 各 20,000 条。

- `dataset_index.json` 文件 SHA256 为 `9b0cdb13e41fee88acb18a3f60091f582842806806fb64657acad31142acbe3b`，与 checkpoint 的准备数据记录一致。
- 27k checkpoint 的词表 K 为 **1,580**，唯一 embedding 参数为 `model.embedding.weight`，形状 **(1580, 32)**。保存的词表条目与发布文件的 `tokens` 完全一致。
- Checkpoint 记录的 canonical 语义 SHA256 为 `25af338d8200eaceedf62bd33dc76d769dcec27f526e6df34280c43c9b3f979a`，等于 `training_vocabulary.json` 内的 `sha256` 字段。发布词表和训练目录词表的整个 JSON 文件 SHA256 均为 `bc383e57f882433099ef4e27832f24e19093649456e2ef217a51df0bc81758d0`。两种哈希分别对应语义条目和完整文件。

**结论为 PASS。** 两个 Clip 按相同语义进入同一张 embedding 表。原始数据、词表、模型权重、训练代码、索引及划分均保留原样。Checkpoint 只在 CPU 上读取元数据和参数形状，Zarr 检查限于上述 400 个样本。

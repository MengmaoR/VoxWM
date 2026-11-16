# Native transition audit — D2 / attempt_2

NO_SOURCE_LOSS_OR_UNEXPLAINED_SOLID_TO_LIQUID_IN_OBSERVED_WINDOW

Original trace: `outputs/vae_native_six_20261116/traces/D2/attempt_2`. Rollout `e14fc750b7614f6887e2b8552ac49b4f`. Requested 18.9–19.2s; 25 observed voxel transitions.

## Findings

审计对象按用户修正为 **D2 / attempt_2**。完整原始VWT1在本地保留；没有重模拟。

- 18.903740935 → 19.177778929s（观察帧15→16）共25个完整48³ Combined-Token转换：15个液位/向下流标志更新、8个邻域Water Flow出现、1个Flowing Lava消退、1个红蘑菇被Flowing Lava取代。
- `(-258,1440,-2)`：`mcl_core:lava_flowing, param2=2 → air,0`。不是Source。其同层邻居`(-257,1440,-2)`由param2=7变为15，开启Flow-Down位；Craftium `src/servermap.cpp:853` 的同层供液规则明确排除向下流的邻居。这与失去横向供给后退流一致，未记录内部每一步的队列归因。
- `(-257,1440,-3)`：`mcl_mushrooms:mushroom_red,0 → mcl_core:lava_flowing,0`。**不是Dirt→Lava**。原生蘑菇定义含`destroy_by_lava_flow=1`，且前一观察存在上方Lava；VoxeLibre原生ABM允许破坏它，再由液体填入。没有采样到中间Air，不能补造这一帧，也不能声称已捕获具体破坏回调。
- 两枚初始化Source `(-257,1439,-2)`、`(-256,1441,-1)` 从观察帧1到31一直保持`lava_source,0`。目标位置前后六邻域均无Water；该Trace的cooling事件列表为空。这里的结论限于已记录观察/写入，不排除所有未记录的原生回调。
- 全32帧共3,538,944个体素的原始Token及(Node Name,param2)与Viewer输入完全一致，世界原点始终`(-288,1424,-32)`；时钟、步数和帧索引一致。没有发现词表或世界坐标映射错位。
- 旧Viewer把Source与Flowing都画成同色全立方体，并把红蘑菇画成默认土灰色立方体。新版Native-State渲染按原始Node Definition选择几何；未知定义明确标注为semantic fallback。

## 本地复查

```bash
# 从VoxWM仓库根目录执行，只读取已有Trace
.venv/bin/python -m vae_data.sharded.audit_transitions \
  --trace outputs/vae_native_six_20261116/traces/D2/attempt_2 \
  --start 18.9 --end 19.2 --output outputs/d2_transition_audit
.venv/bin/python -m vae_data export-traces --dataset outputs/vae_native_six_20261116
.venv/bin/python -m vae_data view --dataset outputs/vae_native_six_20261116 --port 8765
```

[固定红蘑菇坐标的Before/After](http://127.0.0.1:8765/?type=D2&attempt=attempt_2&frame=16&mode=changes&before=15&after=16&world=-257,1440,-3&isolate=1)。点Focus voxel，然后使用East/X剖切upper=31。另一目标坐标为`-258,1440,-2`。黄色选择框是坐标提示；Cyan Ghost仅表示上一观察中被移除的叶子，不是当前GT。

## Viewer QA

使用已存在的Playwright和本机Chrome/SwiftShader（Browser plugin not available；未新增依赖）。桌面1500×1080与390×844窄屏检查完成；原生渲染、三模式、固定坐标Before/After/Difference、实际点击拾取与原始Token对应、火焰/叶子Ghost、页面非空和无运行时错误通过。截图ReadPixels产生的SwiftShader性能警告保存在`viewer_qa.json`，不是应用异常；未验证其它浏览器。旧网站docs未修改。

D3现有成功链的四次相邻状态，真实中心结构变化分别为 **2 / 5 / 2 / 1**。火焰年龄变化被单独分类，不用年龄更新填充五状态。`D3/attempt_2`观察17包含真实leaves→fire，供Ghost预览。

- [Before：红蘑菇](native_before.png)
- [After：Flowing Lava](native_after.png)
- [实际Firelike + 历史叶子Ghost](fire_ghost.png)
- [完整逐体素表CSV](transitions.csv)
- [原始状态、邻接、Source全时序与核对JSON](audit.json)
- [页面交互检查](viewer_qa.json)

All times below are after-observation times. Exact write time is bounded by the corresponding before/after observations in `audit.json`.

| sim_time | world_xyz | before_name | before_param2 | after_name | after_param2 | transition_type |
|---|---|---|---|---|---|---|
| 19.17777892947197 | [-266, 1434, -2] | air | 0 | mcl_core:water_flowing | 7 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-265, 1433, -2] | air | 0 | mcl_core:water_flowing | 15 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-265, 1434, -1] | air | 0 | mcl_core:water_flowing | 15 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-264, 1431, -2] | air | 0 | mcl_core:water_flowing | 7 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-264, 1432, -2] | air | 0 | mcl_core:water_flowing | 15 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-264, 1433, -1] | air | 0 | mcl_core:water_flowing | 15 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-264, 1434, 0] | air | 0 | mcl_core:water_flowing | 15 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-264, 1435, 1] | air | 0 | mcl_core:water_flowing | 14 | flowing_liquid_appearance_observed |
| 19.17777892947197 | [-258, 1439, -3] | mcl_core:lava_flowing | 2 | mcl_core:lava_flowing | 3 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-258, 1440, -2] | mcl_core:lava_flowing | 2 | air | 0 | flowing_liquid_recession_observed |
| 19.17777892947197 | [-257, 1438, -4] | mcl_core:lava_flowing | 8 | mcl_core:lava_flowing | 9 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-257, 1438, -3] | mcl_core:lava_flowing | 10 | mcl_core:lava_flowing | 11 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-257, 1439, -4] | mcl_core:lava_flowing | 2 | mcl_core:lava_flowing | 3 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-257, 1440, -3] | mcl_mushrooms:mushroom_red | 0 | mcl_core:lava_flowing | 0 | lava_replaced_lava_destructible_node_observed |
| 19.17777892947197 | [-257, 1440, -2] | mcl_core:lava_flowing | 7 | mcl_core:lava_flowing | 15 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-257, 1441, -2] | mcl_core:lava_flowing | 2 | mcl_core:lava_flowing | 3 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-256, 1439, -4] | mcl_core:lava_flowing | 8 | mcl_core:lava_flowing | 9 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-256, 1439, -3] | mcl_core:lava_flowing | 10 | mcl_core:lava_flowing | 11 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-256, 1440, -3] | mcl_core:lava_flowing | 8 | mcl_core:lava_flowing | 9 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-256, 1440, -2] | mcl_core:lava_flowing | 10 | mcl_core:lava_flowing | 11 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-256, 1441, -3] | mcl_core:lava_flowing | 2 | mcl_core:lava_flowing | 3 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-255, 1441, -3] | mcl_core:lava_flowing | 0 | mcl_core:lava_flowing | 1 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-255, 1441, -2] | mcl_core:lava_flowing | 10 | mcl_core:lava_flowing | 11 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-255, 1442, -2] | mcl_core:lava_flowing | 0 | mcl_core:lava_flowing | 1 | liquid_level_or_flow_down_update |
| 19.17777892947197 | [-255, 1442, -1] | mcl_core:lava_flowing | 10 | mcl_core:lava_flowing | 11 | liquid_level_or_flow_down_update |

## Evidence and limits

- Both initialization source positions are tracked at every original observation in `audit.json`.
- Raw VWT1 to ABV1 and (Node Name,param2) viewer mappings: {'raw_viewer_tensor_equal': True, 'viewer_pair_mapping_equal': True, 'world_origin_fixed': True, 'native_time_strictly_increasing': True, 'frame_indices_match': True}.
- Red mushroom has native `destroy_by_lava_flow=1` in `mcl_mushrooms/small.lua`; `mcl_core/functions.lua` permits removal beside/under feeding lava. This supports a legal destruction/refill explanation, but the trace does not identify the precise callback or contain an intermediate air observation.
- Flowing-lava withdrawal is distinguished from source deletion. No transition is labelled normal merely because its new name contains lava.
- Full node-write/map-update journal is unavailable; cooling-only evidence cannot exclude every other native write.

## 最终检查与保留范围

51项相关CPU Unit通过（实测5.551秒），包括Source/Flowing、0x07/0x08、Fire Age、真实燃烧/可疑固体转液体分类、负世界坐标、中心/邻接、同坐标Before/After和GT导出重建；Python/JavaScript语法与diff检查通过。具体耗时以执行日志为准。没有重新采集任何Native、1K或100K，也没有修改Craftium/VoxeLibre源码或原GT词表。

原始GT没有保存所有Lua/C++节点写入和未观察的中间状态。本报告不能证明每个内部callback的因果，只确认已采集数据中实际发生的前后节点状态。未覆盖的Node Definition采用显式标注的简化cube；其它浏览器和完整原生材质/粒子系统未验证。

# VoxWM 执行守则

Vox2Pix 当前采用 [v3 数据协议](VOX2PIX_DATA_COLLECTION_PROTOCOL.md)。类别为 Surface、Cave、Underwater，白模保存单层首次非水命中及 Water Count。保留实际 RGB 相机、48³ 脚底锚定、名义眼高与可见性半径 1.6、40 个视角区中至少 20 个可用。空余率只作分析。v2 1K 结果作为独立历史保留。

- 用户已授权本次 v3 最小 Native smoke，仅每类 1 场景、每场景 5 个视角。真实 45 帧、更大采集和训练需单独授权。三类统一使用原生 `display_gamma=3.0`、`lighting_floor=0.35`。同场景同位姿的 Floor 0.0 / 0.35 对照单独保存。水下观察者自然沉底，再从真实位置建立 48³；不在下沉时固定高度或裁剪。

数据采集和 E00 小样本验证阶段已结束。当前活动方向是`voxel_vae/`的 E01-Masked Combined-Token 3D ResNet VAE。E00 的 Plain 与 Masked 产物保留作历史。`python -m vae_data` → `vae_data/sharded/`及[数据协议](VAE_DATA_COLLECTION_PROTOCOL.md)继续定义只读正式数据格式。

- 当前范围为 E01-Masked 的全训练集入口、固定 3D ResNet VAE、balanced sampler、训练和诊断。用户手动执行 E01。架构/beta/latent sweep、alpha=0.75 消融、Dynamics 训练及新数据采集均需要另行授权。
- 正式训练数据为`vae_100k_clip1`与`vae_100k_clip2`两个独立100K roots。不同root的uint16 ID没有跨root语义；训练必须按真实`(Node Name,raw param2)`构建确定性canonical vocabulary并在线remap，不得改原Zarr或设置UNK/静默丢token。
- VAE只输入中心`tokens[16:32,16:32,16:32]`的canonical离散ID，经随机32D `nn.Embedding`进入网络。light、biome/category/time/chain/Δt均不输入；D4 anchor不算训练sample；五个动态状态是独立reconstruction sample。
- E01 复用 96/192/192 3D ResNet Encoder、`mu/logvar:[B,16,4,4,4]`、nearest-upsample Decoder 和 K 类 logits，无 skip/attention/Transformer/dropout/BatchNorm。按步模式的 batch 为 32 Static + D1–D4 各 8。`--epochs` 模式每个完整 epoch 遍历 train split 恰好一次，按各类实际数量分配各 batch，batch 最大为 64。AdamW 1e-4，clip 1.0，无 scheduler。
- Masked 目标为 `0.5 L_static + 0.25 L_dynamic_unchanged + 0.25 L_dynamic_changed + 1e-6 raw KL`。Changed Mask 来自同一完整五状态链的 canonical 中心 16³，只用于 loss。microbatch=2 使用完整 effective batch 的全局样本分母。
- E01 用两个 100K Clip 的完整 `train` split，默认 bf16、20,000 步上限。每 1,000 步在固定的 1,280 条独立验证状态上选最优 checkpoint。160 条训练诊断状态只用于评估。Test split 留出。E01 从 seed 42 初始化，续跑使用 E01 自己的 checkpoint、准备索引和验证子集。
- E01 的 160K/20K/20K 索引调整只按完整 World Seed 从旧 test 向 train 和 validation 补入状态，固定诊断子集和原始数据保持原样。按全局 batch=64 续训时每个 epoch 为 2,500 步。
- E01 双卡用 GPU 2、3 手动启动单节点 DDP。两进程分同一个全局 batch，每卡 microbatch 最大为 2。全局分项分母跨进程求和，只在最后一组 microbatch 同步梯度。主进程负责验证、早停、曲线和 checkpoint，checkpoint 保存各进程的 RNG 状态。epoch 模式的早停只在完整 epoch 边界生效，checkpoint 保存 epoch 计划，按步恢复其采样位置。
- E01 续训指定 `--resume --epochs N` 时从已完成轮数起再完成 N 轮。已有 epoch 计划的中断续跑只传 `--resume`，续跑至已保存的目标。
- 所有真实训练及续跑由用户手动执行。Codex 负责代码与快速本地检查。E01 产物写入 `outputs/voxel_vae/e01/phase_masked/`。训练保存 `loss_curves.svg`、`loss_curves.json` 和可恢复早停状态。`voxel_vae.training_monitor` 独立只读。默认连续两次验证总 loss 改善未超过 0.001 时早停。`--no-early-stop` 在一次启动中关闭自动停止。`checkpoint_best.pt` 对应最低验证总 loss。
- 当前规模100K，100个严格1K片；每片Static500，D1/D2/D3/D4各125，动态每类25条完整五状态链。D4是Water–Lava Solidification，完全替代历史Vegetation，不自动恢复旧200K配方。
- 正式样本保存真实同一时刻48³(27个相邻16³采样窗口)，目标与动态判定仅中心 `[16:32]^3`。Dynamic保持全局16格网格；Static仅高度可逐整数偏移，与引擎MapBlock明确区分，完整Region高度必须在[-62,193]。邻块变化不能加候选、拼链或重置超时；D1/D2/D4为48 Tick，D3为120 Tick。
- Dynamic最多10个中心不同候选，随机连续五状态窗口，不足则拒绝。D4必须完整中心Pre–Reaction–Post，物理帧统一去重，成功场景的Anchor独立保存且不计入1K。
- Static每片25主要Biome各20，Surface8/ElevatedComplex4/Shallow4/DeepCave4，真实Transition尽量每组4且为附加属性。不拼接、不拿其他群落或类别补缺。先按缺额选目标群落/高度，再用原生Biome噪声接口有限定位，命中后加载并复核真实地表群落；失败保留缺额，不退回随机地点凑数，不建Discovery Cache。Static按新高度协议：中心C∈[-38,170]，Surface包含真实S，Elevated从S+9到170且Air≥10%/<98%，Shallow为[S−24,S−8]，Deep为[-38,S−32]（均裁到允许域）；每地点最多16次不重复高度，空旷Elevated只向下，地下允许全实心但拒绝全Void，ignore另作完整性错误。
- 全局uint16 Combined Token=(Node Name,raw param2)，词表只追加；uint8 light辅助。48³无损存储，中心16³仍是VAE目标。World Seed固定80/10/10 split，不跨split。
- 纯Voxel：从原生入口禁非技术实体、掉落及实体引起的Voxel变化；技术player不可见无交互。当前falling-entity动态不支持，出现即排除该rollout，不默默冻结物理后发布数据。
- 使用Craftium/VoxeLibre原生液体、固化与火。不写流体轨迹，不改规则/ABM概率/黏度；固定24Hz物理时间、整数Tick为权威，无墙钟FPS限速。允许维护固定版本的最小Timing/Scheduler补丁，默认旧GT会话不启用。几何可达性只筛源点，不能冒充native事件证据；初始化操作不能计native演化。
- 正式采集并行度按用户11/17要求取消片数/固定CPU上限：`--workers auto`默认使用当前CPU affinity（不可用时CPU数），手动正整数不封顶；超出片数以多个独立世界进程供给每片唯一Writer，不能并发直接操作同一Writer。Linux软件渲染每会话自动分配私有Xvfb并清理，不依赖固定DISPLAY或手改引擎wrapper。词表只在新增Token时发布，序列化/fsync在全局锁外，等待锁有超时；仅正式提交/新增词表算进度，失败重试不算。用户手动启动真实采集；本轮只做轻量合成/进程测试。
- 动态观察批传输按11/17最新指令：默认每批24个真实fixed24 Tick，可选48；旧单拍advance保留，clock-verify默认单拍。批内Server/Client/Lua每拍对齐并post-step捕获，内存中首帧RAW+逐拍稀疏delta经socket一次返回；Python按序逐拍调用原Collector，首次stop之后尾巴不解码为Token、不进候选/事件/五状态链，存在尾巴的世界结束后不复用。48/120 Tick无变化、10候选、30/40/60秒上限和正式五帧/Anchor格式不变。不改并行策略，用户本次实测使用64世界；不得自动启动Native采集或声称已实测提速。
- 正式长跑默认由外部`vae_data/supervisor.py start`按配置启动为脱离SSH终端的独立后台会话：同时至多64个世界且每个未完成Shard恰好1个世界。`dashboard`是只读前台进程，不启动、不拥有采集子进程；关闭Dashboard或SSH不得影响采集。`stop`先请求后台监控器有序清理，再只按本次随机环境标记处理可能被PID 1收养的后代，明确拒绝PID 1，并在成功返回前确认supervisor/dataset/vocabulary/shard锁均可获取。Collector另有Linux parent-death信号和自身树清理兜底。每个活跃Shard独立计算600秒进度；某Shard没有完成attempt、样本提交或READY时重启整次采集，不能由其他健康Shard掩盖。禁止按进程名或全局`pkill`清理，不能碰其他用户任务；真实服务器稳定性仍由用户手动运行验证。
- 缺额补采可用`--only-category Static|D1|D2|D3|D4`；调度器只发所选类别，达到该类别原配额即停止，绝不能用其他类别补位或改100片×1000配额。新高度策略Static必须写入兼容的新目录`outputs/vae_100k_heightadaptive`，不能混入旧`vae_100k_fixed24`；配置为`vae_data/configs/collection_64x1_static.json`，Dashboard单列Focus Static，原动态目录只读。
- 两批完成后的唯一正式合并入口是`scripts/merge_vae_static.py`：保留`vae_100k_fixed24`全部D1-D4并丢弃其旧Static，导入`vae_100k_heightadaptive`全部Static。两个全局词表不得直接混用；必须保持动态词表ID不变、按(Node Name,raw param2)追加并重映射Static。默认只读dry-run；`--apply`只能在两套采集均停止且锁可获取时创建独立staging，逐片完整验证后替换活动路径并保留原目录备份。不得原地删Zarr/SQLite行，也不得自动删除备份或Static源目录。
- 低缓存：Static立即提交或丢弃；Dynamic保留至多10候选和必要Anchor/事件引用；不写完整raw轨迹、不积累Discovery Cache。分片独占写入，全局词表短锁，完成QA后原子READY；READY片只读。
- 动态中心先按真实固体—Air Surface或地下连通Cave选择；D1/D3仅Surface，D2/D4两高度各有50%机会。D1/D2/D3/D4-B不天然目标体素定位，A/C仅对应稳定天然池。全点规划前中心最大六连通Air≥65；A/C最大连通稳定对应流体≥19（Source/Flowing可，不跨中心、不加面积/比例门槛）。保留现有16³对齐与48³ Assembly。按类/高度记录attempt/空间合格/规划合格/接受数。
- 所有人工source仅中心local[1..14]^3，即region[17:31]^3，先域限制再合法枚举；Python调用前后及Lua放置前复核，任何越界整次拒绝。传播仍在完整48³原生世界，不裁剪到14³；天然池不受人工source域约束。
- 全部合法source参与共享图/反向乐观边界/10sim秒bitset批处理；禁止先随机截取256/32/64点。至少四个不同时间阶段才算潜力，不能用四个同刻voxel凑数；D3潜力筛选同样采用5秒间隙，其他2秒；几何估计与实际native证据分开。p=1/N独立采样、零命中均匀保底、有限source cap；D4-B必须联合兼容，记录条件采样偏移。
- 六型Native协议上限仍为每型3次/1成功、整轮900秒，不因时钟修订自动重开。历史两轮结果保留；固定时钟轮只做过约29秒Clock检查；随后用户在11/17明确授权重跑D1/D3/D4-A/B/C并跳过D2，且批准修复坐标接口后重新给五类各3次额度。该轮已完成：修复段7次尝试，五类各1成功，两段累计约254.52秒；D2沿用旧结果并明确标记。完成后不自动追加。更多Water/Lava/Fire受控随机跨吞吐验证须先说明预算并取得授权。11/17用户高度修订另允许有限只读Static核验；`static-verify`单次最多180秒，不创建正式Shard。随后用户明确要求10个不同群落的Static试采，高度总配额4/2/2/2；`static-ten`最多40次位置尝试/600秒，10成功即停。本轮已完成15次尝试、10群落/10不同C、107.29秒，接入旧8766预览；不自动追加。正式1K/100K、广域扫描及训练由用户手动启动。
- Static使用按实际世界坐标读取的Capture，每次Prepare以同Tick P0/VoxelManip逐体素核对Node/param1/param2；Forceload覆盖实际相交的全部引擎块。Static按(seed,真实中心)跨片去重，新高度策略不混入旧目录。固定24Hz正式采集每Tick推进恰好一次、post-server-update捕获，不能用float时间轮询、丢帧或复制状态。干预完成为相对Tick0，绝对/相对时间由整数Tick/24计算。VAE格式v2不混入历史variable-step Shard。
- 六型debug及明确有限Clock验证允许完整原生观察Trace（首帧+稀疏增量），包括失败；正式Shard仍只保存选中五帧/必要Anchor。GT-only Viewer独立于docs，禁止改动或部署现有展示网站。
- 原始GT与词表只读；不改已有Viewer材质/方块渲染。Viewer几何/材质映射独立。检查同一世界坐标的真实Node Name/raw param2，Source丢失或固体变液体不可无写入证据当普通流动。Fire年龄更新不计结构动态；Ghost必须与当前GT明确区分。
- 科研与效率优先：简单可读，不建平台、复杂服务、冗余预检、hash审计或无关重构。只做相关快速Unit/语法/CLI检查，通常累计30秒内；测试Fixture绝不能当Native数据或物理证据。
- 保留 `vae_p0/`、`vae_p1/`、`vae_pilot/`、旧Unified源码/README/协议快照、所有旧结果/日志与docs。旧入口 `vae_data.legacy_entry` 仅历史参考，不混入当前格式。旧CoDL只读，不修改或推送。
- 唯一开发远端 `MengmaoR/VoxWM` / master。每次编码前检查并保留用户改动、`git pull --ff-only`。正常commit/push，不强推、不重写历史，不触碰展示站部署/权限。
- 交付说明实际Native、Unit及未验证功能，写明真实缺口。实现完成即停止，等待用户手动运行。

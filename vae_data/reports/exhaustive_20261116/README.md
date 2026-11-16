# INCONCLUSIVE — Exhaustive Planning / Six-Type Native GT (2026-11-16)

本轮唯一原生验证共12次尝试，202.49秒，900秒总上限。D1/D2/D3/D4-C各得到一条合法五状态链；D4-A/B未成功，不扩次数追成功。正式1K/100K及训练没有执行。

全合法源点一次建立共享CSR图，以乐观反向边界排除确定不可达点，10秒分层稀疏64-source bitset传播；不做256/32/64源点预随机截断。所有合法点有统计或不可达判断，正式只保存选中点细节。D4-B采用隐式接触兼容关系和有界条件采样，避免Water×Lava笛卡尔积。

复杂度按批共享可达状态/边，最坏随ceil(N/64)×R×V×10增长；稀疏frontier只访问激活部分，缓冲按V与range固定复用，没有N×V永久张量。传播是带时间/随机性限制的潜力，不是Native物理替代。

## 现有真实48³ GT上的CPU实测

|任务|合法源点|完整预计算秒|峰值进程RSS MiB|merged wavefront visits|
|---|---:|---:|---:|---:|
|water|{'water': 33852}|0.2758|99.3|1820528|
|lava|{'lava': 33852}|0.3049|87.1|1580411|
|fire|{'fire': 53275}|0.4305|76.2|356173|
|pair|{'water': 56756, 'lava': 56756}|0.6796|118.6|9149122|

RSS含Python、输入和共享图，不是内核独占；双源统计包含实际独立采样/条件查询。额外Native模拟数=0。完整分项在[planning_profile.json](planning_profile.json)。小固定Fixture与独立BFS结果/到达时间一致；相同24源点实测速比仅代表该Fixture，不外推原生采集：
- water: 5.32×（24源点，相同输出，shared graph已构建）。
- lava: 1.84×（24源点，相同输出，shared graph已构建）。
- fire: 2.71×（24源点，相同输出，shared graph已构建）。
- pair: 3.63×（24源点，相同输出，shared graph已构建）。

## 六型实际结果

|类型|尝试|成功|证据|
|---|---:|---:|---|
|D1|1|1|[PNG](D1/contact_sheet.png) · [GIF](D1/evolution.gif) · [报告](D1/report.md)|
|D2|2|1|[PNG](D2/contact_sheet.png) · [GIF](D2/evolution.gif) · [报告](D2/report.md)|
|D3|2|1|[PNG](D3/contact_sheet.png) · [GIF](D3/evolution.gif) · [报告](D3/report.md)|
|D4-A|3|0|[PNG](D4-A/contact_sheet.png) · [GIF](D4-A/evolution.gif) · [报告](D4-A/report.md)|
|D4-B|3|0|[PNG](D4-B/contact_sheet.png) · [GIF](D4-B/evolution.gif) · [报告](D4-B/report.md)|
|D4-C|1|1|[PNG](D4-C/contact_sheet.png) · [GIF](D4-C/evolution.gif) · [报告](D4-C/report.md)|

D4-A三次潜力不足；D4-B原生运行中两次把散落4–5个source误认成池，已在运行后改为连通source池判断并加中心接触halo筛选，只离线复算，不能把新逻辑称为已通过Native。第三次确有自然水池。全部失败Trace保留。

原生采集来自base commit `c9c7bbe3940078b0b56938186224b519c9788ac9`的本轮未提交工作树（原记录dirty=true），交付包含运行后的D4-B离线修正；没有改写原始源码来源。

完整数据在本地`outputs/vae_native_six_20261116/`（不入Git）：`traces/<type>/attempt_N/`保存无损VWT1 tokens/light、真实观察时间/step、全部源点统计、预测时间网格、事件及选中五帧索引；`viewer/`为只读显示用ABV1导出。实际Native、预计算、Accepted五状态与Synthetic Unit数据分离。

查看及服务器手动生成/恢复/检查命令见[vae_data README](../../README.md)。当前[Node-state审计与Viewer改进](../transition_audit_20261116/README.md)使用同一原始GT，未增加原生采集。

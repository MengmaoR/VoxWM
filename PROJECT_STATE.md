# VoxWM 当前状态

**Native Transition Audit + Native-State GT Viewer — IMPLEMENTED / CPU VERIFIED**。

- 用户修正的目标为D2 attempt_2，18.9–19.2秒。原始32帧/完整48³逐体素核对通过；目标变化是Flowing Lava→Air，以及红蘑菇→Flowing Lava。未发现Source→Air或Dirt→Lava；两枚初始化Source保留。没有完整节点写入journal，内部回调归因仍有限。[审计/CSV/预览](vae_data/reports/transition_audit_20261116/README.md)。
- Viewer直接解码原词表，三种模式、液位/Flow-Down、Firelike、可选历史叶子Ghost、点击Inspector和同坐标Before/After/Difference已实现并以已有GT验证；本轮没有新增Native采集、模型或训练，也没有改动原GT/词表/原生物理/docs。
- VAE配方不变：100×1K，Static500+四动态各125/片，完整48³、中心16³、五状态链。此前六型有界Native验证仍为INCONCLUSIVE：202.49秒，D1/D2/D3/D4-C成功，D4-A/B缺口及后续仅离线修正均保留。[前轮实测](vae_data/reports/exhaustive_20261116/README.md)。
- 原始Trace保留在`outputs/vae_native_six_20261116/`，不入Git。正式1K/100K、25群落产能及D4-A/B稳定产出未验证。[手动命令](vae_data/README.md)。

唯一下一步：用户在本地Viewer复查D2固定世界坐标及D3燃烧过程，再决定采集器的下一项有限改动；Codex完成提交后停止，不追加Native生成。

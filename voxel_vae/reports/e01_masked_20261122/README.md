# E01-Masked 30,000 步服务器结果

最后一轮于 2026-11-22 13:36（北京时间）完成。用户在 GPU 4、5、6、7 上从第 27,500 步恢复，关闭本轮早停并跑完 2,500 步，用时 20 分 12 秒。此前从第 25,000 步出发的一轮在第 27,500 步触发了早停。

训练使用 160,000 条状态、全局 batch 64、bf16 和 seed 42。下表来自同一组固定的 1,280 条验证状态。模型共有 23,576,236 个参数，Combined Token 词表有 1,580 个条目，每个条目的 embedding 为 32 维。

| 步数 | 验证总 loss ↓ | Changed CE ↓ | D3 Changed CE ↓ | Changed 准确率 ↑ | D3 Changed 准确率 ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 0.288398 | 0.695618 | 1.617255 | 83.32% | 50.58% |
| 25,000 | 0.256879 | **0.587967** | 1.442314 | 85.66% | 56.12% |
| 27,000 | **0.253373** | 0.594015 | 1.437059 | 85.10% | 54.04% |
| 27,500 | 0.258191 | 0.604236 | 1.462403 | 85.44% | 54.85% |
| 28,000 | 0.255317 | 0.603082 | 1.472677 | 85.28% | 56.93% |
| 29,000 | 0.253502 | 0.588045 | **1.431195** | 85.76% | 54.85% |
| 30,000 | 0.259818 | 0.620825 | 1.573265 | **86.10%** | **57.51%** |

从 27,500 到 30,000 步，固定训练诊断的 Changed CE 从 0.191439 降至 0.160749，验证 Changed CE 从 0.604236 升至 0.620825。29,000 步的验证 D3 Changed CE 短暂降至全程最低。30,000 步录得最高的 Changed 和 D3 精确 token 准确率。

## 结果文件

- [训练与验证曲线](phase_masked/loss_curves.svg) · [曲线数据](phase_masked/loss_curves.json)
- [原始训练与验证日志](phase_masked/metrics.jsonl) · [30,000 步完整指标](phase_masked/final_metrics.json)
- [运行元数据](phase_masked/run_metadata.json) · [原始准备配置](config.yaml) · [最后一轮的实际配置](phase_masked/config.yaml) · [最终早停状态](phase_masked/early_stopping.json)
- [160K/20K/20K 划分补丁](split_160k.json) · [固定诊断和验证样本索引](subset_manifest.json) · [Canonical Token 词表](training_vocabulary.json)
- [27,000 步验证画廊指标](phase_masked/viewer_validation_examples_metrics.json) · [Checkpoint 步数及代码版本](checkpoint_provenance.json)

模型权重保存在服务器的 `/user-data/txj/VoxWM/outputs/voxel_vae/e01/phase_masked/`。`checkpoint_best.pt` 对应最低验证总 loss（27,000 步），`checkpoint_last.pt` 对应最后一步（30,000 步）。

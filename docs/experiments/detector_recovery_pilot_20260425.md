# detector_recovery 试验记录（2026-04-25）

## 目的

上一轮 `clear_det_preserve_pilot` 的问题是：

- detector 回拉没有成功
- fog 任务仍在参与更新，雾类别稳定性明显变差

因此本轮试验进一步收缩目标，只做：

- detector-only recovery
- 只在 clear 图上优化检测损失
- 不再对 fog 分类 / beta 回归任务施加损失

## 运行配置

- 训练脚本：`scripts/run_detector_recovery_train.sh`
- 实际输出目录：`outputs/Fog_Detection_Project_detectorrecovery_pilot`
- 恢复 checkpoint：
  - `outputs/Fog_Detection_Project_fogfocus_full/checkpoints/checkpoint_epoch_0012.pt`
- 本轮设置：
  - `BS_EPOCHS=13`
  - `BS_FRAME_STRIDE=8`
  - `BS_DET_LOSS_WEIGHT=0.0`
  - `BS_CLEAR_DET_LOSS_WEIGHT=1.0`
  - `BS_FOG_CLS_LOSS_WEIGHT=0.0`
  - `BS_FOG_REG_LOSS_WEIGHT=0.0`
  - `BS_FREEZE_YOLO_FOR_FOG=0`

说明：

- 仍然只追加了 **1 个增量 epoch**
- 但这次只让 detector 接收 clear 图检测监督

## 训练结果

来自：

- `outputs/Fog_Detection_Project_detectorrecovery_pilot/runs/smoke_20260425_104228/summary.json`

关键结果：

- Train:
  - `loss = 2.3783`
  - `det = 0.0000`
  - `clear_det = 2.3783`
- Val:
  - `loss = 3.8767`
  - `det = 0.0000`
  - `clear_det = 3.8767`

补充说明：

- 本轮同样没有刷新历史最佳验证损失 `1.0072`
- 生成了：
  - `unified_model.pt`
  - `checkpoints/checkpoint_epoch_0013.pt`

## benchmark_v1 对标结果

候选模型对标目录：

- `outputs/Candidate_Benchmark_detectorrecovery_pilot/`

候选摘要：

- `candidate_benchmark_summary.json`
- `candidate_benchmark_summary.md`

### 候选模型聚合结果

| 指标 | `detectorrecovery_pilot` |
|---|---:|
| `weighted_unified_mean_count_per_frame` | `5.7598` |
| `weighted_unified_frames_with_detections_ratio` | `0.6912` |
| `weighted_mean_count_gap_hybrid_minus_unified` | `12.2672` |
| `weighted_beta_mean` | `0.0536` |
| `weighted_dominant_switch_rate` | `0.0447` |
| `weighted_beta_abs_delta_mean` | `0.001993` |
| 视频级推荐 | `hybrid × 5` |

### 与正式参考点 `fogfocus_full` 对比

| 指标 | `fogfocus_full` | `detectorrecovery_pilot` | 差值（candidate - baseline） |
|---|---:|---:|---:|
| `weighted_unified_mean_count_per_frame` | `6.5490` | `5.7598` | `-0.7892` |
| `weighted_unified_frames_with_detections_ratio` | `0.8235` | `0.6912` | `-0.1324` |
| `weighted_dominant_switch_rate` | `0.0323` | `0.0447` | `+0.0124` |
| `weighted_beta_abs_delta_mean` | `0.000906` | `0.001993` | `+0.001087` |

### baseline gate 结果

- `overall_pass = false`
- `passed_baseline_count = 0 / 4`

未通过原因：

- 相对 `fogfocus_full`：
  - 平均检测数下降
  - 非零检测帧占比下降
  - 虽然雾切换率恶化没有超过阈值，但检测项已经失败
- 相对 `default_baseline` / `videoadapt`：
  - 平均检测数略高
  - 但非零检测帧占比仍然更差
  - 雾切换率也更差

## 分视频观察

`detectorrecovery_pilot` 在 5 段视频上的统一模型检测数/帧：

| 视频 | 统一模型平均检测数/帧 | 混合平均检测数/帧 | 差值（hybrid - unified） | 主导雾类别 |
|---|---:|---:|---:|---|
| `getty_demo_real_fog` | `0.2857` | `1.7679` | `1.4821` | `UNIFORM FOG` |
| `control_clear_sparse_traffic_day` | `5.2833` | `21.0500` | `15.7667` | `PATCHY FOG` |
| `control_clear_medium_traffic_day` | `5.3167` | `29.3000` | `23.9833` | `PATCHY FOG` |
| `control_clear_heavy_traffic_day` | `10.7333` | `29.1500` | `18.4167` | `PATCHY FOG` |
| `control_clear_dense_traffic_day` | `17.0333` | `38.1333` | `21.1000` | `PATCHY FOG` |

观察：

- 在真实 fog 视频上，统一检测进一步退化到 `0.2857`
- 在 4 段 clear-weather control clips 上，检测也没有实质改善
- 所有 4 段 clear control clips 的主导雾类别都偏成了 `PATCHY FOG`

这说明 detector-only recovery 虽然比上一轮的多任务 clear preserve 更“纯”，但仍然没有解决根问题。

## 与上一轮 `clear_det_preserve_pilot` 的比较

| 指标 | `clear_det_preserve_pilot` | `detectorrecovery_pilot` |
|---|---:|---:|
| `weighted_unified_mean_count_per_frame` | `5.6446` | `5.7598` |
| `weighted_unified_frames_with_detections_ratio` | `0.7230` | `0.6912` |
| `weighted_dominant_switch_rate` | `0.0720` | `0.0447` |
| `weighted_beta_abs_delta_mean` | `0.002007` | `0.001993` |

解读：

- detector-only recovery 在平均检测数上略高于上一轮
- 但非零检测帧占比反而更差
- 雾切换率确实比上一轮低了一些
- 整体仍然没有通过 baseline gate

## 结论

本轮 `detectorrecovery_pilot` **仍然失败**，不应进入正式 baseline 集合。

相对上一轮，它说明了一件更明确的事：

- 仅仅把训练目标切到“clear 图检测恢复”，也不足以修复统一模型在 `benchmark_v1` 上的 clear-weather detection deficit

也就是说，当前问题可能不是“损失权重不对”这么简单，而更可能涉及：

- 单类检测头本身的容量/适配问题
- 从多类预训练 detector 迁移到单类 unified detector 的表示损失
- fog / non-fog 场景共享特征的结构冲突

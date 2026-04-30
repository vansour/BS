# clear_det_preserve 试验记录（2026-04-25）

## 目的

本轮试验针对 `benchmark_v1` 暴露出的主要问题：

- 统一模型在 4 段 clear-weather control clips 上的检测能力明显弱于混合路线

为此引入了一条新的继续训练路线：

- 从 `fogfocus_full` 的 checkpoint 继续训练
- 解冻 detector
- 保留 fog 分类 / beta 回归任务
- 在 clear 图上额外加入辅助检测损失

## 运行配置

- 训练脚本：`scripts/run_clear_det_preserve_train.sh`
- 实际输出目录：`outputs/Fog_Detection_Project_clearpreserve_pilot`
- 恢复 checkpoint：
  - `outputs/Fog_Detection_Project_fogfocus_full/checkpoints/checkpoint_epoch_0012.pt`
- 本轮设置：
  - `BS_EPOCHS=13`
  - `BS_FRAME_STRIDE=8`
  - `BS_DET_LOSS_WEIGHT=0.35`
  - `BS_CLEAR_DET_LOSS_WEIGHT=0.75`
  - `BS_FREEZE_YOLO_FOR_FOG=0`

说明：

- `checkpoint_epoch_0012.pt` 恢复后对应的下一轮是 `Epoch 13/13`
- 因此本轮实际只追加了 **1 个增量 epoch**
- 使用 `FRAME_STRIDE=8` 是为了在当前回合内拿到可比较结果，而不是做长时正式训练

## 训练结果

来自：

- `outputs/Fog_Detection_Project_clearpreserve_pilot/runs/smoke_20260425_102038/summary.json`

关键结果：

- Train:
  - `loss = 3.3050`
  - `det = 2.4090`
  - `clear_det = 2.3874`
  - `fog_cls = 0.4474`
- Val:
  - `loss = 4.9379`
  - `det = 3.9487`
  - `clear_det = 3.8939`
  - `fog_cls = 0.4234`

补充说明：

- 本轮没有刷新历史最佳验证损失 `1.0072`
- 因此目录下生成了：
  - `unified_model.pt`
  - `checkpoints/checkpoint_epoch_0013.pt`
- 但**没有**新的 `unified_model_best.pt`

## benchmark_v1 对标结果

候选模型对标目录：

- `outputs/Candidate_Benchmark_clearpreserve_pilot/`

候选摘要：

- `candidate_benchmark_summary.json`
- `candidate_benchmark_summary.md`

### 候选模型聚合结果

| 指标 | `clearpreserve_pilot` |
|---|---:|
| `weighted_unified_mean_count_per_frame` | `5.6446` |
| `weighted_unified_frames_with_detections_ratio` | `0.7230` |
| `weighted_mean_count_gap_hybrid_minus_unified` | `12.3824` |
| `weighted_beta_mean` | `0.0485` |
| `weighted_dominant_switch_rate` | `0.0720` |
| `weighted_beta_abs_delta_mean` | `0.002007` |
| 视频级推荐 | `hybrid × 5` |

### 与正式参考点 `fogfocus_full` 对比

| 指标 | `fogfocus_full` | `clearpreserve_pilot` | 差值（candidate - baseline） |
|---|---:|---:|---:|
| `weighted_unified_mean_count_per_frame` | `6.5490` | `5.6446` | `-0.9044` |
| `weighted_unified_frames_with_detections_ratio` | `0.8235` | `0.7230` | `-0.1005` |
| `weighted_dominant_switch_rate` | `0.0323` | `0.0720` | `+0.0397` |
| `weighted_beta_abs_delta_mean` | `0.000906` | `0.002007` | `+0.001101` |

### baseline gate 结果

候选摘要给出的 gate 结论：

- `overall_pass = false`
- `passed_baseline_count = 0 / 4`

未通过的主要原因：

- 相对 `fogfocus` / `fogfocus_full`
  - 统一模型平均检测数下降
  - 非零检测帧占比下降
  - 雾类别切换率显著上升
- 相对 `default_baseline` / `videoadapt`
  - 虽然平均检测数基本持平
  - 但非零检测帧占比下降
  - 雾类别切换率也显著上升

## 分视频观察

`clearpreserve_pilot` 在 5 段视频上的统一模型检测数/帧：

| 视频 | 统一模型平均检测数/帧 | 混合平均检测数/帧 | 差值（hybrid - unified） | 主导雾类别 |
|---|---:|---:|---:|---|
| `getty_demo_real_fog` | `0.3452` | `1.7679` | `1.4226` | `UNIFORM FOG` |
| `control_clear_sparse_traffic_day` | `5.3667` | `21.0500` | `15.6833` | `PATCHY FOG` |
| `control_clear_medium_traffic_day` | `5.3333` | `29.3000` | `23.9667` | `PATCHY FOG` |
| `control_clear_heavy_traffic_day` | `10.4667` | `29.1500` | `18.6833` | `CLEAR` |
| `control_clear_dense_traffic_day` | `16.2500` | `38.1333` | `21.8833` | `PATCHY FOG` |

可以看到：

- 在真实 fog 视频上，统一检测还进一步下降
- 在 4 段 clear-weather control clips 上，没有出现预期中的检测改善
- 反而在 3 段 clear clips 上，雾分类主导类别偏到了 `PATCHY FOG`

这说明当前这版 clear 辅助检测权重设置把 detector 拉回来的同时，也扰动了雾头稳定性，而且并没有真正弥补 control clips 上的检测差距。

## 结论

本轮 `clearpreserve_pilot` **失败**，不应进入正式 baseline 集合。

结论可以概括为：

1. 仅追加 1 个增量 epoch + `CLEAR_DET_LOSS_WEIGHT=0.75` 这组设置，并没有改善 `benchmark_v1` 上的 clear-weather 检测缺口
2. 候选模型在统一检测覆盖率与雾类别稳定性上都比 `fogfocus_full` 更差
3. 当前这条路线如果继续做，不应直接照抄本轮超参数，而应优先考虑：
   - 降低 `CLEAR_DET_LOSS_WEIGHT`
   - 缩短或取消 fog 分类 / beta 权重的同步更新幅度
   - 或先做 detector-only 的短时恢复实验，再回到多任务联合

# coco_vehicle_head_pilot 试验记录（2026-04-25）

## 目的

前两轮围绕单类检测头的修补路线都失败了：

- `clear_det_preserve_pilot`
- `detector_recovery_pilot`

因此本轮不再继续微调损失权重，而是直接验证结构性假设：

- 不再把 detector head 强制重建成单类头
- 保留 YOLO 预训练的 COCO 80 类检测头
- 训练时把 UA-DETRAC 车辆框映射到 COCO `car` 类
- 推理时再把 COCO vehicle classes 折叠回统一的 `vehicle`

目标是判断：**保留原始多类 detector head 后，统一模型的检测能力是否能明显恢复。**

## 运行配置

- 训练脚本：`scripts/run_coco_vehicle_head_pilot.sh`
- 配置文件：`configs/coco_vehicle_head_pilot.yaml`
- 输出目录：`outputs/Fog_Detection_Project_cocohead_pilot`
- 恢复权重：
  - `outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt`
- 结构设置：
  - `det_head_mode = "coco_vehicle"`
  - `resume.model_only = true`
  - `resume.nonstrict_model_only = true`
  - `resume.reset_epoch = true`
- 本轮训练设置：
  - `BS_EPOCHS=1`
  - `BS_FRAME_STRIDE=16`
  - `BS_LR=1e-5`
  - `BS_DET_LOSS_WEIGHT=0.75`
  - `BS_CLEAR_DET_LOSS_WEIGHT=0.35`
  - `BS_FOG_CLS_LOSS_WEIGHT=1.25`
  - `BS_FOG_REG_LOSS_WEIGHT=1.0`

说明：

- 这仍然只是一个 **低成本结构可行性 pilot**
- 重点是先验证“保留 COCO head”这条路线能否在 `benchmark_v1` 上站住脚

## 训练结果

来自：

- `outputs/Fog_Detection_Project_cocohead_pilot/runs/smoke_20260425_135312/summary.json`

关键结果：

- Train:
  - `loss = 7.6222`
  - `det = 6.2591`
  - `clear_det = 6.2677`
  - `fog_cls = 0.5872`
  - `fog_reg = 0.000201`
- Val:
  - `loss = 8.2286`
  - `det = 6.6054`
  - `clear_det = 6.8206`
  - `fog_cls = 0.7096`
  - `fog_reg = 0.000381`

产物：

- `outputs/Fog_Detection_Project_cocohead_pilot/unified_model.pt`
- `outputs/Fog_Detection_Project_cocohead_pilot/unified_model_best.pt`
- `outputs/Fog_Detection_Project_cocohead_pilot/checkpoints/checkpoint_epoch_0001.pt`

## benchmark_v1 对标结果

候选模型对标目录：

- `outputs/Candidate_Benchmark_cocohead_pilot/`

候选摘要：

- `candidate_benchmark_summary.json`
- `candidate_benchmark_summary.md`

### 候选模型聚合结果

| 指标 | `cocohead_pilot` |
|---|---:|
| `weighted_unified_mean_count_per_frame` | `0.0735` |
| `weighted_unified_frames_with_detections_ratio` | `0.0711` |
| `weighted_mean_count_gap_hybrid_minus_unified` | `17.9534` |
| `weighted_beta_mean` | `0.0494` |
| `weighted_dominant_switch_rate` | `0.0571` |
| `weighted_beta_abs_delta_mean` | `0.002055` |
| 视频级推荐 | `hybrid × 5` |

### 与正式参考点 `fogfocus_full` 对比

| 指标 | `fogfocus_full` | `cocohead_pilot` | 差值（candidate - baseline） |
|---|---:|---:|---:|
| `weighted_unified_mean_count_per_frame` | `6.5490` | `0.0735` | `-6.4755` |
| `weighted_unified_frames_with_detections_ratio` | `0.8235` | `0.0711` | `-0.7525` |
| `weighted_dominant_switch_rate` | `0.0323` | `0.0571` | `+0.0248` |
| `weighted_beta_abs_delta_mean` | `0.000906` | `0.002055` | `+0.001149` |

### baseline gate 结果

- `overall_pass = false`
- `passed_baseline_count = 0 / 4`

未通过原因：

- 相对 `fogfocus_full`：
  - 平均检测数几乎完全崩塌
  - 非零检测帧占比从 `0.8235` 降到 `0.0711`
  - 雾类别切换率也超过允许阈值
- 相对全部 4 个基线：
  - 检测项全部失败
  - 雾切换率检查也全部失败
  - 只有 `beta_abs_delta_mean` 仍在允许范围内

## 分视频观察

`cocohead_pilot` 在 5 段视频上的统一模型检测数/帧：

| 视频 | 统一模型平均检测数/帧 | 混合平均检测数/帧 | 差值（hybrid - unified） | 主导雾类别 |
|---|---:|---:|---:|---|
| `getty_demo_real_fog` | `0.1726` | `1.7679` | `1.5952` | `UNIFORM FOG` |
| `control_clear_sparse_traffic_day` | `0.0000` | `21.0500` | `21.0500` | `PATCHY FOG` |
| `control_clear_medium_traffic_day` | `0.0167` | `29.3000` | `29.2833` | `PATCHY FOG` |
| `control_clear_heavy_traffic_day` | `0.0000` | `29.1500` | `29.1500` | `CLEAR` |
| `control_clear_dense_traffic_day` | `0.0000` | `38.1333` | `38.1333` | `PATCHY FOG` |

与 `fogfocus_full` 相比，最关键的退化是：

- 4 段 clear-weather control clips 上，统一检测几乎全部掉到 `0`
- 其中 3 段 clear clips 的主导雾类别从 `CLEAR` 漂移成了 `PATCHY FOG`
- `control_clear_medium_traffic_day` 的雾切换率升到 `0.1186`
- `control_clear_heavy_traffic_day` 的雾切换率升到 `0.1695`

这说明本轮不仅没有恢复 detector，连 clear-weather 下的雾稳定性也一起退化了。

## 结论

本轮 `cocohead_pilot` **失败**，不应进入正式 baseline 集合。

它给出的结论比前两轮更明确：

- 单纯把 detector 结构改成“保留 COCO head”，并不足以在当前训练设定下恢复统一模型检测能力
- 至少在这次 `1 epoch + frame_stride=16` 的 pilot 中，结构改动后统一检测出现了接近全面失效
- clear-weather control clips 上同时出现了检测崩塌和雾类别漂移，说明这条路线当前还不能作为正式替代方案

更稳妥的下一步不应直接扩大训练规模，而应先排查：

- COCO head 下训练标签映射是否真正匹配当前损失实现
- 结构迁移后 detector 输出分布是否被阈值/NMS 过滤到接近全空
- 当前多任务损失组合是否在 COCO-head 路线下破坏了 clear-weather 表征

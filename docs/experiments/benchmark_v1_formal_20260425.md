# benchmark_v1 正式结果记录（2026-04-25）

## 基本信息

- benchmark ID：`benchmark_v1`
- 运行日期：`2026-04-25`
- 视频集：
  - 1 段真实 fog 视频：`gettyimages-1353950094-640_adpp.mp4`
  - 4 段 UA-DETRAC clear-weather control clips
- 模型集：
  - `default_baseline`
  - `fogfocus`
  - `videoadapt`
  - `fogfocus_full`
- 输出目录：`outputs/Benchmark_Model_Compare_v1`

## 核心结论

- 在当前 `benchmark_v1` 的 5 段视频上，4 套统一模型的推荐路线全部都是 `hybrid`，没有任何一套统一模型在视频级推荐上翻盘。
- 从统一模型自身指标看，`fogfocus` 与 `fogfocus_full` 并列当前最强：
  - `weighted_unified_mean_count_per_frame = 6.549`
  - `weighted_unified_frames_with_detections_ratio = 0.824`
- `default_baseline` 与 `videoadapt` 基本持平：
  - `weighted_unified_mean_count_per_frame = 5.637`
  - `weighted_unified_frames_with_detections_ratio = 0.757`
- 即使是当前最强的 `fogfocus / fogfocus_full`，相对混合路线的平均检测差距仍然很大：
  - `weighted_mean_count_gap_hybrid_minus_unified = 11.478`

这意味着：截至本轮正式 benchmark，最终工程演示路线仍应保持为“雾模型 + 独立 `yolo11n` 检测”的混合方案。

## 模型对比摘要

| 模型 | 角色 | 统一模型平均检测数/帧 | 统一模型非零检测帧占比 | 混合减统一平均差值 | beta 均值 | fog switch rate | beta abs delta mean | 视频级推荐 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `default_baseline` | `historical_reference` | 5.637 | 0.757 | 12.390 | 0.004624 | 0.000000 | 0.000141 | `hybrid` x 5 |
| `fogfocus` | `fog_specialized_reference` | 6.549 | 0.824 | 11.478 | 0.034476 | 0.007444 | 0.000999 | `hybrid` x 5 |
| `videoadapt` | `target_video_reference` | 5.637 | 0.757 | 12.390 | 0.004711 | 0.000000 | 0.000138 | `hybrid` x 5 |
| `fogfocus_full` | `formal_training_reference` | 6.549 | 0.824 | 11.478 | 0.029682 | 0.032258 | 0.000906 | `hybrid` x 5 |

## 对 `fogfocus_full` 的解读

`fogfocus_full` 是本轮最重要的正式训练结果，其表现可以概括为：

- 在统一模型检测指标上，与 `fogfocus` 并列最好
- 相比 `fogfocus`，`weighted_beta_mean` 更低：`0.029682 < 0.034476`
- 相比 `fogfocus`，`weighted_beta_abs_delta_mean` 略好：`0.000906 < 0.000999`
- 但 `weighted_dominant_switch_rate` 更高：`0.032258 > 0.007444`

说明：

- `fogfocus_full` 在 `beta` 平滑性上略优于 `fogfocus`
- 但在雾类别稳定性上反而更容易切换
- 整体上可以认为 `fogfocus_full` 是当前“正式训练参考点”，但仍不足以替代混合路线

## 分视频观察（以 `fogfocus_full` 为例）

| 视频 | 统一模型平均检测数/帧 | 混合平均检测数/帧 | 差值（hybrid - unified） | 主导雾类别 |
|---|---:|---:|---:|---|
| `getty_demo_real_fog` | 0.685 | 1.768 | 1.083 | `UNIFORM FOG` |
| `control_clear_sparse_traffic_day` | 7.117 | 21.050 | 13.933 | `CLEAR` |
| `control_clear_medium_traffic_day` | 5.983 | 29.300 | 23.317 | `CLEAR` |
| `control_clear_heavy_traffic_day` | 11.883 | 29.150 | 17.267 | `CLEAR` |
| `control_clear_dense_traffic_day` | 17.633 | 38.133 | 20.500 | `CLEAR` |

可见：

- 在真实 fog 视频上，统一模型与混合路线之间仍存在稳定差距
- 在 4 段 clear-weather control clips 上，统一模型检测分支明显偏弱
- 因此当前统一模型的主要短板已经非常明确：**clear-weather traffic detection capacity 远弱于独立 `yolo11n`**

## 当前建议

### 工程路线

- 最终演示继续采用混合方案，不建议把统一模型直接作为最终检测路线替代 `yolo11n`

### 研究路线

- 后续若继续优化统一模型，应优先解决 clear-weather control clips 上的检测能力缺口
- 单纯继续调 fog head 或 `beta` 相关参数，不太可能解决当前 benchmark 暴露出来的主问题

## 关联产物

- 多模型总览：`outputs/Benchmark_Model_Compare_v1/model_overview.csv`
- 模型 × 视频矩阵：`outputs/Benchmark_Model_Compare_v1/model_video_matrix.csv`
- 汇总 JSON：`outputs/Benchmark_Model_Compare_v1/multi_model_benchmark_summary.json`
- 汇总 Markdown：`outputs/Benchmark_Model_Compare_v1/multi_model_benchmark_summary.md`

# benchmark_v1 说明

## 目的

`benchmark_v1` 的目标不是追求“视频越多越好”，而是先固定一套可重复运行的真实视频基准，
让以下问题可以被稳定回答：

- 新权重是否真的优于当前 baseline
- 哪个模型更适合统一推理路线
- 哪个模型在雾分类稳定性与 `beta` 抖动上更可接受

当前仓库已经支持：

- 单模型 benchmark
- 多模型 benchmark
- 候选模型对标

而 `benchmark_v1` 的职责是固定这些工具的输入视频集合与采样策略。

## 当前状态

当前工作区里已经可以直接运行的 `benchmark_v1` 资产共有 5 段：

- 1 段真实雾天部署视频
  - `gettyimages-1353950094-640_adpp.mp4`
- 4 段由 UA-DETRAC 序列导出的真实晴天交通 control clips
  - `benchmarks/videos/ua_detrac_mvi40152_clear_sparse_day.mp4`
  - `benchmarks/videos/ua_detrac_mvi20033_clear_medium_day.mp4`
  - `benchmarks/videos/ua_detrac_mvi20011_clear_heavy_day.mp4`
  - `benchmarks/videos/ua_detrac_mvi20035_clear_dense_day.mp4`

因此，当前 `benchmark_v1` 已经从“只有一个真实 fog 视频的模板”升级成了
“1 段真实 fog 视频 + 4 段真实 clear-weather control clips”的可运行版本。

但这仍然不是论文最终形态，因为：

- fog 侧覆盖仍然不足，当前只有 1 段真实 fog 视频
- clear-weather control 已经补齐多段，但真实 fog 的轻/中/局地浓淡变化仍未覆盖充分

## benchmark_v1 的固定规则

### 视频选择规则

- 优先使用真实高速公路或路侧监控视频，不使用离线合成视频替代真实部署场景。
- 每个视频都应长期保留，不应在同一 `benchmark_id` 下静默替换。
- 如果确需调整集合，应新建 `benchmark_v2`，而不是回写 `benchmark_v1`。

### 覆盖要求

建议至少补齐以下场景：

- 轻雾
- 中雾
- 团雾 / 局地浓淡不均
- 车辆稀疏
- 车辆密集
- 不同光照条件下的白天视频

最低建议规模：

- 4 到 6 段视频

当前 `benchmark_v1` 已经具备 5 段 active 资产，但其中 4 段属于 clear-weather control，
后续仍应继续补入新的真实 fog 视频，而不是只增加 clear-weather control。

### 运行策略

- `sample_stride` 固定
- `max_frames` 固定
- 同一轮模型比较必须使用同一份 `configs/benchmark_videos.json`

## 当前推荐命令

### 单模型 benchmark

```bash
bash scripts/run_benchmark.sh
```

### 构建 benchmark 资产

如果需要重新生成 UA-DETRAC control clips，可以执行：

```bash
python scripts/build_benchmark_assets.py
```

该脚本当前会根据 `configs/benchmark_videos.json` 中的 `source_sequence`、
`clip_start_frame` 和 `clip_num_frames` 字段重建 control clips，并输出：

- `outputs/Benchmark_Assets/benchmark_asset_build_report.json`
- `outputs/Benchmark_Assets/benchmark_asset_build_report.md`

### 多模型 benchmark

```bash
bash scripts/run_multi_model_benchmark.sh
```

## benchmark_v1 基线模型集合

当前 `configs/benchmark_models.json` 已经被收口为 `benchmark_v1` 配套 baseline 清单。

当前角色如下：

- `historical_reference`
  对应 `default_baseline`，表示长期默认统一模型参考点
- `fog_specialized_reference`
  对应 `fogfocus`，表示偏雾天气头适配参考点
- `target_video_reference`
  对应 `videoadapt`，表示目标视频适配参考点
- `formal_training_reference`
  对应 `fogfocus_full`，表示当前正式 fog-focused 训练参考点

这些角色会被写入：

- `model_overview.csv`
- `model_video_matrix.csv`
- `multi_model_benchmark_summary.json/.md`

### 候选模型对标

```bash
BS_CANDIDATE_WEIGHTS=/abs/path/to/unified_model_best.pt \
BS_CANDIDATE_LABEL=my_candidate \
bash scripts/run_candidate_benchmark.sh
```

## 候选模型通过规则

当前候选模型默认通过规则由 `scripts/benchmark_candidate_model.py` 内置，比较逻辑是：

- 候选模型的 `weighted_unified_mean_count_per_frame` 不能低于 baseline
- 候选模型的 `weighted_unified_frames_with_detections_ratio` 不能低于 baseline
- 候选模型的 `weighted_dominant_switch_rate` 相对 baseline 的恶化不能超过容忍阈值
- 候选模型的 `weighted_beta_abs_delta_mean` 相对 baseline 的恶化不能超过容忍阈值

当前默认容忍阈值：

- `allowed_unified_mean_drop = 0.0`
- `allowed_unified_ratio_drop = 0.0`
- `allowed_fog_switch_increase = 0.02`
- `allowed_beta_abs_delta_increase = 0.005`

这意味着：

- 检测侧不能退步
- 稳定性侧允许极小幅度恶化，但不允许明显变差

候选模型输出的 `candidate_benchmark_summary.json/.md` 会明确给出：

- overall pass / fail
- 每个 baseline 的 pass / fail
- 每条规则的实际阈值

## 建议的下一次补充

下一轮应优先补齐真实视频，而不是继续扩展脚本数量：

1. 把 `configs/benchmark_videos.json` 扩到 4 到 6 段真实视频
2. 对 `benchmark_v1` 输出做一次正式归档
3. 用同一套 benchmark 对 `default / fogfocus / videoadapt / 新候选权重` 做横向比较

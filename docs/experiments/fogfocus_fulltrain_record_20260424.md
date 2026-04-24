# Fog-Focused Formal Experiment Record

## 1. 实验目标

- 实验名称：`Fog-Focused-Full-01`
- 实验日期：`2026-04-24`
- 实验负责人：`Codex + 用户工作区`
- 实验目的：
  - 验证冻结检测器、关闭检测损失后，雾分类与 beta 回归在完整 epoch 训练中的稳定性与泛化能力。
  - 检查正式 fog-focused 训练完成后，是否能改善真实视频中的雾类型分布与 beta 表达。
  - 在训练结束后自动执行路线评估，确认最终演示路线是否发生变化。

## 2. 起始条件

- 起始 checkpoint：
  - `outputs/Fog_Detection_Project_fogfocus/checkpoints/checkpoint_epoch_0004.pt`
- 是否只恢复模型参数：
  - `BS_RESUME_MODEL_ONLY=1`
- 数据根目录：
  - `data/UA-DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train`
- 深度缓存目录：
  - `outputs/Depth_Cache`
- 训练/验证划分规则：
  - `TRAIN_RATIO = 0.8`
  - `FRAME_STRIDE = 1`
  - `SEED = 42`

## 3. 训练配置

| 项目 | 值 |
|---|---|
| 输出目录 | `outputs/Fog_Detection_Project_fogfocus_full` |
| Checkpoint 目录 | `outputs/Fog_Detection_Project_fogfocus_full/checkpoints` |
| 设备 | `cuda` |
| Batch size | `16` |
| 总 epoch 数 | `12` |
| 学习率 | `1e-5` |
| `FREEZE_YOLO_FOR_FOG` | `1` |
| `DET_LOSS_WEIGHT` | `0.0` |
| `FOG_CLS_LOSS_WEIGHT` | `1.75` |
| `FOG_REG_LOSS_WEIGHT` | `1.35` |
| `FOG_CLEAR_PROB` | `0.15` |
| `FOG_UNIFORM_PROB` | `0.35` |
| `FOG_PATCHY_PROB` | `0.50` |
| `FOG_BETA_MIN` | `0.04` |
| `UNIFORM_DEPTH_SCALE` | `7.0` |
| `PATCHY_DEPTH_BASE` | `2.0` |
| `PATCHY_DEPTH_NOISE_SCALE` | `8.0` |
| `FOG_LABEL_SMOOTHING` | `0.05` |
| `FOG_CLS_CLEAR_WEIGHT` | `0.75` |
| `FOG_CLS_UNIFORM_WEIGHT` | `1.0` |
| `FOG_CLS_PATCHY_WEIGHT` | `1.1` |
| `NUM_WORKERS` | `8` |
| `PREFETCH_FACTOR` | `2` |
| `PERSISTENT_WORKERS` | `1` |
| `SKIP_QAT` | `1` |

## 4. 运行命令

```bash
bash scripts/run_fogfocus_full_train.sh
```

本次实际训练采用的核心环境变量等价于：

```bash
BS_OUTPUT_DIR=outputs/Fog_Detection_Project_fogfocus_full \
BS_CHECKPOINT_DIR=outputs/Fog_Detection_Project_fogfocus_full/checkpoints \
BS_RESUME_CHECKPOINT=outputs/Fog_Detection_Project_fogfocus/checkpoints/checkpoint_epoch_0004.pt \
BS_BATCH_SIZE=16 \
BS_EPOCHS=12 \
BS_LR=1e-5 \
BS_QAT_EPOCHS=0 \
BS_SKIP_QAT=1 \
BS_MAX_TRAIN_BATCHES=0 \
BS_MAX_VAL_BATCHES=0 \
BS_RESUME_MODEL_ONLY=1 \
BS_FREEZE_YOLO_FOR_FOG=1 \
BS_DET_LOSS_WEIGHT=0.0 \
BS_FOG_CLS_LOSS_WEIGHT=1.75 \
BS_FOG_REG_LOSS_WEIGHT=1.35 \
BS_FOG_CLEAR_PROB=0.15 \
BS_FOG_UNIFORM_PROB=0.35 \
BS_FOG_PATCHY_PROB=0.50 \
BS_FOG_BETA_MIN=0.04 \
BS_UNIFORM_DEPTH_SCALE=7.0 \
BS_PATCHY_DEPTH_BASE=2.0 \
BS_PATCHY_DEPTH_NOISE_SCALE=8.0 \
BS_FOG_LABEL_SMOOTHING=0.05 \
BS_FOG_CLS_CLEAR_WEIGHT=0.75 \
BS_FOG_CLS_UNIFORM_WEIGHT=1.0 \
BS_FOG_CLS_PATCHY_WEIGHT=1.1 \
python src/train.py
```

## 5. 训练产物

- 运行日志：
  - `outputs/Fog_Detection_Project_fogfocus_full/fogfocus_full_train_20260424_221613.log`
- `config_snapshot.json`：
  - `outputs/Fog_Detection_Project_fogfocus_full/runs/smoke_20260424_141616/config_snapshot.json`
- `metrics.jsonl`：
  - `outputs/Fog_Detection_Project_fogfocus_full/runs/smoke_20260424_141616/metrics.jsonl`
- `summary.json`：
  - `outputs/Fog_Detection_Project_fogfocus_full/runs/smoke_20260424_141616/summary.json`
- `unified_model.pt`：
  - `outputs/Fog_Detection_Project_fogfocus_full/unified_model.pt`
- `unified_model_best.pt`：
  - `outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt`

## 6. 训练结果摘要

| 指标 | 值 |
|---|---|
| 最佳监控指标（best loss） | `1.007177609544653` |
| 最终 train loss | `0.9346499112394021` |
| 最终 val loss | `1.007177609544653` |
| 最终 fog cls loss（train） | `0.533983662720978` |
| 最终 fog reg loss（train） | `0.00013222367131010554` |
| 最终 fog cls loss（val） | `0.5753997521145598` |
| 最终 fog reg loss（val） | `0.00016891948472983278` |
| 是否出现 non-finite grad | `否` |
| 是否触发 AMP recovery | `否` |

## 7. 后处理评估

训练完成后自动执行：

```bash
python scripts/postprocess_fogfocus_full.py
```

结果如下：

| 项目 | 值 |
|---|---|
| 评估视频 | `gettyimages-1353950094-640_adpp.mp4` |
| fog 权重 | `outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt` |
| 统一方案平均检测数/帧 | `0.6845238095238095` |
| 混合方案平均检测数/帧 | `1.7678571428571428` |
| dominant fog class 直方图 | `{'CLEAR': 2, 'UNIFORM FOG': 102, 'PATCHY FOG': 64}` |
| mean beta | `0.06709203869104385` |
| route recommendation | `hybrid` |

## 8. 结论

- 本次实验是否优于 `fogfocus` 起点权重：
  - 从训练监控指标看，是。最佳 `val loss` 从起点阶段记录的 `1.1711` 降至 `1.0072`，雾任务收敛更充分。
- 是否打破 `clear` 偏置：
  - 是。真实视频上不再塌缩到 `clear`，dominant fog class 分布为 `UNIFORM FOG` 与 `PATCHY FOG` 为主。
- 是否值得纳入论文正式结果：
  - 是。该结果比 smoke 级 fogfocus 权重更适合写入论文“正式 fog-focused 训练”部分。
- 是否需要继续训练或调整参数：
  - 若目标是继续提升论文说服力，可进一步补多视频验证和消融实验；若目标是工程演示，当前结果已足够作为正式 fog 权重使用。

## 9. 备注

- 本次训练通过完整 epoch 跑完，未使用 smoke batch 限制。
- 虽然 `run_dir` 名称仍为 `smoke_*`，但这仅是当前代码命名逻辑遗留，不代表本次仍是 smoke run。
- 检测路线判断未发生变化：最终演示仍应保留 `hybrid`。
- 自动后处理产物：
  - `outputs/Fog_Detection_Project_fogfocus_full/formal_experiment_summary.md`
  - `outputs/Fog_Detection_Project_fogfocus_full/formal_experiment_summary.json`

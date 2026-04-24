# Fog-Focused Formal Experiment Record

本文件用于记录一次“正式 fog-focused 训练”的完整实验条件与结果，建议每次正式训练复制一份独立记录。

## 1. 实验目标

- 实验名称：
- 实验日期：
- 实验负责人：
- 实验目的：
  - 例：验证冻结检测器、关闭检测损失后，雾分类与 beta 回归在完整 epoch 训练中的稳定性与泛化能力。

## 2. 起始条件

- 起始 checkpoint：
  - 例：`outputs/Fog_Detection_Project_fogfocus/checkpoints/checkpoint_epoch_0004.pt`
- 是否只恢复模型参数：
  - 例：`BS_RESUME_MODEL_ONLY=1`
- 数据根目录：
- 深度缓存目录：
- 训练/验证划分规则：
  - `TRAIN_RATIO`
  - `FRAME_STRIDE`
  - `SEED`

## 3. 训练配置

| 项目 | 值 |
|---|---|
| 输出目录 | |
| Checkpoint 目录 | |
| 设备 | |
| Batch size | |
| 总 epoch 数 | |
| 学习率 | |
| `FREEZE_YOLO_FOR_FOG` | |
| `DET_LOSS_WEIGHT` | |
| `FOG_CLS_LOSS_WEIGHT` | |
| `FOG_REG_LOSS_WEIGHT` | |
| `FOG_CLEAR_PROB` | |
| `FOG_UNIFORM_PROB` | |
| `FOG_PATCHY_PROB` | |
| `FOG_BETA_MIN` | |
| `UNIFORM_DEPTH_SCALE` | |
| `PATCHY_DEPTH_BASE` | |
| `PATCHY_DEPTH_NOISE_SCALE` | |
| `FOG_LABEL_SMOOTHING` | |
| `FOG_CLS_CLEAR_WEIGHT` | |
| `FOG_CLS_UNIFORM_WEIGHT` | |
| `FOG_CLS_PATCHY_WEIGHT` | |
| `NUM_WORKERS` | |
| `PREFETCH_FACTOR` | |
| `PERSISTENT_WORKERS` | |
| `SKIP_QAT` | |

## 4. 运行命令

```bash
bash scripts/run_fogfocus_full_train.sh
```

若本次实验做了额外覆盖，请完整粘贴实际执行命令：

```bash
# 在此粘贴本次实际执行命令
```

## 5. 训练产物

- 运行日志：
- `config_snapshot.json`：
- `metrics.jsonl`：
- `summary.json`：
- `unified_model.pt`：
- `unified_model_best.pt`：

## 6. 训练结果摘要

| 指标 | 值 |
|---|---|
| 最佳监控指标（best loss） | |
| 最终 train loss | |
| 最终 val loss | |
| 最终 fog cls loss（train） | |
| 最终 fog reg loss（train） | |
| 最终 fog cls loss（val） | |
| 最终 fog reg loss（val） | |
| 是否出现 non-finite grad | |
| 是否触发 AMP recovery | |

## 7. 后处理评估

建议在训练完成后立刻执行：

```bash
python scripts/evaluate_inference_routes.py \
  --video gettyimages-1353950094-640_adpp.mp4 \
  --fog-weights <本次 best 权重路径> \
  --sample-stride 5 \
  --max-frames 300 \
  --output-dir <本次实验的路线评估目录>
```

填写结果：

| 项目 | 值 |
|---|---|
| 评估视频 | |
| fog 权重 | |
| 统一方案平均检测数/帧 | |
| 混合方案平均检测数/帧 | |
| dominant fog class 直方图 | |
| mean beta | |
| route recommendation | |

## 8. 结论

- 本次实验是否优于 `fogfocus` 起点权重：
- 是否打破 `clear` 偏置：
- 是否值得纳入论文正式结果：
- 是否需要继续训练或调整参数：

## 9. 备注

- 记录异常情况、显存占用、训练中断、视频表现观察等：

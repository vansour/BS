# Thesis Experiment Result Tables (Filled)

以下表格已根据 `2026-04-24` 完成的正式 fog-focused 训练与自动后处理结果回填，可直接复制到论文第八章再做格式化。

## 表 1  Fog-Focused 正式训练结果表

| 实验名称 | 起始权重 | 训练策略 | epoch | 最佳 val loss | 最终 fog cls loss | 最终 fog reg loss | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| Fog-Focused-Full-01 | `outputs/Fog_Detection_Project_fogfocus/checkpoints/checkpoint_epoch_0004.pt` | 冻结检测器，`DET_LOSS_WEIGHT=0` | 12 | 1.0072 | 0.5754（val） | 0.000169（val） | 完整 epoch，无 smoke 限制，未触发 AMP recovery，建议作为论文正式雾权重结果 |

## 表 2  方法消融实验表

| 编号 | 设置 | 是否使用深度 | 是否使用 patchy | 任务组织 | fog cls 指标 | beta 回归指标 | 说明 |
|---|---|---|---|---|---:|---:|---|
| A1 | 无深度基线 | 否 | 否 | 雾分类 + 回归 | 待补 | 待补 | 建议后续补正式消融 |
| A2 | 仅 uniform 合成 | 是 | 否 | 雾分类 + 回归 | 待补 | 待补 | 建议后续补正式消融 |
| A3 | uniform + patchy | 是 | 是 | 雾分类 + 回归 | 待补 | 待补 | 建议后续补正式消融 |
| A4 | 单任务雾分类 | 是 | 是 | 仅分类 | 待补 | - | 建议后续补正式消融 |
| A5 | 双任务雾属性 | 是 | 是 | 分类 + 回归 | 待补 | 待补 | 建议后续补正式消融 |
| A6 | 完整多任务 | 是 | 是 | 检测 + 分类 + 回归 | 待补 | 待补 | 建议后续补正式消融 |

说明：截至当前工作区，正式完整结果最扎实的是 `Fog-Focused-Full-01`；系统性消融仍建议在后续继续补齐。

## 表 3  真实视频路线对比表

| 路线 | fog 权重 | 车辆检测器 | 统一方案平均检测数/帧 | 非零检测帧占比 | mean beta | dominant fog class | 结论 |
|---|---|---|---:|---:|---:|---|---|
| 默认统一模型 | `outputs/Fog_Detection_Project/unified_model_best.pt` | 统一模型检测头 | 0.411 | 0.411 | 0.00570 | `CLEAR` | 不推荐作为最终结果 |
| 视频适配统一模型 | `outputs/Fog_Detection_Project_videoadapt_formal/unified_model_best.pt` | 统一模型检测头 | 0.411 | 0.411 | 0.00575 | `CLEAR` | 未改善 clear 偏置 |
| 偏雾天统一模型 | `outputs/Fog_Detection_Project_fogfocus/unified_model_best.pt` | 统一模型检测头 | 0.685 | 0.571 | 0.06782 | `PATCHY FOG` | 可作为阶段性最佳统一雾权重 |
| Fog-Focused 正式训练后统一模型 | `outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt` | 统一模型检测头 | 0.685 | 0.571 | 0.06709 | `UNIFORM FOG` | 正式 fog 权重，更适合论文正式结果 |
| 最终演示混合方案 | `outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt` | `yolo11n.pt` | 0.685（统一侧） / 1.768（混合侧） | 0.571（统一侧） / 0.488（混合侧） | 0.06709 | `UNIFORM FOG` | 最终采用 |

## 表 4  检测性能对比表

| 方法 | 权重/设置 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 备注 |
|---|---|---:|---:|---:|---:|---|
| `yolo11n` 车辆基线 | `yolo11n.pt` | 0.8077 | 0.7344 | 0.8102 | 0.5714 | 最终混合路线使用的独立检测器 |
| 统一模型检测分支（默认） | `Fog_Detection_Project/unified_model_best.pt` | 0.8836 | 0.8038 | 0.8937 | 0.6570 | UA-DETRAC clear-weather val 上当前最佳 |
| 统一模型检测分支（fogfocus） | `Fog_Detection_Project_fogfocus/unified_model_best.pt` | 0.8720 | 0.7700 | 0.8645 | 0.6204 | 偏雾天气头适配后检测指标略降 |
| 统一模型检测分支（完整训练后） | `Fog_Detection_Project_fogfocus_full/unified_model_best.pt` | 0.8720 | 0.7700 | 0.8645 | 0.6204 | 正式训练冻结检测器，因此与 `fogfocus` 一致 |

说明：

- 上表由 `scripts/evaluate_uadetrac_detection_metrics.py` 在 UA-DETRAC 验证集上统计得到。
- 若论文保留“最终演示混合方案”这一表述，可直接说明其检测器部分继承 `yolo11n` 车辆基线指标。

## 表 5  多视频验证汇总表

| 视频编号 | 场景类型 | 天气特征 | fog 权重 | mean beta | dominant fog class | 统一平均检测数/帧 | 混合平均检测数/帧 | route recommendation |
|---|---|---|---|---:|---|---:|---:|---|
| V1 | 固定监控视频 | 浓雾、车流中等 | `fogfocus_full` | 0.06709 | `UNIFORM FOG` | 0.685 | 1.768 | `hybrid` |
| V2 | 山区桥隧 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |
| V3 | 轻雾 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |
| V4 | 中重雾 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |
| V5 | 夜间/低照度 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |

## 可直接写进论文的简短结论

- 正式 fog-focused 训练在完整 epoch 条件下将最佳验证损失降低到 `1.0072`，相较起始 fogfocus 权重继续改善了雾任务收敛状态。
- 训练完成后的新 best 权重在真实视频上保持了非 `clear` 主导的雾分类结果，dominant fog class 分布为 `UNIFORM FOG: 102`、`PATCHY FOG: 64`。
- 尽管正式 fog-focused 训练强化了雾任务表现，但路线评估结论仍未改变：最终工程演示仍应采用 `fogfocus_full + yolo11n` 的混合方案。

# Thesis Experiment Result Table Templates

以下表格可直接复制进论文第八章，填入正式实验结果。

## 表 1  Fog-Focused 正式训练结果表

| 实验名称 | 起始权重 | 训练策略 | epoch | 最佳 val loss | 最终 fog cls loss | 最终 fog reg loss | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| Fog-Focused-Full-01 | `fogfocus/checkpoint_epoch_0004.pt` | 冻结检测器，`DET_LOSS_WEIGHT=0` | | | | | |
| Fog-Focused-Full-02 |  |  |  |  |  |  |  |

建议备注中写明：
- 是否完整 epoch
- 是否使用 smoke 限制
- 是否触发 AMP recovery
- 是否作为论文正式模型

## 表 2  方法消融实验表

| 编号 | 设置 | 是否使用深度 | 是否使用 patchy | 任务组织 | fog cls 指标 | beta 回归指标 | 说明 |
|---|---|---|---|---|---:|---:|---|
| A1 | 无深度基线 | 否 | 否 | 雾分类 + 回归 | | | |
| A2 | 仅 uniform 合成 | 是 | 否 | 雾分类 + 回归 | | | |
| A3 | uniform + patchy | 是 | 是 | 雾分类 + 回归 | | | |
| A4 | 单任务雾分类 | 是 | 是 | 仅分类 | | - | |
| A5 | 双任务雾属性 | 是 | 是 | 分类 + 回归 | | | |
| A6 | 完整多任务 | 是 | 是 | 检测 + 分类 + 回归 | | | |

建议：
- `fog cls 指标` 可填写 Accuracy / Macro-F1
- `beta 回归指标` 可填写 MSE / MAE / RMSE 中一种或多种

## 表 3  真实视频路线对比表

| 路线 | fog 权重 | 车辆检测器 | 统一方案平均检测数/帧 | 非零检测帧占比 | mean beta | dominant fog class | 结论 |
|---|---|---|---:|---:|---:|---|---|
| 默认统一模型 | `Fog_Detection_Project/unified_model_best.pt` | 统一模型检测头 | | | | | |
| 视频适配统一模型 | `Fog_Detection_Project_videoadapt_formal/unified_model_best.pt` | 统一模型检测头 | | | | | |
| 偏雾天统一模型 | `Fog_Detection_Project_fogfocus/unified_model_best.pt` | 统一模型检测头 | | | | | |
| 最终混合方案 | `Fog_Detection_Project_fogfocus/unified_model_best.pt` | `yolo11n.pt` | | | | | |

建议：
- 若论文保留“最终演示路线”论证，这张表是核心表
- 结论列可以写“保留 / 不推荐 / 最终采用”

## 表 4  检测性能对比表

| 方法 | 权重/设置 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 备注 |
|---|---|---:|---:|---:|---:|---|
| 纯 YOLO 单类检测基线 |  |  |  |  |  | |
| 统一模型检测分支（默认） |  |  |  |  |  | |
| 统一模型检测分支（完整训练后） |  |  |  |  |  | |
| 最终演示混合方案 | `yolo11n.pt` + fog 权重 |  |  |  |  | 若无法严格定义联合 mAP，可只在备注说明演示用途 |

## 表 5  多视频验证汇总表

| 视频编号 | 场景类型 | 天气特征 | fog 权重 | mean beta | dominant fog class | 统一平均检测数/帧 | 混合平均检测数/帧 | route recommendation |
|---|---|---|---|---:|---|---:|---:|---|
| V1 | 平原高流量 | | | | | | | |
| V2 | 山区桥隧 | | | | | | | |
| V3 | 轻雾 | | | | | | | |
| V4 | 中重雾 | | | | | | | |
| V5 | 夜间/低照度 | | | | | | | |

## 写作建议

- `表 1` 和 `表 2` 更偏方法论说服力；
- `表 3` 和 `表 5` 更偏工程落地说服力；
- `表 4` 用于补足“检测分支”的论文严谨性。

如果论文篇幅有限，至少保留：

1. Fog-Focused 正式训练结果表  
2. 方法消融实验表  
3. 真实视频路线对比表  
4. 检测性能对比表

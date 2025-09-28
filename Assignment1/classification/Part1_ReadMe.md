# Assignment 1 - Part 1: 分类评测指标 

本实验围绕分类任务的评测指标展开，要求补全基础与二分类指标函数，并用给定脚本自测与查看输出。

分数占比 - 20%

仓库主要文件: `classification/`
- 基础指标实现：`accuracy_error.py`
- 二分类指标实现：`evaluation_metrics.py`
- 测试脚本：`test.py`
（本部分不依赖额外数据集）

## 实验目标
- 理解并实现分类的基础指标
    - accuracy_score（准确率）
    - mean_squared_error（均方误差，用于数值预测/对比）
- 理解并实现二分类指标
    - precision、recall、f1（基于 TP/FP/FN/TN）
- 使用测试脚本验证实现是否正确

## 实验内容
1) 数据与文件说明
- 本实验不提供额外数据文件
- `test.py` 内置了若干玩具用例用于自测。

2) 你需要在代码中完成 `TODO` 部分
- 在 `accuracy_error.py` 中实现：
    - accuracy_score(y_true, y_pred)
    - mean_squared_error(y_true, y_pred)

- 在 `evaluation_metrics.py` 中实现二分类指标：
    - precision_score、recall_score、f1_score（已提供 _binary_counts 与 EPS）

- 注意：完成后删除/注释占位的 raise NotImplementedError(...)

## 评测与运行
- 运行 `test.py`,脚本将打印期望值与实际值，并标记 PASS/FAIL，在示例用例下：
    `accuracy` 期望 0.75
    `mse` 期望约 0.6667
    `precision / recall / f1` 期望各 0.50

## 提交内容
- 提交完成后的 `accuracy_error.py` 与 `evaluation_metrics.py`。
- 附上一次 `test.py` 运行的截图或日志（显示 PASS 结果）。
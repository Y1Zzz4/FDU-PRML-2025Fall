# Assignment 1 - Part 2: 决策树 

本实验围绕决策树分类展开，要求实现多种分裂准则（信息增益、信息增益率、Gini、分类误差率），并用给定脚本在玩具数据上验证预测结果与接口正确性。

分数占比 - 40%

仓库主要文件: `decision_tree/`
- 分裂准则实现：`criterion.py`
- 决策树主体：`decision_tree.py`
- 测试脚本：`test_decision_tree.py`
（本部分不依赖额外数据集）

## 实验目标
- 理解并实现四种分裂度量
    - Information_Gain（信息增益）
    - Information_Gain_Ratio（信息增益率）、
    - Gini_Index（Gini 指数）
    - Classification_Error（分类误差率）
- 理解常见超参数的作用

## 实验内容
1) 数据与文件说明
- 本实验不提供额外数据文件
- `test_decision_tree.py` 内置了 2D 玩具数据 X, y, 会分别以四种 criterion 训练并断言预测结果。

2) 你需要在代码中完成 `TODO` 部分
- 在 `criterion.py` 中补全四个函数的计算逻辑：
    - `__info_gain(y, l_y, r_y)`：父节点熵 − 子节点加权熵
    - `__info_gain_ratio(y, l_y, r_y)`：信息增益 ÷ 分裂信息（对子集比例的熵）
    - `__gini_index(y, l_y, r_y)`：分裂前 Gini − 分裂后 加权 Gini
    - `__error_rate(y, l_y, r_y)`：分裂前 分类误差 − 分裂后 加权分类误差

- 注意：完成后删除/注释占位的 raise NotImplementedError(...)

## 评测与运行
运行 `test_decision_tree.py`：脚本将用 {`info_gain`,`info_gain_ratio`,`gini`,`error_rate`} 逐一训练并断言 predict(T) 等于期望结果 [-1, 1, 1]。
你应看到四种准则均通过断言（无异常退出）。

## 提交内容
- 提交你完成后的 `criterion.py`
- 一次测试脚本运行的截图/日志
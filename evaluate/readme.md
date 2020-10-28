# 位置预测评估

2020.9.27 更新：测试了不同评估方法的执行，测试了评估结果的保存。

2020.9.30 更新：增加列表类型数据的评估

2020.10.18更新：调整基类

2020.10.23更新：更新文件结构，修复bug

2020.10.28更新：增加模型评估的计算，整理评估结果

## 文件组织格式

| 文件目录                               | 作用                   |
| -------------------------------------- | ---------------------- |
| dir_path/evaluate/eval_next_loc.py     | 评估下一跳位置预测     |
| dir_path/evaluate/eval_funcs.py        | 实现各种评估方法       |
| dir_path/evaluate/utils.py             | 功能/工具型函数        |
| dir_path/evaluate/config.json          | 评估类默认参数         |
| dir_path/evaluate/readme.md            | 评估部分说明文档       |
| dir_path/runtimeFiles/evaluate/res.txt | 评估结果（待整理格式） |
| dir_path/test/lzh_test                 | 运行样例/测试方法      |

## 使用说明

### 调用方法

当需要进行位置预测评估时：

1. 引入 `/evaluate` 目录下的 `eval_next_loc.py` 文件。
2. 创建 `Evaluate` 类。
3. 调用 `evaluate` 方法。
4. 可参考 `/test`目录下的 `lzh_test.py` 文件。

```python
from evaluate import eval_next_loc as enl

if __name__ == '__main__':
    data = '{' \
           '"uid1": { ' \
           '"trace_id1":' \
           '{ "loc_true": [1], "loc_pred": [[0.01, 0.91, 0.8]] }, ' \
           '"trace_id2":' \
           '{ "loc_true": [2], "loc_pred": [[0.2, 0.13, 0.08]] } ' \
           '},' \
           '"uid2": { ' \
           '"trace_id1":' \
           '{ "loc_true": [0], "loc_pred": [[0.4, 0.5, 0.7]] }' \
           '}' \
           '}'
    data2 = '{' \
            '"uid1": { ' \
            '"trace_id1":' \
            '{ "loc_true": [1], "loc_pred": [[0.01, 0.1, 0.8]] }, ' \
            '"trace_id2":' \
            '{ "loc_true": [2], "loc_pred": [[0.2, 0.13, 0.08]] } ' \
            '},' \
            '"uid2": { ' \
            '"trace_id1":' \
            '{ "loc_true": [0], "loc_pred": [[0.4, 0.5, 0.7]] }' \
            '}' \
            '}'
    config = {
        'model': 'DeepMove',
        'mode': ['ACC', 'top-2', 'top-3']
    }
    var = enl.EvaluateNextLoc(config)
    # 正常写法
    #var.evaluate(data=data)
    # iterator/yield 写法
    data_list = [data, data2]
    for data in data_list:
        var.evaluate(data)
    var.save_result(r'D:\Users\12908\Documents\git\Bigscity-Human-Mobility-Prediction-Toolkit/runtimeFiles/evaluate')
```

### 参数含义

#### config.json

评估类的默认配置。

2. mode_list：所有支持的评估方法。
2. mode：评估方法，以列表形式。
3. output_gate：控制是否在评估过程中输出评估信息，可选项**"true"**或**"false"**。
4. data_path：若采用从文件中读取数据进行评估，则该参数为待评估文件的**绝对路径**。
5. model：评估模型，因为我们对某些如DeepMove这样的模型的输出需要进行**预处理**。

#### __init__(config)

初始化函数的参数意义。

1. config：用于传递 global_config。

#### evaluate(self, data=None)

evaluate函数对应的参数含义。

1. data：用户可选择传入数据进行评估，**json类型或者可转换为json的str类型**，也可以是列表类型的数据（列表元素为分batch的评估数据）；若不传入，则选择从文件中读取数据进行评估；若文件路径也没有可评估数据，则抛出异常。

#### save_result(self, result_path="")

save_result函数对应的参数含义。

1. result_path：评估结束后将评估结果保存到文件中，给出文件的绝对路径。

## 数据格式说明

附：关于位置的表示，参考DeepMove模型对所有的位置进行独热编码，每个位置具有一个编号。

```
{
	uid1: {
		trace_id1: {
			loc_true: 一个list类型列表，列表中元素代表真实位置(编号)
			loc_pred: 一个list类型列表，列表中的元素又是一个list类型列表，代表 [模型预测出的位置(编号)]。按照预测可能性的大小排列，最有可能的位置排在第1位；也可以是列表中元素代表 [模型预测出的位置(编号)的置信度]，比如DeepMove模型的输出。
		},
		trace_id2: {
			...
		},
		...
	},
	uid2: {
		...
	},
	...
}
```

样例：

```
data = '{' \
        '"uid1": { ' \
        '"trace_id1":' \
        '{ "loc_true": [1], "loc_pred": [[0.01, 0.91, 0.8]] }, ' \
        '"trace_id2":' \
        '{ "loc_true": [2], "loc_pred": [[0.2, 0.13, 0.08]] } ' \
        '},' \
        '"uid2": { ' \
        '"trace_id1":' \
        '{ "loc_true": [0], "loc_pred": [[0.4, 0.5, 0.7]] }' \
        '}' \
        '}'
```

## 评估方法

mode：评估选项/指标

1. ACC, 计算预测准确度（Accuracy）
2. MSE：均方误差（Mean Square Error）
3. MAE：平均绝对误差（Mean Absolute Error）
4. RMSE：均方根误差（Root Mean Square Error）
5. MAPE：平均绝对百分比误差（Mean Absolute Percentage Error）
6. MARE：平均绝对和相对误差（Mean Absolute Relative Error）
7. top-k：check whether the ground-truth location v appears in the top-k candidate locations，即真实位置在前k个预测位置中准确的准确度，若某个数据集上所有模型整体预测准确的偏低的情况，可采用top-5。
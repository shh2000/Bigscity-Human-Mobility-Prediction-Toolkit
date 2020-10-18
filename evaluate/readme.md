# 位置预测评估

2020.9.27 更新：测试了不同评估方法的执行，测试了评估结果的保存。

2020.9.30 更新：增加列表类型数据的评估

2020.10.18更新：调整基类

## 文件组织格式

|-- evaluate -- basic.py 评估基类

​					  -- evaluate.py 评估类（继承基类）

​					  -- config.json 评估类的配置文件

​					  -- readme.md 评估部分说明文档

|-- runtimeFiles -- evaluate -- res.txt 评估结果（待整理格式）

|-- test -- lzh_test.py	运行样例/测试方法

## 使用说明

### 调用方法

当需要进行位置预测评估时：

1. 引入 `/evaluate` 目录下的 `evaluate.py` 文件。
2. 创建 `EvalPredLoc` 类。
3. 调用 `evaluate` 方法。
4. 可参考 `/test/lzh_test.py` 文件。

```python
from evaluate import evaluate as epl

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
    var = epl.EvalPredLoc(r'D:\Users\12908\Documents\git\Bigscity-Human-Mobility-Prediction-Toolkit')
    # var.evaluate()
    var.evaluate(data=data, mode=['ACC', 'MAE', 'top-2', 'top-3'])
    var.save_result('/runtimeFiles/evaluate')
```

### 参数含义

#### config.json

1. all_mode：所有支持的评估方法
2. mode_list：对模型采用的评估方法，以列表形式。
3. data_path：若采用从文件中读取数据进行评估，则该参数必填，为待评估文件的**绝对路径**。
4. data_type：评估模型的数据类型，如对DeepMove模型的输出需要进行**预处理**。
5. output_gate：控制是否在评估过程中输出评估信息，可选项**"True"**或**"False"**。

#### __init__(dir_path)

初始化函数的参数意义。

1. dir_path：给出整个项目工程的绝对路径。

#### evaluate(self, data=None, config=None, mode=None)

evaluate函数对应的参数含义。

1. data：用户可选择传入数据进行评估，**json类型或者可转换为json的str类型**，也可以是列表类型，列表元素为分batch的评估数据，若不传入，则默认从config.json配置文件中读入待评估数据的路径进行评估。
2. config：对应用户配置的个性化参数，对应 **global_config**，可覆盖config配置文件中的参数。
3. mode：用户选择 **评估方法**，以列表形式传入，可不传入，从配置文件中读取。

后两个参数的意义是考虑用户既可以通过配置文件设置评估参数，也可以直接传入评估方法，由于不是必须参数，所以可取舍，只是希望方便用户操作。

#### save_result(self, result_path="")

save_result函数对应的参数含义。

1. result_path：评估结束后将评估结果保存到文件中，给出文件的相对路径（相对于项目工程）。

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
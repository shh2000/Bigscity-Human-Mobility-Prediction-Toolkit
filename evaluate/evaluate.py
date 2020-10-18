import json
import numpy as np
import os
import shutil
import re

from evaluate.basic import Evaluate


class EvalPredLoc(Evaluate):

    def __init__(self, dir_path):
        """
        Initialize the creation of the Evaluate Class
        :param dir_path: 项目工程的绝对路径
        """
        super().__init__(dir_path)
        # 从配置文件中读取相关参数
        self.all_mode = self.config['all_mode']
        self.mode_list = self.config['mode_list']
        self.data_path = self.config['data_path']
        self.data_type = self.config['data_type']
        self.output_switch = self.config['output_switch'] == 'True'
        # 初始化类的内部变量
        self.mode = "ACC"
        self.topK = 1
        self.maxK = 1
        self.data = {}
        self.metrics = {}
        self.trace_metrics = {}

    def evaluate(self, data=None, config=None, mode=None):
        """
        The entrance of evaluation (user-oriented)
        :param data: 待评估数据, 可以直接是dict类型或者str形式的dict类型，也可以是列表类型(分batch)
        :param config: 用户传进来的个性化参数
        :param mode: 指标，列表形式, 如 [MSE“，”MAPE“], 默认从配置文件中读入
        :return: 对应指标的结果
        """
        if mode is not None:
            self.mode_list = mode
        if config is not None:
            self.load_config(config)
        pattern = re.compile("top-[1-9]\\d*$")
        for mode in self.mode_list:
            if mode in self.all_mode:
                self.metrics[mode] = []
                self.trace_metrics[mode] = []
            elif re.match(pattern, mode) is not None:
                k = int(mode.split('-')[1])
                self.metrics[mode] = []
                self.trace_metrics[mode] = []
                self.maxK = k if k > self.maxK else self.maxK
            else:
                raise ValueError("{}: 不支持的评估方法".format(mode))
        if data is not None:
            self.data = data
        try:
            self.data = json.load(open(self.data_path)) if data is None else data
        except Exception:
            raise ValueError('待评估数据的路径无效')
        if isinstance(self.data, list):
            t_data = self.data
            for batch_data in t_data:
                self.data = batch_data
        self.evaluate_data()

    def save_result(self, result_path=""):
        """
        :param result_path: 相对路径，存放结果json
        :return:
        """
        path = self.dir_path + result_path
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.mkdir(path)
        file = open(path + '/res.txt', 'w')
        file.write(json.dumps(self.metrics))
        file.write('\n')
        file.write(json.dumps(self.trace_metrics))
        file.close()
        pass

    def evaluate_data(self):
        """
        evaluate data batch (internal)
        """
        self.init_data()
        loc_true = []
        loc_pred = []
        user_ids = self.data.keys()
        for user_id in user_ids:
            user = self.data[user_id]
            trace_ids = user.keys()
            for trace_id in trace_ids:
                trace = user[trace_id]
                t_loc_true = trace['loc_true']
                t_loc_pred = trace['loc_pred']
                loc_true.extend(t_loc_true)
                loc_pred.extend(t_loc_pred)
                self.run_mode(t_loc_pred, t_loc_true, 'trace')
        self.run_mode(loc_pred, loc_true, 'model')

    def load_config(self, config):
        """
        from global_config settings
        :param config: 用户配置的个性化参数
        :return:
        """
        if 'mode_list' in config.keys():
            self.mode_list = config['mode_list']
        if 'data_path' in config.keys():
            self.data_path = config['data_path']
        if 'data_type' in config.keys():
            self.data_type = config['data_type']
        if 'output_switch' in config.keys():
            self.output_switch = config['output_switch'] == 'True'

    def run_mode(self, loc_pred, loc_true, field):
        """
        The method of run evaluate (internal)
        :param loc_pred: 模型预测出位置的结果
        :param loc_true: 数据集的真实位置
        :param field: 是对轨迹进行评估还是对整个模型进行评估 (internal)
        """
        assert len(loc_pred) == len(loc_true), "评估的预测数据与真实数据大小不一致"
        t_loc_pred = [[] for i in range(self.maxK)]
        for i in range(len(loc_true)):
            assert len(loc_pred[i]) >= self.maxK, "模型的位置预测结果少于top-{}评估方法的k值".format(self.maxK)
            for j in range(self.maxK):
                t_loc_pred[j].append(loc_pred[i][j])
        for mode in self.mode_list:
            self.mode = mode
            if self.mode == 'ACC':
                t, avg_acc = self.ACC(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_acc, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_acc)
                else:
                    self.trace_metrics[self.mode].append(avg_acc)
            elif self.mode == 'RMSE':
                avg_loss = self.RMSE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_loss, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_loss)
                else:
                    self.trace_metrics[self.mode].append(avg_loss)
            elif self.mode == 'MSE':
                avg_loss = self.MSE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_loss, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_loss)
                else:
                    self.trace_metrics[self.mode].append(avg_loss)
            elif self.mode == "MAE":
                avg_loss = self.MAE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_loss, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_loss)
                else:
                    self.trace_metrics[self.mode].append(avg_loss)
            elif self.mode == "MAPE":
                avg_loss = self.MAPE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_loss, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_loss)
                else:
                    self.trace_metrics[self.mode].append(avg_loss)
            elif self.mode == "MARE":
                avg_loss = self.MARE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_loss, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_loss)
                else:
                    self.trace_metrics[self.mode].append(avg_loss)
            elif self.mode == "SMAPE":
                avg_loss = self.SMAPE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.output(self.mode, avg_loss, field)
                if field == "model":
                    self.metrics[self.mode].append(avg_loss)
                else:
                    self.trace_metrics[self.mode].append(avg_loss)
            else:
                self.topK = int(mode.split('-')[1])
                avg_acc = self.top_k(np.array(t_loc_pred, dtype=object), np.array(loc_true, dtype=object))
                self.output(self.mode, avg_acc, field)
                if field == 'model':
                    self.metrics[self.mode].append(avg_acc)
                else:
                    self.trace_metrics[self.mode].append(avg_acc)

    # 均方误差（Mean Square Error）
    @staticmethod
    def MSE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'MSE: 预测数据与真实数据大小不一致'
        return np.mean(sum(pow(loc_pred - loc_true, 2)))

    # 平均绝对误差（Mean Absolute Error）
    @staticmethod
    def MAE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'MAE: 预测数据与真实数据大小不一致'
        return np.mean(sum(loc_pred - loc_true))

    # 均方根误差（Root Mean Square Error）
    @staticmethod
    def RMSE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'RMSE: 预测数据与真实数据大小不一致'
        return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))

    # 平均绝对百分比误差（Mean Absolute Percentage Error）
    @staticmethod
    def MAPE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'MAPE: 预测数据与真实数据大小不一致'
        assert 0 not in loc_true, "MAPE: 真实数据有0，该公式不适用"
        return np.mean(abs(loc_pred - loc_true) / loc_true)

    # 平均绝对和相对误差（Mean Absolute Relative Error）
    @staticmethod
    def MARE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), "MARE：预测数据与真实数据大小不一致"
        assert np.sum(loc_true) != 0, "MARE：真实位置全为0，该公式不适用"
        return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)

    # 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
    @staticmethod
    def SMAPE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'SMAPE: 预测数据与真实数据大小不一致'
        assert 0 in (loc_pred + loc_true), "SMAPE: 预测数据与真实数据有0，该公式不适用"
        return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) + np.abs(loc_true)))

    # 对比真实位置与预测位置获得预测准确率
    @staticmethod
    def ACC(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), "accuracy: 预测数据与真实数据大小不一致"
        loc_diff = loc_pred - loc_true
        loc_diff[loc_diff != 0] = 1
        return loc_diff, np.mean(loc_diff == 0)

    def top_k(self, loc_pred, loc_true):
        assert self.topK > 0, "top-k ACC评估方法：k值应不小于1"
        assert len(loc_pred) >= self.topK, "top-k ACC评估方法：没有提供足够的预测数据做评估"
        assert len(loc_pred[0]) == len(loc_true), "top-k ACC评估方法：预测数据与真实数据大小不一致"
        if self.topK == 1:
            t, avg_acc = self.ACC(loc_pred[0], loc_true)
            return avg_acc
        else:
            tot_list = np.zeros(len(loc_true), dtype=int)
            for i in range(self.topK):
                t, avg_acc = self.ACC(loc_pred[i], loc_true)
                tot_list = tot_list + t
            return np.mean(tot_list < self.topK)

    def init_data(self):
        """
        Here we transform specific data types to standard input type
        """
        # 加载json类型的数据为字典类型
        if type(self.data) == str:
            self.data = json.loads(self.data)
        assert type(self.data) == dict, "待评估数据的类型/格式不合法"
        # 对相应评估模型的输入数据做特殊处理
        if self.data_type == 'DeepMove':
            user_idx = self.data.keys()
            for user_id in user_idx:
                trace_idx = self.data[user_id].keys()
                for trace_id in trace_idx:
                    trace = self.data[user_id][trace_id]
                    loc_pred = trace['loc_pred']
                    new_loc_pred = []
                    for t_list in loc_pred:
                        new_loc_pred.append(self.sort_confidence_ids(t_list))
                    self.data[user_id][trace_id]['loc_pred'] = new_loc_pred

    def sort_confidence_ids(self, confidence_list):
        """
        Here we convert the prediction results of the DeepMove model
        DeepMove model output: confidence of all locations
        Evaluate model input: location ids based on confidence
        :param confidence_list:
        :return: ids_list
        """
        assert self.data_type == 'DeepMove', '非法调用sort_confidence_ids函数.'
        sorted_list = sorted(confidence_list, reverse=True)
        mark_list = [0 for i in confidence_list]
        ids_list = []
        for item in sorted_list:
            for i in range(len(confidence_list)):
                if confidence_list[i] == item and mark_list[i] == 0:
                    mark_list[i] = 1
                    ids_list.append(i)
                    break
            if len(ids_list) == self.maxK:
                break
        return ids_list

    def output(self, method, value, field):
        """
        :param method: 评估方法
        :param value: 对应评估方法的评估结果值
        :param field: 评估的范围, 对一条轨迹或是整个模型
        """
        if self.output_switch:
            if method == 'ACC':
                if field == 'model':
                    print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method, value))
                else:
                    print('{} avg_acc={:.3f}'.format(method, value))
            elif method == 'MSE' or method == 'RMSE' or method == 'MAE' \
                    or method == 'MAPE' or method == 'MARE' or method == 'SMAPE':
                if field == 'model':
                    print('---- 该模型在 {} 评估方法下 avg_loss={:.3f} ----'.format(method, value))
                else:
                    print('{} avg_loss={:.3f}'.format(method, value))
            else:
                if field == 'model':
                    print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method, value))
                else:
                    print('{} avg_acc={:.3f}'.format(method, value))

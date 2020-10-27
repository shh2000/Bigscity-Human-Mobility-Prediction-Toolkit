import json
import numpy as np
import os
import shutil
import re
import sys

from evaluate.eval_funcs import ACC, top_k, SMAPE, RMSE, MAPE, MARE, MSE, MAE
from evaluate.utils import output, transfer_data


class EvaluateNextLoc:

    def __init__(self, config):
        """
        Initialize the creation of the Evaluate Class
        :param config: 用于传递 global_config
        """
        self.config_path = './config/evaluate/eval_next_loc.json'
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except Exception:
            raise ValueError('评估类的配置文件路径无效')
        # 从配置文件中读取相关参数
        self.output_switch = self.config['output_switch'].lower() == 'true'
        self.mode_list = self.config['mode_list']
        self.data_path = self.config['data_path']
        self.data_type = self.config['data_type']
        self.mode = self.config['mode']
        self.load_config(config)
        # 初始化类的内部变量
        self.topK = 1
        self.maxK = 1
        self.topK_pattern = re.compile("top-[1-9]\\d*$")
        self.data = None
        self.metrics = {}
        self.trace_metrics = {}
        # 检查是否有不支持的配置
        self.check_config()

    def load_config(self, config):
        """
        from global_config settings
        :param config: 用户配置的个性化参数
        :return:
        """
        if 'output_switch' in config.keys():
            self.output_switch = config['output_switch'].lower() == 'true'
        if 'mode_list' in config.keys():
            self.mode_list = config['mode_list']
        if 'data_path' in config.keys():
            self.data_path = config['data_path']
        if 'data_type' in config.keys():
            self.data_type = config['data_type']
        if 'mode' in config.keys():
            self.mode = config['mode']

    def check_config(self):
        # check mode
        for mode in self.mode:
            if mode in self.mode_list:
                self.metrics[mode] = []
                self.trace_metrics[mode] = []
            elif re.match(self.topK_pattern, mode) is not None:
                k = int(mode.split('-')[1])
                self.metrics[mode] = []
                self.trace_metrics[mode] = []
                self.maxK = k if k > self.maxK else self.maxK
            else:
                raise ValueError("{} 是不支持的评估方法".format(mode))

    def evaluate(self, data=None):
        """
        The entrance of evaluation (user-oriented)
        :param data: 待评估数据, 可以直接是dict类型或者str形式的dict类型，也可以是列表类型(分batch)
        :return: 对应指标的结果
        """
        if data is not None:
            self.data = data
        else:
            try:
                with open(self.data_path) as f:
                    self.data = json.load(f)
            except Exception:
                raise ValueError('待评估数据的路径无效')
        if isinstance(self.data, list):
            data_list = self.data
            for batch_data in data_list:
                self.data = batch_data
                self.evaluate_data()
        else:
            self.evaluate_data()

    def save_result(self, result_path=None):
        """
        :param result_path: 绝对路径，存放结果json
        :return:
        """
        if result_path is None:
            raise ValueError('请正确指定保存评估结果的绝对路径')
        # if os.path.exists(result_path):
        #     shutil.rmtree(result_path)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        with open(result_path + '/res.txt', "w") as f:
            f.write(json.dumps(self.metrics))
            f.write('\n')
            f.write(json.dumps(self.trace_metrics))

    def evaluate_data(self):
        """
        evaluate data batch (internal)
        """
        self.data = transfer_data(self.data, self.data_type, self.maxK)
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
        for mode in self.mode:
            if mode == 'ACC':
                t, avg_acc = ACC(np.array(t_loc_pred[0]), np.array(loc_true))
                self.add_metrics(mode, field, avg_acc)
            elif re.match(self.topK_pattern, mode) is not None:
                avg_acc = top_k(np.array(t_loc_pred, dtype=object), np.array(loc_true, dtype=object),
                                int(mode.split('-')[1]))
                self.add_metrics(mode, field, avg_acc)
            else:
                avg_loss = 0
                if mode == "SMAPE":
                    avg_loss = SMAPE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == 'RMSE':
                    avg_loss = RMSE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == "MAPE":
                    avg_loss = MAPE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == "MARE":
                    avg_loss = MARE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == 'MSE':
                    avg_loss = MSE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == "MAE":
                    avg_loss = MAE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.add_metrics(mode, field, avg_loss)

    def add_metrics(self, method, field, avg):
        """
        save every trace metrics or the whole model metrics
        :param method: evaluate method
        :param field: trace or model
        :param avg: avg_acc or avg_loss
        """
        if self.output_switch:
            output(method, avg, field)
        if field == 'model':
            self.metrics[method].append(avg)
        else:
            self.trace_metrics[method].append(avg)

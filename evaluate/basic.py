import json
import numpy as np
import os
import shutil


class Evaluate(object):

    def __init__(self, dir_path):
        self.dir_path = dir_path
        try:
            f = json.load(open(dir_path + '/evaluate/config.json'))
        except Exception:
            raise ValueError('评估类的配置文件路径无效')
        else:
            self.config = f
        self.all_mode = self.config['all_mode']
        self.data_path = self.config['data_path']
        self.data_type = self.config['data_type']
        self.output_gate = self.config['output_gate'] == 'True'
        self.mode = "ACC"
        self.topK = 1
        self.data = {}
        self.metrics = {}
        self.trace_metrics = {}

    def init_data(self):
        """
        Here we transform specific data types to standard input type
        """
        # 加载json类型的数据为字典类型
        if type(self.data) == str:
            self.data = json.loads(self.data)
        assert type(self.data) == dict, "待评估数据的类型/格式不合法"
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

    def evaluate(self, mode=None, topK=None, data=None):
        """
        :param mode: 指标，如”MSE“，”MAPE“, 默认ACC
        :param topK: 评估指标的另一个参数, top-k
        :param data: 待评估数据，也可以传入待评估数据的路径
        :return: 对应指标的结果
        """
        if mode is not None:
            self.mode = mode
        if topK is not None:
            self.topK = topK
        if data is not None:
            self.data = data
        else:
            try:
                self.data = json.load(open(self.data_path))
            except Exception:
                raise ValueError('待评估数据的路径无效')
        assert self.mode in self.all_mode, "不支持的评估方法"
        self.init_data()
        if self.mode == 'ACC' and self.topK > 1:
            self.trace_metrics['top-' + str(self.topK) + ' ' + self.mode] = []
        else:
            self.trace_metrics[self.mode] = []
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
        assert len(loc_pred) == len(loc_true), "评估的预测数据与真实数据大小不一致"
        t_loc_pred = [[] for i in range(self.topK)]
        for i in range(len(loc_true)):
            for j in range(self.topK):
                t_loc_pred[j].append(loc_pred[i][j])
        if self.mode == 'ACC':
            avg_acc = self.top_k(np.array(t_loc_pred, dtype=object), np.array(loc_true, dtype=object))
            self.output(self.mode, avg_acc, field)
            if field == 'model':
                if self.topK > 1:
                    self.metrics['top-' + str(self.topK) + ' ' + self.mode] = avg_acc
                else:
                    self.metrics[self.mode] = avg_acc
            else:
                if self.topK > 1:
                    self.trace_metrics['top-' + str(self.topK) + ' ' + self.mode] = avg_acc
                else:
                    self.trace_metrics[self.mode].append(avg_acc)
        elif self.mode == 'RMSE':
            avg_loss = self.RMSE(np.array(t_loc_pred[0]), np.array(loc_true))
            self.output(self.mode, avg_loss, field)
            if field == 'model':
                self.metrics[self.mode] = avg_loss
            else:
                self.trace_metrics[self.mode].append(avg_loss)
        elif self.mode == 'MSE':
            avg_loss = self.MSE(np.array(t_loc_pred[0]), np.array(loc_true))
            self.output(self.mode, avg_loss, field)
            if field == 'model':
                self.metrics[self.mode] = avg_loss
            else:
                self.trace_metrics[self.mode].append(avg_loss)
        elif self.mode == "MAE":
            avg_loss = self.MAE(np.array(t_loc_pred[0]), np.array(loc_true))
            self.output(self.mode, avg_loss, field)
            if field == 'model':
                self.metrics[self.mode] = avg_loss
            else:
                self.trace_metrics[self.mode].append(avg_loss)
        elif self.mode == "MAPE":
            avg_loss = self.MAPE(np.array(t_loc_pred[0]), np.array(loc_true))
            self.output(self.mode, avg_loss, field)
            if field == 'model':
                self.metrics[self.mode] = avg_loss
            else:
                self.trace_metrics[self.mode].append(avg_loss)
        elif self.mode == "MARE":
            avg_loss = self.MARE(np.array(t_loc_pred[0]), np.array(loc_true))
            self.output(self.mode, avg_loss, field)
            if field == 'model':
                self.metrics[self.mode] = avg_loss
            else:
                self.trace_metrics[self.mode].append(avg_loss)
        elif self.mode == "SMAPE":
            avg_loss = self.SMAPE(loc_pred, loc_true)
            self.output(self.mode, avg_loss, field)
            if field == "model":
                self.metrics[self.mode] = avg_loss
            else:
                self.trace_metrics[self.mode].append(avg_loss)

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
            if len(ids_list) == self.topK:
                break
        return ids_list

    def output(self, method, value, field):
        """
        :param method: 评估方法
        :param value: 对应评估方法的评估结果值
        :param field: 评估的范围, 对一条轨迹或是整个模型
        """
        if self.output_gate:
            if method == 'ACC':
                if field == 'model':
                    print('---- 该模型在 ', end="")
                    if self.topK > 1:
                        print('top-{} '.format(self.topK), end="")
                    print('ACC 评估方法下 avg_acc={:.3f} ----'.format(value))
                else:
                    print('accuracy={:.3f}'.format(value))
            elif method == 'MSE' or method == 'RMSE' or method == 'MAE' \
                    or method == 'MAPE' or method == 'MARE' or method == 'SMAPE':
                if field == 'model':
                    print('---- 该模型在 {} 评估方法下 avg_loss={:.3f} ----'.format(method, value))
                else:
                    print('{} avg_loss={:.3f}'.format(method, value))
            else:
                raise ValueError('不支持的评估方法')

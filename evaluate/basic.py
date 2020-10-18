import json


class Evaluate(object):

    def __init__(self, dir_path):
        """
        Initialize the creation of the Evaluate Class
        :param dir_path: 项目工程的绝对路径
        """
        self.dir_path = dir_path
        try:
            f = json.load(open(dir_path + '/evaluate/config.json'))
        except Exception:
            raise ValueError('评估类的配置文件路径无效')
        else:
            self.config = f

    def evaluate(self, data=None, config=None, mode=None):
        """
        The entrance of evaluation (user-oriented)
        :param data: 待评估数据, 可以直接是dict类型或者str形式的dict类型，也可以是列表类型(分batch)
        :param config: 用户传进来的个性化参数
        :param mode: 指标，列表形式, 如 [MSE“，”MAPE“], 默认从配置文件中读入
        :return: 对应指标的结果
        """
        raise NotImplementedError

    def save_result(self, result_path=""):
        """
        :param result_path: 相对路径，存放结果json
        :return:
        """
        raise NotImplementedError

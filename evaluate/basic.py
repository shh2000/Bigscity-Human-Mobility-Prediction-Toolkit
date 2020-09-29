import json


class Evaluate(object):
    dirPath = ''
    config = {}

    def __init__(self, dirPath):
        self.dirPath = dirPath
        try:
            f = json.load(open(dirPath + '/datasets/config.json'))
        except Exception:
            raise ValueError('config.dirPath is wrong')
        else:
            self.config = f

    def evaluate(self, index='MSE', resultPath=''):
        """
        :param index: 指标，如”MSE“，”MAPE“
        :param resultPath: 相对路径，存放结果json，格式同老库
        :return: 对应指标的结果
        """
        pass

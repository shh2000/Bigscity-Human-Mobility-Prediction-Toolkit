import json


class DataLoader(object):
    config = {}

    def __init__(self):
        try:
            f = json.load(open('./datasets/config.json'))
        except:
            """
            没有config文件
            """
            raise FileNotFoundError('No Config File')
        self.config = f

    def load(self, dataset_name):
        """
        :param dataset_name: 数据集名
        :return: 正常返回json（dict形式），否则一定会抛某个异常
        """

        try:
            f = json.load(open('./datasets/data/' + dataset_name + '.json'))
        except:
            """
            没有这个json文件
            """
            raise FileNotFoundError('No file')
        return f

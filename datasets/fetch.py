import json
from datasets.translate import Translater


class Fetcher(object):
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

    def fetch(self, dataset_name):
        """
        :param dataset_name: 数据集名
        :return: 能够返回说明正常fetch，否则一定会抛某个异常
        """
        if dataset_name not in self.config.keys():
            """
            没有这个数据集
            """
            raise ValueError('No Such Dataset')
        else:
            load_type = self.config[dataset_name]
            if load_type == 'local':
                try:
                    f = open('./datasets/data/' + dataset_name + '.json')
                except:
                    """
                    没有这个json文件
                    """
                    raise FileNotFoundError('No file')
            else:
                """
                download from server
                """
                ori = open('')
                """
                Translate origin data to formatted data
                """
                translater = Translater(dataset_name)
                f = translater.trans(ori)
                """
                f是格式化文件，保存f到本地
                """
                self.config[dataset_name] = 'local'
                json.dump(self.config, open('./datasets/config.json', 'w'))
            return

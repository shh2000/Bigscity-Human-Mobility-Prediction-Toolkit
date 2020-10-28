from datasets.basic import Dataset
from datasets.Translate import Translater
import json


class Fetcher(Dataset):
    def load(self, dataset_name):
        if dataset_name in self.config.keys() and self.config[dataset_name] == 'online/origin' or \
                self.config[dataset_name] == 'online/json':
            """
            TODO:download origin to cache/
            """
            df = {}  # 应该是下载后的json文件
            if self.config[dataset_name] == 'online/origin':
                translator = Translater()
                df = translator.load(dataset_name=dataset_name)
            json.dump(df, open(self.RELATIVE_PATH + '/datasets/data/' + dataset_name + '.json', 'w'))
            self.config[dataset_name] = 'local'
            json.dump(df, open(self.RELATIVE_PATH + '/datasets/config.json', 'w'))  # 保存更改到config.json
            return
        else:
            raise ValueError('Needn\'t fetch this dataset')

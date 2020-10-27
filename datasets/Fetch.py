from datasets.basic import Dataset, RELATIVE_PATH
from datasets.Translate import Translater
import json


class Fetcher(Dataset):
    def load(self, dataset_name):
        if dataset_name not in self.config.keys():
            raise ValueError('No such dataset')
        if self.config[dataset_name] == 'online/origin' or self.config[dataset_name] == 'online/json':
            """
            download origin to cache/
            """
            df = {}  # 应该是下载后的json文件
            if self.config[dataset_name] == 'online/origin':
                translater = Translater()
                df = translater.load(dataset_name=dataset_name)
            """
            save df to data/
            """
            self.config[dataset_name] = 'local'
            json.dump(df, open(RELATIVE_PATH + '/datasets/config.json', 'w'))  # 保存更改到config.json
            return

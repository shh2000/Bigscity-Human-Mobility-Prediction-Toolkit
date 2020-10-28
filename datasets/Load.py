from datasets.basic import Dataset
import json
from random import random


class Loader(Dataset):
    def load(self, dataset_name):
        if dataset_name in self.config.keys():
            if self.config[dataset_name] == 'local':
                f = json.load(open(self.RELATIVE_PATH + '/datasets/data/' + dataset_name + '.json'))
                return f
            else:
                raise ValueError('Must fetch before load')
        else:
            raise ValueError('No such dataset')

    def load_divide(self, dataset_name, ratio: float = 0.2):
        if dataset_name in self.config.keys():
            if self.config[dataset_name] == 'local':
                f = json.load(open(self.RELATIVE_PATH + '/datasets/data/' + dataset_name + '.json'))
                df1 = {'type': 'FeatureCollection', 'features': []}
                df2 = {'type': 'FeatureCollection', 'features': []}
                for item in f['features']:
                    if random() > ratio:
                        df1['features'].append(item)
                    else:
                        df2['features'].append(item)
                return df1, df2
            else:
                raise ValueError('Must fetch before load')
        else:
            raise ValueError('No such dataset')

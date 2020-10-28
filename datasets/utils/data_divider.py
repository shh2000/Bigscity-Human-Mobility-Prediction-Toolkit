import os
import json
import math
from random import shuffle

class DataDivider():

    def __init__(self, data_path, rate=None):
        '''
        data_path: full path to the undivided json dataset
        rate: the divided rate
        '''
        if rate:
            self.rate = rate
        else:
            self.rate = {
                'train': [0, 0.85],
                'eval': [0.85, 0.9],
                'test': [0.9, 1],
            }
        self.data_path = data_path
        path, ext = os.path.splitext(self.data_path)
        self.dataset_type = ['train', 'eval', 'test']
        self.divided_path = {t: path + '_' + t + ext for t in self.dataset_type}

    def check(self):
        '''
        function:
            check if divided json dataset exist
        output:
            True/False
        '''
        for v in self.divided_path.values():
            if not os.path.isfile(v):
                return False
        return True

    def divide(self):
        '''
        function:
            load data from self.data_path and divide
        output:
            divided data
        '''
        with open(self.data_path, 'r') as f:
            df = json.load(f)
        features = df['features']
        shuffle(features)
        f_len = len(features)
        if f_len < 10:
            raise Exception('dataset length to small')
        idx = {k: list(map(lambda x: math.floor(f_len * x), v)) for k, v in self.rate.items()}
        data = {
            k: {
                'type': 'FeatureCollection',
                'features': features[v[0]:v[1]]
            }
            for k, v in idx.items()
        }
        return data

    def dump(self):
        '''
        function:
            load data from self.data_path, divide and dump to divided json dataset
        output:
            None
        '''
        data = self.divide()
        dataset_type = ['train', 'eval', 'test']
        for t in dataset_type:
            with open(self.divided_path[t], 'w') as f:
                json.dump(data[t], f)

    def load_divided(self):
        '''
        function:
            load data from divided json dataset
        output:
            divided dataset
        '''
        data = {}
        for t in self.dataset_type:
            with open(self.divided_path[t], 'r') as f:
                data[t] = json.load(f)
        return data

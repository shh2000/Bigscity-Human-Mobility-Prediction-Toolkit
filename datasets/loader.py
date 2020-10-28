import os
import json
from random import random
from datasets.basic import Dataset
from datasets.utils.translator import Translator
from datasets.utils.data_divider import DataDivider

class Loader(Dataset):
    '''
    Load data from local, only accept json dataset
    '''

    def __init__(self, dataset_name, data_type, config_dir='config/datasets', data_dir='data'):
        super().__init__(dataset_name, data_type, config_dir=config_dir, data_dir=data_dir)
        if self.data_type != 'json':
            raise ValueError('must translate before load data')
        self.dataset_json = os.path.join(self.data_dir, self.dataset_name + '.json')
        if not os.path.isfile(self.dataset_json):
            raise ValueError(f'file {self.dataset_json} not exist')
        self.data_divider = DataDivider(self.dataset_json)

    def load(self):
        '''
        function:
            get divided dataset. if divided json dataset not exist, divided the origin dataset
        output:
            df: divided dataset
        '''
        if not self.data_divider.check():
            self.data_divider.dump()
        df = self.data_divider.load_divided()
        return df

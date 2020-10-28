import os
import json
import datasets.utils as utils

class Dataset(object):

    def __init__(self, dataset_name, data_type, config_dir='config/datasets', data_dir='data'):
        '''
        dataset_name: the name of the dataset
        data_type: the type of data, in ['origin', 'json']
        config_dir: dataset config directory path
        data_dir: dataset file directory path
        '''
        self.dataset_name = dataset_name
        if data_type in ['origin', 'json']:
            self.data_type = data_type
        else:
            raise ValueError(f'data_type "{data_type}" error')
        if os.path.exists(config_dir):
            self.config_dir = config_dir
        else:
            raise ValueError(f'path {config_dir} not exist')
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise ValueError(f'path {data_dir} not exist')


    def load(self, dataset_name):
        raise NotImplementedError('Basic cannot be called directly')

import os
import time
import json
from datasets.basic import Dataset
from datasets.utils.translator import Translator
from datasets.utils.data_divider import DataDivider


class Fetcher(Dataset):
    '''
    Fetch data from web
    '''

    def __init__(self, dataset_name, data_type, config_dir='config/datasets', data_dir='data'):
        super().__init__(dataset_name, data_type, config_dir=config_dir, data_dir=data_dir)
        link_path = os.path.join(config_dir, 'download_link.json')
        with open(link_path, 'r') as f:
            links = json.load(f)
            self.link = links[self.dataset_name][self.data_type]

        now = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        self.dataset_origin = os.path.join(self.data_dir, self.dataset_name + '_' + now + '_origin' )
        self.dataset_json = os.path.join(self.data_dir, self.dataset_name + '_' + now + '.json')
        self.data_divider = DataDivider(self.dataset_json)
        if self.data_type == 'origin':
            self.translator = Translator(self.dataset_name, self.dataset_origin, self.dataset_json)
            self.download_path = self.dataset_origin
        else:
            self.translator = None
            self.download_path = self.dataset_json

    def load(self):
        '''
        function:
            download, translate and divide the data
        output:
            df: divided data
        '''
        self.download()
        if self.translator:
            self.translator.translate()
        self.data_divider.dump()
        df = self.data_divider.load_divided()
        return df

    def download(self):
        '''
        TODO: download self.link to self.download_path
        '''
        pass

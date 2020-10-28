import os
import json
import datasets.utils.utils as utils


class Translator():

    def __init__(self, dataset_name, dataset_path, dump_path):
        '''
        dataset_name: the name(type) of the dataset
        dataset_path: full path of the origin dataset
        dump_path: full path of the json dataset
        '''
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dump_path = dump_path
        self.trans_method = {
            'foursquare-tky': utils.trans_foursquare_tky,
            'foursquare': utils.trans_foursquare,
            'gowalla': utils.trans_gowalla,
            'sample': utils.trans_sample,
            'format': utils.trans_format
        }

    def translate(self):
        '''
        function:
            load data from dataset_path, translate and dump to dump_path
        output:
            None
        '''
        if self.dataset_name == 'format':
            origin = 'BIGSCITY'
            df = self.trans_method[self.dataset_name](origin)
        else:
            with open(self.dataset_path, 'r') as f:
                df = self.trans_method[self.dataset_name](f)
        with open(self.dump_path, 'w') as f:
            json.dump(df, f)

if __name__ == '__main__':
    translator = Translator('sample_name', 'sample_path', 'sample_dump_path')
    translator.translate()

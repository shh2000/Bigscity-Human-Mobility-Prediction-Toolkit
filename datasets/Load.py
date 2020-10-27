from datasets.basic import Dataset, RELATIVE_PATH
import json


class Loader(Dataset):
    def load(self, dataset_name):

        if dataset_name in self.config.keys():
            if self.config[dataset_name] == 'local':
                f = json.load(open(RELATIVE_PATH + '/datasets/data/' + dataset_name + '.json'))
                return f
            else:
                raise ValueError('Must fetch before load')
        else:
            raise ValueError('No such dataset')

from datasets.basic import Dataset
from datasets.utils import trans_foursquare, trans_gowalla, trans_format, trans_sample, trans_foursquare_tky

name2method = {'foursquare-tky': trans_foursquare_tky, 'foursquare': trans_foursquare,
               'gowalla': trans_gowalla, 'sample': trans_sample, 'format': trans_format}


class Translater(Dataset):
    def load(self, dataset_name):
        if dataset_name == 'format':
            origin = 'BIGSCITY'
        else:
            origin = open(self.RELATIVE_PATH + '/datasets/cache/' + dataset_name + '.csv')
        df = name2method[dataset_name](origin)
        return df

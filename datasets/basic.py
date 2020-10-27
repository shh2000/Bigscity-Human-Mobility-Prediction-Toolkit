import json

RELATIVE_PATH = '.'
#shell（调用者）相对于根目录的路径


class Dataset(object):
    config = {}

    def __init__(self):
        try:
            f = json.load(open(RELATIVE_PATH + '/datasets/config.json'))
        except:
            raise ValueError('config.dirPath is wrong')
        else:
            self.config = f

    def load(self, dataset_name):
        pass

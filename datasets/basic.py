import json


class Dataset(object):
    config = {}
    RELATIVE_PATH = '.'  # shell（调用者）相对于根目录的路径

    def __init__(self, RELATIVE_PATH='.'):
        self.RELATIVE_PATH = RELATIVE_PATH
        try:
            f = json.load(open(RELATIVE_PATH + '/datasets/config.json'))
        except:
            raise ValueError('RELATIVE_PATH is wrong')
        else:
            self.config = f

    def load(self, dataset_name):
        raise NotImplementedError('Basic cannot be called directly')

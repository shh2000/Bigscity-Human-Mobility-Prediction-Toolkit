import json


class Dataset(object):
    datasets = {}
    dirPath = ''

    def __init__(self, dirPath):
        self.dirPath = dirPath
        try:
            f = json.load(open(dirPath + '/datasets/config.json'))
        except:
            raise ValueError('config.dirPath is wrong')
        else:
            self.datasets = f

    def load(self, name='format'):
        if name in self.datasets.keys():
            f = json.load(open(self.dirPath + '/datasets/' + name + '.json'))
            return f
        else:
            raise ValueError('No such dataset')

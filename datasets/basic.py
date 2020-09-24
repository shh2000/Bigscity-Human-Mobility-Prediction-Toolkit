import json


class Dataset(object):
    config = {}
    dirPath = ''

    def __init__(self, dirPath):
        self.dirPath = dirPath
        try:
            f = json.load(open(dirPath + '/datasets/config.json'))
        except:
            raise ValueError('config.dirPath is wrong')
        else:
            self.config = f

    def load(self, name='format'):
        if name in self.config.keys():
            if self.config[name] == 'local':
                f = json.load(open(self.dirPath + '/datasets/' + name + '.json'))
                return f
            else:
                pass
        else:
            raise ValueError('No such dataset')

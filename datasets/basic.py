import json


class Dataset(object):
    config = {}
    dir_path = ''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        try:
            f = json.load(open(dir_path + '/datasets/data/config.json'))
        except:
            raise ValueError('config.dir_path is wrong')
        else:
            self.config = f

    def load(self, name='format'):
        if name in self.config.keys():
            if self.config[name] == 'local':
                f = json.load(open(self.dir_path + '/datasets/data/' + name + '.json'))
                return f
            else:
                pass
        else:
            raise ValueError('No such dataset')

import json


class Dataset(object):
    config = {}

    def __init__(self):
        try:
            f = json.load(open('../datasets/config.json'))
        except:
            raise ValueError('config.dir_path is wrong')
        else:
            self.config = f

    def load(self, name='format'):
        if name in self.config.keys():
            if self.config[name] == 'local':
                f = json.load(open('../datasets/data/' + name + '.json'))
                return f
            else:
                pass
        else:
            raise ValueError('No such dataset')

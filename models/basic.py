import json


class Model(object):
    dirPath = ''

    def __init__(self, dirPath):
        self.dirPath = dirPath
        try:
            f = json.load(open(dirPath + '/models/config.json'))
        except:
            raise ValueError('config.dirPath is wrong')

    def train(self, X):
        return True

    def test(self, X):
        Y = []
        return Y

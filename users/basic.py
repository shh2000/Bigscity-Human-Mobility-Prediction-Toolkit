import json


class Task(object):
    dirPath = ''

    def __init__(self, dirPath):
        self.dirPath = dirPath

    def run(self, modelName, generalPreName, datasetName):
        if True:
            """
            如果在模型对应的地址可以查到训练好的模型文件，则直接调预测，否则先训练再预测
            返回值就是模型本身test的返回值？
            """
            return True
        else:
            return False

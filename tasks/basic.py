import json


class Task(object):

    def __init__(self, dirPath, config):
        '''
        dirPath: 根目录的绝对路径
        config: global_config
        '''
        self.dirPath = dirPath
        self.config = config

    def run(self, modelName, preName, datasetName, train):
        '''
        train: true 必要训练，false 视 cache 的有无决定是否训练
        模型的 cache 命名： modelName_preName_datasetName.m 至少要保证这三个阶段相同的 cache 才能够使用不然可能报错
        '''
import json


class Task(object):

    def __init__(self, dir_path, config):
        '''
        dir_path: 根目录的绝对路径
        config: global_config
        '''
        self.dir_path = dir_path
        self.config = config

    def run(self, model_name, pre_name, dataset_name, train):
        '''
        train: true 必要训练，false 视 cache 的有无决定是否训练
        模型的 cache 命名： modelName_preName_datasetName.m 至少要保证这三个阶段相同的 cache 才能够使用不然可能报错
        '''
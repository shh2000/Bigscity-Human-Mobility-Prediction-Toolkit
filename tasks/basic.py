import json


class Task(object):

    def __init__(self, config, model_name, pre_name, dataset_name):
        '''
        config: global_config
        根据 model_name pre_name dataset_name 去初始化对应的模块
        evaluate 模块由于是一个 task 对应一个评估模块，所以不需要指定 evaluate_name，但也要在 init 中初始化
        '''
        raise NotImplementedError("Task not implemented")

    def run(self, train):
        '''
        train: true 必要训练，false 视 cache 的有无决定是否训练
        模型的 cache 命名： modelName_preName_datasetName.m 至少要保证这三个阶段相同的 cache 才能够使用不然可能报错
        '''
        raise NotImplementedError("Task run not implemented")

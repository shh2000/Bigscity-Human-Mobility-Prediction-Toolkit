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
        因为模型的参数过多所以做 cache 的话只会保留最近一次在此训练集上训练的结果，既无法保证当前 cache 的结果是用当前 config 训练的
        返回一个 file_name 表示模型预测输出的中间文件名，格式为 modelName_preName_datasetName_timestamps.json
        '''
from .basic import Task
from presentation import GenHistoryPre
from models import TrajPreLocalAttnLong
from runner import DeepMoveRunner
import json
import os
import torch

class NextLocPred(Task):

    def __init__(self, dirPath, config):
        super(NextLocPred, self).__init__(dirPath, config)
    
    def run(self, modelName, preName, datasetName, train):
        # 需要检查模型是否已经训练过了
        pre = self.getPre(preName, datasetName)
        # model 有几个参数需要根据 dataset 去设置
        self.config['model']['pre_feature'] = pre.get_data_feature()
        model, runner = self.getModel(modelName)
        if train or not os.path.exists(self.dirPath + 'runtimeFiles/save_model/{}.m'.format(modelName)):
            # 如果 train 设置为 true 或者不存在 cache 则要进行训练
            model = runner.train(model, pre)
            # 缓存 model
            torch.save(model.state_dict(), self.dirPath + 'runtimeFiles/save_model/{}.m'.format(modelName))
            print('finish train {}'.format(modelName))
        else:
            # load model from cache
            model.load_state_dict(torch.load(self.dirPath + 'runtimeFiles/save_model/{}.m'.format(modelName)))
            print('load {} from cache'.format(modelName))
        res = runner.predict(model, pre)
        # 实例化 evaluate 类
        '''
        TODO: 等待具体的 evaluate 类
        evaluate = evaluate(self.dirPath, self.config['evaluate'])
        evaluator.evalute(res)
        '''
    def getPre(self, preName, datasetName):
        if preName == 'GenHistoryPre':
            return GenHistoryPre(self.dirPath, self.config['presentation'], datasetName)
        else:
            raise ValueError('no this presentation!')
    
    def getModel(self, modelName):
        if modelName == 'deepMove':
            return TrajPreLocalAttnLong(self.dirPath, self.config['model']), DeepMoveRunner(self.dirPath, self.config['train'])
        else:
            raise ValueError('no this model!')
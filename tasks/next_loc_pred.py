import json
import os
import torch

from tasks.basic import Task
from presentation.gen_history_pre import GenHistoryPre
from runner.run_deepmove import DeepMoveRunner
from datasets.basic import Dataset

class NextLocPred(Task):

    def __init__(self, dir_path, config):
        super(NextLocPred, self).__init__(dir_path, config)
    
    def run(self, model_name, pre_name, dataset_name, train):
        # 需要检查模型是否已经训练过了
        dataset = Dataset(self.dir_path)
        data = dataset.load(dataset_name)
        pre = self.getPre(pre_name, data, cache_name=dataset_name)
        # model 有几个参数需要根据 dataset 去设置
        self.config['model']['pre_feature'] = pre.get_data_feature()
        runner = self.getRunner(model_name)
        model_cache = self.dir_path + '/cache/model_cache/{}_{}_{}.m'.format(model_name, pre_name, dataset_name)
        if train or not os.path.exists(model_cache):
            # 如果 train 设置为 true 或者不存在 cache 则要进行训练
            runner.train(train_data=pre.get_data('train'), eval_data=pre.get_data('eval'))
            # 缓存 model
            runner.save_cache(model_cache)
            print('finish train {}'.format(model_name))
        else:
            # load model from cache
            runner.load_cache(model_cache)
        res = runner.predict(pre.get_data('test'))
        # 实例化 evaluate 类
        '''
        TODO: 等待具体的 evaluate 类
        evaluate = evaluate(self.dir_path, self.config['evaluate'])
        evaluator.evalute(res)
        '''
    def getPre(self, pre_name, data, cache_name):
        if pre_name == 'GenHistoryPre':
            return GenHistoryPre(self.dir_path, self.config['presentation'], data, cache_name)
        else:
            raise ValueError('no this presentation!')
    
    def getRunner(self, model_name):
        if model_name == 'deepMove':
            return DeepMoveRunner(self.dir_path, self.config['train'], self.config['model'])
        else:
            raise ValueError('no this model!')
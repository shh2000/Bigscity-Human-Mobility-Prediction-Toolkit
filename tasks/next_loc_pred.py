import json
import os
import torch

from tasks.basic import Task
from presentation.gen_history_pre import GenHistoryPre
from presentation.hst_lstm_pre import HSTLSTMPre
from runner.run_deepmove import DeepMoveRunner
from runner.run_hst_lstm import HSTLSTMRunner
from datasets.basic import Dataset
from evaluate.eval_next_loc import EvaluateNextLoc


class NextLocPred(Task):

    def __init__(self, config, model_name, pre_name, dataset_name):
        self.config = config
        self.dataset = Dataset()
        self.dataset_name = dataset_name
        self.pre = self.get_pre(pre_name, cache_name=dataset_name)
        self.runner = self.get_runner(model_name)
        self.model_cache = './cache/model_cache/{}_{}_{}.m'.format(model_name, pre_name, dataset_name)
        if 'model' not in self.config['evaluate']:
            self.config['evaluate']['model'] = self.get_data_type(model_name)
        self.evaluate = EvaluateNextLoc(self.config['evaluate'])
        self.evaluate_res_dir = 'cache/evaluate_cache'

    def run(self, train):
        # 需要检查模型是否已经训练过了
        print('Loading data...')
        data = self.dataset.load(self.dataset_name)
        print("Done")
        print('Transferring data...')
        self.pre.transfer_data(data)
        print("Done")
        # model 有几个参数需要根据 dataset 去设置
        self.config['model']['pre_feature'] = self.pre.get_data_feature()
        self.runner.init_model(self.config['model'])
        if train or not os.path.exists(os.path.join(self.config['dir_path'],self.model_cache)):
            # 如果 train 设置为 true 或者不存在 cache 则要进行训练
            print("Training...")
            self.runner.train(train_data=self.pre.get_data('train'), eval_data=self.pre.get_data('eval'))
            # 缓存 model
            self.runner.save_cache(self.model_cache)
        else:
            # load model from cache
            self.runner.load_cache(self.model_cache)
        res = [self.runner.predict(self.pre.get_data('test'))]
        for evaluate_input in res:
            with open('./test.json', 'w') as f:
                json.dump(evaluate_input, f)
            self.evaluate.evaluate(evaluate_input)
        self.evaluate.save_result(os.path.join(self.config['dir_path'], self.evaluate_res_dir))
        
    def get_pre(self, pre_name, cache_name):
        if pre_name == 'GenHistoryPre':
            return GenHistoryPre(cache_name)
        elif pre_name == 'HSTLSTMPre':
            pre = HSTLSTMPre(self.config, cache_name)
            pre_config = pre.get_config()
            for key in pre_config.keys():
                if key not in self.config.keys():
                    self.config[key] = pre_config[key]
            return pre
        else:
            raise ValueError('no this presentation!')
    
    def get_runner(self, model_name):
        if model_name == 'deepMove':
            return DeepMoveRunner(self.config['train'])
        elif model_name == 'HST-LSTM':
            return HSTLSTMRunner(self.config)
        else:
            raise ValueError('no this model!')

    def get_data_type(self, model_name):
        if model_name == 'deepMove':
            return 'DeepMove'
        else:
            raise ValueError('no this model!')

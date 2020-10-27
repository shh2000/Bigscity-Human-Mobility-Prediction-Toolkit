import os

from tasks.basic import Task
from presentation.strnn_pre import StrnnPre
from runner.run_strnn import StrnnRunner
from datasets.basic import Dataset
from evaluate.eval_next_loc import EvaluateNextLoc


class NextLocPred(Task):

    def __init__(self, config, model_name, pre_name, dataset_name):
        super(NextLocPred, self).__init__(config, model_name, pre_name, dataset_name)
        self.dataset = Dataset()
        self.config = config
        self.dataset_name = dataset_name
        self.pre = self.get_pre(pre_name, cache_name=dataset_name)
        self.runner = self.get_runner(model_name)
        self.model_cache = './cache/model_cache/{}_{}_{}.m'.format(model_name, pre_name, dataset_name)
        self.evaluate = EvaluateNextLoc(self.config['evaluate'])
        self.evaluate_res_dir = './cache/evaluate_cache'

    def run(self, train):
        # 需要检查模型是否已经训练过了
        data = self.dataset.load(self.dataset_name)
        self.pre.transfer_data(data, use_cache=True)
        # model 有几个参数需要根据 dataset 去设置
        self.config['model']['pre_feature'] = self.pre.get_data_feature()
        self.runner.init_model(self.config['model'])
        if train or not os.path.exists(self.model_cache):
            # 如果 train 设置为 true 或者不存在 cache 则要进行训练
            self.runner.train(train_data=self.pre.get_data('train'), eval_data=self.pre.get_data('eval'))
            # 缓存 model
            self.runner.save_cache(self.model_cache)
        else:
            # load model from cache
            self.runner.load_cache(self.model_cache)
        res = self.runner.predict(self.pre.get_data('test'))
        # 实例化 evaluate 类
        # print(res)
        self.evaluate.evaluate(res)
        self.evaluate.save_result(self.evaluate_res_dir)

    def get_pre(self, pre_name, cache_name):
        if pre_name == 'STRNNPre':
            return StrnnPre(self.config['presentation'], cache_name)
        else:
            raise ValueError('no this presentation!')

    def get_runner(self, model_name):
        if model_name == 'strnn':
            return StrnnRunner(self.config['train'])
        else:
            raise ValueError('no this model!')

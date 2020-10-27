import os
import json
from models.fpmc import FPMC
from utils.fpmc_utils import *
from random import shuffle

from runner.basic import Runner

class FPMCRunner(Runner):

    def __init__(self, dir_path, config):
        self.dir_path = dir_path
        with open(os.path.join(dir_path, 'config/run/fpmc.json'), 'r') as config_file:
            self.config = json.load(config_file)
            # 全局 config 可以覆写 loc_config
            if config:
                for key in self.config:
                    if key in config:
                        self.config[key] = config[key]
        self.model = None
    
    def train(self, train_data, eval_data):
        acc, mrr = self.model.learnSBPR_FPMC(train_data, eval_data, n_epoch=self.config['n_epoch'],
                                       neg_batch_size=self.config['n_neg'], eval_per_epoch=False)

    def predict(self, data):
        data_list, user_set, item_set = load_data_from_dir(data)
        acc, mrr = self.model.evaluation(data_list)
        print('In sample:%.4f\t%.4f' % (acc, mrr))


    def init_model(self, model_config):
        f_dir = self.config['input_dir']

        data_list, user_set, item_set = load_data_from_dir(f_dir)
        shuffle(data_list)

        train_ratio = 0.8
        split_idx = int(len(data_list) * train_ratio)
        tr_data = data_list[:split_idx]
        te_data = data_list[split_idx:]

        fpmc = FPMC(n_user=max(user_set) + 1, n_item=max(item_set) + 1,
                    n_factor=self.config['n_factor'], learn_rate=self.config['learn_rate'], regular=self.config['regular'])
        fpmc.user_set = user_set
        fpmc.item_set = item_set
        fpmc.init_model()

        self.model = fpmc

    def load_cache(self, cache_name):
        pass

    def save_cache(self, cache_name):
        pass



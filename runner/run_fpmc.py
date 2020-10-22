import os
import json

from runner.basic import Runner

class FpmcRunner(Runner):

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
        pass

    def predict(self, data):
        pass

    def init_model(self, model_config):
        pass

    def load_cache(self, cache_name):
        pass

    def save_cache(self, cache_name):
        pass

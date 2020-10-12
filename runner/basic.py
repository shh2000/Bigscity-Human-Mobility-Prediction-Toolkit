
class Runner(object):

    def __init__(self, dir_path, config, model_config):
        '''
        model 的实例化放到 Runner 内部
        '''
        self.dir_path = dir_path
        self.config = config

    def train(self, train_data, eval_data):
        '''
        use data to train model with config
        '''

    def predict(self, data):
        '''
        use model to predict data
        使用 yield 机制，返回一个迭代类
        '''
    
    def load_cache(self, cache_name):
        '''
        加载对应模型的 cache 
        '''

    def save_cache(self, cache_name):
        '''
        将当前的模型保存到文件内
        '''
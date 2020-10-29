
class Runner(object):

    def __init__(self, config):
        '''
        model 的实例化放到 Runner 内部，但由于无法做到模型与表示层的完全解耦（即 embedding 还是在模型内部的）
        所以模型的初始化是依赖于表示层处理完的数据特征的，所以额外提供一个 init_model 来在 run 的时候初始化模型
        '''
        raise NotImplementedError("Runner not implemented")

    def train(self, train_data, eval_data):
        '''
        use data to train model with config
        '''
        raise NotImplementedError("Runner train not implemented")

    def predict(self, data):
        '''
        use model to predict data
        使用 yield 机制，返回一个迭代类
        '''
        raise NotImplementedError("Runner predict not implemented")
    
    def init_model(self, model_config):
        '''
        根据 model_config 初始化 runner 对应的模型
        '''
        raise NotImplementedError("Runner init_model not implemented")

    def load_cache(self, cache_name):
        '''
        加载对应模型的 cache 
        '''
        raise NotImplementedError("Runner load cache not implemented")

    def save_cache(self, cache_name):
        '''
        将当前的模型保存到文件内
        '''
        raise NotImplementedError("Runner save cache not implemented")


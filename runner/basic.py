
class Runner(object):

<<<<<<< HEAD
    def __init__(self, dir_path, config):
=======
    def __init__(self, config):
>>>>>>> master
        '''
        model 的实例化放到 Runner 内部，但由于无法做到模型与表示层的完全解耦（即 embedding 还是在模型内部的）
        所以模型的初始化是依赖于表示层处理完的数据特征的，所以额外提供一个 init_model 来在 run 的时候初始化模型
        '''
<<<<<<< HEAD
        self.dir_path = dir_path
        self.config = config
=======
        raise NotImplementedError("Runner not implemented")
>>>>>>> master

    def train(self, train_data, eval_data):
        '''
        use data to train model with config
        '''
<<<<<<< HEAD
=======
        raise NotImplementedError("Runner train not implemented")
>>>>>>> master

    def predict(self, data):
        '''
        use model to predict data
        使用 yield 机制，返回一个迭代类
        '''
<<<<<<< HEAD
=======
        raise NotImplementedError("Runner predict not implemented")
>>>>>>> master
    
    def init_model(self, model_config):
        '''
        根据 model_config 初始化 runner 对应的模型
        '''
<<<<<<< HEAD
=======
        raise NotImplementedError("Runner init_model not implemented")
>>>>>>> master

    def load_cache(self, cache_name):
        '''
        加载对应模型的 cache 
        '''
<<<<<<< HEAD
=======
        raise NotImplementedError("Runner load cache not implemented")
>>>>>>> master

    def save_cache(self, cache_name):
        '''
        将当前的模型保存到文件内
        '''
<<<<<<< HEAD
=======
        raise NotImplementedError("Runner save cache not implemented")
>>>>>>> master

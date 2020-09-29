
class Runner(object):

    def __init__(self, dirPath, config):
        self.dirPath = dirPath
        self.config = config

    def train(self, model, pre):
        '''
        use data to train model with config
        模型训练结果的缓存不在该方法中完成，在 task 中完成
        '''
        return model

    def predict(self, model, pre):
        '''
        use model to predict data
        由外部指定保存预测结果的 fileName
        使用 yield 机制，返回一个迭代类
        '''
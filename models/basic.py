import json
import torch.nn as nn

class Model(nn.Module):
    '''
    对于深度的 Model 就继承 pytorch Module 的子类，按照其要求覆写相应的方法即可
    对于非深度的 Model 自行参考对应的开源代码实现，因为 Model 的调用是在 runner 内的，所以不需要对 Model 做统一的接口要求
    '''
    def __init__(self, dirPath, config):
        '''
        dirPath: 用于加载模型的 config 获取超参
        config: 为外部传入的 global config，global config 将会覆盖 config 中的同名参数
        '''
        super(Model, self).__init__()
import json
import torch.nn as nn

class Model(nn.Module):
    '''
    Model 作为 pytorch Module 的子类，按照其要求覆写相应的方法即可
    '''
    def __init__(self, dirPath, config):
        '''
        dirPath: 用于加载模型的 config 获取超参
        config: 为外部传入的 global config，global config 将会覆盖 config 中的同名参数
        '''
        super(Model, self).__init__()

    def forward(self, batch):
        '''
        batch 为从 pre.get_loader() 获取到的 dataloader 处拿到的 batch，可以保证模型拿到的 batch 是符合模型自己输入格式
        具体格式上由模型实现者自行决定
        '''
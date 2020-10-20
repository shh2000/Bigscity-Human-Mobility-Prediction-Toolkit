import json


class Evaluate(object):

    def __init__(self, dir_path, config):
        '''
        dir_path: 用于寻找 loc_config，如果有必要做 loc_config 的话
        config: 用于传递 global_config
        '''
        raise NotImplementedError("Evaluate init not implemented")

    def evaluate(self, data):
        '''
        data: 是根据 yield 机制生成的一个迭代类可以通过（对于没有做 batch 的模型的话，传一个 list 就行）
        for batch in data:
            获取到每一个 batch 的输出
        '''
        raise NotImplementedError("Evaluate not implemented")

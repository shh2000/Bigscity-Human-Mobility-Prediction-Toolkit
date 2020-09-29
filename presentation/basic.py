
class Presentation(object):

    def __init__(self, dirPath, config, datasetName):
        '''
        dirPath: 用于加载表示层的 config 获取超参
        config: 为外部传入的 global config，global config 将会覆盖 config 中的同名参数
        datasetName: 用于加载原始数据集，数据的初步预处理在 Presentation 类 init 阶段完成，之后 get_data 可以只负责划分 test/train 集，做 batch 这些
        '''
        self.dirPath = dirPath

    def get_data(self, mode):
        '''
        深度模型：建议该方法返回一个 torch.utils.data.Dataloader 对象
        该方法的调用在对应模型的 Runner，故没有过多的约束条件
        mode: train / test
        '''
        return None

    def get_data_feature(self):
        '''
        如果模型使用了 embedding 层，一般是需要根据数据集的 loc_size、tim_size、uid_size 等特征来确定 embedding 层的大小的
        故该方法返回一个 dict，包含表示层能够提供的数据集特征
        该 feature 将会通过 global_config['model']['pre_feature'] 传递给 model 的 init 函数
        '''
        return {}

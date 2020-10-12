
class Presentation(object):

    def __init__(self, dir_path, config, data, cache_name):
        '''
        dir_path: 用于加载表示层的 config 获取超参
        config: 为外部传入的 global config，global config 将会覆盖 config 中的同名参数
        data: 从 dataset load 出来的原始数据集
        cache_name: 可以传递 datasetName，并不一定是最后的 cache 文件名，因为还需要将 pre 的参数也写到 cache 的文件名里面
        '''
        self.dir_path = dir_path

    def get_data(self, mode, use_cache=True):
        '''
        返回值参考(gen_history_pre):
        {
            'loader': 一个 DataLoader 类的实例，用于支持 batch 训练,
            'total_batch': 主要用于做 verbose 输出进度信息
        }
        mode: train / test / eval
        '''
        return None

    def get_data_feature(self):
        '''
        如果模型使用了 embedding 层，一般是需要根据数据集的 loc_size、tim_size、uid_size 等特征来确定 embedding 层的大小的
        故该方法返回一个 dict，包含表示层能够提供的数据集特征
        该 feature 将会通过 global_config['model']['pre_feature'] 传递给 model 的 init 函数
        '''
        return {}

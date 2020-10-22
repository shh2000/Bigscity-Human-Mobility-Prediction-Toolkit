from presentation.basic import Presentation

class FpmcPresentation(Presentation):

    def __init__(self, dir_path, config, cache_name):
        '''
        dir_path: 用于加载表示层的 config 获取超参
        config: 为外部传入的 global config，global config 将会覆盖 config 中的同名参数
        cache_name: 可以传递 datasetName，并不一定是最后的 cache 文件名，因为还需要将 pre 的参数也写到 cache 的文件名里面
        '''
        super(FpmcPresentation, self).__init__(dir_path)
        parameters_str = ''
        self.cache_file_name = 'gen_history_{}{}.json'.format(cache_name, parameters_str)
        self.data = None

    def get_data(self, mode):
        return None

    def transfer_data(self, data, use_cache=True):
        '''
        不再在 init 中传入原始数据集，而是单独做一个接口接受原始数据集。因为数据的加载是在 run 的时候，而模块的初始化是在 init 阶段
        数据加载之后，建议在这一步就做切片、过滤，get_data 只负责划分 eval/train/test 数据集，并返回即可
        '''
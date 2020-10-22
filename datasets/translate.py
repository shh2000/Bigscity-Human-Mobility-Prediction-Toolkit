class Translater(object):
    dataset_name = ''

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def trans(self,origin):
        """
        :param origin: 原始文件句柄
        :return: 格式化json
        """
        #TODO: 在集成的时候把老代码拿过来跑通

from presentation.basic import Presentation
import json
import math

class FpmcPresentation(Presentation):

    def __init__(self, dir_path, config, cache_name):
        '''
        dir_path: 用于加载表示层的 config 获取超参
        config: 为外部传入的 global config，global config 将会覆盖 config 中的同名参数
        cache_name: 可以传递 datasetName，并不一定是最后的 cache 文件名，因为还需要将 pre 的参数也写到 cache 的文件名里面
        '''
        super(FpmcPresentation, self).__init__(dir_path, cache_name)
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
        grid_width = 0.01
        with open(data, 'rb') as file:
            ori_data = json.load(file)
        #print(ori_data)

        cnt = 0
        ori2new = dict()
        new2ori = dict()
        min_latitude = 10000
        max_latitude = -10000
        min_longtitude = 10000
        max_longtitude = -10000
        for user in ori_data['features']:
            ori_uid = user['properties']['uid']
            if ori_uid not in ori2new:
                ori2new[ori_uid] = cnt
                new2ori[cnt] = ori_uid
                cnt += 1
            for loc in user['geometry']['coordinates']:
                latitude = loc['location'][0]
                longtitude = loc['location'][1]
                min_latitude = min(latitude, min_latitude)
                max_latitude = max(latitude, max_latitude)
                min_longtitude = min(longtitude, min_longtitude)
                max_longtitude = max(longtitude, max_longtitude)
        '''print(ori2new)
        print(new2ori)
        print('{} {} {} {}'.format(min_latitude, max_latitude, min_longtitude, max_longtitude))'''

        row_num = math.floor((max_latitude-min_latitude)/grid_width)+1
        column_num = math.floor((max_longtitude-min_longtitude)/grid_width)+1

        with open('./data/user_list.txt', 'w') as f:
            f.write('\"User_Index\"\n')
            for key in new2ori:
                f.write(str(key)+'\n')

        with open('./data/location_list.txt', 'w') as f:
            f.write('\"Location_Index\"\n')
            for i in range(row_num*column_num):
                f.write(str(i) + '\n')

        with open('./data/loc_seq.txt', 'w') as f:
            for user in ori_data['features']:
                uid = user['properties']['uid']
                f.write(str(ori2new[uid])+' ')
                for loc in user['geometry']['coordinates']:
                    i = math.floor((loc['location'][0]-min_latitude)/grid_width)+1
                    j = math.floor((loc['location'][1]-min_longtitude)/grid_width)+1
                    grid = i*j-1
                    f.write(str(grid)+' ')
                f.write('\n')
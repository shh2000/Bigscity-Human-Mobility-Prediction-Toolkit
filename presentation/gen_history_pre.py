import os
import json
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from presentation.basic import Presentation
from presentation.list_dataset import ListDataset
from utils.presentation_helper import encodeLoc, parseTime, calculateBaseTime, calculateTimeOff

class GenHistoryPre(Presentation):

    def __init__(self, dir_path, config, cache_name):
        self.dir_path = dir_path
        with open(os.path.join(dir_path, 'config/presentation/gen_history.json'), 'r') as config_file:
            self.config = json.load(config_file)
        # 全局 config 可以覆写 loc_config
        parameters_str = ''
        for key in self.config:
            if key in config:
                self.config[key] = config[key]
            if key != "batch_size" and key != "num_workers":
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = 'gen_history_{}{}.json'.format(cache_name, parameters_str)
        self.data = None
        self.pad_item = None

    def get_data(self, mode):
        '''
        return dataloader and total_batch
        '''
        if mode == 'eval':
            mode = 'test' # TODO: 暂时用测试集当 eval 吧，虽然不太好的样子
        history_data = self.gen_history(mode)
        dataset = ListDataset(history_data)
        def collactor(batch):
            loc = []
            tim = []
            history_loc = []
            history_tim = []
            history_count = []
            uid = []
            target = []
            session_id = []
            for item in batch:
                loc.append(item['loc'])
                tim.append(item['tim'])
                history_loc.append(item['history_loc'])
                history_tim.append(item['history_tim'])
                history_count.append(item['history_count'])
                uid.append(item['uid'])
                target.append(item['target'])
                session_id.append(item['session_id'])
            return loc, tim, history_loc, history_tim, history_count, uid, target, session_id
        data_loader = DataLoader(dataset=dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], collate_fn=collactor)
        total_batch = dataset.__len__() / self.config['batch_size']
        return {'loader': data_loader, 'total_batch': total_batch}

    def get_data_feature(self):
        res = {
            'loc_size': self.data['loc_size'],
            'tim_size': self.data['tim_size'],
            'uid_size': self.data['uid_size'],
            'target_len': self.config['pad_len'] - self.config['history_len']
        }
        return res

    def transfer_data(self, data, use_cache=True):
        if use_cache and os.path.exists(os.path.join(self.dir_path, 'cache/pre_cache/', self.cache_file_name)):
            # load cache
            f = open(os.path.join(self.dir_path, 'cache/pre_cache/', self.cache_file_name), 'r')
            self.data = json.load(f)
            loc_pad = self.data['loc_size'] - 1
            tim_pad = self.data['tim_size'] - 1
            self.pad_item = (loc_pad, tim_pad)
            f.close()
        else:
            # 因为对 data 的切片过滤只需要进行一次
            # 对 data 进行切片与过滤
            transformed_data = self.cutter_filter(data)
            # pad parameter
            loc_pad = transformed_data['loc_size']
            transformed_data['loc_size'] += 1
            tim_pad = transformed_data['tim_size']
            transformed_data['tim_size'] += 1
            self.pad_item = (loc_pad, tim_pad)
            self.data = transformed_data
            # 做 cache
            if not os.path.exists(os.path.join(self.dir_path, 'cache/pre_cache')):
                os.makedirs(os.path.join(self.dir_path, 'cache/pre_cache'))
            with open(os.path.join(self.dir_path, 'cache/pre_cache/', self.cache_file_name), 'w') as f:
                json.dump(transformed_data, f)

    def cutter_filter(self, data):
            '''
            data: raw data which obey the trajectory data format
            min_session_len: the min number of nodes in a session
            min_sessions: the min number of sessions for a user
            time_length: use for cut raw trajectory into sessions (需为 12 的整数倍)
            output:
            {
                uid: {
                    sessions: {
                        session_id: [
                            [loc, tim],
                            [loc, tim]
                        ],
                        ....
                    }, 按照 time_length 的时间间隔将一个用户的 trace 划分成多段 session
                    train: [0, 1, 2],
                    test: [3, 4] 按照一定比例，划分 train 与 test 合集。目前暂定是后 25% 的 session 作为 test
                }
            }
            '''
            min_session_len = self.config['min_session_len'] 
            min_sessions = self.config['min_sessions']
            time_length = self.config['time_length']
            base_zero = time_length > 12 # 只对以半天为间隔的做特殊处理
            features = data['features']
            # 因为 DeepMove 将对 loc 进行了 labelEncode 所以需要先获得 loc 的全集
            loc_set = []
            data_transformed = {}
            for feature in features:
                uid = feature['properties']['uid']
                sessions = {}
                traj_data = feature['geometry']['coordinates']
                session_id = 1
                session = {
                    'loc': [],
                    'tim': []
                }
                if len(traj_data) == 0:
                    # TODO: shouldn't happen this, throw error ?
                    continue
                start_time = parseTime(traj_data[0]['time'], traj_data[0]['time_format'])
                base_time = calculateBaseTime(start_time, base_zero)
                for index, node in enumerate(traj_data):
                    loc_hash = encodeLoc(node['location'])
                    loc_set.append(loc_hash)
                    if index == 0:
                        session['loc'].append(loc_hash)
                        session['tim'].append(start_time.hour - base_time.hour) # time encode from 0 ~ time_length
                    else:
                        now_time = parseTime(node['time'], node['time_format'])
                        time_off = calculateTimeOff(now_time, base_time)
                        if time_off < time_length and time_off >= 0: # should not be 乱序
                            # stay in the same session
                            session['loc'].append(loc_hash)
                            session['tim'].append(time_off)
                        else:
                            if len(session['loc']) >= min_session_len:
                                # session less than 2 point should be filtered, because this will cause target be empty
                                # new session will be created
                                sessions[str(session_id)] = session
                                # clear session and add session_id
                                session_id += 1
                            session = {
                                'loc': [],
                                'tim': []
                            }
                            start_time = now_time
                            base_time = calculateBaseTime(start_time, base_zero)
                            session['loc'].append(loc_hash)
                            session['tim'].append(start_time.hour - base_time.hour)
                if len(session['loc']) >= min_session_len:
                    sessions[str(session_id)] = session
                else:
                    session_id -= 1
                # TODO: there will be some trouble with only one session user
                if len(sessions) >= min_sessions:
                    data_transformed[str(uid)] = {}
                    data_transformed[str(uid)]['sessions'] = sessions
                    # 25% session will be test session
                    split_num = math.ceil(session_id*0.6) + 1
                    data_transformed[str(uid)]['train'] = [str(i) for i in range(1, split_num)]
                    if split_num < session_id:
                        data_transformed[str(uid)]['test'] = [str(i) for i in range(split_num, session_id + 1)]
                    else:
                        data_transformed[str(uid)]['test'] = []
            # label encode
            print('start encode')
            print('loc size ', len(loc_set))
            encoder = LabelEncoder()
            encoder.fit(loc_set)
            print('finish encode')

            # do loc labelEncoder
            verbose = 100
            cnt = 0
            total_len = len(data_transformed)
            for user in data_transformed.keys():
                for session in data_transformed[user]['sessions'].keys():
                    temp = []
                    # TODO: any more effecient way to do this ?
                    length = len(data_transformed[user]['sessions'][session]['tim'])
                    loc_tmp = encoder.transform(data_transformed[user]['sessions'][session]['loc']).reshape(length, 1).astype(int)
                    tim_tmp = np.array(data_transformed[user]['sessions'][session]['tim']).reshape(length, 1).astype(int)
                    data_transformed[user]['sessions'][session] = np.hstack((loc_tmp, tim_tmp)).tolist()
                cnt += 1
                if cnt % verbose == 0:
                    print('data encode finish: {}/{}'.format(cnt, total_len))
            res = {
                'data_neural': data_transformed,
                'loc_size': encoder.classes_.shape[0],
                'uid_size': len(data_transformed),
                'tim_size': time_length
            }
            return res

    def gen_history(self, mode):
        '''
        pad_item: (loc_pad, tim_pad)
        return list of data
        (loc, tim, history_loc, hisory_tim, history_count, uid, target)
        '''
        data = []
        user_set = self.data['data_neural'].keys()
        pad_len = self.config['pad_len']
        history_len = self.config['history_len']
        for u in user_set:
            if mode == 'test' and len(self.data['data_neural'][u][mode]) == 0:
                # 当一用户 session 过少时会发生这个现象
                continue
            sessions = self.data['data_neural'][u]['sessions']
            if mode == 'all':
                train_id = self.data['data_neural'][u]['train'] + self.data['data_neural'][u]['test']
            else:
                train_id = self.data['data_neural'][u][mode]
            for c, i in enumerate(train_id):
                trace = {}
                if mode == 'train' and c == 0 or mode == 'all' and c == 0:
                    continue
                session = sessions[i]
                if len(session) <= 1:
                    continue
                ## refactor target
                target = [s[0] for s in session[1:]]
                if len(target) < pad_len - history_len:
                    pad_list = [self.pad_item[0] for i in range(pad_len - history_len - len(target))]
                    target = target + pad_list
                else:
                    target = target[-(pad_len - history_len):]
                history = []
                if mode == 'test':
                    test_id = self.data['data_neural'][u]['train']
                    for tt in test_id:
                        history.extend([(s[0], s[1]) for s in sessions[tt]])
                for j in range(c):
                    history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
                # refactor history
                if len(history) >= history_len:
                    # 取后 history_len 个点
                    history = history[-history_len:]
                else:
                    # 将 history 填充足够
                    pad_history = [self.pad_item for i in range(history_len - len(history))]
                    history = pad_history + history
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
                history_loc = [s[0] for s in history]  # 把多个 history 路径合并成一个？
                history_tim = [s[1] for s in history]
                trace['history_loc'] = history_loc
                trace['history_tim'] = history_tim
                trace['history_count'] = history_count
                loc_tim = history
                loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
                # refactor loc tim
                if len(loc_tim) < pad_len:
                    pad_list = [self.pad_item for i in range(pad_len - len(loc_tim))]
                    loc_tim = loc_tim + pad_list
                else:
                    # 截断
                    loc_tim = loc_tim[-pad_len:]
                loc_np = [s[0] for s in loc_tim]
                tim_np = [s[1] for s in loc_tim]
                trace['loc'] = loc_np # loc 会与 history loc 有重合， loc 的前半部分为 history loc
                trace['tim'] = tim_np
                trace['target'] = target  # target 会与 loc 有一段的重合，只有 target 的最后一位 loc 没有
                trace['uid'] = int(u)
                trace['session_id'] = i
                data.append(trace)
        return data

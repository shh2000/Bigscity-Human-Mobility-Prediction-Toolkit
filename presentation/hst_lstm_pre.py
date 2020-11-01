import os
import json
import torch
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import datetime
from presentation.basic import Presentation
from presentation.list_dataset import ListDataset
from utils.presentation_helper import encodeLoc, parseTime, calculateBaseTime, calculateTimeOff
from presentation.gen_history_pre import GenHistoryPre

class HSTLSTMPre(Presentation):

    def __init__(self, config, cache_name):
        with open(os.path.join(config['dir_path'], 'config/presentation/hst_lstm.json'), 'r') as config_file:
            self.config = json.load(config_file)
        # 全局 config 可以覆写 loc_config
        parameters_str = ''
        for key in self.config:
            if key in config:
                self.config[key] = config[key]
            if key != "batch_size" and key != "num_workers":
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = 'cache/pre_cache/' + 'hst_lstm_{}{}.json'.format(cache_name, parameters_str)
        self.cache_file_folder = 'cache/pre_cache/'
        self.data = None

    def get_data(self, mode):
        last_train = int(math.floor(self.config['user_size'] * self.config['train_rate']))
        last_eval = last_train + int(math.floor(self.config['user_size'] * self.config['eval_rate']))
        last_test = last_eval + int(math.floor(self.config['user_size']*self.config['test_rate']))
        if mode == 'train':
            return self.data[:last_train]
        if mode == 'eval':
            return self.data[last_train:last_eval]
        if mode == 'test':
            return self.data[last_eval:last_test]

    def get_data_feature(self):
        return self.data.shape

    def transfer_data(self, raw_data, use_cache=True):
        extracted_data = self.extract_sessions(raw_data)
        self.data = self.transform_data(extracted_data)


    def extract_sessions(self, data):
        '''
        从raw_data里边提取有效数据
        :param data: ['features'][user]{'properties':{'uid':}, 'geometry':{'coordinates':list[]}}
                    list[step]{'location', 'time_format', time'}
        :return: extracted_data : {'uid':{'sessions': [list[step]*session_size], 'session_num'}}
        '''
        session_size = self.config['session_size']
        step_size = self.config['step_size']
        user_size = self.config['user_size']
        time_slot_size = self.config["time_slot_size"]
        aoi_size = self.config["aoi_size"]
        time_span = self.config['time_span']
        features = data['features'][:min(user_size*4, len(data['features']))]
        extracted_data = {}

        for feature in features:  # 这层对user进行遍历
            uid = feature['properties']['uid']
            if str(uid) in extracted_data.keys():
                if extracted_data[str(uid)]['session_num'] == session_size:
                    continue
                session_left = session_size - extracted_data[str(uid)]['session_num']
            else:
                session_left = session_size
                extracted_data[str(uid)] = {'session_num': 0, 'sessions': []}

            sessions = []
            session = []
            traj_data = feature['geometry']['coordinates']
            if len(traj_data) == 0:
                raise IndexError
            prev_time = parseTime(traj_data[-1]['time'], traj_data[0]['time_format'])
            prev_date = prev_time.date()
            for step in range(len(traj_data)-1, -1, -1):  # 这层对每个user的record进行遍历，并且从中提取指定个session
                cur_time = parseTime(traj_data[step]['time'], traj_data[step]['time_format'])
                cur_date = cur_time.date()
                if time_span >= (cur_date-prev_date).days >= 0:
                    session.append(traj_data[step])
                    prev_date = cur_date
                    if len(session) == step_size:
                        prev_date = cur_date + datetime.timedelta(1)  # 当天剩余数据就丢弃了，保证session间的独立性
                        sessions.append(session)
                        session = []
                        if len(sessions) == session_left:
                            extracted_data[str(uid)]['sessions'] = sessions
                            extracted_data[str(uid)]['session_num'] = session_size
                            break
                else:
                    session = []
                    prev_date = cur_date
            extracted_data[str(uid)]['sessions'].append(x for x in sessions)  # 单个user数据处理完毕
            if len(extracted_data) == user_size:
                delete = []
                for user in extracted_data.keys():
                    if extracted_data[user]['session_num']<session_size:
                        delete.append(user)
                for key in delete:
                    extracted_data.pop(key)  # 确保没有session不够的项
            if len(extracted_data) == user_size:
                break
        if len(extracted_data) < user_size:
            raise RuntimeError('data not enough')
        # 更新数据feature
        return extracted_data

    def transform_data(self, extracted_data):
        '''
        将extracted_data转化成为tensor的形式，并且对time和location做预处理
        :param extracted_data
        :return: tensor of shape [user_size, session_size, step_size, 3]
        '''
        transformed_data = np.zeros([self.config['user_size'], self.config['session_size'], self.config['step_size'], 3])
        scaler = MinMaxScaler((0, self.config['space_slot_size']-1))
        for u, user in enumerate(extracted_data.keys()):
            for session in range(self.config['session_size']):
                start_time = parseTime(extracted_data[user]['sessions'][session][0]['time'],
                                       extracted_data[user]['sessions'][session][0]['time_format'])
                for step in range(self.config['step_size']):
                    step_record = extracted_data[user]['sessions'][session][step]
                    cur_time = parseTime(step_record['time'], step_record['time_format'])
                    delta_time = cur_time-start_time
                    ratio = delta_time.total_seconds()/float(self.config['time_span']*86400)
                    transformed_data[u, session, step, 2] = ratio
                    transformed_data[u, session, step, 0] = step_record['location'][0]
                    transformed_data[u, session, step, 1] = step_record['location'][1]

        shape = transformed_data[:, :, :, 0].shape
        transformed_data[:, :, :, 0] = scaler.fit_transform(transformed_data[:, :, :, 0].reshape([-1, 1])).reshape(shape)
        transformed_data[:, :, :, 1] = scaler.fit_transform(transformed_data[:, :, :, 1].reshape([-1, 1])).reshape(shape)
        transformed_data[:, :, :, 1] += transformed_data[:, :, :, 0]*(self.config['space_slot_size']-1)
        transformed_data[:, :, :, 0] = transformed_data[:, :, :, 1]*(self.config['aoi_size']-1)//pow(self.config['space_slot_size']-1, 2)
        data = torch.from_numpy(transformed_data).long()

        return data

    def get_config(self):
        return self.config










import os
import datetime
import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import json

from models.strnn import STRNNModule
from presentation.basic import Presentation


class StrnnPre(Presentation):
    def __init__(self, dir_path, config, cache_name):
        super(StrnnPre, self).__init__(dir_path, config, cache_name)
        with open(os.path.join(dir_path, 'config/presentation/strnn_pre.json'), 'r') as config_file:
            self.config = json.load(config_file)
        # 全局 config 可以覆写 loc_config
        parameters_str = ''
        for key in self.config:
            if key in config:
                self.config[key] = config[key]
            parameters_str += '_' + str(self.config[key])
        self.cache_file_name = 'strnn_pre_{}{}.json'.format(cache_name, parameters_str)
        self.data = dict()
        self.pad_item = None

    def get_data(self, mode):
        '''
        return dataloader and total_batch
        '''
        if mode == 'train':
            return self.data['train']
        elif mode == 'test':
            return self.data['test']
        elif mode == 'eval':
            return self.data['valid']
        else:
            return {}

    def get_data_feature(self):
        res = {
            'user_cnt': self.data['user_cnt'],
            'loc_cnt': self.data['loc_cnt'],
        }
        return res

    def transfer_data(self, data, use_cache=True):
        if use_cache and os.path.exists(os.path.join(self.dir_path, 'cache/pre_cache/', self.cache_file_name)):
            # load cache
            pass
        else:
            self.pre_data()
        train_file = os.path.join(self.dir_path, "cache/strnn/prepro_train_%s.txt" % self.config['lw_time'])
        valid_file = os.path.join(self.dir_path, "cache/strnn/prepro_valid_%s.txt" % self.config['lw_time'])
        test_file = os.path.join(self.dir_path, "cache/strnn/prepro_test_%s.txt" % self.config['lw_time'])
        train_user, train_td, train_ld, train_loc, train_dst = self.treat_prepro(train_file, step=1)
        valid_user, valid_td, valid_ld, valid_loc, valid_dst = self.treat_prepro(valid_file, step=2)
        test_user, test_td, test_ld, test_loc, test_dst = self.treat_prepro(test_file, step=3)
        self.data['train'] = [train_user, train_td, train_ld, train_loc, train_dst]
        self.data['valid'] = [valid_user, valid_td, valid_ld, valid_loc, valid_dst]
        self.data['test'] = [test_user, test_td, test_ld, test_loc, test_dst]

    def load_data(self, dataset_path, read_line=6500000):
        user2id = {}
        poi2id = {}

        train_user = []
        train_time = []
        train_lati = []
        train_longi = []
        train_loc = []
        valid_user = []
        valid_time = []
        valid_lati = []
        valid_longi = []
        valid_loc = []
        test_user = []
        test_time = []
        test_lati = []
        test_longi = []
        test_loc = []

        train_f = open(dataset_path, 'r')
        lines = train_f.readlines()[:read_line]

        user_time = []
        user_lati = []
        user_longi = []
        user_loc = []
        visit_thr = 30

        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= visit_thr:  # 只记录超过30行数据的user
                    # 记录每个user的序号 因为会忽略数据少于30的user 所以序号变了
                    user2id[prev_user] = len(user2id)
                prev_user = user
                visit_cnt = 1

        train_f = open(dataset_path, 'r')
        lines = train_f.readlines()[:read_line]

        prev_user = int(lines[0].split('\t')[0])
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user2id.get(user) is None:
                continue
            user = user2id.get(user)

            time = (datetime.datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
                    - datetime.datetime(2009, 1, 1)).total_seconds() / 60  # minutes
            lati = float(tokens[2])
            longi = float(tokens[3])
            location = int(tokens[4])
            if poi2id.get(location) is None:
                poi2id[location] = len(poi2id)  # 记录每个位置的序号
            location = poi2id.get(location)  # 把数据集中原始的位置修改成自定的编号

            if user == prev_user:
                user_time.insert(0, time)
                user_lati.insert(0, lati)
                user_longi.insert(0, longi)
                user_loc.insert(0, location)
            else:
                train_thr = int(len(user_time) * 0.7)
                valid_thr = int(len(user_time) * 0.8)
                train_user.append(user)  # 源代码是user 我觉得是prev_user
                train_time.append(user_time[:train_thr])
                train_lati.append(user_lati[:train_thr])
                train_longi.append(user_longi[:train_thr])
                train_loc.append(user_loc[:train_thr])
                valid_user.append(user)
                valid_time.append(user_time[train_thr:valid_thr])
                valid_lati.append(user_lati[train_thr:valid_thr])
                valid_longi.append(user_longi[train_thr:valid_thr])
                valid_loc.append(user_loc[train_thr:valid_thr])
                test_user.append(user)
                test_time.append(user_time[valid_thr:])
                test_lati.append(user_lati[valid_thr:])
                test_longi.append(user_longi[valid_thr:])
                test_loc.append(user_loc[valid_thr:])

                prev_user = user
                user_time = [time]
                user_lati = [lati]
                user_longi = [longi]
                user_loc = [location]

        if user2id.get(user) is not None:
            train_thr = int(len(user_time) * 0.7)
            valid_thr = int(len(user_time) * 0.8)
            train_user.append(user)
            train_time.append(user_time[:train_thr])
            train_lati.append(user_lati[:train_thr])
            train_longi.append(user_longi[:train_thr])
            train_loc.append(user_loc[:train_thr])
            valid_user.append(user)
            valid_time.append(user_time[train_thr:valid_thr])
            valid_lati.append(user_lati[train_thr:valid_thr])
            valid_longi.append(user_longi[train_thr:valid_thr])
            valid_loc.append(user_loc[train_thr:valid_thr])
            test_user.append(user)
            test_time.append(user_time[valid_thr:])
            test_lati.append(user_lati[valid_thr:])
            test_loc.append(user_loc[valid_thr:])

        f = open(os.path.join(self.dir_path, 'cache/strnn/train_file.csv'), 'w')
        f.write('useid' + '\t' + 'time' + '\t' + 'lat' + '\t' + 'lon' + '\t' + 'locid' + '\n')
        for i in range(len(train_user)):
            for j in range(len(train_time[i])):
                f.write(str(train_user[i]) + '\t' + str(train_time[i][j]) + '\t'
                        + str(train_lati[i][j]) + '\t' + str(train_longi[i][j]) + '\t' + str(train_loc[i][j]) + '\n')
        f.close()

        f = open(os.path.join(self.dir_path, 'cache/strnn/test_file.csv'), 'w')
        f.write('useid' + '\t' + 'time' + '\t' + 'lat' + '\t' + 'lon' + '\t' + 'locid' + '\n')
        for i in range(len(test_user)):
            for j in range(len(test_time[i])):
                f.write(str(test_user[i]) + '\t' + str(test_time[i][j]) + '\t'
                        + str(test_lati[i][j]) + '\t' + str(test_longi[i][j]) + '\t' + str(test_loc[i][j]) + '\n')
        f.close()

        f = open(os.path.join(self.dir_path, 'cache/strnn/valid_file.csv'), 'w')
        f.write('useid' + '\t' + 'time' + '\t' + 'lat' + '\t' + 'lon' + '\t' + 'locid' + '\n')
        for i in range(len(valid_user)):
            for j in range(len(valid_time[i])):
                f.write(str(valid_user[i]) + '\t' + str(valid_time[i][j]) + '\t'
                        + str(valid_lati[i][j]) + '\t' + str(valid_longi[i][j]) + '\t' + str(valid_loc[i][j]) + '\n')
        f.close()

        return len(user2id), poi2id, train_user, train_time, train_lati, \
               train_longi, train_loc, valid_user, valid_time, valid_lati, \
               valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc

    def pre_data(self):
        # Data loading params
        train_file = os.path.join(self.dir_path, "datasets/data/Gowalla_totalCheckins.txt")

        print("Loading data...")
        user_cnt, poi2id, \
        train_user, train_time, train_lati, train_longi, train_loc, \
        valid_user, valid_time, valid_lati, valid_longi, valid_loc, \
        test_user, test_time, test_lati, test_longi, test_loc = self.load_data(train_file, 5000)

        print("User/Location: {:d}/{:d}".format(user_cnt, len(poi2id)))
        self.data['user_cnt'] = user_cnt
        self.data['loc_cnt'] = len(poi2id)

        data_model = STRNNModule(self.config['dim'], self.data['loc_cnt'],
                                 self.data['user_cnt'], self.config['ww']).cuda()

        print("Making train file...")
        f = open(os.path.join(self.dir_path, "cache/strnn/prepro_train_%s.txt" % self.config['lw_time']), 'w')
        # Training
        # 不同user的time,lat,lon,loc是不一样多的 这里进行了合并
        # 效果是 time[0],lat[0],lon[0],loc[0]合并到一起  time[1],lat[1],lon[1],loc[1]合并到一起...
        # 也就是每一个用户的相关数据合并到一个元组中 len(train_batches)=len(train_time)
        train_batches = list(zip(train_time, train_lati, train_longi, train_loc))
        # 这里进行分割 每个用户的数据分别训练
        for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
            batch_time, batch_lati, batch_longi, batch_loc = train_batch  # 获取每个用户的四维数据
            self.run(f, data_model, train_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=1)  # 处理原始数据
        f.close()

        print("Making valid file...")
        f = open(os.path.join(self.dir_path, "cache/strnn/prepro_valid_%s.txt" % self.config['lw_time']), 'w')
        # Eavludating
        valid_batches = list(zip(valid_time, valid_lati, valid_longi, valid_loc))
        for j, valid_batch in enumerate(tqdm.tqdm(valid_batches, desc="valid")):
            batch_time, batch_lati, batch_longi, batch_loc = valid_batch
            self.run(f, data_model, valid_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=2)
        f.close()

        print("Making test file...")
        f = open(os.path.join(self.dir_path, "cache/strnn/prepro_test_%s.txt" % self.config['lw_time']), 'w')
        # Testing
        test_batches = list(zip(test_time, test_lati, test_longi, test_loc))
        for j, test_batch in enumerate(tqdm.tqdm(test_batches, desc="test")):
            batch_time, batch_lati, batch_longi, batch_loc = test_batch
            self.run(f, data_model, test_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=3)
        f.close()

    def run(self, f, data_model, user, time, lati, longi, loc, step):
        user = Variable(torch.from_numpy(np.asarray([user]))).type(torch.cuda.LongTensor)
        time = Variable(torch.from_numpy(np.asarray(time))).type(torch.cuda.FloatTensor)
        lati = Variable(torch.from_numpy(np.asarray(lati))).type(torch.cuda.FloatTensor)
        longi = Variable(torch.from_numpy(np.asarray(longi))).type(torch.cuda.FloatTensor)
        loc = Variable(torch.from_numpy(np.asarray(loc))).type(torch.cuda.LongTensor)
        rnn_output = data_model(f, user, time, lati, longi, loc, step)

    def treat_prepro(self, train, step):
        train_f = open(train, 'r')
        # Need to change depending on threshold
        if step == 1:
            lines = train_f.readlines()  # [:86445] #659 #[:309931]
        elif step == 2:
            lines = train_f.readlines()  # [:13505]#[:309931]
        elif step == 3:
            lines = train_f.readlines()  # [:30622]#[:309931]

        train_user = []
        train_td = []
        train_ld = []
        train_loc = []
        train_dst = []

        user = 1
        user_td = []
        user_ld = []
        user_loc = []
        user_dst = []

        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            if len(tokens) < 3:
                if user_td:
                    train_user.append(user)
                    train_td.append(user_td)
                    train_ld.append(user_ld)
                    train_loc.append(user_loc)
                    train_dst.append(user_dst)
                user = int(tokens[0])
                user_td = []
                user_ld = []
                user_loc = []
                user_dst = []
                continue
            td = np.array([float(t) for t in tokens[0].split(',')])
            ld = np.array([float(t) for t in tokens[1].split(',')])
            loc = np.array([int(t) for t in tokens[2].split(',')])
            dst = int(tokens[3])
            user_td.append(td)
            user_ld.append(ld)
            user_loc.append(loc)
            user_dst.append(dst)

        if user_td:
            train_user.append(user)
            train_td.append(user_td)
            train_ld.append(user_ld)
            train_loc.append(user_loc)
            train_dst.append(user_dst)
        return train_user, train_td, train_ld, train_loc, train_dst

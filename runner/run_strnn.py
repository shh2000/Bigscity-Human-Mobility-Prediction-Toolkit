import json
import os
import tqdm
import numpy as np
import torch
from torch.autograd import Variable

from runner.basic import Runner
from models.strnn import STRNNCell


class StrnnRunner(Runner):
    def __init__(self, config):
        '''
        model 的实例化放到 Runner 内部，但由于无法做到模型与表示层的完全解耦（即 embedding 还是在模型内部的）
        所以模型的初始化是依赖于表示层处理完的数据特征的，所以额外提供一个 init_model 来在 run 的时候初始化模型
        '''
        with open('./config/run/strnn.json', 'r') as config_file:
            self.config = json.load(config_file)
            # 全局 config 可以覆写 loc_config
            if config:
                for key in self.config:
                    if key in config:
                        self.config[key] = config[key]
        self.model = None

    def train(self, train_data, eval_data):
        '''
        use data to train model with config
        '''
        for i in range(self.config['num_epochs']):
            # Training
            total_loss = 0.
            train_batches = list(zip(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]))
            for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
                # inner_batches = data_loader.inner_iter(train_batch, batch_size)
                # for k, inner_batch in inner_batches:
                batch_user, batch_td, batch_ld, batch_loc, batch_dst = train_batch  # inner_batch)
                if len(batch_loc) < 3:
                    continue
                total_loss += self.run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
                # if (j+1) % 2000 == 0:
                #    print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()
            # Evaluation
            if (i + 1) % self.config['evaluate_every'] == 0:
                print("==================================================================================")
                # print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j,
                valid_batches = list(zip(eval_data[0], eval_data[1], eval_data[2], eval_data[3], eval_data[4]))
                self.print_score(valid_batches, step=2)

    def predict(self, data):
        '''
        use model to predict data
        使用 yield 机制，返回一个迭代类
        '''
        print("Training End..")
        print("==================================================================================")
        print("Test: ")
        test_batches = list(zip(data[0], data[1], data[2], data[3], data[4]))
        self.print_score(test_batches, step=3)
        evaluate_input = {}
        for batch in tqdm.tqdm(test_batches, desc="validation"):
            batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
            if len(batch_loc) < 3:
                continue
            batch_o, target = self.run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=3)
            if batch_user not in evaluate_input:
                evaluate_input[batch_user] = {}
            trace_input = {}
            trace_input['loc_true'] = [target]
            trace_input['loc_pred'] = [list(batch_o)]
            evaluate_input[batch_user]['0'] = trace_input
        return evaluate_input

    def init_model(self, model_config):
        '''
        根据 model_config 初始化 runner 对应的模型
        '''
        self.model = STRNNCell(self.config['dim'], model_config['pre_feature']['loc_cnt'], model_config['pre_feature']['user_cnt']).cuda()
        for key in model_config:
            self.config[key] = model_config[key]
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.config['learning_rate'],
                                         momentum=self.config['momentum'], weight_decay=self.config['reg_lambda'])

    def load_cache(self, cache_name):
        '''
        加载对应模型的 cache
        '''
        self.model.load_state_dict(torch.load(cache_name))

    def save_cache(self, cache_name):
        '''
        将当前的模型保存到文件内
        '''
        torch.save(self.model.state_dict(), cache_name)

    def parameters(self):
        params = []
        for model in [self.model]:
            params += list(model.parameters())
        return params

    # 传入了每一个用户的td loc loc dst数据(都是list)
    def run(self, user, td, ld, loc, dst, step):
        self.optimizer.zero_grad()

        seqlen = len(td)
        user = Variable(torch.from_numpy(np.asarray([user]))).type(torch.cuda.LongTensor)

        # 每一次使用pre文件中的一行数据
        # 也就是一个窗口内的数据进行训练
        # rnn_output是RNN的循环的状态h
        h_0 = Variable(torch.randn(self.config['dim'], 1), requires_grad=False).type(torch.cuda.FloatTensor)
        rnn_output = h_0
        up_time = self.config['up_time']
        lw_time = self.config['lw_time']
        up_dist = self.config['up_dist']
        lw_dist = self.config['lw_dist']
        for idx in range(seqlen - 1):
            td_upper = Variable(torch.from_numpy(np.asarray(up_time - td[idx]))).type(torch.cuda.FloatTensor)
            td_lower = Variable(torch.from_numpy(np.asarray(td[idx] - lw_time))).type(torch.cuda.FloatTensor)
            ld_upper = Variable(torch.from_numpy(np.asarray(up_dist - ld[idx]))).type(torch.cuda.FloatTensor)
            ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx] - lw_dist))).type(torch.cuda.FloatTensor)
            location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(torch.cuda.LongTensor)
            rnn_output = self.model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)

        # 计算S T矩阵的时候用到
        td_upper = Variable(torch.from_numpy(np.asarray(up_time - td[-1]))).type(torch.cuda.FloatTensor)
        td_lower = Variable(torch.from_numpy(np.asarray(td[-1] - lw_time))).type(torch.cuda.FloatTensor)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist - ld[-1]))).type(torch.cuda.FloatTensor)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[-1] - lw_dist))).type(torch.cuda.FloatTensor)
        location = Variable(torch.from_numpy(np.asarray(loc[-1]))).type(torch.cuda.LongTensor)

        # dst[-1]是实际的下一跳编号 validation返回的是预测的可能的下一跳列表
        if step > 1:
            return self.model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, rnn_output), dst[-1]

        destination = Variable(torch.from_numpy(np.asarray([dst[-1]]))).type(torch.cuda.LongTensor)
        J = self.model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location, destination, rnn_output)
        J.backward()
        self.optimizer.step()
        return J.data.cpu().numpy()

    def print_score(self, batches, step):
        recall1 = 0.
        recall5 = 0.
        recall10 = 0.
        recall100 = 0.
        recall1000 = 0.
        recall10000 = 0.
        iter_cnt = 0

        for batch in tqdm.tqdm(batches, desc="validation"):
            batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
            if len(batch_loc) < 3:
                continue
            iter_cnt += 1
            batch_o, target = self.run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)
            print('batch_user: ', end=' ')
            print(batch_user)
            print('len(batch_o)= ', end=' ')
            print(len(batch_o))
            print('batch_o= ', end=' ')
            print(batch_o)
            print('target= ', end=' ')
            print(target)
            recall1 += target in batch_o[:1]
            recall5 += target in batch_o[:5]
            recall10 += target in batch_o[:10]
            recall100 += target in batch_o[:100]
            recall1000 += target in batch_o[:1000]
            recall10000 += target in batch_o[:10000]

        print("recall@1: ", recall1 / iter_cnt)
        print("recall@5: ", recall5 / iter_cnt)
        print("recall@10: ", recall10 / iter_cnt)
        print("recall@100: ", recall100 / iter_cnt)
        print("recall@1000: ", recall1000 / iter_cnt)
        print("recall@10000: ", recall10000 / iter_cnt)

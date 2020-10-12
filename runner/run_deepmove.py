import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from runner.basic import Runner
from models.deepmove import TrajPreLocalAttnLong

class DeepMoveRunner(Runner):

    def __init__(self, dir_path, config, model_name, model_config):
        self.dir_path = dir_path
        with open(os.path.join(dir_path, 'config/run/deepMove.json'), 'r') as config_file:
            self.config = json.load(config_file)
            # 全局 config 可以覆写 loc_config
            if config:
                for key in self.config:
                    if key in config:
                        self.config[key] = config[key]
        self.model = TrajPreLocalAttnLong(self.dir_path, model_config)
    
    def train(self, train_data, eval_data):
        if self.config['use_cuda']:
            criterion = nn.NLLLoss().cuda()
        else:
            criterion = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config['lr'],
                            weight_decay=self.config['L2'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.config['lr_step'],
                                                 factor=self.config['lr_decay'], threshold= self.config['schedule_threshold'])
        tmp_path = '/tmp/checkpoint/'
        if not os.path.exits(self.dir_path + tmp_path):
            os.makedirs(self.dir_path + tmp_path)
        metrics = {}
        metrics['train_loss'] = []
        metrics['accuracy'] = []
        train_data_loader, train_total_batch = train_data['loader'], train_data['total_batch']
        test_data_loader,  test_total_batch = eval_data['loader'], eval_data['total_batch']
        lr = self.config['lr']
        for epoch in range(self.config['max_epoch']):
            self.model, avg_loss = self.run(train_data_loader, self.model, self.config['use_cuda'], optimizer, criterion, 
                                        self.config['lr'], self.config['clip'], train_total_batch, self.config['verbose'])
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            metrics['train_loss'].append(avg_loss)
            # eval stage
            avg_loss, avg_acc = self.evaluate(test_data_loader, self.model, self.config['use_cuda'], test_total_batch, self.config['verbose'], criterion)
            print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))
            metrics['accuracy'].append(avg_acc)
            save_name_tmp = 'ep_' + str(epoch) + '.m'
            torch.save(self.model.state_dict(), self.dir_path + tmp_path + save_name_tmp)
            scheduler.step(avg_acc)
            lr_last = lr
            lr = optimizer.param_groups[0]['lr']
            if lr_last > lr:
                load_epoch = np.argmax(metrics['accuracy'])
                load_name_tmp = 'ep_' + str(load_epoch) + '.m'
                self.model.load_state_dict(torch.load(self.dir_path + tmp_path + load_name_tmp))
                print('load epoch={} model state'.format(load_epoch))
            if lr <= 0.9 * 1e-5:
                break
        best = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
        avg_acc = metrics['accuracy'][best]
        load_name_tmp = 'ep_' + str(best) + '.m'
        self.model.load_state_dict(torch.load(self.dir_path + tmp_path + load_name_tmp))
        # 删除之前创建的临时文件夹
        for rt, dirs, files in os.walk(self.dir_path + tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.dir_path + tmp_path)

    def load_cache(self, cache_name):
        self.model.load_state_dict(torch.load(cache_name))
    
    def save_cache(self, cache_name):
        torch.save(self.model.state_dict(), cache_name)

    def predict(self, data):
        self.model.train(False)
        test_data_loader, test_total_batch = data['loader'], data['total_batch']
        cnt = 0
        for loc, tim, history_loc, history_tim, history_count, uid, target, session_id in test_data_loader:
            if self.config['use_cuda']:
                loc = torch.LongTensor(loc).cuda()
                tim = torch.LongTensor(tim).cuda()
                target = torch.LongTensor(target).cuda()
            else:
                loc = torch.LongTensor(loc)
                tim = torch.LongTensor(tim)
                target = torch.LongTensor(target)
            scores = self.model(loc, tim)
            # elif model_mode == 'simple':
            #     scores = model(loc, tim)
            #     scores = scores[:, -target_len:, :]
            evaluate_input = {}
            for i in range(len(uid)):
                u = uid[i]
                s = session_id[i]
                trace_input = {}
                trace_input['loc_true'] = target[i].tolist()
                trace_input['loc_pred'] = scores[i].tolist()
                if u not in evaluate_input:
                    evaluate_input[u] = {}
                evaluate_input[u][s] = trace_input
            yield evaluate_input
            cnt += 1
            if cnt % self.config['verbose'] == 0:
                print('finish batch {}/{}'.format(cnt, test_total_batch))

    def run(self, data_loader, model, use_cuda, optimizer, criterion, lr, clip, total_batch, verbose):
        model.train(True)
        total_loss = []
        cnt = 0
        loc_size = model.loc_size
        for loc, tim, history_loc, history_tim, history_count, uid, target in data_loader:
            # use accumulating gradients
            # one batch, one step
            optimizer.zero_grad()
            if use_cuda:
                loc = torch.LongTensor(loc).cuda()
                tim = torch.LongTensor(tim).cuda()
                target = torch.LongTensor(target).cuda()
            else:
                loc = torch.LongTensor(loc)
                tim = torch.LongTensor(tim)
                target = torch.LongTensor(target)
            scores = model(loc, tim) # batch_size * target_len * loc_size
            # elif model_mode == 'simple':
            #     scores = model(loc, tim)
            #     scores = scores[:, -target_len:, :] 这个可以想办法在 model 里做了
            # change to batch_size x target_len * loc_size
            scores = scores.reshape(-1, loc_size)
            target = target.reshape(-1)
            loss = criterion(scores, target)
            loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def evaluate(self, data_loader, model, use_cuda, total_batch, verbose, criterion):
        model.train(False)
        total_loss = []
        total_acc = []
        cnt = 0
        loc_size = model.loc_size
        for loc, tim, history_loc, history_tim, history_count, uid, target in data_loader:
            if use_cuda:
                loc = torch.LongTensor(loc).cuda()
                tim = torch.LongTensor(tim).cuda()
                target = torch.LongTensor(target).cuda()
            else:
                loc = torch.LongTensor(loc)
                tim = torch.LongTensor(tim)
                target = torch.LongTensor(target)
            scores = model(loc, tim) # batch_size * target_len * loc_size
            # elif model_mode == 'simple':
            #     scores = model(loc, tim)
            #     scores = scores[:, -target_len:, :]
            scores = scores.reshape(-1, loc_size)
            target = target.reshape(-1)
            loss = criterion(scores, target)
            total_loss.append(loss.data.cpu().numpy().tolist())
            acc = self.get_acc(target, scores)
            total_acc.append(acc)
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
        avg_loss = np.mean(total_loss, dtype=np.float64)
        avg_acc = np.mean(total_acc, dtype=np.float64)
        return avg_loss, avg_acc
    
    def get_acc(self, target, scores, topk = 1):
        """target and scores are torch cuda Variable"""
        target = target.data.cpu().numpy()
        val, idxx = scores.data.topk(topk, 1)
        predx = idxx.cpu().numpy()
        correct_cnt = 0
        for i, p in enumerate(predx):
            t = target[i]
            if t in p:
                correct_cnt += 1
        acc = correct_cnt / target.shape[0]
        return acc
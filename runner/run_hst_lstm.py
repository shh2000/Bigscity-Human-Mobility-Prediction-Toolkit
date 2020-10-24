import json
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from runner.basic import Runner
from models import HST_LSTM


class HSTLSTMRunner(Runner):
    def __init__(self, config, dir_path):
        self.dir_path = dir_path
        self.model = None
        with open(os.path.join(dir_path, "config/run/hst-lstm.json")) as f:
            self.config = json.load(f)
            if config:
                for key in self.config:
                    if key in config:
                        self.config[key] = config[key]

    def init_model(self, model_config):
        if self.config['use_gpu']:
            self.model = HST_LSTM.HSTLSTM(model_config).to(self.config["device"])
        else:
            self.model = HST_LSTM.HSTLSTM(model_config)

    def train(self, train_data, eval_data=None):
        """
        train function
        :param train_data: 已经预处理好的数据 shape = (batch, session, step, 3)
        :param eval_data: 暂时不用
        """
        if self.config["use_gpu"]:
            train_data = train_data.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), self.config['lr'])
        loss_func = torch.nn.CrossEntropyLoss()
        data_set = TensorDataset(train_data, train_data[:, :, :, 0])  # y就是aoi数据
        data_loader = DataLoader(data_set, batch_size=self.config["batch_size"], shuffle=True)

        for epoch in range(self.config['epochs']):
            current_loss = 0.
            current_acc = 0.
            i = 0
            for _, (batch_x, batch_y) in enumerate(data_loader):
                for session in range(1, batch_x.size(1)):
                    x = batch_x[:, :session, :, :]
                    distribution, prediction = self.model(x, batch_x[:, session, :, :])
                    optimizer.zero_grad()
                    loss = loss_func(distribution[:, :-1, :].flatten(0, 1), batch_y[:, session, 1:].flatten())
                    loss.backward()
                    current_loss += loss.sum().detach().to("cpu").item()
                    current_acc += prediction[:, :-1].eq(batch_y[:, session, 1:]).detach().to(
                        "cpu").sum().item() / prediction.numel()
                    i = i + 1
                    optimizer.step()
            print("epoch{} : loss: {}       acc: {}".format(epoch, current_loss, current_acc / i))

    def predict(self, pre):  # 默认最后一个session是要预测的
        if self.config["use_gpu"]:
            pre = pre.cuda()
        return self.model(pre[:, :-1, :, :], pre[:, -1, :, :])

    def load_cache(self, cache_name):
        if os.path.exists(os.path.join(self.dir_path + "test/", cache_name)):
            raise FileNotFoundError
        else:
            self.model = torch.load(os.path.join(self.dir_path + "test/", cache_name))

    def save_cache(self, cache_name):
        torch.save(self.model, os.path.join(self.dir_path + "test/", cache_name))

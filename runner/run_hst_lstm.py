import json
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from runner.basic import Runner


class HSTLSTMRunner(Runner):
    def __init__(self, dir_path, config):
        super().__init__(dir_path, config)
        self.dir_path = dir_path
        with open(os.path.join(dir_path, "config/run/hst-lstm.json")) as f:
            self.config = json.load(f)
            if config:
                for key in self.config:
                    if key in config:
                        self.config[key] = config[key]

    def train(self, model, pre):
        """
        train function
        :param model: HST-LSTM model
        :param pre: 已经预处理好的数据 shape = (batch, session, step, 3)
        :return:
        """
        if self.config["use_gpu"]:
            model.to(torch.device(self.config["device"]))
            pre.cuda()
        optimizer = torch.optim.Adam(model.parameters(), self.config['lr'])
        loss_func = torch.nn.CrossEntropyLoss()
        data_set = TensorDataset(pre, pre[:, :, :, 0])  # y就是aoi数据
        data_loader = DataLoader(data_set, batch_size=self.config["batch_size"], shuffle=True)

        for epoch in range(self.config['epochs']):
            current_loss = 0.
            for _, (batch_x, batch_y) in enumerate(data_loader):
                for session in range(batch_x.size(1)):
                    for step in range(1, batch_x.size(2)):
                        x = torch.cat((batch_x[:, :session, :, :], batch_x[:, session + 1:, :, :]),
                                      dim=1)  # 把第session个记录剔掉
                        distribution, prediction = model(x, batch_x[:, session, :step, :])
                        optimizer.zero_grad()
                        loss = loss_func(distribution, batch_y[:, session, step])
                        loss.backward()
                        current_loss += loss.sum().detach().to("cpu").item()
                        optimizer.step()
            print("epoch{} : {}".format(epoch, current_loss))

    def predict(self, model, pre):  # 默认最后一个session是要预测的
        if self.config["use_gpu"]:
            model.to(torch.device(self.config["device"]))
            pre.cuda()
        return model(pre[:, :-1, :, :], pre[:, -1, :, :])

import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from runner.basic import Runner

dirPath = "/Users/marc-antoine/Documents/Github/Bigscity-Human-Mobility-Prediction-Toolkit/"

class HSTLSTMRunner(Runner):

    def __init__(self, dirPath, config):
        super().__init__(dirPath, config)
        self.dirPath = dirPath
        with open(os.path.join(dirPath, "config/run/hst-lstm.json")) as f:
            self.config = json.load(f)
            if config:
                for key in self.config:
                    if key in config:
                        self.config[key] = config[key]

    def train(self, model, pre):
        optimizer = torch.optim.Adam(model.parameters(), self.config['lr'])
        loss_func = torch.nn.CrossEntropyLoss()
        dataset = TensorDataset(pre[0], pre[1])
        data_loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        for epoch in range(self.config['epochs']):
            current_loss = 0.
            for _, (batch_x, batch_y) in enumerate(data_loader):
                for session in range(batch_x.size(1)):
                    for step in range(1, batch_x.size(2)):
                        x = torch.cat((batch_x[:,:session,:,:,:], batch_x[:,session+1:,:,:,:]), dim=1)  # 把第session个记录剔掉
                        distribution, predict = model(x[:,:,:,0,:], x[:,:,:,1,:], x[:,:,:,2,:],
                                                      batch_x[:,session,:step,0,:],
                                                      batch_x[:,session,:step,1,:],
                                                      batch_x[:,session,:step,2,:])
                        optimizer.zero_grad()
                        loss = loss_func(distribution, batch_y[:, session, step])
                        loss.backward()
                        current_loss += loss.sum().item()
                        optimizer.step()
            print("epoch{} : {}".format(epoch, current_loss))



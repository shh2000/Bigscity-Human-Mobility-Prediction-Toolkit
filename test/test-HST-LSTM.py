import torch
import json
import os
from models.HST_LSTM.HST_LSTM import HSTLSTM
from models.HST_LSTM import HST_LSTM
from runner.run_hst_lstm import HSTLSTMRunner


dirPath = "/Users/marc-antoine/Documents/Github/Bigscity-Human-Mobility-Prediction-Toolkit/"

with open(os.path.join(dirPath, "config/run/hst-lstm.json")) as f:
    config = json.load(f)

pre = torch.randint(0, config["aoi_size"], [config['batch_size']*10, config['session_size'], config['step_size'], 3]).long()
pre[:, :, :, 1] = torch.rand(config['batch_size']*10, config['session_size'], config['step_size']) * config["temporal_slot_size"]
pre[:, :, :, 2] = torch.rand(config['batch_size']*10, config['session_size'], config['step_size']) * config["spacial_slot_size"]

model = HST_LSTM.HSTLSTM(dirPath, config)
runner = HSTLSTMRunner(dirPath, config)
runner.train(model, pre)







import torch
import json
import os
import numpy as np
from models import HST_LSTM
from runner.run_hst_lstm import HSTLSTMRunner


with open("../global_config_sample.json") as f:
    path = json.load(f)
    dir_path = path['server_path']

with open(os.path.join(dir_path, "config/run/hst-lstm.json")) as f:
    config = json.load(f)
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pre = torch.randint(0, config["aoi_size"], [config['batch_size']*10, config['session_size'], config['step_size'], 3]).long()
pre[:, :, :, 1] = torch.rand(config['batch_size']*10, config['session_size'], config['step_size']) * config["temporal_slot_size"]
pre[:, :, :, 2] = torch.rand(config['batch_size']*10, config['session_size'], config['step_size']) * config["spacial_slot_size"]

model = HST_LSTM.HSTLSTM(dir_path, config)
runner = HSTLSTMRunner(dir_path, config)
runner.train(model, pre)

torch.save(model.cpu(), './trained_model.pkl')
np.save("pre.npy", pre.numpy())

print(runner.predict(model, pre))

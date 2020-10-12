import torch
import json
import os
from models.HST_LSTM.HST_LSTM import HSTLSTM
from models.HST_LSTM import HST_LSTM
from runner.run_hst_lstm import HSTLSTMRunner


dirPath = "/Users/marc-antoine/Documents/Github/Bigscity-Human-Mobility-Prediction-Toolkit/"

with open(os.path.join(dirPath, "config/run/hst-lstm.json")) as f:
    config = json.load(f)

test = torch.randn(3,3, 3, 3)


pre = [torch.randn(config['batch_size']*10, config['session_size'], config['step_size'], 3, config['input_size'])]
aoi = torch.randint(config['aoi_size'], [config['batch_size']*10, config['session_size'], config['step_size']])
# pre.append(torch.nn.functional.one_hot(aoi, config['aoi_size']))
pre.append(aoi)
model = HST_LSTM.HSTLSTM(dirPath, config)

runner = HSTLSTMRunner(dirPath, config)
runner.train(model, pre)







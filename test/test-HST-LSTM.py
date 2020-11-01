import torch
import json
import os
import numpy as np
from models import HST_LSTM
from runner.run_hst_lstm import HSTLSTMRunner
from sklearn.preprocessing import MinMaxScaler
from presentation.hst_lstm_pre import HSTLSTMPre
from tasks.next_loc_pred import NextLocPred



with open("../global_config_sample.json") as f:
    global_config = json.load(f)
    dir_path = global_config['dir_path']
'''''
with open(os.path.join(dir_path, "config/run/hst-lstm.json")) as f:
    config = json.load(f)
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["dir_path"] = dir_path

pre = torch.randint(0, config["aoi_size"], [config['batch_size']*4, config['session_size'], config['step_size'], 3]).long()
pre[:, :, :, 1] = torch.rand(config['batch_size']*4, config['session_size'], config['step_size']) * config["time_slot_size"]
pre[:, :, :, 2] = torch.rand(config['batch_size']*4, config['session_size'], config['step_size']) * config["space_slot_size"]

runner = HSTLSTMRunner(config, dir_path)
runner.init_model(config)
runner.train(pre)

# torch.save(model.cpu(), './trained_model.pkl')
runner.save_cache("trained_model.pkl")
np.save("pre.npy", pre.numpy())

print(runner.predict(pre))
data['features'][user]{'properties':{'uid':}, 'geometry':{'coordinates':list[]}}
list[step]{'location', 'time_format', time'}


with open(dir_path+"datasets/data/gowalla.json") as f:
    data = json.load(f)
print(data["features"])
'''

task = NextLocPred(global_config, 'HST-LSTM', 'HSTLSTMPre', 'gowalla')
task.run(train=True)



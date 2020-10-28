import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from runner.run_fpmc import *

if __name__ == '__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    model = FPMCRunner(os.getcwd())
    train_data, eval_data = model.init_model()
    model.train(train_data, eval_data)
    eval_dict = model.predict(model.config['input_dir'])
    print(eval_dict)

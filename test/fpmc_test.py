import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from runner.run_fpmc import *
from evaluate.eval_next_loc import EvaluateNextLoc
from presentation.fpmc_presentation import *

if __name__ == '__main__':
    # 测试模型
    model = FPMCRunner(os.getcwd())
    fpmc_pre = FpmcPresentation(None, None, None)
    fpmc_pre.transfer_data('./datasets/foursquare-tky.json')
    train_data, eval_data = model.init_model()
    model.train(train_data, eval_data)
    eval_dict = model.predict(model.config['input_dir'])
    #print(eval_dict)
    #print('{} {} {}'.format(type(eval_dict), type(eval_dict[0]), type[eval_dict[0]['loc_pred']]))

    # 测试evaluation
    config = {
        "model": {},
        "presentation": {},
        "train": {},
        "evaluate": {}
    }
    evaluate = EvaluateNextLoc(config['evaluate'])
    evaluate_res_dir = './cache/evaluate_cache'
    evaluate.evaluate(eval_dict)
    evaluate.save_result(evaluate_res_dir)




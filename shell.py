"""
这个文件不需要写类
命令行接口
创建Task
"""
from tasks import NextLocPred
import sys
import json

if __name__ == "__main__":
    # shell 搬运的工作以及设计工作就交给其他人来做，这里只给出一个调用 task 的样例
    if len(sys.argv) != 4:
        print('wrong format parameters!', file=sys.stderr)
        exit(1)
    model_name = sys.argv[1] # deepMove / SimpleRNN / FPMC
    dataset_name = sys.argv[2]
    pre_name = sys.argv[3]
    config_file = open('./global_config.json', 'r')
    global_config = json.load(config_file)
    config_file.close()
    task = NextLocPred(config=global_config['task'])
    task.run(model_name=model_name, pre_name=pre_name, dataset_name=dataset_name, train=True)

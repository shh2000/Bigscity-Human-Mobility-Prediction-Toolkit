import sys
sys.path.append('..')

from tasks.next_loc_pred import NextLocPred
# 读取 config 文件
dir_path = "D:/ubantu_shared/human mobility toolkit/Bigscity-Human-Mobility-Prediction-Toolkit/"
config = {
    "model" : {},
    "presentation": {},
    "train": {},
    "evaluate": {}
}
task = NextLocPred(dir_path=dir_path, config=config, model_name="deepMove", pre_name="GenHistoryPre", dataset_name="foursquare-tky")
# 测试 task 的 run
self = task
data = self.dataset.load(self.dataset_name)
self.pre.transfer_data(data)
self.config['model']['pre_feature'] = self.pre.get_data_feature()
self.runner.init_model(self.config['model'])
self.runner.train(train_data=self.pre.get_data('train'), eval_data=self.pre.get_data('eval'))
self.runner.save_cache(self.model_cache)
# res = self.runner.predict(self.pre.get_data('test'))

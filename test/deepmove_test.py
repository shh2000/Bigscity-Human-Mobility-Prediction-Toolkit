from tasks.next_loc_pred import NextLocPred
import pickle
# 读取 config 文件
config = {
    "model" : {},
    "presentation": {},
    "train": {},
    "evaluate": {}
}
task = NextLocPred(config=config, model_name="deepMove", pre_name="GenHistoryPre", dataset_name="foursquare-tky")
task.run(True)
# 测试 task 的 run
# self = task
# data = self.dataset.load(self.dataset_name)
# self.pre.transfer_data(data)
# self.config['model']['pre_feature'] = self.pre.get_data_feature()
# self.runner.init_model(self.config['model'])
# self.runner.train(train_data=self.pre.get_data('train'), eval_data=self.pre.get_data('eval'))
# self.runner.save_cache(self.model_cache)
# res = self.runner.predict(self.pre.get_data('test'))

# 调试 deepmove 的性能，使用源码的数据集试试
# picklefile=open('./datasets/data/foursquare.pk','rb')
# data=pickle.load(picklefile,encoding='iso-8859-1')
# self = task
# pre = self.pre
# x = {}
# x['data_neural'] = data['data_neural']
# x['loc_size'] = len(data['vid_list'])
# x['uid_size'] = len(data['uid_list'])
# x['tim_size'] = 48
# pre.data = x
# # pre.pad_item = (x['loc_size'] - 1, x['tim_size'] - 1)
# # 数据准备好了
# self.config['model']['pre_feature'] = self.pre.get_data_feature()
# self.runner.init_model(self.config['model'])
# # 如果 train 设置为 true 或者不存在 cache 则要进行训练
# self.runner.train(train_data=self.pre.get_data('train'), eval_data=self.pre.get_data('test'))
# # # 缓存 model
# self.runner.save_cache(self.model_cache)
# # self.runner.load_cache(self.model_cache)
# res = self.runner.predict(self.pre.get_data('test'))
# for evaluate_input in res:
#     self.evaluate.evaluate(evaluate_input)
# self.evaluate.save_result(self.evaluate_res_dir)
#

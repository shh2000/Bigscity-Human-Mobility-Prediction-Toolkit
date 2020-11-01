from tasks.next_loc_pred import NextLocPred
# import torch
# import pickle
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence
# import torch.nn.functional as F
# 读取 config 文件
config = {
    "model" : {},
    "presentation": {},
    "train": {},
    "evaluate": {}
}

task = NextLocPred(config=config, model_name="deepMove", pre_name="GenHistoryPre", dataset_name="foursquare-tky")
task.run(False)
# # 测试 task 的 run
# self = task
# data = self.dataset.load(self.dataset_name)
# self.pre.transfer_data(data)
# self.config['model']['pre_feature'] = self.pre.get_data_feature()
# self.runner.init_model(self.config['model'])
# self.runner.load_cache(self.model_cache)
# self.runner.train(train_data=self.pre.get_data('train'), eval_data=self.pre.get_data('eval'))
# # self.runner.save_cache(self.model_cache)
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
# max_sessions = 0
# a = []
# data_neural = data['data_neural']
# for u in data_neural:
#     for s in data_neural[u]['sessions']:
#         for loc, tim in data_neural[u]['sessions'][s]:
#             a.append(loc)

# 测试补齐
# train_data=self.pre.get_data('train')
# data_loader = train_data['loader']
# loc, tim, loc_len, history_loc, history_tim, history_len, uid, target, session_id = data_loader.__iter__().__next__()
# # self = self.runner.model
# loc = torch.LongTensor(loc).cuda()
# tim = torch.LongTensor(tim).cuda()
# history_loc = torch.LongTensor(history_loc).cuda()
# history_tim = torch.LongTensor(history_tim).cuda()
# model

# batch_size = loc.shape[0]
# h1 = torch.zeros(1, batch_size, self.hidden_size)
# h2 = torch.zeros(1, batch_size, self.hidden_size)
# c1 = torch.zeros(1, batch_size, self.hidden_size)
# c2 = torch.zeros(1, batch_size, self.hidden_size)
# if self.use_cuda:
#     h1 = h1.cuda()
#     h2 = h2.cuda()
#     c1 = c1.cuda()
#     c2 = c2.cuda()

# loc_emb = self.emb_loc(loc)
# tim_emb = self.emb_tim(tim)
# x = torch.cat((loc_emb, tim_emb), 2).permute(1, 0, 2) # change batch * seq * input_size to seq * batch * input_size
# x = self.dropout(x)

# history_loc_emb = self.emb_loc(history_loc)
# history_tim_emb = self.emb_tim(history_tim)
# history_x = torch.cat((history_loc_emb, history_tim_emb), 2).permute(1, 0, 2)
# history_x = self.dropout(history_x)

# # pack x and history_x
# pack_x = pack_padded_sequence(x, lengths=loc_len, enforce_sorted=False)
# pack_history_x = pack_padded_sequence(history_x, lengths=history_len, enforce_sorted=False)
# if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
#     hidden_history, h1 = self.rnn_encoder(pack_history_x, h1)
#     hidden_state, h2 = self.rnn_decoder(pack_x, h2)
# elif self.rnn_type == 'LSTM':
#     hidden_history, (h1, c1) = self.rnn_encoder(pack_history_x, (h1, c1))
#     hidden_state, (h2, c2) = self.rnn_decoder(pack_x, (h2, c2))
# #unpack
# hidden_history, hidden_history_len = pad_packed_sequence(hidden_history, batch_first=True)
# hidden_state, hidden_state_len = pad_packed_sequence(hidden_state, batch_first=True)
# # hidden_history = hidden_history.permute(1, 0, 2) # change history_len * batch_size * input_size to batch_size * history_len * input_size
# # hidden_state = hidden_state.permute(1, 0, 2)
# attn_weights = self.attn(hidden_state, hidden_history) # batch_size * state_len * history_len
# context = attn_weights.bmm(hidden_history) # batch_size * state_len * input_size
# out = torch.cat((hidden_state, context), 2)  # batch_size * state_len * 2 x input_size
# out = self.dropout(out)

# y = self.fc_final(out) # batch_size * state_len * loc_size
# score = F.log_softmax(y, dim=2)

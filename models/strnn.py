import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


class STRNNCell(nn.Module):
    def __init__(self, hidden_size, loc_cnt, user_cnt):
        super(STRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # C
        self.weight_th_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_th_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_sh_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S
        self.weight_sh_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S

        self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        self.permanet_weight = nn.Embedding(user_cnt, hidden_size)

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        loc_len = len(loc)
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])
                / (td_upper[i]+td_lower[i])) for i in range(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])
                / (ld_upper[i]+ld_lower[i])) for i in range(loc_len)]

        loc = self.location_weight(loc).view(-1,self.hidden_size,1)
        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i]))\
                .view(1,self.hidden_size,1) for i in range(loc_len)], dim=0), dim=0)
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = loc_vec + usr_vec
        return self.sigmoid(hx)

    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        q_v = self.location_weight(dst)
        output = torch.mm(q_v, (h_tq + torch.t(p_u)))
        return torch.log(1+torch.exp(torch.neg(output)))

    def validation(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        # error exist in distance (ld_upper, ld_lower)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        user_vector = h_tq + torch.t(p_u)
        ret = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        # 返回的列表按照概率从高到低的顺序，列出了预测的下一跳的位置编号
        return np.argsort(np.squeeze(-1*ret))


class STRNNModule(nn.Module):
    def __init__(self, dim, loc_cnt, user_cnt, ww):
        super(STRNNModule, self).__init__()
        # embedding:
        self.user_weight = Variable(torch.randn(user_cnt, dim), requires_grad=False).type(torch.cuda.FloatTensor)
        self.h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(torch.cuda.FloatTensor)
        self.location_weight = nn.Embedding(loc_cnt, dim)
        self.perm_weight = nn.Embedding(user_cnt, dim)
        self.ww = ww
        # attributes:
        self.time_upper = nn.Parameter(torch.randn(dim, dim).type(torch.cuda.FloatTensor))
        self.time_lower = nn.Parameter(torch.randn(dim, dim).type(torch.cuda.FloatTensor))
        self.dist_upper = nn.Parameter(torch.randn(dim, dim).type(torch.cuda.FloatTensor))
        self.dist_lower = nn.Parameter(torch.randn(dim, dim).type(torch.cuda.FloatTensor))
        self.C = nn.Parameter(torch.randn(dim, dim).type(torch.cuda.FloatTensor))
        # modules:
        self.sigmoid = nn.Sigmoid()

    # find the most closest value to w, w_cap(index)
    def find_w_cap(self, times, i):
        trg_t = times[i] - self.ww
        tmp_t = times[i]
        tmp_i = i - 1
        for idx, t_w in enumerate(reversed(times[:i]), start=1):  # times是升序 翻过来是降序
            if t_w.data.cpu().numpy() == trg_t.data.cpu().numpy():
                return i - idx
            elif t_w.data.cpu().numpy() > trg_t.data.cpu().numpy():  # t_w大于目标 记录下来 降序 下次记录的会越来越小
                tmp_t = t_w
                tmp_i = i - idx
            elif t_w.data.cpu().numpy() < trg_t.data.cpu().numpy():  # 此时t_w是第一个比目标小的 tmp_t是比目标大的中最小的那个
                if trg_t.data.cpu().numpy() - t_w.data.cpu().numpy() \
                        < tmp_t.data.cpu().numpy() - trg_t.data.cpu().numpy():  # t_w比tmp_t更接近目标 选择t_w
                    return i - idx
                else:  # 选择tmp_t
                    return tmp_i
        return 0

    def return_h_tw(self, f, times, latis, longis, locs, idx):
        w_cap = self.find_w_cap(times, idx)
        if w_cap is 0:
            return self.h_0
        else:
            self.return_h_tw(f, times, latis, longis, locs, w_cap)

        lati = latis[idx] - latis[w_cap:idx]
        longi = longis[idx] - longis[w_cap:idx]
        td = times[idx] - times[w_cap:idx]
        ld = self.euclidean_dist(lati, longi)

        # 把原始的一个时间点一个时间点的数据按照窗口宽度w整合
        # 把同一个窗口内的数据合并到一起
        # 预处理的结果 td,ld,loc,dst输出到文件中
        # 因为宽度w可能包含多个时间点，所以td ld loc都可以能是list dst是t时刻的位置 是目标
        data = ','.join(str(e) for e in td.data.cpu().numpy()) + "\t"
        f.write(data)
        data = ','.join(str(e) for e in ld.data.cpu().numpy()) + "\t"
        f.write(data)
        # data = ','.join(str(e.data.cpu().numpy()[0]) for e in locs[w_cap:idx])+"\t"
        data = ','.join(str(e.data.cpu().numpy()) for e in locs[w_cap:idx]) + "\t"  # q ti u
        f.write(data)
        # data = str(locs[idx].data.cpu().numpy()[0])+"\n"
        data = str(locs[idx].data.cpu().numpy()) + "\n"  # q t u
        f.write(data)

    # get transition matrices by linear interpolation
    # def get_location_vector(self, td, ld, locs):
    #     tud = up_time - td
    #     tdd = td - lw_time
    #     lud = up_dist - ld
    #     ldd = ld - lw_dist
    #     loc_vec = 0
    #     for i in range(len(tud)):
    #         Tt = torch.div(torch.mul(self.time_upper, tud[i])
    #                        + torch.mul(self.time_lower, tdd[i]), tud[i] + tdd[i])
    #         Sl = torch.div(torch.mul(self.dist_upper, lud[i])
    #                        + torch.mul(self.dist_lower, ldd[i]), lud[i] + ldd[i])
    #         loc_vec += torch.mm(Sl, torch.mm(Tt,
    #                                          torch.t(self.location_weight(locs[i]))))
    #     return loc_vec

    def euclidean_dist(self, x, y):
        return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

    # neg_lati, neg_longi, neg_loc, step):
    # user, times, latis, longis, locs全是一维的
    def forward(self, f, user, times, latis, longis, locs, step):
        f.write(str(user.data.cpu().numpy()[0]) + "\n")
        # positive sampling
        pos_h = self.return_h_tw(f, times, latis, longis, locs, len(times) - 1)

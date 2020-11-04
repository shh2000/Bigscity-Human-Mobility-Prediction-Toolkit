import config
import torch.nn as nn
import torch


class serm_torch(nn.Module):
    user_dim = 0
    len = 0
    word_vec = {}
    place_dim = 0
    time_dim = 0
    pl_d = 0
    time_k = 0
    hidden_neurons = 0
    learning_rate = 0.0
    use_cuda = False

    def __init__(self, user_dim, len_serm, word_vec, place_dim=config.GRID_COUNT * config.GRID_COUNT,
                 time_dim=config.time_dim, pl_d=config.pl_d, time_k=config.time_k, hidden_neurons=config.hidden_neurons,
                 learning_rate=config.learning_rate, use_cuda=False):
        super(serm_torch, self).__init__()
        self.user_dim = user_dim
        self.len = len_serm
        self.word_vec = word_vec
        self.place_dim = place_dim
        self.time_dim = time_dim
        self.pl_d = pl_d
        self.time_k = time_k
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda

    def forward(self, input_X):
        pl_input = input_X[0]
        time_input = input_X[1]
        user_input = input_X[2]
        pl_embedding = nn.Embedding(num_embeddings=(self.place_dim + 1), embedding_dim=self.pl_d,
                                    padding_idx=0)(torch.from_numpy(pl_input).long())
        time_embedding = nn.Embedding(num_embeddings=(self.time_dim + 1), embedding_dim=self.time_k,
                                      padding_idx=0)(torch.from_numpy(time_input).long())
        user_embedding = nn.Embedding(num_embeddings=(self.user_dim + 1), embedding_dim=self.place_dim + 1,
                                      padding_idx=0)(torch.from_numpy(user_input).long())

        attrs_latent = torch.cat([pl_embedding, time_embedding], dim=2)
        lstm_out, hidden = nn.LSTM(input_size=100, hidden_size=self.hidden_neurons)(attrs_latent)
        dense = nn.Linear(in_features=50, out_features=self.place_dim + 1)(lstm_out)

        out_vec = torch.add(dense, user_embedding)
        # print(torch.Tensor.size(out_vec))
        pred = nn.Softmax(dim=2)(out_vec)
        return pred

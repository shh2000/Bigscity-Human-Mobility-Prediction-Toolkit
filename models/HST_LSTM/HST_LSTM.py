import math
import json
import torch
import os
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter




def st_lstm_cell(input_l, input_s, input_q, hidden, cell, w_ih, w_hh, w_s, w_q, b_ih, b_hh):
    """
    Proceed calculation of one step of STLSTM.
    :param input_l: input of location embedding, shape (batch_size, input_size)
    :param input_s: input of spatial embedding, shape (batch_size, input_size)
    :param input_q: input of temporal embedding, shape (batch_size, input_size)
    :param hidden: hidden state from previous step, shape (batch_size, hidden_size)
    :param cell: cell state from previous step, shape (batch_size, hidden_size)
    :param w_ih: chunk of weights for process input tensor, shape (4 * hidden_size, input_size)
    :param w_hh: chunk of weights for process hidden state tensor, shape (4 * hidden_size, hidden_size)
    :param w_s: chunk of weights for process input of spatial embedding, shape (3 * hidden_size, input_size)
    :param w_q: chunk of weights for process input of temporal embedding, shape (3 * hidden_size, input_size)
    :param b_ih: chunk of biases for process input tensor, shape (4 * hidden_size)
    :param b_hh: chunk of biases for process hidden state tensor, shape (4 * hidden_size)
    :return: hidden state and cell state of this step.
    """
    gates = torch.mm(input_l, w_ih.t()) + torch.mm(hidden,
                                                   w_hh.t()) + b_ih + b_hh  # Shape (batch_size, 4 * hidden_size)
    in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

    ifo_gates = torch.cat((in_gate, forget_gate, out_gate), 1)  # shape (batch_size, 3 * hidden_size)
    ifo_gates += torch.mm(input_s, w_s.t()) + torch.mm(input_q, w_q.t())
    in_gate, forget_gate, out_gate = ifo_gates.chunk(3, 1)

    in_gate = torch.sigmoid(in_gate)
    forget_gate = torch.sigmoid(forget_gate)
    cell_gate = torch.tanh(cell_gate)
    out_gate = torch.sigmoid(out_gate)

    next_cell = (forget_gate * cell) + (in_gate * cell_gate)
    next_hidden = out_gate * torch.tanh(cell_gate)

    return next_hidden, next_cell


class STLSTMCell(nn.Module):
    """
    A Spatial-Temporal Long Short Term Memory (HST_LSTM) cell.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTMCell(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hc = (torch.randn(3, 20), torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        >>>     hc = st_lstm(input_l[i], input_s[i], input_q[i], hc)
        >>>     output.append(hc[0])
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """
        super(STLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_ih = Parameter(torch.randn(4 * hidden_size, input_size), requires_grad=True)
        self.w_hh = Parameter(torch.randn(4 * hidden_size, hidden_size), requires_grad=True)
        self.w_s = Parameter(torch.randn(3 * hidden_size, input_size), requires_grad=True)
        self.w_q = Parameter(torch.randn(3 * hidden_size, input_size), requires_grad=True)
        if bias:
            self.b_ih = Parameter(torch.randn(4 * hidden_size), requires_grad=True)
            self.b_hh = Parameter(torch.randn(4 * hidden_size), requires_grad=True)
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.reset_parameters()

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):    # type:(Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_l, input_s, input_q, hc=None):
        """
        Proceed one step forward propagation of HST_LSTM.
        :param input_l: input of location embedding vector, shape (batch_size, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, input_size)
        :param hc: tuple containing hidden state and cell state of previous step.
        :return: hidden state and cell state of this step.
        """
        self.check_forward_input(input_l)
        self.check_forward_input(input_s)
        self.check_forward_input(input_q)
        if hc is None:
            zeros = torch.zeros(input_l.size(0), self.hidden_size, dtype=input_l.dtype, device=input_l.device)
            hc = (zeros, zeros)
        self.check_forward_hidden(input_l, hc[0], '[input_l, hidden]')
        self.check_forward_hidden(input_l, hc[1], '[input_l, cell]')
        self.check_forward_hidden(input_s, hc[0], '[input_s, hidden]')
        self.check_forward_hidden(input_s, hc[1], '[input_s, cell]')
        self.check_forward_hidden(input_q, hc[0], '[input_q, hidden]')
        self.check_forward_hidden(input_q, hc[1], '[input_q, cell]')
        return st_lstm_cell(input_l=input_l, input_s=input_s, input_q=input_q,
                            hidden=hc[0], cell=hc[1],
                            w_ih=self.w_ih, w_hh=self.w_hh, w_s=self.w_s, w_q=self.w_q,
                            b_ih=self.b_ih, b_hh=self.b_hh)


class STLSTM(nn.Module):
    """
    One layer, batch-first Spatial-Temporal LSTM network.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTM(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hidden_out, cell_out = st_lstm(input_l, input_s, input_q)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """
        super(STLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.cell = STLSTMCell(input_size, hidden_size, bias)

    def check_forward_input(self, input_l, input_s, input_q):
        if not (input_l.size(1) == input_s.size(1) == input_q.size(1)):
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    (input_l.size(1), input_s.size(1), input_q.size(1)), self.input_size))

    def forward(self, input_l, input_s, input_q, hc=None):
        """
        Proceed forward propagation of HST_LSTM network.
        :param input_l: input of location embedding vector, shape (batch_size, step, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, step, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, step, input_size)
        :param hc: tuple containing initial hidden state and cell state, optional.
        :return: hidden states and cell states produced by iterate through the steps.
        """
        self.check_forward_input(input_l, input_s, input_q)
        for step in range(input_l.size(1)):
            hc = self.cell(input_l[:, step, :], input_s[:, step, :], input_q[:, step, :], hc)
        return hc[0], hc[1]


class HSTLSTM(nn.Module):

    def __init__(self, dirPath, config):
        super(HSTLSTM, self).__init__()
        with open(os.path.join(dirPath, "config/model/hst-lstm.json"), 'r') as f:
            parameters = json.load(f)
        for key in parameters:
            if key in config:
                parameters[key] = config[key]
        self.input_size = parameters['input_size']
        self.hidden_size = parameters['hidden_size']
        self.aoi_size = parameters['aoi_size']
        self.bias = parameters['bias']
        self.lr = parameters['lr']
        self.device = parameters['device']
        self.use_gpu = parameters['use_gpu']
        self.w_p = Parameter(torch.randn(self.hidden_size, self.aoi_size), requires_grad=True)
        self.b_p = Parameter(torch.randn(self.aoi_size), requires_grad=True)
        self.soft_max = nn.Softmax(dim=1)
        self.encoding_stlstm = STLSTM(self.input_size, self.hidden_size, self.bias)
        self.context_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.decoding_stlstm = STLSTM(self.input_size, self.hidden_size, self.bias)
        if self.use_gpu:
            self.to(self.device)

    def data_to_gpu(self, input_l, input_s, input_q, input_l_u, input_s_u, input_q_u, hc_e=None, hc_c=None):
        input_l.to(self.device)
        input_s.to(self.device)
        input_q.to(self.device)
        input_l_u.to(self.device)
        input_s_u.to(self.device)
        input_q_u.to(self.device)
        hc_c.to(self.device)
        hc_e.to(self.device)

    def forward(self, input_l, input_s, input_q, input_l_u, input_s_u, input_q_u, hc_e=None, hc_c=None):
        """
        Proceed forward propagation of HST-LSTM network.
        :param input_l: input tensor of location embedding vector, shape (batch_size, session_size, step, input_size)
        :param input_s: input tensor of spatial embedding vector, shape (batch_size, session_size, step, input_size)
        :param input_q: input tensor of temporal embedding vector, shape (batch_size, session_size, step, input_size)
        :param input_q_u: input tensor of unknown location embedding vector, shape (batch_size, step, input_size)
        :param input_s_u: input tensor of unknown location embedding vector, shape (batch_size, step, input_size)
        :param input_l_u: input tensor of unknown location embedding vector, shape (batch_size, step, input_size)
        :param hc_e: initial tuple of hidden state and cell state for encoding st-lstm
        :param hc_c: initial tuple of hidden state and cell state for context lstm
        :return: hidden states and cell states produced by iterate through the steps.
        """
        if self.use_gpu:
            self.data_to_gpu(input_l, input_s, input_q, input_l_u, input_s_u, input_q_u, hc_e, hc_c)
        output_hidden_e = []
        for session in range(input_l.size(1)):
            h, _ = self.encoding_stlstm(input_l[:, session, :, :].squeeze(1), input_s[:, session, :, :].squeeze(1),
                                        input_q[:, session, :, :].squeeze(1))
            output_hidden_e.append(h)  # output_hidden_e.size = (session_size, batch_size, hidden_size)
        input_context = torch.stack(output_hidden_e)  # size = (session_size, batch_size, hidden_size)
        _, hc = self.context_lstm(input_context, hc_c)
        output_context = (hc[0].squeeze(), torch.zeros(input_l.size(0), self.hidden_size))
        output_decoding, _ = self.decoding_stlstm(input_l_u, input_s_u, input_q_u, hc=output_context)

        distribution = self.soft_max(torch.mm(output_decoding, self.w_p) + self.b_p)
        prediction = torch.max(distribution, dim=1)

        return distribution, prediction

    














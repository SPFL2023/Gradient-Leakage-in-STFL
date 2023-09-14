import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j][0])
        return F.softmax(attn_energies, dim=1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            if len(hidden.size()) == 1:
                energy = hidden.dot(energy)
            else:
                energy = hidden[0].dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        # self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        # self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        # self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        # self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_emb_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc_emb, tim_emb, history_loc_emb, history_tim_emb, history_count, uid, target_len, batch_size):
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c1 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)
        x = x.to(torch.float32)

        history = torch.cat((history_loc_emb, history_tim_emb), 2)
        history = torch.tanh(self.fc_attn(history))

        self.rnn.flatten_parameters()
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out_state, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out_state, (h1, c1) = self.rnn(x, (h1, c1))
        out_state = out_state.squeeze(1)

        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0)
        his_len = history.size()[0]
        
        attn_weights = attn_weights.expand(batch_size, -1, -1)
        
        if batch_size == 1:
            context = attn_weights.bmm(history.reshape((batch_size, his_len, self.hidden_size))).squeeze(0)
        else:
            context = attn_weights.bmm(history.reshape((batch_size, his_len, self.hidden_size)))

        # print("context.shape: {}".format(context.shape))

        if batch_size == 1:
            out = torch.cat((out_state[-target_len:], context), 1)  # no need for fc_attn
        else:
            out = torch.cat((out_state.reshape(batch_size, self.hidden_size), context.reshape(batch_size, self.hidden_size)), 1)
            out = out[0].reshape(1, len(out[0]))

        uid_emb = self.emb_uid(uid)

        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y, dim=1)

        return score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from mdrnns import *
import os

class SeqEncoding(nn.Module):

    def __init__(self, config, emb_dim=None, n_heads=8):
        super().__init__()

        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = 512
        self.n_heads = n_heads
        self.config.hidden_dim = 512
        self.config.dropout = 0.5

        self.linear_qk = nn.Linear(self.emb_dim, self.n_heads, bias=None)
        self.linear_v = nn.Linear(self.emb_dim, self.hidden_dim, bias=True)
        self.linear_o = nn.Linear(self.emb_dim, self.hidden_dim, bias=True)

        self.linear_q = nn.Linear(self.emb_dim, self.n_heads, bias=None)
        self.linear_k = nn.Linear(self.emb_dim, self.n_heads, bias=None)
        self.tanh = nn.Tanh()

        self.norm0 = nn.LayerNorm(self.config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.config.dropout, inplace=True),
            nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(self.config.hidden_dim)

        self.dropout_layer = nn.Dropout(self.config.dropout, inplace=True)


    def forward(
            self,
            S: torch.Tensor,
            T: torch.Tensor,
            masks: torch.Tensor = None,
    ):
        B, N, H = S.shape
        n_heads = self.n_heads
        subH = H // n_heads

        if masks is not None:
            masks = masks.unsqueeze(1) & masks.unsqueeze(2)
            masks_addictive = -1000. * (1. - masks.float())
            masks_addictive = masks_addictive.unsqueeze(-1)

        S_res = S

        # X = self.seq2mat(S, S)  # (B, N, N, H)
        # X = F.relu(X, inplace=True)
        # q = self.linear_q(X)
        # k = self.linear_k(X)
        # attn = q + k
        # attn = attn + masks_addictive
        # attn = self.tanh(attn)
        # attn = attn.permute(0, -1, 1, 2)  # (B, n_heads, N, T)
        # attn = attn.softmax(-1)  # (B, n_heads, N, T)

        # Table-Guided Attention
        attn = self.linear_qk(T)
        # attn = attn + masks_addictive
        attn = attn
        attn = attn.permute(0, -1, 1, 2)  # (B, n_heads, N, T)
        attn = attn.softmax(-1)  # (B, n_heads, N, T)

        v = self.linear_v(S)  # (B, N, H)
        v = v.view(B, N, n_heads, subH).permute(0, 2, 1, 3)  # B, n_heads, N, subH

        S = attn.matmul(v)  # B, n_heads, N, subH
        S = S.permute(0, 2, 1, 3).reshape(B, N, subH * n_heads)

        S = self.linear_o(S)
        S = F.relu(S, inplace=False)
        S = self.dropout_layer(S)

        S = S_res + S
        S = self.norm0(S)

        S_res = S

        # Position-wise FeedForward
        S = self.ffn(S)
        S = self.dropout_layer(S)
        S = S_res + S
        S = self.norm1(S)

        return S, attn


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s

class Triaffine(nn.Module):
    """
    Triaffine layer for second-order scoring.
    This function has a tensor of weights `W` and bias terms if needed.
    The score `s(x, y, z)` of the vector triple `(x, y, z)` is computed as `x^T z^T W y`.
    Usually, `x` and `y` can be concatenated with bias terms.
    References:
        - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
          Efficient Second-Order TreeCRF for Neural Dependency Parsing
          https://www.aclweb.org/anthology/2020.acl-main.302/
        - Xinyu Wang, Jingxian Huang, and Kewei Tu (ACL'19)
          Second-Order Semantic Dependency Parsing with End-to-End Neural Networks
          https://www.aclweb.org/anthology/P19-1454/
    Args:
        n_in (int):
            The dimension of the input feature.
        bias_x (bool):
            If True, add a bias term for tensor x. Default: False.
        bias_y (bool):
            If True, add a bias term for tensor y. Default: False.
    """

    def __init__(self, n_in, bias_x=False, bias_y=False):
        super().__init__()

        self.n_in = n_in
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in + bias_x,
                                                n_in,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, n_in]
            y (torch.Tensor): [batch_size, seq_len, n_in]
            z (torch.Tensor): [batch_size, seq_len, n_in]
        Returns:
            s (torch.Tensor): [batch_size, seq_len, seq_len, seq_len]
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = torch.einsum('bzk,ikj->bzij', z, self.weight)
        # [batch_size, seq_len, seq_len, seq_len]
        s = torch.einsum('bxi,bzij,byj->bzxy', x, w, y)

        return s


class TriaffineScorer(nn.Module):
    def __init__(self, n_in=800, n_out=400, bias_x=True, bias_y=False, dropout=0.33):
        super(TriaffineScorer, self).__init__()
        self.l = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.m = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.attn = Triaffine(n_in=n_out, bias_x=bias_x, bias_y=bias_y)

    def forward(self, h):
        left = self.l(h)
        mid = self.m(h)
        right = self.r(h)
        return self.attn(left, mid, right).permute(0, 2, 3, 1)

    def forward2(self, word, span):
        left = self.l(word)
        mid = self.m(span)
        right = self.r(span)
        return self.attn(mid, right, left).permute(0, 2, 3, 1)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)
        self.triaffine = TriaffineScorer(n_in=biaffine_size, n_out=cls_num)
        self.linear2 = MLP(n_in=384, n_out=cls_num, dropout=dropout)

    def forward(self, x, y, z):
        ent_sub = self.dropout(self.mlp1(x))
        ent_obj = self.dropout(self.mlp2(y))

        o1 = self.biaffine(ent_sub, ent_obj)  #[8, 28, 28, 7]

        z = self.dropout(self.mlp_rel(z))  #[8, 28, 28, 384]
        o2 = self.linear(z)

        return o1 + o2


def get_word2idx():
    path = './glove/glove.6B.300d.txt'
    data = {'<pad>':0, '<unk>':1, '<cls>': 2, '<sep>': 3 }
    i = 4
    with open(path, 'r') as f:
        for line in f.readlines():
            data[line.strip().split(' ')[0]] = i
            i = i + 1
    return data


def get_numpy_word_embed(word2ix):
    row = 0
    file = 'glove.6B.300d.txt'
    path = './glove'
    whole = os.path.join(path, file)
    words_embed = {}
    with open(whole, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            # print(len(line.split()))
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 20000:
                break
            row += 1
    # word2ix = {}
    ix2word = {ix: w for w, ix in word2ix.items()}
    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 300
    data = [id2emb[ix] for ix in range(len(word2ix))]

    return data


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

        self.lstm_hid_size = config.lstm_hid_size

        lstm_input_size = 0

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        self.dropout = nn.Dropout(config.emb_dropout)
        self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size,
                                     config.channels, config.ffnn_hid_size,
                                     config.out_dropout)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

        self.mdrnn = get_mdrnn_layer(
                config, direction='Q', norm='', Md='25d', first_layer=True)

        self.mdrnn2 = get_mdrnn_layer(
            config, direction='Q', norm='', Md='25d', first_layer=False)


        if 'msra' in config.config:
            self.reduce = torch.nn.Linear(696, 512, bias=True)
        elif 'ace05' in config.config or 'ace04' in config.config or 'NNE' in config.config:
            self.reduce = torch.nn.Linear(1320, 256, bias=True)  #1320
        elif 'genia' in config.config:
            self.reduce = torch.nn.Linear(676, 256, bias=True)  #696

        self.up_mlp = nn.Linear(256, 512)

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''


        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float(), output_attentions=True)
        attention_state = bert_embs[3]
        attention = None

        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

            for _item in attention_state:
                if attention == None:
                    attention = _item
                    attention = attention.permute(0, 2, 3, 1)
                else:
                    attention = torch.cat([attention, _item.permute(0, 2, 3, 1)], dim=-1)

        attention = attention[:,:max(sent_length),:max(sent_length),:]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())   #[8, 28, 512]

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([reg_emb, dis_emb, cln, attention], dim=-1)

        conv_inputs = self.reduce(conv_inputs)

        T, Tstates = self.mdrnn(conv_inputs, states=None)

        T, Tstates = self.mdrnn2(T, states=Tstates)

        T = self.up_mlp(T)

        outputs = self.predictor(word_reps, word_reps, T)

        return outputs

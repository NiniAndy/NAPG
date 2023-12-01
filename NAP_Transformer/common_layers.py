# config:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"""
General purpose functions
"""


def pad_list(xs, pad_value, length):
    # 把label都pad成max_len长度
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    # max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, length, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


""" 
Transformer common layers
"""


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    encoder:获得一个仅在有数据的地方是1，pad出来的位置为0的一个mask，维度为[batch_size, T, 1]
    其中T为padded_input的最后一维长度，即为卷积后的宽度，根据两次pooling为longest_n_frame/4
    decoder: padded_input是pad以后的label，pad_idx是2
    padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)  # N=batch_size
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])
        # [batch_size, T]一个高度为batch_size的长度为T的全为1的tensor
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0  # 没有啥用
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()  # 等于pad_idx的赋值为0
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)  # 在最后加一个维度[batch_size, T, 1]


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    """
    For masking out the padding part of key sequence.
    seq_k：seq_in_pad
    seq_q: seq_in_pad  [batch_size, tgt_max_len]
    """
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)  # tgt_max_len
    padding_mask = seq_k.eq(pad_idx)  # pad_idx是True其他为False
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # [batch_size, tgt_max_len, tgt_max_len]
    return padding_mask


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """
    mask position is set to 1
    0代表没有mask为False，1代表有mask为True
    decoder: padded_input--->encoder_padded_outputs: [batch_size, T, dim_model]
             input_lengths--->encoder_input_lengths: 记录每个频谱的n_frame  [batch_size]
             expand_length: tgt_max_len
    """
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # 全都是1 [batch_size, T, 1]
    # N x Ti, lt(1) like not operation
    # 小于1为1，大于1为0
    pad_mask = non_pad_mask.squeeze(-1).lt(1)  # 全都是0=False [batch_size, T]
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)  # 全都是0=False [batch_size, T, T]
    return attn_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    # seq：[batch_size, tgt_max_len]
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    # torch.triu返回三角矩阵，上对角线为1，其他为0，[tgt_max_len, tgt_max_len]
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # [batch_size, tgt_max_len, tgt_max_len]
    subsequent_mask = subsequent_mask.bool()  # 0:False, 1:True
    return subsequent_mask


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    """unfold T x B x C to T x B x C x K"""
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class PositionalEncoding(nn.Module):
    """
    Positional Encoding class
    """

    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term)  # take the odd (jump by 2)
        pe[:, 1::2] = torch.cos(position * exp_term)  # take the even (jump by 2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 位置编码[1, max_length, dim_model]

    def forward(self, input):
        """
        args:
            input输入VGG的输出: [batch_size, T, H]
        output:
            tensor: B x T
        """
        return self.pe[:, :input.size(1)]  # 位置编码[batch_size, T]


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Layer class
    FFN(x) = max(0, xW1 + b1) W2+ b2
    """

    def __init__(self, dim_model, dim_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(dim_model, dim_hidden)
        self.linear_2 = nn.Linear(dim_hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        """
        args:
            x: tensor
        output:
            y: tensor
        """
        residual = x
        # x = self.layer_norm(x)
        output = self.dropout(self.linear_2(F.relu(self.linear_1(x))))
        # output = output + residual
        output = self.layer_norm(output + residual)
        return output


class PositionwiseFeedForwardWithConv(nn.Module):
    """
    Position-wise Feedforward Layer Implementation with Convolution class
    """

    def __init__(self, dim_model, dim_hidden, dropout=0.1):
        super(PositionwiseFeedForwardWithConv, self).__init__()
        self.conv_1 = nn.Conv1d(dim_model, dim_hidden, 1)
        self.conv_2 = nn.Conv1d(dim_hidden, dim_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        residual = x
        # x = self.layer_norm(x)
        output = x.transpose(1, 2)
        output = self.conv_2(F.relu(self.conv_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output  # output: [batch_size, T, dim_model]




class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dim_key, dim_value, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        # dim_model是输入的最后一个维度，num_heads*dim_key是输出的最后一个维度，前面的维度保持不变
        self.query_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.value_linear = nn.Linear(dim_model, num_heads * dim_value)

        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.value_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_value)))

        self.attention = ScaledDotProductAttention(temperature=np.power(dim_key, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.output_linear = nn.Linear(num_heads * dim_value, dim_model)

        nn.init.xavier_normal_(self.output_linear.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        dim_model = H
        query: B x T_Q x H, key: B x T_K x H, value: B x T_V x H
        mask: B x T x T (attention mask)
        return:
            output: [batch_size, T, dim_model]
            attention: [batch_size*num_heads, T, T]
        """
        batch_size, len_query, _ = query.size()  # batch_size, T, dim_model
        batch_size, len_key, _ = key.size()
        batch_size, len_value, _ = value.size()
        residual = query  # 残差网络输入 [batch_size, T, dim_model]
        # query = self.layer_norm(query)
        # q,k,v:[batch_size, T, dim_model]   query_linear(query):[batch_size, T, num_heads*dim_query]
        # view就是reshape query:[batch_size, T, num_heads, dim_query]
        query = self.query_linear(query).view(batch_size, len_query, self.num_heads, self.dim_key)  # B x T_Q x num_heads x H_K
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads, self.dim_key)  # B x T_K x num_heads x H_K
        value = self.value_linear(value).view(batch_size, len_value, self.num_heads, self.dim_value)  # B x T_V x num_heads x H_V
        # query.permute(2, 0, 1, 3)就是更换维度位置[num_heads, batch_size, T, dim_query]
        # view后就是[num_heads*batch_size, T, dim_query]
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, self.dim_key)  # (num_heads * B) x T_Q x H_K
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.dim_key)  # (num_heads * B) x T_K x H_K
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_value, self.dim_value)  # (num_heads * B) x T_V x H_V
        # 扩充mask，将mask的第一维扩充成[batch_size*num_heads, T, T]
        if mask is not None:
            mask = mask.repeat(self.num_heads*batch_size, 1, 1)  # (B * num_head) x T x T
        output, attn = self.attention(query, key, value, mask=mask)
        # 将输出重新reshape
        output = output.view(self.num_heads, batch_size, len_query, self.dim_value)  # num_heads x B x T_Q x H_V
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_query, -1)
        attn = attn.view(self.num_heads, batch_size, len_query, len_value)# B x T_Q x (num_heads * H_V)
        # output.shape: [batch_size, T, num_heads*dim_value]
        output = self.dropout(self.output_linear(output))  # B x T_Q x H_O
        # output.shape: [batch_size, T, dim_model]
        output = self.layer_norm(output + residual)
        # output = output + residual
        return output, attn

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v :[num_heads*batch_size, T, dim_query]
        mask: [num_heads*batch_size, T, dim_query]
        return:
            output: [batch_size*num_heads, T, dim_value]
            attention: [batch_size*num_heads, T, T]
        """
        attn = torch.bmm(q, k.transpose(1, 2))  # [num_heads*batch_size, T, T]
        attn = attn / attn.abs().max(-1, keepdim=True)[0] + 1
        attn = attn / (self.temperature * 8)

        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)  # mask为True的地方用-np.inf填充，其他不变
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # 3维矩阵相乘
        return output, attn


class ScoreAttention(nn.Module):
    def __init__(self, dim_model, dim_key, dropout=0.1):
        super(ScoreAttention, self).__init__()
        num_heads = 1
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_key = dim_key

        # dim_model是输入的最后一个维度，num_heads*dim_key是输出的最后一个维度，前面的维度保持不变
        self.query_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * dim_key)
        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        self.attention = TopKScaledDotProductAttentionScore(temperature=np.power(dim_key, 0.5), attn_dropout=dropout)

    def forward(self, query, key, mask=None, topk=4, ngram=3):
        """
        dim_model = H
        query: B x T_Q x H, key: B x T_K x H, value: B x T_V x H
        mask: B x T x T (attention mask)
        return:
            output: [batch_size, T, dim_model]
            attention: [batch_size*num_heads, T, T]
        """
        batch_size, len_query, _ = query.size()  # batch_size, T, dim_model
        batch_size, len_key, _ = key.size()
        query = self.query_linear(query).view(batch_size, len_query, self.num_heads, self.dim_key)  # B x T_Q x num_heads x H_K
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads, self.dim_key)  # B x T_K x num_heads x H_K
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, self.dim_key)  # (num_heads * B) x T_Q x H_K
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.dim_key)  # (num_heads * B) x T_K x H_K
        if mask is not None:
            mask = mask.repeat(self.num_heads*batch_size, 1, 1)  # (B * num_head) x T x T
        topk_score = self.attention(query, key, batch_size, mask, topk, ngram)
        return topk_score

class TopKScaledDotProductAttentionScore(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q, k, batch_size,mask=None, topk=4, ngram=3):
        """
        q, k, v :[num_heads*batch_size, T, dim_query]
        mask: [num_heads*batch_size, T, dim_query]
        return:
            output: [batch_size*num_heads, T, dim_value]
            attention: [batch_size*num_heads, T, T]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) # [num_heads*batch_size, T, T]
        attn = attn / 256
        # attn = attn / attn.abs().max(dim=-1, keepdim=True)[0] + 1
        attn = torch.sigmoid(attn)
        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)  # mask为True的地方用-np.inf填充，其他不变
        scores_prob = attn[attn !=  -torch.inf].view(batch_size, -1,  topk)
        scores_prob_init = scores_prob[:, :topk, :]
        scores_prob_init = scores_prob_init.unsqueeze(-2).repeat(1, 1, topk, 1).view(batch_size, 1, topk, topk*topk)
        scores_prob_end = scores_prob[:, -topk:, :]
        scores_prob_end = scores_prob_end.unsqueeze(-1).repeat(1, 1, 1, topk).view(batch_size, 1, topk, topk*topk)
        scores_prob = scores_prob[:, topk:-topk, :]
        scores_prob = scores_prob.reshape(batch_size, -1, topk, (ngram-1), topk)
        scores_prob_before = scores_prob[:, :, :, 0, :].contiguous().unsqueeze(-1).view(-1, topk, 1)
        scores_prob_after = scores_prob[:, :, :, 1, :].contiguous().unsqueeze(-2).view(-1, 1, topk)
        score_matrix = torch.bmm(scores_prob_before, scores_prob_after)
        score_matrix = score_matrix.view(batch_size, -1, topk, topk*topk)
        score_matrix = torch.cat([scores_prob_init, score_matrix, scores_prob_end], dim=1)
        topk_score = score_matrix.sum(dim=-1)
        # topk_score = F.normalize(topk_score, p=1, dim=-1)
        return topk_score


class DotProductAttention(nn.Module):
    """
    Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()
        # self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)
        # # concat -> (N, To, 2*H)
        # concated = torch.cat((attention_output, queries), dim=2)
        # # output -> (N, To, H)
        # output = torch.tanh(self.linear_out(
        #     concated.view(-1, 2*hidden_size))).view(batch_size, -1, hidden_size)

        return attention_output, attention_distribution

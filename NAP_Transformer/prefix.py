import torch
import torch.nn as nn
import torch.nn.functional as F

from wenet.NAP_Transformer.common_layers import ScoreAttention, MultiHeadAttention

import numpy as np
import heapq

def search_top_paths(a_np, top_k=10):
    rows, cols = a_np.shape
    paths = [(-a_np[0][i], [i]) for i in range(cols)]  # 使用负数因为heapq是最小堆，我们希望根据得分从大到小排序
    heapq.heapify(paths)
    top_paths = []
    while paths:
        score, path = heapq.heappop(paths)
        row = len(path)
        if row == rows:
            top_paths.append((score, path))
            if len(top_paths) == top_k:
                break
        else:
            for i in range(cols):
                new_score = score * -a_np[row][i]
                heapq.heappush(paths, (-new_score, path + [i]))
    return [(score * -1, path) for score, path in top_paths]


class PREFIX(nn.Module):
    def __init__(self, topk, ngram, tgt_maxlen, decoder):
        super().__init__()

        self.topk = topk
        self.ngram = ngram
        self.tgt_maxlen = tgt_maxlen
        self.embed = decoder.embed
        self.score_attn = ScoreAttention(256, 256, 0)

        self_chunk_mask = torch.zeros(self.topk, self.topk)
        ngram_chunk_mask = torch.ones(self.topk, self.ngram * self.topk)
        self.seq_len_ngram_mask = torch.zeros(self.tgt_maxlen * self.topk, self.tgt_maxlen * self.topk + self.topk * self.ngram)

        for i in range(self.tgt_maxlen):
            sid_self = i * self.topk
            eid_self = (i + 1) * self.topk
            self.seq_len_ngram_mask[sid_self:eid_self, sid_self:sid_self + self.ngram * self.topk] = ngram_chunk_mask
            self.seq_len_ngram_mask[sid_self:eid_self, sid_self + self.topk:eid_self + self.topk] = self_chunk_mask
        self.seq_len_ngram_mask = self.seq_len_ngram_mask[:, self.topk: self.topk + eid_self]


    def forward(self, bsz, seq_len, na_pred, device) :
        na_prefix_prob, na_prefix = torch.topk(torch.softmax(na_pred, dim=-1), k=self.topk, dim=2)
        # na_prefix_prob, na_prefix [bsz, seq_len, k]
        na_prefix_prob = torch.softmax(na_prefix_prob, dim=-1)
        prefix_emb, _ = self.embed(na_prefix.reshape(-1, self.topk).view(-1, 1))
        # [bsz, seq_len, k]->[bsz*seq_len, k]->[bsz*seq_len*k, 1]->[bsz*seq_len*k, d]
        prefix_emb = prefix_emb.view(bsz, seq_len*self.topk, -1)  # (bsz, seq_len*k, d)

        scores_mask = ~self.seq_len_ngram_mask[:seq_len * self.topk, :seq_len * self.topk].bool().to(device)
        na_score = self.score_attn(prefix_emb, prefix_emb, scores_mask, self.topk, self.ngram)
        prefix_score = torch.mul(na_score, na_prefix_prob)
        prefix_score = F.normalize(prefix_score, p=1, dim=-1)
        prefix_emb = prefix_emb.view(bsz, seq_len, self.topk, -1)
        prefix_score = prefix_score.view(bsz, seq_len, self.topk, -1)
        scored_prefix = (prefix_emb * prefix_score).sum(dim=2)
        return scored_prefix, prefix_score, na_prefix

    def one_step_forward(self, bsz, seq_len, na_pred, device) :

        na_prefix_prob, na_prefix = torch.topk(torch.softmax(na_pred, dim=-1), k=self.topk, dim=2)
        one_prefix_prob = na_prefix_prob[0].numpy()
        top_10_paths = search_top_paths(one_prefix_prob)
        lst = []
        for i in range(len(top_10_paths)):
            score, path = top_10_paths[i]
            path_na_prefix = torch.zeros((seq_len,), dtype=torch.long)
            for j in range(len(path_na_prefix)):
                path_na_prefix[j] = na_prefix[i, j, path[j]]

            lst.append((path_na_prefix[:-1], torch.tensor(np.log(score))))

        return lst





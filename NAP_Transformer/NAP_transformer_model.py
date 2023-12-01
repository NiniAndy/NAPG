# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import time

from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add, remove_duplicates_and_blank, th_accuracy, reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds, mask_finished_scores, subsequent_mask)
from wenet.utils.context_graph import ContextGraph

from wenet.CTC_NA.decoder import CTCNA_TransformerDecoder
from wenet.NAP_Transformer.common_layers import ScoreAttention, MultiHeadAttention
from wenet.NAP_Transformer.decoder import NAP_TransformerDecoder
from wenet.NAP_Transformer.prefix import PREFIX



class NapTransformerModel(nn.Module):
    """NA prefix hybrid Encoder-Decoder model"""

    def __init__(self,
                 vocab_size,
                 encoder: TransformerEncoder,
                 ardecoder: NAP_TransformerDecoder,
                 nadecoder: CTCNA_TransformerDecoder,
                 ctc: CTC,
                 ctc_weight= 0.5,
                 ignore_id= IGNORE_ID,
                 lsm_weight = 0.0,
                 length_normalized_loss = False,
                 lfmmi_dir = ''):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        self.topk = 4
        self.ngram = 3
        self.tgt_maxlen = 60
        self.placeholder_emb = nn.Embedding(self.tgt_maxlen, nadecoder.attention_dim)
        self.spend = 0

        self.encoder = encoder
        self.nadecoder = nadecoder
        self.decoder = ardecoder
        self.prefix_gen = PREFIX(self.topk, self.ngram, self.tgt_maxlen, self.decoder)
        self.emendatory_attn = MultiHeadAttention(4, 256, 64, 64, 0.1)

        self.ctc = ctc
        self.sploss_lo = nn.Linear(256, vocab_size)
        self.criterion_att = LabelSmoothingLoss(size=vocab_size,
                                                padding_idx=ignore_id,
                                                smoothing=lsm_weight,
                                                normalize_length=length_normalized_loss)

    def forward(self, speech, speech_lengths, text, text_lengths):
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]), \
            (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        # 1. Encoder

        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2. ctc loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)

        # 3. NA decoder
        loss_na, na_pred = self._calc_na_loss(encoder_out, encoder_mask, text, text_lengths) # (B, L, V)

        seq_len = na_pred.size(1)
        bsz = encoder_out.size(0)
        device = encoder_out.device

        # 4. prefix
        # na_prefix_prob, na_prefix = torch.topk(na_pred, k=1, dim=2)
        # na_prefix = na_prefix.squeeze(-1)
        # na_prefix, _ = add_sos_eos(na_prefix[:, :-1], self.sos, self.eos, self.ignore_id)
        # prefix_emb, _ = self.decoder.embed(na_prefix)
        # truth_text_in_pad, _ = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        # truth_text_in_pad_emb, _ = self.decoder.embed(truth_text_in_pad)
        # truth_text_in_lens = text_lengths + 1
        # maxlen = truth_text_in_pad.size(1)
        # # 对角mask的生成
        # tgt_mask = ~make_pad_mask(truth_text_in_lens, maxlen).unsqueeze(1)
        # tgt_mask = tgt_mask.to(truth_text_in_pad.device)
        # re_teacher_mask = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device)
        # teacher_mask = ~re_teacher_mask
        # re_teacher_mask = re_teacher_mask.unsqueeze(-1).repeat(1, 1, prefix_emb.shape[-1])
        # teacher_mask = teacher_mask.unsqueeze(-1).repeat(1, 1, truth_text_in_pad_emb.shape[-1])
        # # 扩充emb
        # text_emb = truth_text_in_pad_emb.unsqueeze(1).repeat(1, seq_len, 1, 1)
        # prefix_emb = prefix_emb.unsqueeze(1).repeat(1, seq_len, 1, 1)
        # teacher_tgt = text_emb.masked_fill(teacher_mask, 0.0)
        # prefix_tgt = prefix_emb.masked_fill(re_teacher_mask, 0.0)
        # emendatory_prefix = teacher_tgt + prefix_tgt
        #
        # emendatory_prefix = emendatory_prefix.view(-1, seq_len, emendatory_prefix.size(-1))
        # truth_text_in_pad_emb = truth_text_in_pad_emb.view(-1,  1, truth_text_in_pad_emb.size(-1))
        # prefix, prefix_attn = self.emendatory_attn(truth_text_in_pad_emb, emendatory_prefix, emendatory_prefix, None)
        # prefix = prefix.view(bsz, seq_len, -1)

        # '''
        # rescore na_prefix
        # na_prefix_prob, na_prefix = torch.topk(torch.softmax(na_pred, dim=-1), k=self.topk, dim=2)
        # # na_prefix_prob, na_prefix [bsz, seq_len, k]
        # na_prefix_prob = torch.softmax(na_prefix_prob, dim=-1)
        # prefix_emb, _ = self.decoder.embed(na_prefix.reshape(-1, self.topk).view(-1, 1))
        # # [bsz, seq_len, k]->[bsz*seq_len, k]->[bsz*seq_len*k, 1]->[bsz*seq_len*k, d]
        # prefix_emb = prefix_emb.view(bsz, seq_len*self.topk, -1)  # (bsz, seq_len*k, d)
        #
        # scores_mask = ~self.seq_len_ngram_mask[:seq_len * self.topk, :seq_len * self.topk].bool().to(encoder_out.device)
        # na_score = self.score_attn(prefix_emb, prefix_emb, scores_mask, self.topk, self.ngram)
        # prefix_score = torch.mul(na_score, na_prefix_prob)
        # prefix_score = F.normalize(prefix_score, p=1, dim=-1)
        # prefix_emb = prefix_emb.view(bsz, seq_len, self.topk, -1)
        # prefix_score = prefix_score.view(bsz, seq_len, self.topk, -1)
        # scored_prefix = (prefix_emb * prefix_score).sum(dim=2)
        scored_prefix, prefix_score, na_prefix = self.prefix_gen(bsz, seq_len, na_pred, device)
        loss_score = self._calc_scored_prefix_loss(scored_prefix, text, text_lengths)
        # 得到真值的emb
        emendatory_prefix, truth_text_in_pad_emb, text_out_pad, truth_text_in_lens = self._calc_teacher_force(text,
                                                                                                              text_lengths,
                                                                                                              scored_prefix,
                                                                                                              seq_len)
        teacher_force_prefix, prefix_attn = self.emendatory_attn(truth_text_in_pad_emb, emendatory_prefix, emendatory_prefix, None)
        teacher_force_prefix = teacher_force_prefix.view(bsz, seq_len, -1)

        loss_ar, _ = self._calc_ar_loss(encoder_out, encoder_mask, teacher_force_prefix, truth_text_in_lens, text_out_pad)

        loss = 0.3* ( 0.3 * loss_na + 0.6 * loss_score + 0.1* loss_ctc) + 0.7 * loss_ar
        loss_2 = 0.3 * loss_ctc  + 0.7 * loss_ar
        # loss = self.ctc_weight * ( 0.5 * loss_na + 0.5 * loss_score) + (1 - self.ctc_weight) * loss_ar
        return {"loss": loss, "loss_2": loss_2,"loss_ar": loss_ar, "loss_ctc": loss_ctc, "loss_na": loss_na, "loss_score": loss_score}

    def _calc_na_loss(self, encoder_out, encoder_mask, ys_pad, ys_pad_lens):
        _, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_out_pad = torch.where(ys_out_pad == self.ignore_id, self.eos, ys_out_pad)
        query_need_len = ys_out_pad.size(1)
        # 构建NA的输入query
        bsz = encoder_out.size(0)
        query = torch.arange(self.tgt_maxlen).type_as(encoder_out).long()
        query = self.placeholder_emb(query)[: query_need_len]
        query = query.unsqueeze(0).repeat(bsz, 1, 1)
        query_len = torch.full((bsz,), query_need_len, dtype=torch.long)
        query_mask = ~make_pad_mask(query_len, query_need_len).unsqueeze(1).to(encoder_out.device)
        # 1. Forward decoder
        na_pred = self.nadecoder(encoder_out, encoder_mask, query, query_mask)
        loss = self.criterion_att(na_pred, ys_out_pad)
        return loss, na_pred

    def _calc_ar_loss(self, encoder_out, encoder_mask, teacher_force_prefix, ys_in_lens, ys_out_pad):
        # 1. Forward decoder
        decoder_out, *_ = self.decoder(encoder_out, encoder_mask, teacher_force_prefix, ys_in_lens)
        # 2. Compute attention loss
        loss_ar = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id)
        return loss_ar, acc_att

    def _calc_scored_prefix_loss(self, scored_prefix, ys_pad, ys_pad_lens):
        ys_hat = self.sploss_lo(scored_prefix)
        _, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        loss_scr = self.criterion_att(ys_hat, ys_out_pad)
        return loss_scr

    def _calc_teacher_force(self, text, text_lengths, scored_prefix, seq_len):
        truth_text_in_pad, text_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        truth_text_in_pad_emb, _ = self.decoder.embed(truth_text_in_pad)
        truth_text_in_lens = text_lengths + 1
        # 为scored_prefix加入<sos>
        prefix_emb = torch.cat((truth_text_in_pad_emb[:, :1, :], scored_prefix[:, :-1, :]), dim=1)
        # 对角mask的生成
        tgt_mask = ~make_pad_mask(truth_text_in_lens, seq_len).unsqueeze(1)
        tgt_mask = tgt_mask.to(truth_text_in_pad.device)
        re_teacher_mask = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device)
        teacher_mask = ~re_teacher_mask
        re_teacher_mask = re_teacher_mask.unsqueeze(-1).repeat(1, 1, prefix_emb.shape[-1])
        teacher_mask = teacher_mask.unsqueeze(-1).repeat(1, 1, truth_text_in_pad_emb.shape[-1])
        # 扩充emb
        text_emb = truth_text_in_pad_emb.unsqueeze(1).repeat(1, seq_len, 1, 1)
        prefix_emb = prefix_emb.unsqueeze(1).repeat(1, seq_len, 1, 1)
        teacher_tgt = text_emb.masked_fill(teacher_mask, 0.0)
        prefix_tgt = prefix_emb.masked_fill(re_teacher_mask, 0.0)
        emendatory_prefix = teacher_tgt + prefix_tgt
        # 获得前置假设的teacher force结果
        emendatory_prefix = emendatory_prefix.view(-1, seq_len, emendatory_prefix.size(-1))
        truth_text_in_pad_emb = truth_text_in_pad_emb.view(-1, 1, truth_text_in_pad_emb.size(-1))
        return emendatory_prefix, truth_text_in_pad_emb, text_out_pad, truth_text_in_lens



    def _forward_encoder(self, speech, speech_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1, simulate_streaming = False):
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(speech,
                                                                            decoding_chunk_size=decoding_chunk_size,
                                                                            num_decoding_left_chunks=num_decoding_left_chunks
                                                                            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(speech,
                                                     speech_lengths,
                                                     decoding_chunk_size=decoding_chunk_size,
                                                     num_decoding_left_chunks=num_decoding_left_chunks
                                                     )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def _forward_na_decoder(self, encoder_out, encoder_mask, len):
        bsz = encoder_out.size(0)
        query = torch.arange(self.tgt_maxlen).type_as(encoder_out).long()
        query = self.placeholder_emb(query)[: len]
        query = query.unsqueeze(0).repeat(bsz, 1, 1)
        query_len = torch.full((bsz,), len, dtype=torch.long)
        query_mask = ~make_pad_mask(query_len, len).unsqueeze(1).to(encoder_out.device)
        # 1. Forward decoder
        na_pred = self.nadecoder(encoder_out, encoder_mask, query, query_mask)
        return na_pred



    def recognize(self,
                  speech,
                  speech_lengths,
                  tgt_len,
                  beam_size: int = 10,
                  decoding_chunk_size: int = -1,
                  num_decoding_left_chunks: int = -1,
                  simulate_streaming: bool = False,
                  ):
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        start_time = time.time()
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(speech,
                                                          speech_lengths,
                                                          decoding_chunk_size,
                                                          num_decoding_left_chunks,
                                                          simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, 1, maxlen)  # (B*N, 1, max_len)
        # 1.1 Forward NA decoder
        na_pred = self._forward_na_decoder(encoder_out, encoder_mask, tgt_len+1)
        seq_len = na_pred.size(1)
        bsz = encoder_out.size(0)
        device = encoder_out.device
        scored_prefix, prefix_score, na_prefix = self.prefix_gen(bsz, seq_len, na_pred, device)
        # scored_prefix:[bsz, seq_len, d]
        na_prefix = na_prefix[:, :, 0]
        scored_prefix, _= self.decoder.embed(na_prefix)

        # 2. Init Hypothesis
        hyps = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(self.sos)  # (B*N, 1)
        sos_emb, _ = self.decoder.embed(hyps)  # (B*N, 1, d)
        input_prefix = torch.cat((sos_emb, scored_prefix), dim=1)  # (B*N, seq_len, d)
        # 2.1 Init scores
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None

        # 2. Decoder forward step by step
        for i in range(1, seq_len + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1).to(device)  # (B*N, i, i)

            last_time_output = hyps  # (B*N, 1)
            last_time_output_emb, _ = self.decoder.embed(last_time_output)
            input_prefix[:, :i, :].copy_(last_time_output_emb)# (B*N, 1, d) K V
            decoder_input, _ = self.emendatory_attn(last_time_output_emb, input_prefix, input_prefix, None)

            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(encoder_out, encoder_mask, decoder_input, input_prefix, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            # Update cache to be consistent with new topk scores / hyps
            cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
            base_cache_index = (torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
            cache_index = base_cache_index + cache_index
            cache = [torch.index_select(c, dim=0, index=cache_index) for c in cache]
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        end_time = time.time()
        self.spend += end_time - start_time
        print ("time spend: ", self.spend, "s:", end_time - start_time)
        return best_hyps, best_scores

    def ctc_greedy_search(self, speech,speech_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1, simulate_streaming=False):
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(speech,
                                                          speech_lengths,
                                                          decoding_chunk_size,
                                                          num_decoding_left_chunks,
                                                          simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def _ctc_prefix_beam_search(self,
                                speech,
                                speech_lengths,
                                beam_size,
                                decoding_chunk_size=-1,
                                num_decoding_left_chunks=-1,
                                simulate_streaming=False,
                                context_graph: ContextGraph = None,
                                ):
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, decoding_chunk_size,
                                                          num_decoding_left_chunks,
                                                          simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score, context_state, context_score))
        cur_hyps = [(tuple(), (0.0, -float('inf'), 0, 0.0))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb, context_state, context_score),
            # default value(-inf, -inf, 0, 0.0)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf'), 0, 0.0))
            # 用于存储下一步的结果, key是前缀, value是(pb, pnb, context_state, context_score)
            # 大小为beam_size x beam_size
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()  # 获得s的概率
                for prefix, (pb, pnb, c_state, c_score) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None  # 前缀prefix的最后一个字符
                    if s == 0:  # blank
                        n_pb, n_pnb, _, _ = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb, _, _ = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb, _, _ = next_hyps[n_prefix]
                        new_c_state, new_c_score = 0, 0
                        if context_graph is not None:
                            new_c_state, new_c_score = context_graph.find_next_state(c_state, s)
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state, c_score + new_c_score)
                    else:  # s不是空且不等于前一个字符
                        n_prefix = prefix + (s,)  # 前缀加上s
                        n_pb, n_pnb, _, _ = next_hyps[n_prefix]  # 得到前缀的概率
                        new_c_state, new_c_score = 0, 0
                        if context_graph is not None:
                            new_c_state, new_c_score = context_graph.find_next_state(c_state, s)
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])  # 把s的概率更新到前缀的概率
                        next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state, c_score + new_c_score)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add([x[1][0], x[1][1]]) + x[1][3],
                               reverse=True)  # 所有的按照概率排列组合排序取beam_size个
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]]) + y[1][3]) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(self,
                               speech: torch.Tensor,
                               speech_lengths: torch.Tensor,
                               beam_size: int,
                               decoding_chunk_size: int = -1,
                               num_decoding_left_chunks: int = -1,
                               simulate_streaming: bool = False,
                               context_graph: ContextGraph = None,
                               ):
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming,
                                               context_graph)
        return hyps[0]

    def attention_rescoring(self,
                            speech: torch.Tensor,
                            speech_lengths: torch.Tensor,
                            tgt_len: int,
                            beam_size: int,
                            decoding_chunk_size: int = -1,
                            num_decoding_left_chunks: int = -1,
                            ctc_weight: float = 0.0,
                            simulate_streaming: bool = False,
                            reverse_weight: float = 0.0,
                            context_graph: ContextGraph = None,
                            ) :
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        # CTC输出的结果称为hyps
        start_time = time.time()
        encoder_out, encoder_mask = self._forward_encoder(speech,
                                                          speech_lengths,
                                                          decoding_chunk_size,
                                                          num_decoding_left_chunks,
                                                          simulate_streaming)  # (B, maxlen, encoder_dim)
        end_time = time.time()
        self.spend += end_time - start_time
        print("time spend: ", self.spend, "s:", end_time - start_time)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, 1, maxlen)  # (B*N, 1, max_len)
        # 1.1 Forward NA decoder
        na_pred = self._forward_na_decoder(encoder_out, encoder_mask, tgt_len + 1)
        seq_len = na_pred.size(1)
        bsz = encoder_out.size(0)
        device = encoder_out.device
        hyps = self.prefix_gen.one_step_forward(bsz, seq_len, na_pred, device)


        assert len(hyps) == beam_size
        # 找到hyps中最长的长度并对序列进行pad, hyp[0]是ctc预测文本的序列,hyp[1]是概率，pad_id是-1
        hyps_pad = pad_sequence([torch.tensor(hyp[0], device=device, dtype=torch.long) for hyp in hyps], True, self.ignore_id)
        # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps], device=device,  dtype=torch.long)  # (beam_size,)
        # 为decoder的输入添加<sos>
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        # (beam_size, max_hyps_len)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining

        hyps_pad, _ = self.decoder.embed(hyps_pad)
        decoder_out, *_ = self.decoder(encoder_out, encoder_mask, hyps_pad, hyps_lens)

        # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i

        return hyps[best_index][0], best_score


    @torch.jit.ignore(drop=True)
    def load_lfmmi_resource(self):
        with open('{}/tokens.txt'.format(self.lfmmi_dir), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                if arr[0] == '<sos/eos>':
                    self.sos_eos_id = int(arr[1])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_compiler = MmiTrainingGraphCompiler(self.lfmmi_dir, device=device, oov="<UNK>", sos_id=self.sos_eos_id, eos_id=self.sos_eos_id, )
        self.lfmmi = LFMMILoss(graph_compiler=self.graph_compiler, den_scale=1, use_pruned_intersect=False, )
        self.word_table = {}
        with open('{}/words.txt'.format(self.lfmmi_dir), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.word_table[int(arr[1])] = arr[0]

    @torch.jit.ignore(drop=True)
    def _calc_lfmmi_loss(self, encoder_out, encoder_mask, text):
        ctc_probs = self.ctc.log_softmax(encoder_out)
        supervision_segments = torch.stack((torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
                                            encoder_mask.squeeze(dim=1).sum(dim=1).to('cpu'),), 1).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(ctc_probs, supervision_segments, allow_truncate=3)
        text = [' '.join([self.word_table[j.item()] for j in i if j != -1]) for i in text]
        loss = self.lfmmi(dense_fsa_vec=dense_fsa_vec, texts=text) / len(text)
        return loss

    def load_hlg_resource_if_necessary(self, hlg, word):
        if not hasattr(self, 'hlg'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hlg = k2.Fsa.from_dict(torch.load(hlg, map_location=device))
        if not hasattr(self.hlg, "lm_scores"):
            self.hlg.lm_scores = self.hlg.scores.clone()
        if not hasattr(self, 'word_table'):
            self.word_table = {}
            with open(word, 'r') as fin:
                for line in fin:
                    arr = line.strip().split()
                    assert len(arr) == 2
                    self.word_table[int(arr[1])] = arr[0]

    @torch.no_grad()
    def hlg_onebest(self,
                    speech: torch.Tensor,
                    speech_lengths: torch.Tensor,
                    decoding_chunk_size: int = -1,
                    num_decoding_left_chunks: int = -1,
                    simulate_streaming: bool = False,
                    hlg: str = '',
                    word: str = '',
                    symbol_table: Dict[str, int] = None,
                    ):
        self.load_hlg_resource_if_necessary(hlg, word)
        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, decoding_chunk_size,
                                                          num_decoding_left_chunks,
                                                          simulate_streaming)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (1, maxlen, vocab_size)
        supervision_segments = torch.stack(
            (torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
             encoder_mask.squeeze(dim=1).sum(dim=1).cpu()),
            1,
        ).to(torch.int32)
        lattice = get_lattice(nnet_output=ctc_probs,
                              decoding_graph=self.hlg,
                              supervision_segments=supervision_segments,
                              search_beam=20,
                              output_beam=7,
                              min_active_states=30,
                              max_active_states=10000,
                              subsampling_factor=4)
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        hyps = get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps

    @torch.no_grad()
    def hlg_rescore(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
            lm_scale: float = 0,
            decoder_scale: float = 0,
            r_decoder_scale: float = 0,
            hlg: str = '',
            word: str = '',
            symbol_table: Dict[str, int] = None,
    ) -> List[int]:
        self.load_hlg_resource_if_necessary(hlg, word)
        device = speech.device
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        supervision_segments = torch.stack(
            (torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
             encoder_mask.squeeze(dim=1).sum(dim=1).cpu()),
            1,
        ).to(torch.int32)
        lattice = get_lattice(nnet_output=ctc_probs,
                              decoding_graph=self.hlg,
                              supervision_segments=supervision_segments,
                              search_beam=20,
                              output_beam=7,
                              min_active_states=30,
                              max_active_states=10000,
                              subsampling_factor=4)
        nbest = Nbest.from_lattice(
            lattice=lattice,
            num_paths=100,
            use_double_scores=True,
            nbest_scale=0.5,
        )
        nbest = nbest.intersect(lattice)
        assert hasattr(nbest.fsa, "lm_scores")
        assert hasattr(nbest.fsa, "tokens")
        assert isinstance(nbest.fsa.tokens, torch.Tensor)

        tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
        tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
        tokens = tokens.remove_values_leq(0)
        hyps = tokens.tolist()

        # cal attention_score
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out_repeat = []
        tot_scores = nbest.tot_scores()
        repeats = [tot_scores[i].shape[0] for i in range(tot_scores.dim0)]
        for i in range(len(encoder_out)):
            encoder_out_repeat.append(encoder_out[i:i + 1].repeat(
                repeats[i], 1, 1))
        encoder_out = torch.concat(encoder_out_repeat, dim=0)
        encoder_mask = torch.ones(encoder_out.size(0),
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        reverse_weight = 0.5
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out

        decoder_scores = torch.tensor([
            sum([decoder_out[i, j, hyps[i][j]] for j in range(len(hyps[i]))])
            for i in range(len(hyps))
        ],
            device=device)  # noqa
        r_decoder_scores = []
        for i in range(len(hyps)):
            score = 0
            for j in range(len(hyps[i])):
                score += r_decoder_out[i, len(hyps[i]) - j - 1, hyps[i][j]]
            score += r_decoder_out[i, len(hyps[i]), self.eos]
            r_decoder_scores.append(score)
        r_decoder_scores = torch.tensor(r_decoder_scores, device=device)

        am_scores = nbest.compute_am_scores()
        ngram_lm_scores = nbest.compute_lm_scores()
        tot_scores = am_scores.values + lm_scale * ngram_lm_scores.values + \
                     decoder_scale * decoder_scores + r_decoder_scale * r_decoder_scores
        ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)
        hyps = get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
            self,
            xs: torch.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
            cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
            self,
            hyps: torch.Tensor,
            hyps_lens: torch.Tensor,
            encoder_out: torch.Tensor,
            reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out

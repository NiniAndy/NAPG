# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
"""Decoder definition."""
from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from wenet.transformer.attention import MultiHeadedAttention
from wenet.NAP_Transformer.decoder_layer import DecoderLayer
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.mask import (subsequent_mask, make_pad_mask)


class NAP_TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, encoder-decoder cross attention is not
                       applied, such as CIF model
    """

    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 input_layer: str = "embed",
                 use_output_layer: bool = True,
                 normalize_before: bool = True,
                 src_attention: bool = True,
                 ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = nn.Sequential(nn.Embedding(vocab_size, attention_dim),
                                       PositionalEncoding(attention_dim, positional_dropout_rate))
        elif input_layer == 'none':
            self.embed = NoPositionalEncoding(attention_dim, positional_dropout_rate)
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.channel_attn_layer = ChannelCrossAttention(1, 4, 1)
        self.decoders = nn.ModuleList([DecoderLayer(attention_dim,
                                                    MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate),
                                                    MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate)
                                                    if src_attention else None,
                                                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                                                    dropout_rate,
                                                    normalize_before,
                                                    ) for _ in range(self.num_blocks)])

    def forward(self, memory, memory_mask, prefix, ys_in_lens):
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        maxlen = prefix.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(prefix.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m  # 对角矩阵
        x = prefix

        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_one_step(self,
                         memory: torch.Tensor,
                         memory_mask: torch.Tensor,
                         refreshed_last_time_ouput_emb,  # [bsz, 1, dim]
                         refreshed_prefix,  # [bsz, se_len, dim]
                         cache: Optional[List[torch.Tensor]] = None,
                         ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = refreshed_last_time_ouput_emb
        kv = refreshed_prefix
        bsz, len, _ = kv.size()
        query_len = torch.full((bsz,), len, dtype=torch.long)
        tgt_mask = ~make_pad_mask(query_len, len).unsqueeze(1).to(kv.device)
        new_cache = []

        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            # if i == 0:
            #     x, _, memory, memory_mask = decoder.inference(x, kv, tgt_mask, memory, memory_mask, cache=c)
            # else:
            x, _, memory, memory_mask = decoder.inference(x, None, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache


class ChannelCrossAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(ChannelCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels2, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # x1: [batch_size, num_channels1, seq_len, dim]
        # x2: [batch_size, num_channels2, height2, width2]

        # Generate query from x1
        q = self.query_conv(x1)

        # Generate key and value from x2
        k = self.key_conv(x2)
        v = self.value_conv(x2)

        # Reshape for matmul
        q = q.view(q.size(0), q.size(1), -1)  # [batch_size, out_channels, seq_len * dim]
        k = k.view(k.size(0), k.size(1), -1)  # [batch_size, out_channels,  seq_len * dim]
        v = v.view(v.size(0), v.size(1), -1)  # [batch_size, out_channels,  seq_len * dim]

        # Transpose k for matmul
        k = k.permute(0, 2, 1)  # [batch_size, seq_len * dim, out_channels]

        # Attention score
        attn_score = torch.matmul(q, k)  # [batch_size, seq_len * dim, height2 * width2]

        # Softmax for normalization
        attn_score = F.softmax(attn_score, dim=-1)

        # Weighted sum of v
        out = torch.matmul(attn_score, v)  # [batch_size, seq_len * dim, out_channels]

        # Reshape back to original size
        out = out.view(q.size(0), q.size(1), x1.size(2), x1.size(3))  # [batch_size, out_channels, seq_len, dim]

        return out

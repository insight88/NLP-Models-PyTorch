# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax

logger = logging.getLogger(__name__)


def Linear(i_dim, o_dim, bias=True, std=0.02):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=std)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.hidden_size, self.all_head_size, std=config.initializer_range)
        self.key = Linear(self.hidden_size, self.all_head_size, std=config.initializer_range)
        self.value = Linear(self.hidden_size, self.all_head_size, std=config.initializer_range)

        self.dropout = Dropout(config.dropout_prob)

        self.fc = Linear(self.hidden_size, self.hidden_size, std=config.initializer_range)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.fc(context_layer)

        return attention_output

class Positionwise_ff(nn.Module):
    def __init__(self, config):
        super(Positionwise_ff, self).__init__()
        self.hidden_size = config.hidden_size
        self.ff_dim = config.ff_dim
        self.dropout_prob = config.dropout_prob

        self.fc_1 = Linear(self.hidden_size, self.ff_dim, std=config.initializer_range)
        self.fc_2 = Linear(self.ff_dim, self.hidden_size, std=config.initializer_range)

        self.dropout = Dropout(self.dropout_prob)
        self.act_fn = ACT2FN[config.act_fn] \
            if isinstance(config.act_fn, str) else config.act_fn

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc_1(x)))
        x = self.fc_2(x)

        return x


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("import basic LayerNorm version"
          "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class LayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

            self.init_weights()

        def init_weights(self):
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class Config(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 hidden_size=768,
                 num_hidden_layers=12,
                 ff_dim=3072,
                 num_heads=12,
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 LN_eps=1e-12,
                 ln_pos=1
                 ):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.act_fn = act_fn
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.ff_dim = ff_dim
            self.num_heads = num_heads
            self.dropout_prob = dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.LN_eps = LN_eps
            self.initializer_range = initializer_range
            self.ln_pos = ln_pos
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        config = Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.dropout = Dropout(config.dropout_prob)

        self.init_weights(config)

    def init_weights(self, config):
        self.word_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        layer = EncoderLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, src_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_mask)

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config, embedding_weights):
        super(Decoder, self).__init__()
        self.vocab_size = config.vocab_size
        layer = DecoderLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.decoder = Linear(embedding_weights.size(1),
                              embedding_weights.size(0),
                              bias=False)
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))

    def forward(self, hidden_states, src, trg_mask, src_trg_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, src, trg_mask, src_trg_mask)

        logits = self.decoder(hidden_states) + self.bias

        return logits    
    
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.attn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)

        self.self_attention = Attention(config)
        self.ffn = Positionwise_ff(config)
        self.ln_pos = config.ln_pos

    def forward(self, src, src_mask):
        # Self-Attention
        h = src
        if not self.ln_pos:
            src = self.attn_norm(src)
        src = self.self_attention(src, src, src, src_mask)
        src = h + src
        if self.ln_pos:
            src = self.attn_norm(src)

        # FFN
        h = src
        if not self.ln_pos:
            src = self.ffn_norm(src)
        src = self.ffn(src)
        src = h + src
        if self.ln_pos:
            src = self.ffn_norm(src)

        return src
    
    
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.attn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.m_attn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)

        self.self_attention = Attention(config)
        self.masked_self_attention = Attention(config)

        self.ffn = Positionwise_ff(config)
        self.ln_pos = config.ln_pos

    def forward(self, trg, src, trg_mask, src_mask):
        # Masked Self-Attention
        h = trg
        if not self.ln_pos:
            trg = self.m_attn_norm(trg)
        trg = self.masked_self_attention(trg, trg, trg, trg_mask)
        trg = h + trg
        if self.ln_pos:
            trg = self.m_attn_norm(trg)

        # Self-Attention
        h = trg
        if not self.ln_pos:
            trg = self.attn_norm(trg)
        trg = self.self_attention(trg, src, src, src_mask)
        trg = h + trg
        if self.ln_pos:
            trg = self.attn_norm(trg)

        # FFN
        h = trg
        if not self.ln_pos:
            trg = self.ffn_norm(trg)
        trg = self.ffn(trg)
        trg = h + trg
        if self.ln_pos:
            trg = self.ffn_norm(trg)

        return trg
    

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, self.embedding.word_embeddings.weight)
        self.loss_fn = CrossEntropyLoss(ignore_index=0)

    def forward(self, src, trg, src_mask, trg_mask, gt=None):
        # Attention 3D Mask
        extended_src_mask = src_mask[:, None, None, :]
        #extended_src_mask = extended_src_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_src_mask = (1.0 - extended_src_mask) * -10000.0

        extended_trg_mask = trg_mask[:, None, :, :]
        #extended_trg_mask = extended_trg_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_trg_mask = (1.0 - extended_trg_mask) * -10000.0

        # Source Embedding & Encoding
        src_emd = self.embedding(src)
        enc_src = self.encoder(src_emd, extended_src_mask)

        # Target Embedding & Decoding
        trg_emd = self.embedding(trg)
        logits = self.decoder(trg_emd, enc_src, extended_trg_mask, extended_src_mask)

        if gt is not None:
            logits = logits.contiguous().view(-1, logits.size(-1))
            gt = gt.contiguous().view(-1)
            loss = self.loss_fn(logits, gt)
            return loss
        else:
            return logits
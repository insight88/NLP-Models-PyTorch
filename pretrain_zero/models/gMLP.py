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
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("import basic LayerNorm version"
          "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm

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

class Atten(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, causal=False, init_eps=1e-3):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal
        self.max_seq_length = dim_seq

        self.norm = LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        self.act = ACT2FN["gelu"]

        init_eps /= dim_seq
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x, gate_res = None):
        
        device = x.device
        res, gate = x.chunk(2, dim = -1)
        
        pad_size = self.max_seq_length - gate.size(1)
        gate = nn.ConstantPad1d((0, pad_size), 0)(gate.transpose(-2, -1)).transpose(-2, -1)
        gate = self.norm(gate)
        
        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:self.max_seq_length, :self.max_seq_length], bias[:self.max_seq_length]
            mask = torch.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)
        
        gate = F.conv1d(gate, weight, bias)
        gate = gate[:, pad_size:]

        if gate_res != None:
            gate = gate + gate_res

        return self.act(gate) * res


class MDConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=9, dilation=1, stride=1, groups=4, bias=True, std=0.02):
        super(MDConv, self).__init__()
        
        self.groups = groups
        self.padding = ((input_size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv1d(
            input_size, 
            output_size, 
            kernel_size=kernel_size, 
            dilation=dilation,
            groups=groups, 
            padding=self.padding, 
            bias=bias
        )
        nn.init.normal_(self.conv.weight, std=std)
        if bias:
            nn.init.constant_(self.conv.bias, 0.)
        
        k = np.arange(3, kernel_size + 1, 2)
        w = []
        for i in range(groups - 1, -1, -1):
            p = (kernel_size - k[i]) // 2
            w.append(torch.cat([torch.zeros(p), torch.ones(k[i]), torch.zeros(p)], dim=0).unsqueeze(0).unsqueeze(0).repeat(output_size//groups, 1, 1))
        
        self.w_mask = nn.Parameter(torch.cat(w, dim=0), requires_grad=False)
        self.conv.weight.data.mul_(self.w_mask.data)
        
    def forward(self, x):
        self.conv.weight.data.mul_(self.w_mask.data)
        return self.conv(x)

    
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.max_seq_length = config.max_position_embeddings
        self.ff_dim = config.ff_dim
        self.n_experts = config.n_experts
        
        self.proj_in = nn.Sequential(
            nn.Linear(self.hidden_size, self.ff_dim),
            nn.GELU()
        )
        
        self.attn = Atten(self.hidden_size, self.ff_dim // 2, self.hidden_size, causal=config.causal)
        self.sgu = SpatialGatingUnit(self.ff_dim, self.max_seq_length, causal=config.causal, init_eps=1e-3)
        self.proj_out = Linear(self.ff_dim // 2, self.hidden_size, std=config.initializer_range)
        
    def forward(self, x):
        
        gate_res = self.attn(x)
        x = self.proj_in(x)
        x = self.sgu(x, gate_res = gate_res)
        x = self.proj_out(x)
        
        return x

class Positionwise_ff(nn.Module):
    def __init__(self, config):
        super(Positionwise_ff, self).__init__()

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.ff_dim = config.ff_dim
        self.n_experts = config.n_experts
        
        self.fc1 = nn.ModuleList()
        for i in range(self.n_experts):
            layer = MDConv(
                input_size = self.hidden_size // self.n_experts, 
                output_size = self.ff_dim // self.n_experts, 
                kernel_size = self.kernel_size,
                groups = self.num_heads,
                dilation = int(i) + 1,
                bias = True,
                std=config.initializer_range
            )
            self.fc1.append(copy.deepcopy(layer))

        self.fc2 = Linear(
            self.ff_dim, 
            self.hidden_size,
            bias = True,
            std=config.initializer_range
        )
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.dropout_prob)

    def forward(self, x):
        
        x = x.transpose(-1, -2)
        x = torch.split(x, self.hidden_size // self.n_experts, dim=1)
        intermediate = torch.cat([self.fc1[i](x[i]) for i in range(self.n_experts)], dim=1).transpose(-1, -2)
        ff_out = self.dropout(self.fc2(self.act_fn(intermediate)))

        return ff_out



class Block(nn.Module):
    def __init__(self, config, i):
        super(Block, self).__init__()
        self.attn = Attention(config)
        self.attention_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        
        self.attention_norm_r = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.ffn_norm_r = LayerNorm(config.hidden_size, eps=config.LN_eps)
        
        self.ffn = None
        if i % config.layer_group == 0:
            self.ffn = Positionwise_ff(config)
            
        self.ln_pos = config.ln_pos

    def forward(self, x, ffn):
        
        ## Attention
        h = x
        if not self.ln_pos:
            x = self.attention_norm(x)
        x = self.attn(x)
        
        x = self.attention_norm(x + h)
        x = self.attention_norm_r(x + h)

        ## FFN
        h = x
        if not self.ln_pos:
            x = self.ffn_norm(x)
        x = ffn(x)
        
        x = self.ffn_norm(x + h)
        x = self.ffn_norm_r(x + h)
        
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.layer = nn.ModuleList()
        for l in range(config.num_hidden_layers):
            self.layer.append(Block(config, l))
            
    def forward(self, hidden_states):
        
        all_encoder_layers = []
        for layer_block in self.layer:
            if layer_block.ffn != None:
                ffn = layer_block.ffn
            
            hidden_states = layer_block(hidden_states, ffn)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class Config(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 embedding_size=128,
                 hidden_size=768,
                 n_experts=8,
                 causal=0,
                 kernel_size=9,
                 num_hidden_layers=12,
                 ff_dim=3072,
                 layer_group=3,
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
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.n_experts = n_experts
            self.causal == True if causal == 1 else False
            self.kernel_size = kernel_size
            self.ff_dim = ff_dim
            self.layer_group = layer_group
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
        self.word_embeddings = Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.LayerNorm = LayerNorm(config.embedding_size, eps=config.LN_eps)
        self.dropout = Dropout(config.dropout_prob)
        
        self.embedding_hidden_mapping_in = None
        if config.embedding_size < config.hidden_size:
            self.embedding_hidden_mapping_in = Linear(config.embedding_size, config.hidden_size, std=config.initializer_range)

        self.init_weights(config)

    def init_weights(self, config):
        self.word_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, input_ids):
        
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        if self.embedding_hidden_mapping_in is not None:
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        
        return embeddings

class PreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(PreTrainingHeads, self).__init__()
        
        self.dense = Linear(config.hidden_size, config.embedding_size, std=config.initializer_range)
        self.act_fn = ACT2FN[config.act_fn]
        self.LayerNorm = LayerNorm(config.embedding_size, eps=config.LN_eps)
        
        self.decoder = Linear(config.embedding_size, config.vocab_size, bias=False, std=config.initializer_range)
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        pdtype = hidden_states.dtype
        hidden_states = self.dense(hidden_states)  # float32
        hidden_states = self.act_fn(hidden_states)  # torch.float16
        
        hidden_states = hidden_states.to(pdtype)  
        hidden_states = self.LayerNorm(hidden_states)  # torch.float32
        hidden_states = self.decoder(hidden_states)  # torch.float32
        return hidden_states  

def _average_query_doc_embeddings(sequence_output, token_type_ids, valid_mask):
    
    t = token_type_ids.to(torch.bool)
    m = valid_mask.to(torch.bool)
    
    query_flags = ~t * m
    doc_flags = t
    
    #query_flags = (token_type_ids==0)*(valid_mask==1)
    #doc_flags = (token_type_ids==1)*(valid_mask==1)

    query_embeddings = sequence_output * query_flags[:,:,None]
    doc_embeddings = sequence_output * doc_flags[:,:,None]

    return query_embeddings, doc_embeddings
    
class Decoder(nn.Module):
    def __init__(self, max_seq_length, std=0.02):
        super(Decoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.linear = Linear(self.max_seq_length, 1, bias=False, std=std)

    def forward(self, x):
        
        pad_size = self.max_seq_length - x.size(1)
        padded = nn.ConstantPad1d((0, pad_size), 0)(x.transpose(-2, -1))
        pooled_x = self.linear(padded).squeeze(-1)
        return pooled_x
    
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        embedding_output = self.embeddings(input_ids)
        encoded_layers = self.encoder(embedding_output)

        return encoded_layers[-1]


class BertMultiTask_SOP(nn.Module):
    def __init__(self, config, train_mode):
        super(BertMultiTask_SOP, self).__init__()
        self.bert = Model(config)
        self.decoder = Linear(config.hidden_size, config.hidden_size, std=config.initializer_range)
        self.activation = nn.Tanh()
        
        self.cls = PreTrainingHeads(config)
        self.vocab_size = config.vocab_size
        self.train_mode = train_mode
        
        self.dropout = Dropout(config.dropout_prob)
        self.qa_outputs = Linear(config.hidden_size, 2, std=config.initializer_range)
        
        self.set_output_embeddings(self.get_input_embeddings())
        
    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings.weight

    def set_output_embeddings(self, new_embeddings):
        self.cls.decoder.weight = new_embeddings

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, start_positions=None, end_positions=None):
        
        if self.train_mode == 'mrc':
            sequence_output = self.bert(input_ids)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                return total_loss
            else:
                return start_logits, end_logits
        else:
            sequence_output = self.bert(input_ids)
            prediction_scores = self.cls(sequence_output)
            
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.decoder(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            seq_relationship_score = self.qa_outputs(pooled_output)
            
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
                sop_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

                return masked_lm_loss, sop_loss
            else:
                return prediction_scores, seq_relationship_score

            
class BertMultiTask_ICT(nn.Module):
    def __init__(self, config, train_mode):
        super(BertMultiTask_ICT, self).__init__()
        self.bert = Model(config)
        self.decoder = Decoder(config.max_position_embeddings, std=config.initializer_range)
        self.cls = PreTrainingHeads(config)
        self.vocab_size = config.vocab_size
        self.train_mode = train_mode
        
        self.dropout = Dropout(config.dropout_prob)
        self.qa_outputs = Linear(config.hidden_size, 2)
        
        self.set_output_embeddings(self.get_input_embeddings())
        
    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings.weight

    def set_output_embeddings(self, new_embeddings):
        self.cls.decoder.weight = new_embeddings

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                c_input_ids=None, c_token_type_ids=None, c_attention_mask=None,
                next_sentence_label=None, start_positions=None, end_positions=None, use_answer=None):
        
        if self.train_mode == 'ict':
            
            s_encode = self.bert(input_ids) #q인코딩
            c_encode = self.bert(c_input_ids) #c인코딩
            s_encode_pooled = self.decoder(s_encode)
            c_encode_pooled = self.decoder(c_encode)

            logits = torch.matmul(s_encode_pooled, c_encode_pooled.transpose(-2, -1)) # 매트릭스 곱
            target = torch.arange(0, input_ids.size(0)).to(torch.long).to(input_ids.device)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target *= use_answer.to(torch.long)

            loss = loss_fct(logits, target)

            return loss
        
        elif self.train_mode == 'encoder':
            
            encode = self.bert(input_ids) #q인코딩
            encode_pooled = self.decoder(encode)

            return encode_pooled
    
        elif self.train_mode == 'mrc':
            
            sequence_output = self.bert(input_ids)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                return total_loss
            else:
                return start_logits, end_logits
        else:
            sequence_output = self.bert(input_ids)
            prediction_scores = self.cls(sequence_output)

            q_x, d_x = _average_query_doc_embeddings(sequence_output, token_type_ids, attention_mask)

            s_encode_pooled = self.decoder(q_x)
            c_encode_pooled = self.decoder(d_x)

            logits = torch.matmul(s_encode_pooled, c_encode_pooled.transpose(-2, -1))

            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))

                #target = torch.from_numpy(np.array([i for i in range(input_ids.size(0))])).long().to(input_ids.device)
                target = torch.arange(input_ids.size(0), device=input_ids.device)
                ICT_loss = loss_fct(logits, target)
                
                return masked_lm_loss, ICT_loss
            else:
                return prediction_scores, logits
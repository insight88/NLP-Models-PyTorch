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

class MDConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=9, groups=4, bias=True, std=0.02):
        super(MDConv, self).__init__()
        
        self.groups = groups
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            input_size, 
            output_size, 
            kernel_size=kernel_size, 
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
        self.n_experts = config.n_experts
        
        self.key = nn.ModuleList()
        for _ in range(self.n_experts):
            layer = MDConv(
                input_size = self.hidden_size // self.n_experts, 
                output_size = self.hidden_size // self.n_experts,
                kernel_size = self.kernel_size,
                groups = self.num_heads,
                bias = True,
                std=config.initializer_range
            )
            self.key.append(copy.deepcopy(layer))
            
        self.o_proj = Linear(self.hidden_size, self.hidden_size, std=config.initializer_range)

        self.dropout = Dropout(config.dropout_prob)
        self.softmax = Softmax(dim=-1)
        
    def forward(self, hidden_states, attention_mask, at=None):
        
        H = torch.split(hidden_states, self.hidden_size // self.n_experts, dim=-1)
        
        pdtype = H[0].dtype
        attention_mask = attention_mask.to(pdtype)
        
        context_layer, attention_scores = [], []
        for i in range(self.n_experts):
            a_s = torch.matmul(H[i], self.key[i](H[i].transpose(-1, -2))) / math.sqrt(H[i].size(-1))
            a_s = a_s + attention_mask + at[i] if at != None else a_s + attention_mask
                
            #a_p = self.dropout(self.softmax(a_s))
            a_p = self.softmax(a_s)
            c_l = torch.matmul(a_p, H[i])
            
            attention_scores.append(a_s)
            context_layer.append(c_l)
        
        context_layer = torch.cat(context_layer, dim=-1)
        context_layer = self.o_proj(context_layer)
        
        return context_layer, attention_scores

class Positionwise_ff(nn.Module):
    def __init__(self, config):
        super(Positionwise_ff, self).__init__()

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.ff_dim = config.ff_dim
        self.n_experts = config.n_experts
        
        self.fc1 = nn.ModuleList()
        for _ in range(self.n_experts):
            layer = MDConv(
                input_size = self.hidden_size // self.n_experts, 
                output_size = self.ff_dim // self.n_experts, 
                kernel_size = self.kernel_size,
                groups = self.num_heads,
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
        #ff_out = self.dropout(self.fc2(self.act_fn(intermediate)))
        ff_out = self.fc2(self.act_fn(intermediate))

        return ff_out


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        
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


class Block(nn.Module):
    def __init__(self, config, i):
        super(Block, self).__init__()
        self.attn = Attention(config)
        self.attention_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        
        self.ffn = None
        if i % config.layer_group == 0:
            self.ffn = Positionwise_ff(config)
            
        self.ln_pos = config.ln_pos

    def forward(self, x, attention_mask, ffn, at=None):
        ## Attention
        h = x
        if not self.ln_pos:
            x = self.attention_norm(x)
        x, at = self.attn(x, attention_mask, at)
        
        x = h + x
        if self.ln_pos:
            x = self.attention_norm(x)

        ## FFN
        h = x
        if not self.ln_pos:
            x = self.ffn_norm(x)
        x = ffn(x)
        x = x + h
        if self.ln_pos:
            x = self.ffn_norm(x)

        return x, at


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for l in range(config.num_hidden_layers):
            self.layer.append(Block(config, l))
            
    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        at = None
        for layer_block in self.layer:
            if layer_block.ffn != None:
                ffn = layer_block.ffn
            
            hidden_states, at = layer_block(hidden_states, attention_mask, ffn, at)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers





class Config(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 hidden_size=768,
                 n_experts=8,
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
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.n_experts = n_experts
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
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        self.dropout = Dropout(config.dropout_prob)

        self.init_weights(config)

    def init_weights(self, config):
        self.word_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        #embeddings = self.dropout(embeddings)

        return embeddings

class PreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(PreTrainingHeads, self).__init__()
        
        self.dense = Linear(config.hidden_size, config.hidden_size, std=config.initializer_range)
        self.act_fn = ACT2FN[config.act_fn]
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.LN_eps)
        
        self.decoder = Linear(config.hidden_size, config.vocab_size, bias=False, std=config.initializer_range)
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
    query_flags = (token_type_ids==0)*(valid_mask==1)
    doc_flags = (token_type_ids==1)*(valid_mask==1)

    query_embeddings = sequence_output * query_flags[:,:,None]
    doc_embeddings = sequence_output * doc_flags[:,:,None]

    return query_embeddings, doc_embeddings
    
class Decoder(nn.Module):
    def __init__(self, max_seq_length, std=0.02):
        super(Decoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.linear = Linear(self.max_seq_length, 1, bias=False, std=std)

    def forward(self, x):

        pdtype = x.dtype
        pad_size = self.max_seq_length - x.size(1)
        if pad_size > 0:
            padding = torch.zeros((x.size(0), pad_size, x.size(2))).to(pdtype).to(x.device)
            x = torch.cat([x, padding], 1)
            
        pooled_x = self.linear(x.transpose(-2, -1)).squeeze(-1)
        return pooled_x
    
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)

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
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        elif self.train_mode == 'mrc':
            sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
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
            sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
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
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        if self.train_mode == 'ict':
            
            if c_attention_mask is None:
                c_attention_mask = torch.ones_like(input_ids)
            if c_token_type_ids is None:
                c_token_type_ids = torch.zeros_like(input_ids)
                
            s_encode = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask) #q인코딩
            c_encode = self.bert(input_ids = c_input_ids, token_type_ids = c_token_type_ids, attention_mask = c_attention_mask) #c인코딩
            s_encode_pooled = self.decoder(s_encode)
            c_encode_pooled = self.decoder(c_encode)

            logits = torch.matmul(s_encode_pooled, c_encode_pooled.transpose(-2, -1)) # 매트릭스 곱
            target = torch.arange(0, input_ids.size(0)).to(torch.long).to(input_ids.device)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target *= use_answer.to(torch.long)

            loss = loss_fct(logits, target)

            return loss
        
        elif self.train_mode == 'encoder':
            
            encode = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask) #q인코딩
            encode_pooled = self.decoder(encode)

            return encode_pooled
    
        elif self.train_mode == 'mrc':
            sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
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
            sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
            prediction_scores = self.cls(sequence_output)

            q_x, d_x = _average_query_doc_embeddings(sequence_output, token_type_ids, attention_mask)

            s_encode_pooled = self.decoder(q_x)
            c_encode_pooled = self.decoder(d_x)

            logits = torch.matmul(s_encode_pooled, c_encode_pooled.transpose(-2, -1))

            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))

                target = torch.from_numpy(np.array([i for i in range(input_ids.size(0))])).long().to(input_ids.device)
                ICT_loss = loss_fct(logits, target)
                
                return masked_lm_loss, ICT_loss
            else:
                return prediction_scores, logits
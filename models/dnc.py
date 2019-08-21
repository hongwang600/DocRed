import functools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from .schmidhuber_dnc import DNC, LSTMController, FeedforwardController


class sDNC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb_size = config.data_word_vec.shape[1]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        self.sent_limit = config.sent_limit
        self.batch_size = config.batch_size
        self.word_size = config.word_size

        self.word_emb.weight.requires_grad = False
        self.use_entity_type = True
        self.use_coreference = True
        self.use_distance = True

        hidden_size = 128
        input_size = config.data_word_vec.shape[1]
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            # self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
        self.rnn = EncoderLSTM(
            input_size=input_size,
            num_units=hidden_size,
            nlayers=3,
            concat=True,
            bidir=True, # Not yet implemented in pytorch-dnc
            dropout=1 - config.keep_prob,
            return_last=False
        )
        self.linear_re = nn.Linear(hidden_size*2, hidden_size)

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
        else:
            self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h):

        sent = self.word_emb(context_idxs)
        if self.use_coreference:
            sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)

        if self.use_entity_type:
            sent = torch.cat([sent, self.ner_emb(context_ner)], dim=-1)

        # sent = torch.cat([sent, context_ch], dim=-1)
        context_output = self.rnn(sent, context_lens)
        context_output = torch.relu(self.linear_re(context_output))


        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)


        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            predict_re = self.bili(s_rep, t_rep)
        else:
            predict_re = self.bili(start_re_output, end_re_output)

        return predict_re


class LockedDropout(nn.Module):
	def __init__(self, dropout):
		super().__init__()
		self.dropout = dropout

	def forward(self, x):
		dropout = self.dropout
		if not self.training:
			return x
		m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
		mask = Variable(m.div_(1 - dropout), requires_grad=False)
		mask = mask.expand_as(x)
		return mask * x


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer_sizes = [512, 256, 256]
        layer_sizes = [256, 256, 256]
        # controller = FeedforwardController(layer_sizes)
        controller = LSTMController(layer_sizes)
        self.dnc = DNC(
            input_size=input_size,
            output_size=num_units+12,
            word_length=512, # config.max_length
            # cell_count=64,
            cell_count=32,
            n_read_heads=8,
            controller=controller,
            batch_first=True,
            clip_controller=20,
            bias=True,
            # mask=False,
            mask=True,
            dealloc_content=True,
            link_sharpness_control=True,
            disable_content_norm=False,
            # mask_min=0.0,
            mask_min=0.1,
            disable_key_masking=False
        )
        output_size = num_units * 2 #num_units if not bidir else num_units * 2
        self.dropout = LockedDropout(dropout)
        self.final_layer = nn.Linear(input_size, output_size).to(self.device)

        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last
        self.memory = None

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        output = self.dnc(input)
        output = self.dropout(output)
        output = self.final_layer(output)
        return output

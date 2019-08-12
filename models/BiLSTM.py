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


class GCNSimpleDecoder(nn.Module):
  """for ontonotes"""
  def __init__(self, output_dim, entity_num):
    super(GCNSimpleDecoder, self).__init__()
    self.entity_num = entity_num
    self.linear = nn.Linear(output_dim, entity_num, bias=False)

    # gcn on label vectors
    self.transform = nn.Linear(output_dim, output_dim, bias=False)


  def forward(self, connection_matrix):
    transform = self.transform(torch.matmul(connection_matrix, self.linear.weight)  / connection_matrix.sum(-1, keepdim=True))
    label_vectors = transform + self.linear.weight # residual
    return label_vectors

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb_size = config.data_word_vec.shape[1]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        self.sent_limit = config.sent_limit
        self.batch_size = config.batch_size
        self.word_size = config.word_size
        self.entity_size = config.max_entity_size

        self.word_emb.weight.requires_grad = False
        self.use_entity_type = True
        self.use_coreference = True
        self.use_distance = True

        # performance is similar with char_embed
        # self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
        # self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))

        # char_dim = config.data_char_vec.shape[1]
        # char_hidden = 100
        # self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

        hidden_size = 128
        input_size = config.data_word_vec.shape[1]
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            # self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        # input_size += char_hidden
        #input_size += hidden_size*2

        self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
        self.sent_enc = EncoderLSTM(self.word_emb_size, hidden_size, 1, True, True, 1 - config.keep_prob, True)
        self.linear_re = nn.Linear(hidden_size*2, hidden_size)
        self.gcn = GCNSimpleDecoder(hidden_size, self.entity_size)

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            self.bili = torch.nn.Bilinear(2*hidden_size+config.dis_size, 2*hidden_size+config.dis_size, config.relation_num)
            #self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
            #self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, hidden_size)
        else:
            self.bili = torch.nn.Bilinear(2*hidden_size, 2*hidden_size, config.relation_num)
            #self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)
            #self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)
        #self.linear_cls = torch.nn.Linear(hidden_size*3, config.relation_num)

    def update_rel_matrix(self, rel_matrix, cooccur_matrix):
        cooccur_matrix = cooccur_matrix.float()
        new_cooccur_matrix = torch.mm(cooccur_matrix, cooccur_matrix)
        entity_size = cooccur_matrix.size(0)
        embed_size = rel_matrix.size(-1)
        expanded_rel_matrix = rel_matrix.unsqueeze(0).expand(entity_size, entity_size, entity_size, embed_size)
        expanded_cooccur_matrix = cooccur_matrix.unsqueeze(-1).unsqueeze(-1).expand(expanded_rel_matrix.size())
        #print(expanded_cooccur_matrix.size(), expanded_rel_matrix.size())
        matrix_to_add = torch.mul( expanded_rel_matrix, expanded_cooccur_matrix.float().cuda())
        base_matrix = rel_matrix.unsqueeze(-2).expand(entity_size, entity_size, entity_size, embed_size)
        new_rel_matrix = (base_matrix + matrix_to_add).sum(-2).squeeze(-2)
        flatten_new_cooccur = new_cooccur_matrix.view(-1)
        valid_cooccur_idx = flatten_new_cooccur > 0
        flatten_new_cooccur[valid_cooccur_idx] = 1 / flatten_new_cooccur[valid_cooccur_idx]
        fraction = flatten_new_cooccur.view(new_cooccur_matrix.size())
        fraction = fraction.unsqueeze(-1).expand(entity_size, entity_size, embed_size)
        new_rel_matrix = new_rel_matrix * fraction
        new_cooccur_matrix = (new_cooccur_matrix > 0).long()
        #print(matrix_to_add.size())
        #assert(False)
        return new_rel_matrix, cooccur_matrix


    def get_rel_matrix(self, sents_embed, cooccur_matrix):
        #cooccur_matrix = torch.from_numpy(cooccur_matrix).long()
        embed_size = sents_embed.size(-1)
        sents_embed = torch.cat([sents_embed, torch.zeros(1, embed_size).cuda()])
        entity_size = len(cooccur_matrix)
        rel_matrix = torch.rand(entity_size, entity_size, embed_size).cuda()
        #print(sents_embed.size(), cooccur_matrix.size())
        #print(cooccur_matrix)
        #print(len(sents_embed))
        #assert(False)
        rel_matrix = sents_embed[cooccur_matrix].contiguous()
        cooccur_matrix = (cooccur_matrix >= 0).long()
        #print(cooccur_matrix)
        #print(rel_matrix.size())
        max_hop = 2
        for i in range(max_hop):
            rel_matrix, cooccur_matrix = self.update_rel_matrix(rel_matrix, cooccur_matrix)
        return rel_matrix

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sents_idx, cooccur_matrix, num_entities, h_t_idx):
        # para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
        # context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        # context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

        #print(sents_idx.size, sents_idx[0])
        #print(len(cooccur_matrix), cooccur_matrix[0])

        '''
        batch_sents = self.word_emb(sents_idx).view(-1, self.word_size, self.word_emb_size)
        sents_lengths = (sents_idx > 0).long().sum(dim=-1)
        doc_lengths = (sents_lengths > 0).long().sum(dim=-1)
        #print(doc_lengths)
        #print(cooccur_matrix[0][0])
        #print(context_idxs.size(), sents_idx.size(), doc_lengths.size())
        #print(sents_idx[0][0])
        #assert(False)
        flatten_sents_lengths = sents_lengths.view(-1)
        #print(doc_lengths)
        valid_sents = batch_sents[flatten_sents_lengths>0]
        valid_sents_lengths = flatten_sents_lengths[flatten_sents_lengths>0]
        #print(valid_sents.size())
        #print(sents_lengths, sents_lengths.size())
        #print(batch_sents.size())
        sents_embed = self.sent_enc(valid_sents, valid_sents_lengths)
        doc_sents_embed_list = []
        start_idx = 0
        for doc_len in doc_lengths:
            doc_sents_embed_list.append(sents_embed[start_idx:start_idx+doc_len])
            start_idx += doc_len
        #print(len(doc_sents_embed_list), doc_sents_embed_list[0].size())
        #print(sents_embed.size())
        rel_matrix = []
        ins_rel_embed = []
        for i in range(len(doc_sents_embed_list)):
            rel_matrix.append(self.get_rel_matrix(doc_sents_embed_list[i], cooccur_matrix[i][:num_entities[i], :num_entities[i]]))
            ins_rel_embed.append(rel_matrix[-1][h_t_idx[i][:,0],h_t_idx[i][:,1]])
        #print(rel_matrix.size())
        ins_rel_embed = torch.stack(ins_rel_embed)
        #print(ins_rel_embed.size())
        #assert(False)
        '''
        corr_entity_embed = self.gcn(cooccur_matrix)
        batch_size, h_t_size, _ = h_t_idx.size()
        dim_1_idx = torch.arange(batch_size).view(-1,1).expand(batch_size, h_t_size)
        #print(dim_1_idx[:5,:5])
        #print(corr_entity_embed)
        corr_h_embed = corr_entity_embed[dim_1_idx, h_t_idx[:,:,0]]
        corr_t_embed = corr_entity_embed[dim_1_idx, h_t_idx[:,:,1]]
        #print(corr_entity_embed.size(), corr_h_embed.size())

        sent = self.word_emb(context_idxs)
        if self.use_coreference:
            sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)

        if self.use_entity_type:
            sent = torch.cat([sent, self.ner_emb(context_ner)], dim=-1)

        # sent = torch.cat([sent, context_ch], dim=-1)
        context_output = self.rnn(sent, context_lens)

        context_output = torch.relu(self.linear_re(context_output))


        #print(h_mapping.size(), context_output.size())
        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)
        #print('end_rel',end_re_output.size())

        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            #rel_rep = self.bili(s_rep, t_rep)
            #rel_rep = torch.cat([rel_rep, ins_rel_embed], dim=-1)
            #predict_re = self.linear_cls(rel_rep)
            s_rep = torch.cat([s_rep, corr_h_embed], dim=-1)
            t_rep = torch.cat([t_rep, corr_t_embed], dim=-1)
            predict_re = self.bili(s_rep, t_rep)
        else:
            #rel_rep = self.bili(start_re_output, end_re_output)
            #rel_rep = torch.cat([rel_rep, ins_rel_embed], dim=-1)
            #predict_re = self.linear_cls(rel_rep)
            #s_rep = torch.cat([start_re_output, ins_rel_embed], dim=-1)
            #t_rep = torch.cat([end_re_output, ins_rel_embed], dim=-1)
            s_rep = start_re_output
            t_rep = end_re_output
            s_rep = torch.cat([s_rep, corr_h_embed], dim=-1)
            t_rep = torch.cat([t_rep, corr_t_embed], dim=-1)
            predict_re = self.bili(s_rep, t_rep)

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

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        lengths = torch.tensor(input_lengths)
        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, hidden)


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        if self.concat:
            return torch.cat(outputs, dim=2)

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        lengths = torch.tensor(input_lengths)
        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        #if input_lengths is not None:
        #    lens = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, (hidden, c) = self.rnns[i](output, (hidden, c))


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        if self.concat:
            return torch.cat(outputs, dim=-1)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

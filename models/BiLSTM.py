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
from models.attention import SimpleEncoder
from pytorch_transformers import *
from torch.nn.utils.rnn import pad_sequence


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

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
        #hidden_size = 768
        bert_hidden_size = 768
        input_size = config.data_word_vec.shape[1]
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            # self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        # input_size += char_hidden

        self.input_size = input_size

        #self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
        #self.sent_rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
        #self.att_enc = SimpleEncoder(input_size, 4, 1)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.linear_re = nn.Linear(bert_hidden_size+config.coref_size+config.entity_type_size, hidden_size)
        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)
        #self.ent_att_enc = SimpleEncoder(hidden_size*2, 4, 1)

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
            #self.linear_re = nn.Linear((hidden_size+config.dis_size)*2, config.relation_num)
        else:
            #self.linear_re = nn.Linear(hidden_size*2, config.relation_num)
            self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

    def mask_lengths(self, batch_size, doc_size, lengths):
        masks = torch.ones(batch_size, doc_size).cuda()
        index_matrix = torch.arange(0, doc_size).expand(batch_size, -1).cuda()
        index_matrix = index_matrix.long()
        #doc_lengths = torch.tensor(lengths).view(-1,1)
        doc_lengths = lengths.view(-1, 1)
        doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
        masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
        return masks

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks, context_starts):
        # para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
        # context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        # context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

        '''
        sent = self.word_emb(context_idxs)
        if self.use_coreference:
            sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)

        if self.use_entity_type:
            sent = torch.cat([sent, self.ner_emb(context_ner)], dim=-1)

        batch_size, sent_limit, word_size = sent_idxs.size()
        #print(batch_size, h_t_num, word_size)
        i_ = torch.arange(batch_size).view(-1,1,1).expand(-1,sent_limit,word_size)
        #print(i_[:5,:5,:5])
        batch_sents = sent[i_, sent_idxs]
        #print(batch_sents.size())
        batch_sents = batch_sents.view(-1, word_size, self.input_size)
        flatten_sents_lengths = sent_lengths.view(-1)
        sent_emb = self.rnn(batch_sents, flatten_sents_lengths)
        #print(sent_emb.size())
        sent_emb = sent_emb.view(batch_size, sent_limit*word_size, -1)
        i_ = torch.arange(batch_size).view(-1,1).expand(reverse_sent_idxs.size())
        sent_emb = sent_emb[i_, reverse_sent_idxs]
        '''

        # sent = torch.cat([sent, context_ch], dim=-1)
        #print(context_idxs.size())
        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
        #print('output_1',context_output[0])
        context_output = [layer[starts.nonzero().squeeze(1)]
                   for layer, starts in zip(context_output, context_starts)]
        #print('output_2',context_output[0])
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        #print('output_3',context_output[0])
        #print(context_output.size())
        context_output = torch.nn.functional.pad(context_output,(0,0,0,context_idxs.size(-1)-context_output.size(-2)))
        #print('output_4',context_output[0])
        #print(context_output.size())
        #assert(False)
        #context_output = self.rnn(sent, context_lens)
        #context_output = torch.cat([context_output, sent_emb], dim=-1)
        #context_output = sent_emb

        '''
        batch_size, doc_size = context_idxs.size()[:2]
        mask = self.mask_lengths(batch_size, doc_size, context_lens)
        context_output = self.att_enc(sent, mask)
        '''

        '''
        ent_mask = torch.zeros(context_idxs.size()[:2]).cuda()
        ent_mask[pos>0] = 1
        context_output = self.ent_att_enc(context_output, ent_mask)
        '''

        #context_output = torch.relu(self.linear_re(context_output))
        #print('output_4',context_output[0])
        '''
        if self.use_coreference:
            context_output = torch.cat([context_output, self.entity_embed(pos)], dim=-1)

        if self.use_entity_type:
            context_output = torch.cat([context_output, self.ner_emb(context_ner)], dim=-1)
        '''
        context_output = self.linear_re(context_output)


        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)


        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            #rep = torch.cat([s_rep, t_rep], dim=-1)
            #rep = s_rep - t_rep
            #predict_re = self.linear_re(rep)
            predict_re = self.bili(s_rep, t_rep)
        else:
            #rep = s_rep - t_rep
            #rep = torch.cat([s_rep, t_rep], dim=-1)
            #predict_re = self.linear_re(rep)
            predict_re = self.bili(start_re_output, end_re_output)

        #print(predict_re[0])

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
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]




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
        lens[lens==0] = 1

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

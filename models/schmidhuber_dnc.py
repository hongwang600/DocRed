# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn.init as init
import functools
import math


def oneplus(t):
    return F.softplus(t, 1, 20) + 1.0

def get_next_tensor_part(src, dims, prev_pos=0):
    if not isinstance(dims, list):
        dims=[dims]
    n = functools.reduce(lambda x, y: x * y, dims)
    data = src.narrow(-1, prev_pos, n)
    return data.contiguous().view(list(data.size())[:-1] + dims) if len(dims)>1 else data, prev_pos + n

def split_tensor(src, shapes):
    pos = 0
    res = []
    for s in shapes:
        d, pos = get_next_tensor_part(src, s, pos)
        res.append(d)
    return res

def dict_get(dict,name):
    return dict.get(name) if dict is not None else None


def dict_append(dict, name, val):
    if dict is not None:
        l = dict.get(name)
        if not l:
            l = []
            dict[name] = l
        l.append(val)


def init_debug(debug, initial):
    if debug is not None and not debug:
        debug.update(initial)

def merge_debug_tensors(d, dim):
    if d is not None:
        for k, v in d.items():
            if isinstance(v, dict):
                merge_debug_tensors(v, dim)
            elif isinstance(v, list):
                d[k] = torch.stack(v, dim)


def linear_reset(module, gain=1.0):
    assert isinstance(module, torch.nn.Linear)
    init.xavier_uniform_(module.weight, gain=gain)
    s = module.weight.size(1)
    if module.bias is not None:
        module.bias.data.zero_()

_EPS = 1e-6

class AllocationManager(torch.nn.Module):
    def __init__(self):
        super(AllocationManager, self).__init__()
        self.usages = None
        self.zero_usages = None
        self.debug_sequ_init = False
        self.one = None

    def _init_sequence(self, prev_read_distributions):
        # prev_read_distributions size is [batch, n_heads, cell count]
        s = prev_read_distributions.size()
        if self.zero_usages is None or list(self.zero_usages.size())!=[s[0],s[-1]]:
            self.zero_usages = torch.zeros(s[0], s[-1], device = prev_read_distributions.device)
            if self.debug_sequ_init:
                self.zero_usages += torch.arange(0, s[-1]).unsqueeze(0) * 1e-10

        self.usages = self.zero_usages

    def _init_consts(self, device):
        if self.one is None:
            self.one = torch.ones(1, device=device)

    def new_sequence(self):
        self.usages = None

    def update_usages(self, prev_write_distribution, prev_read_distributions, free_gates):
        # Read distributions shape: [batch, n_heads, cell count]
        # Free gates shape: [batch, n_heads]

        self._init_consts(prev_read_distributions.device)
        phi = torch.addcmul(self.one, -1, free_gates.unsqueeze(-1), prev_read_distributions).prod(-2)
        # Phi is the free tensor, sized [batch, cell count]

        # If memory usage counter if doesn't exists
        if self.usages is None:
            self._init_sequence(prev_read_distributions)
            # in first timestep nothing is written or read yet, so we don't need any further processing
        else:
            self.usages = torch.addcmul(self.usages, 1, prev_write_distribution.detach(), (1 - self.usages)) * phi

        return phi

    def forward(self, prev_write_distribution, prev_read_distributions, free_gates):
        phi = self.update_usages(prev_write_distribution, prev_read_distributions, free_gates)
        sorted_usage, free_list = (self.usages*(1.0-_EPS)+_EPS).sort(-1)

        u_prod = sorted_usage.cumprod(-1)
        one_minus_usage = 1.0 - sorted_usage
        sorted_scores = torch.cat([one_minus_usage[..., 0:1], one_minus_usage[..., 1:] * u_prod[..., :-1]], dim=-1)

        return sorted_scores.clone().scatter_(-1, free_list, sorted_scores), phi


class ContentAddressGenerator(torch.nn.Module):
    def __init__(self, disable_content_norm=False, mask_min=0.0, disable_key_masking=False):
        super(ContentAddressGenerator, self).__init__()
        self.disable_content_norm = disable_content_norm
        self.mask_min = mask_min
        self.disable_key_masking = disable_key_masking

    def forward(self, memory, keys, betas, mask=None):
        # Memory shape [batch, cell count, word length]
        # Key shape [batch, n heads*, word length]
        # Betas shape [batch, n heads]
        if mask is not None and self.mask_min != 0:
            mask = mask * (1.0-self.mask_min) + self.mask_min

        single_head = keys.dim() == 2
        if single_head:
            # Single head
            keys = keys.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        memory = memory.unsqueeze(1)
        keys = keys.unsqueeze(-2)

        if mask is not None:
            mask = mask.unsqueeze(-2)
            memory = memory * mask
            if not self.disable_key_masking:
                keys = keys * mask

        # Shape [batch, n heads, cell count]
        norm = keys.norm(dim=-1)
        if not self.disable_content_norm:
            norm = norm * memory.norm(dim=-1)

        scores = (memory * keys).sum(-1) / (norm + _EPS)
        scores *= betas.unsqueeze(-1)

        res = F.softmax(scores, scores.dim()-1)
        return res.squeeze(1) if single_head else res


class WriteHead(torch.nn.Module):
    @staticmethod
    def create_write_archive(write_dist, erase_vector, write_vector, phi):
        return dict(write_dist=write_dist, erase_vector=erase_vector, write_vector=write_vector, phi=phi)

    def __init__(self, dealloc_content=True, disable_content_norm=False, mask_min=0.0, disable_key_masking=False):
        super(WriteHead, self).__init__()
        self.write_content_generator = ContentAddressGenerator(disable_content_norm, mask_min=mask_min, disable_key_masking=disable_key_masking)
        self.allocation_manager = AllocationManager()
        self.last_write = None
        self.dealloc_content = dealloc_content
        self.new_sequence()

    def new_sequence(self):
        self.last_write = None
        self.allocation_manager.new_sequence()

    @staticmethod
    def mem_update(memory, write_dist, erase_vector, write_vector, phi):
        # In original paper the memory content is NOT deallocated, which makes content based addressing basically
        # unusable when multiple similar steps should be done. The reason for this is that the memory contents are
        # still there, so the lookup will find them, unless an allocation clears it before the next search, which is
        # completely random. So I'm arguing that erase matrix should also take in account the free gates (multiply it
        # with phi)
        write_dist = write_dist.unsqueeze(-1)

        erase_matrix = 1.0 - write_dist * erase_vector.unsqueeze(-2)
        if phi is not None:
            erase_matrix = erase_matrix * phi.unsqueeze(-1)

        update_matrix = write_dist * write_vector.unsqueeze(-2)
        return memory * erase_matrix + update_matrix

    def forward(self, memory, write_content_key, write_beta, erase_vector, write_vector, alloc_gate, write_gate,
                free_gates, prev_read_dist, write_mask=None, debug=None):
        last_w_dist = self.last_write["write_dist"] if self.last_write is not None else None

        content_dist = self.write_content_generator(memory, write_content_key, write_beta, mask = write_mask)
        alloc_dist, phi = self.allocation_manager(last_w_dist, prev_read_dist, free_gates)

        # Shape [batch, cell count]
        write_dist = write_gate * (alloc_gate * alloc_dist + (1-alloc_gate)*content_dist)
        self.last_write = WriteHead.create_write_archive(write_dist, erase_vector, write_vector, phi if self.dealloc_content else None)

        dict_append(debug, "alloc_dist", alloc_dist)
        dict_append(debug, "write_dist", write_dist)
        dict_append(debug, "mem_usages", self.allocation_manager.usages)
        dict_append(debug, "free_gates", free_gates)
        dict_append(debug, "write_betas", write_beta)
        dict_append(debug, "write_gate", write_gate)
        dict_append(debug, "write_vector", write_vector)
        dict_append(debug, "alloc_gate", alloc_gate)
        dict_append(debug, "erase_vector", erase_vector)
        if write_mask is not None:
            dict_append(debug, "write_mask", write_mask)

        return WriteHead.mem_update(memory, **self.last_write)

class RawWriteHead(torch.nn.Module):
    def __init__(self, n_read_heads, word_length, use_mask=False, dealloc_content=True, disable_content_norm=False,
                 mask_min=0.0, disable_key_masking=False):
        super(RawWriteHead, self).__init__()
        self.write_head = WriteHead(dealloc_content = dealloc_content, disable_content_norm = disable_content_norm,
                                    mask_min=mask_min, disable_key_masking=disable_key_masking)
        self.word_length = word_length
        self.n_read_heads = n_read_heads
        self.use_mask = use_mask
        self.input_size = 3*self.word_length + self.n_read_heads + 3 + (self.word_length if use_mask else 0)

    def new_sequence(self):
        self.write_head.new_sequence()

    def get_prev_write(self):
        return self.write_head.last_write

    def forward(self, memory, nn_output, prev_read_dist, debug):
        shapes = [[self.word_length]] * (4 if self.use_mask else 3) + [[self.n_read_heads]] + [[1]] * 3
        tensors = split_tensor(nn_output, shapes)

        if self.use_mask:
            write_mask = torch.sigmoid(tensors[0])
            tensors=tensors[1:]
        else:
            write_mask = None

        write_content_key, erase_vector, write_vector, free_gates, write_beta, alloc_gate, write_gate = tensors

        erase_vector = torch.sigmoid(erase_vector)
        free_gates = torch.sigmoid(free_gates)
        write_beta = oneplus(write_beta)
        alloc_gate = torch.sigmoid(alloc_gate)
        write_gate = torch.sigmoid(write_gate)

        return self.write_head(memory, write_content_key, write_beta, erase_vector, write_vector,
                               alloc_gate, write_gate, free_gates, prev_read_dist, debug=debug, write_mask=write_mask)

    def get_neural_input_size(self):
        return self.input_size


class TemporalMemoryLinkage(torch.nn.Module):
    def __init__(self):
        super(TemporalMemoryLinkage, self).__init__()
        self.temp_link_mat = None
        self.precedence_weighting = None
        self.diag_mask = None

        self.initial_temp_link_mat = None
        self.initial_precedence_weighting = None
        self.initial_diag_mask = None
        self.initial_shape = None

    def new_sequence(self):
        self.temp_link_mat = None
        self.precedence_weighting = None
        self.diag_mask = None

    def _init_link(self, w_dist):
        s = list(w_dist.size())
        if self.initial_shape is None or s != self.initial_shape:
            self.initial_temp_link_mat = torch.zeros(s[0], s[-1], s[-1]).to(w_dist.device)
            self.initial_precedence_weighting = torch.zeros(s[0], s[-1]).to(w_dist.device)
            self.initial_diag_mask = (1.0 - torch.eye(s[-1]).unsqueeze(0).to(w_dist)).detach()

        self.temp_link_mat = self.initial_temp_link_mat
        self.precedence_weighting = self.initial_precedence_weighting
        self.diag_mask = self.initial_diag_mask

    def _update_precedence(self, w_dist):
        # w_dist shape: [ batch, cell count ]
        self.precedence_weighting = (1.0 - w_dist.sum(-1, keepdim=True)) * self.precedence_weighting + w_dist

    def _update_links(self, w_dist):
        if self.temp_link_mat is None:
            self._init_link(w_dist)

        wt_i = w_dist.unsqueeze(-1)
        wt_j = w_dist.unsqueeze(-2)
        pt_j = self.precedence_weighting.unsqueeze(-2)

        self.temp_link_mat = ((1 - wt_i - wt_j) * self.temp_link_mat + wt_i * pt_j) * self.diag_mask

    def forward(self, w_dist, prev_r_dists, debug = None):
        self._update_links(w_dist)
        self._update_precedence(w_dist)

        # prev_r_dists shape: [ batch, n heads, cell count ]
        # Emulate matrix-vector multiplication by broadcast and sum. This way we don't need to transpose the matrix
        tlm_multi_head = self.temp_link_mat.unsqueeze(1)

        forward_dist = (tlm_multi_head * prev_r_dists.unsqueeze(-2)).sum(-1)
        backward_dist = (tlm_multi_head * prev_r_dists.unsqueeze(-1)).sum(-2)

        dict_append(debug, "forward_dists", forward_dist)
        dict_append(debug, "backward_dists", backward_dist)
        dict_append(debug, "precedence_weights", self.precedence_weighting)

        # output shapes [ batch, n_heads, cell_count ]
        return forward_dist, backward_dist


class ReadHead(torch.nn.Module):
    def __init__(self, disable_content_norm=False, mask_min=0.0, disable_key_masking=False):
        super(ReadHead, self).__init__()
        self.content_addr_generator = ContentAddressGenerator(disable_content_norm=disable_content_norm,
                                                              mask_min=mask_min,
                                                              disable_key_masking=disable_key_masking)
        self.read_dist = None
        self.read_data = None
        self.new_sequence()

    def new_sequence(self):
        self.read_dist = None
        self.read_data = None

    def forward(self, memory, read_content_keys, read_betas, forward_dist, backward_dist, gates, read_mask=None, debug=None):
        content_dist = self.content_addr_generator(memory, read_content_keys, read_betas, mask=read_mask)

        self.read_dist = backward_dist * gates[..., 0:1] + content_dist * gates[...,1:2] + forward_dist * gates[..., 2:]

        # memory shape: [ batch, cell count, word_length ]
        # read_dist shape: [ batch, n heads, cell count ]
        # result shape: [ batch, n_heads, word_length ]
        self.read_data = (memory.unsqueeze(1) * self.read_dist.unsqueeze(-1)).sum(-2)

        dict_append(debug, "content_dist", content_dist)
        dict_append(debug, "balance", gates)
        dict_append(debug, "read_dist", self.read_dist)
        dict_append(debug, "read_content_keys", read_content_keys)
        if read_mask is not None:
            dict_append(debug, "read_mask", read_mask)
        dict_append(debug, "read_betas", read_betas.unsqueeze(-2))
        if read_mask is not None:
            dict_append(debug, "read_mask", read_mask)

        return self.read_data


class RawReadHead(torch.nn.Module):
    def __init__(self, n_heads, word_length, use_mask=False, disable_content_norm=False, mask_min=0.0,
                 disable_key_masking=False):
        super(RawReadHead, self).__init__()
        self.read_head = ReadHead(disable_content_norm=disable_content_norm, mask_min=mask_min,
                                  disable_key_masking=disable_key_masking)
        self.n_heads = n_heads
        self.word_length = word_length
        self.use_mask = use_mask
        self.input_size = self.n_heads * (self.word_length*(2 if use_mask else 1) + 3 + 1)

    def get_prev_dist(self, memory):
        if self.read_head.read_dist is not None:
            return self.read_head.read_dist
        else:
            m_shape = memory.size()
            return torch.zeros(m_shape[0], self.n_heads, m_shape[1]).to(memory)

    def get_prev_data(self, memory):
        if self.read_head.read_data is not None:
            return self.read_head.read_data
        else:
            m_shape = memory.size()
            return torch.zeros(m_shape[0], self.n_heads, m_shape[-1]).to(memory)

    def new_sequence(self):
        self.read_head.new_sequence()

    def forward(self, memory, nn_output, forward_dist, backward_dist, debug):
        shapes = [[self.n_heads, self.word_length]] * (2 if self.use_mask else 1) + [[self.n_heads], [self.n_heads, 3]]
        tensors = split_tensor(nn_output, shapes)

        if self.use_mask:
            read_mask = torch.sigmoid(tensors[0])
            tensors = tensors[1:]
        else:
            read_mask = None

        keys, betas, gates = tensors

        betas = oneplus(betas)
        gates = F.softmax(gates, gates.dim()-1)

        return self.read_head(memory, keys, betas, forward_dist, backward_dist, gates, debug=debug, read_mask=read_mask)

    def get_neural_input_size(self):
        return self.input_size


class DistSharpnessEnhancer(torch.nn.Module):
    def __init__(self, n_heads):
        super(DistSharpnessEnhancer, self).__init__()
        self.n_heads = n_heads if isinstance(n_heads, list) else [n_heads]
        self.n_data = sum(self.n_heads)

    def forward(self, nn_input, *dists):
        assert len(dists) == len(self.n_heads)
        nn_input = oneplus(nn_input[..., :self.n_data])
        factors = split_tensor(nn_input, self.n_heads)

        res = []
        for i, d in enumerate(dists):
            s = list(d.size())
            ndim = d.dim()
            f  = factors[i]
            if ndim==2:
                assert self.n_heads[i]==1
            elif ndim==3:
                f = f.unsqueeze(-1)
            else:
                assert False

            d += _EPS
            d = d / d.max(dim=-1, keepdim=True)[0]
            d = d.pow(f)
            d = d / d.sum(dim=-1, keepdim=True)
            res.append(d)
        return res

    def get_neural_input_size(self):
        return self.n_data


class DNC(torch.nn.Module):
    def __init__(self, input_size, output_size, word_length, cell_count, n_read_heads, controller, batch_first=False, clip_controller=20,
                 bias=True, mask=False, dealloc_content=True, link_sharpness_control=True, disable_content_norm=False,
                 mask_min=0.0, disable_key_masking=False):
        super(DNC, self).__init__()

        self.clip_controller = clip_controller

        self.read_head = RawReadHead(n_read_heads, word_length, use_mask=mask, disable_content_norm=disable_content_norm,
                                     mask_min=mask_min, disable_key_masking=disable_key_masking)
        self.write_head = RawWriteHead(n_read_heads, word_length, use_mask=mask, dealloc_content=dealloc_content,
                                       disable_content_norm=disable_content_norm, mask_min=mask_min,
                                       disable_key_masking=disable_key_masking)
        self.temporal_link = TemporalMemoryLinkage()
        self.sharpness_control = DistSharpnessEnhancer([n_read_heads, n_read_heads]) if link_sharpness_control else None

        in_size = input_size + n_read_heads * word_length
        control_channels = self.read_head.get_neural_input_size() + self.write_head.get_neural_input_size() +\
                           (self.sharpness_control.get_neural_input_size() if self.sharpness_control is not None else 0)

        self.controller = controller
        controller.init(in_size)
        self.controller_to_controls = torch.nn.Linear(controller.get_output_size(), control_channels, bias=bias)
        self.controller_to_out = torch.nn.Linear(controller.get_output_size(), output_size, bias=bias)
        self.read_to_out = torch.nn.Linear(word_length * n_read_heads, output_size, bias=bias)

        self.cell_count = cell_count
        self.word_length = word_length

        self.memory = None
        self.reset_parameters()

        self.batch_first = batch_first
        self.zero_mem_tensor = None

    def reset_parameters(self):
        linear_reset(self.controller_to_controls)
        linear_reset(self.controller_to_out)
        linear_reset(self.read_to_out)
        self.controller.reset_parameters()

    def _step(self, in_data, debug):
        init_debug(debug, {
            "read_head": {},
            "write_head": {},
            "temporal_links": {}
        })

        # input shape: [ batch, channels ]
        batch_size = in_data.size(0)

        # run the controller
        prev_read_data = self.read_head.get_prev_data(self.memory).view([batch_size, -1])

        control_data = self.controller(torch.cat([in_data, prev_read_data], -1))

        # memory ops
        controls = self.controller_to_controls(control_data).contiguous()
        controls = controls.clamp(-self.clip_controller, self.clip_controller) if self.clip_controller is not None else controls

        shapes = [[self.write_head.get_neural_input_size()], [self.read_head.get_neural_input_size()]]
        if self.sharpness_control is not None:
            shapes.append(self.sharpness_control.get_neural_input_size())

        tensors = split_tensor(controls, shapes)

        write_head_control, read_head_control = tensors[:2]
        tensors = tensors[2:]

        prev_read_dist = self.read_head.get_prev_dist(self.memory)

        self.memory = self.write_head(self.memory, write_head_control, prev_read_dist, debug=dict_get(debug,"write_head"))

        prev_write = self.write_head.get_prev_write()
        forward_dist, backward_dist = self.temporal_link(prev_write["write_dist"] if prev_write is not None else None, prev_read_dist, debug=dict_get(debug, "temporal_links"))

        if self.sharpness_control is not None:
            forward_dist, backward_dist = self.sharpness_control(tensors[0], forward_dist, backward_dist)

        read_data = self.read_head(self.memory, read_head_control, forward_dist, backward_dist, debug=dict_get(debug,"read_head"))

        # output:
        return self.controller_to_out(control_data) + self.read_to_out(read_data.view(batch_size,-1))

    def _mem_init(self, batch_size, device):
        if self.zero_mem_tensor is None or self.zero_mem_tensor.size(0)!=batch_size:
            self.zero_mem_tensor = torch.zeros(batch_size, self.cell_count, self.word_length).to(device)

        self.memory = self.zero_mem_tensor

    def forward(self, in_data, debug=None):
        self.write_head.new_sequence()
        self.read_head.new_sequence()
        self.temporal_link.new_sequence()
        self.controller.new_sequence()

        self._mem_init(in_data.size(0 if self.batch_first else 1), in_data.device)

        out_tsteps = []

        if self.batch_first:
            # input format: batch, time, channels
            for t in range(in_data.size(1)):
                out_tsteps.append(self._step(in_data[:,t], debug))
        else:
            # input format: time, batch, channels
            for t in range(in_data.size(0)):
                out_tsteps.append(self._step(in_data[t], debug))

        merge_debug_tensors(debug, dim=1 if self.batch_first else 0)
        return torch.stack(out_tsteps, dim=1 if self.batch_first else 0)

class LSTMController(torch.nn.Module):
    def __init__(self, layer_sizes, out_from_all_layers=True):
        super(LSTMController, self).__init__()
        self.out_from_all_layers = out_from_all_layers
        self.layer_sizes = layer_sizes
        self.states = None
        self.outputs = None

    def new_sequence(self):
        self.states = [None] * len(self.layer_sizes)
        self.outputs = [None] * len(self.layer_sizes)

    def reset_parameters(self):
        def init_layer(l, index):
            size = self.layer_sizes[index]
            # Initialize all matrices to sigmoid, just data input to tanh
            a=math.sqrt(3.0)*self.stdevs[i]
            l.weight.data[0:-size].uniform_(-a,a)
            a*=init.calculate_gain("tanh")
            l.weight.data[-size:].uniform_(-a, a)
            if l.bias is not None:
                l.bias.data[self.layer_sizes[i]:].fill_(0)
                # init forget gate to large number.
                l.bias.data[:self.layer_sizes[i]].fill_(1)

        # xavier init merged input weights
        for i in range(len(self.layer_sizes)):
            init_layer(self.in_to_all[i], i)
            init_layer(self.out_to_all[i], i)
            if i>0:
                init_layer(self.prev_to_all[i-1], i)

    def _add_modules(self, name, m_list):
        for i, m in enumerate(m_list):
            self.add_module("%s_%d" % (name,i), m)

    def init(self, input_size):
        self.layer_sizes = self.layer_sizes

        # Xavier init: input to all gates is layers_sizes[i-1] + layer_sizes[i] + input_size -> layer_size big.
        # So use xavier init according to this.
        self.input_sizes = [(self.layer_sizes[i - 1] if i>0 else 0) + self.layer_sizes[i] + input_size
                            for i in range(len(self.layer_sizes))]
        self.stdevs = [math.sqrt(2.0 / (self.layer_sizes[i] + self.input_sizes[i])) for i in range(len(self.layer_sizes))]
        self.in_to_all= [torch.nn.Linear(input_size, 4*self.layer_sizes[i]) for i in range(len(self.layer_sizes))]
        self.out_to_all = [torch.nn.Linear(self.layer_sizes[i], 4 * self.layer_sizes[i], bias=False) for i in range(len(self.layer_sizes))]
        self.prev_to_all = [torch.nn.Linear(self.layer_sizes[i-1], 4 * self.layer_sizes[i], bias=False) for i in range(1,len(self.layer_sizes))]

        self._add_modules("in_to_all", self.in_to_all)
        self._add_modules("out_to_all", self.out_to_all)
        self._add_modules("prev_to_all", self.prev_to_all)

        self.reset_parameters()

    def get_output_size(self):
        return sum(self.layer_sizes) if self.out_from_all_layers else self.layer_sizes[-1]

    def forward(self, data):
        for i, size in enumerate(self.layer_sizes):
            d = self.in_to_all[i](data)
            if self.outputs[i] is not None:
                d+=self.out_to_all[i](self.outputs[i])
            if i>0:
                d+=self.prev_to_all[i-1](self.outputs[i-1])

            input_data = torch.tanh(d[...,-size:])
            forget_gate, input_gate, output_gate = torch.sigmoid(d[...,:-size]).chunk(3,dim=-1)

            state_update = input_gate * input_data

            if self.states[i] is not None:
                self.states[i] = self.states[i]*forget_gate + state_update
            else:
                self.states[i] = state_update

            self.outputs[i] = output_gate * torch.tanh(self.states[i])

        return torch.cat(self.outputs, -1) if self.out_from_all_layers else self.outputs[-1]


class FeedforwardController(torch.nn.Module):
    def __init__(self, layer_sizes=[]):
        super(FeedforwardController, self).__init__()
        self.layer_sizes = layer_sizes

    def new_sequence(self):
        pass

    def reset_parameters(self):
        for module in self.model:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))

    def get_output_size(self):
        return self.layer_sizes[-1]

    def init(self, input_size):
        self.layer_sizes = self.layer_sizes

        # Xavier init: input to all gates is layers_sizes[i-1] + layer_sizes[i] + input_size -> layer_size big.
        # So use xavier init according to this.
        self.input_sizes = [input_size] + self.layer_sizes[:-1]

        layers = []
        for i, size in enumerate(self.layer_sizes):
            layers.append(torch.nn.Linear(self.input_sizes[i], self.layer_sizes[i]))
            layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, data):
        return self.model(data)
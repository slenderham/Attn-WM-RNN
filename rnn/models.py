import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def _get_activation_function(func_name):
    if func_name=='relu':
        return F.relu
    elif func_name=='softplus':
        return F.softplus
    elif func_name=='retanh':
        return lambda x: F.tanh(F.relu(x))
    elif func_name=='sigmoid':
        return F.sigmoid()
    else:
        raise RuntimeError(F"{func_name} is an invalid activation function.")

def _get_pos_function(func_name):
    if func_name=='relu':
        return F.relu
    elif func_name=='softplus':
        return F.softplus
    elif func_name=='abs':
        return torch.abs
    else:
        raise RuntimeError(F"{func_name} is an invalid function enforcing positive weight.")

class EILinear(nn.Module):
    def __init__(self, input_size, output_size, remove_diag, 
                    e_prop=0.8, bias=True, pos_function='relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        assert(e_prop<=1 and e_prop>=0)
        self.e_prop = e_prop
        self.e_size = int(e_prop * input_size)
        self.i_size = input_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        mask = torch.tensor([1]*self.e_size+[-1]*self.i_size, dtype=torch.float32).reshape(1, input_size);
        self.mask = mask.repeat([output_size, 1])
        self.pos_func = _get_pos_function(pos_function)
        if remove_diag:
            assert(input_size==output_size)
            self.mask[torch.eye(input_size)>0.5]=0.0
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def effective_weight(self):
        return self.pos_func(self.weight) * self.mask

    def effective_weight(self, w):
        return (self.pos_func(w)) * self.mask

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)

    def forward(self, input, w):
        # weight is non-negative
        return F.linear(input, self.effective_weight(w), self.bias)

class PosWLinear(nn.Module):
    def __init__(self, in_features, out_features, zero_rows_prop=0.0, bias=True):
        super(PosWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.zero_rows = int(zero_rows_prop*in_features)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.bias_mask = torch.cat([torch.ones(in_features-self.zero_rows), torch.zeros(self.zero_rows)])
        else:
            self.register_parameter('bias', None)
        self.weight_mask = torch.cat([torch.ones(in_features-self.zero_rows), torch.zeros(self.zero_rows)]).reshape([1, in_features])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # weight is non-negative
        if self.bias:
            bias = self.bias*self.bias_mask
        return F.linear(input, self.mask*torch.abs(self.weight), bias)

class GroupedEILinear(nn.Module):
    def __init__(self, input_size, output_size, group_size, remove_diag, e_prop=0.8, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size
        assert(e_prop<=1 and e_prop>=0)
        self.e_prop = e_prop
        self.e_size = int(e_prop * input_size)
        self.i_size = input_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(group_size, output_size, input_size))
        mask = torch.tensor([1]*self.e_size+[-1]*self.i_size, dtype=torch.float32).reshape(1, 1, input_size);
        self.mask = mask.repeat([group_size, output_size, 1])
        if remove_diag:
            assert(input_size==output_size)
            self.mask[torch.eye(input_size)>0.5]=0.0
        if bias:
            self.bias = nn.Parameter(torch.Tensor(group_size, output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        self.weight.data[:, :, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def effective_weight(self):
        return torch.abs(self.weight) * self.mask

    def forward(self, input):
        # weight is non-negative
        return torch.einsum('gij, bj->bgi', self.effective_weight(), input) + self.bias

class LeakyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                attn_group_size=None, plastic=True, attention=True, activation='retanh',
                dt=20, tau_x=100, tau_w=200, c_plasticity=None,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = PosWLinear(input_size, hidden_size, zero_rows_prop=0)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, e_prop=e_prop)
        self.h2o = PosWLinear(hidden_size, output_size, zero_rows_prop=1-e_prop)
        self.num_layers = 1
        self.tau_x = tau_x
        self.tau_w = tau_w
        if dt is None:
            alpha_x = 1
            alpha_w = 1
        else:
            alpha_x = dt / self.tau_x
            alpha_w = dt / self.tau_w
        self.alpha_x = alpha_x
        self.alpha_w = alpha_w
        self.oneminusalpha_x = 1 - alpha_x
        self.oneminusalpha_w = 1 - alpha_w
        self._sigma_rec = np.sqrt(2*alpha_x) * sigma_rec
        self._sigma_in = np.sqrt(2*alpha_x) * sigma_in
        self._sigma_w = np.sqrt(2*alpha_w) * sigma_w
        
        self.plastic = plastic
        if plastic:
            if c_plasticity is not None:
                assert(c_plasticity.shape==(6,))
                self.c_plas = torch.from_numpy(c_plasticity)
            else:
                self.c_plas = nn.Parameter(torch.zeros(6))

        self.attention = attention
        # TODO: mixed selectivity is required for the soltani et al 2016 model, what does it mean here? add separate layer
        if attention:
            self.attn_func = nn.Sequential([
                PosWLinear(hidden_size, len(attn_group_size), 1-e_prop, bias=True),
                nn.Softmax(dim=-1)
            ])
            self.attn_group_size = attn_group_size
        else:
            self.attn_func = None
            self.attn_group_size = None

        self.activation = _get_activation_function(activation)

    def init_hidden(self, x):
        batch_size = x.shape[1]
        if self.plastic:
            return (torch.zeros(batch_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size).to(x.device),
                    self.x2h.weight.to(x.device),
                    self.h2h.weight.to(x.device))
        else:
            return (torch.zeros(batch_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size).to(x.device))

    def recurrence(self, x, h, R):
        if self.plastic:
            state, output, wx, wh = h
        else:
            state, output = h
        
        if self.attention:
            attn_weights = torch.repeat_interleave(self.attn_func(output), self.attn_group_size, dim=-1)
            x = x * attn_weights
        else:
            x = x / len(self.attn_group_size)

        x += self._sigma_in * torch.randn_like(x)
        total_input = self.x2h(x, wx) + self.h2h(h, wh)
        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)
        
        if self.plastic:
            wx = wx * self.oneminusalpha_w + self.alpha_w*R*(
                self.c_plas[0].abs()*torch.einsum('bj->bij', x) +
                self.c_plas[1].abs()*torch.einsum('bi->bij', new_output) +
                self.c_plas[2].abs()*torch.einsum('bi, bj->bij', new_output, x)) + \
                self._sigma_w * torch.randn_like(wx)
            wh = wh * self.oneminusalpha_w + self.alpha_w*R*(
                self.c_plas[3].abs()*torch.einsum('bj->bij', output) +
                self.c_plas[4].abs()*torch.einsum('bi->bij', new_output) +
                self.c_plas[5].abs()*torch.einsum('bi, bj->bij', new_output, output)) + \
                self._sigma_w * torch.randn_like(wh)
            return new_state, new_output, wx, wh
        else:
            return new_state, new_output

    def forward(self, x, Rs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x)

        hs = []
        steps = range(x[0].size(0))
        for i in steps:
            hidden = self.recurrence(x[i], hidden, Rs[i])
            hs.append(hidden[1])

        hs = torch.stack(hs, dim=0)
        output = self.h2o(hs)

        return output, hs
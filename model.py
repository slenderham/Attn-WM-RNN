import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class EILinear(nn.Module):
    def __init__(self, input_size, output_size, remove_diag, e_prop=0.8, bias=True):
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
        return torch.abs(self.weight) * self.mask

    def effective_weight(self, dw):
        return (torch.abs(self.weight)+dw) * self.mask

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)

    def forward(self, input, dw):
        # weight is non-negative
        return F.linear(input, self.effective_weight(dw), self.bias)

class PosWLinear(nn.Module):
    def __init__(self, in_features, out_features, zero_rows_prop=0.0, bias=True):
        super(PosWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.zero_rows = int(zero_rows_prop*in_features)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.mask = torch.diag(torch.cat([torch.ones(in_features-self.zero_rows), torch.ones(self.zero_rows)]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.mask*torch.abs(self.weight), self.bias)

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
    def __init__(self, input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, e_prop=1)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, e_prop=e_prop)
        self.num_layers = 1
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        # Recurrent noise
        self._sigma_rec = np.sqrt(2*alpha) * sigma_rec

    def init_hidden(self, x):
        batch_size = x.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device))

    def recurrence(self, x, h):
        state, output = h
        total_input = self.x2h(x) + self.h2h(h)
        state = state * self.oneminusalpha + total_input * self.alpha + self._sigma_rec * torch.randn_like(state)
        output = torch.relu(state)
        return state, output

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x)

        output = []
        steps = range(x.size(0))
        for i in steps:
            hidden = self.recurrence(x[i], hidden)
            output.append(hidden[1])

        output = torch.stack(output, dim=0)
        return output, hidden

class LeakyPlasticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, lrx=0.1, lrh = 0.1,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, e_prop=1)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, e_prop=e_prop)
        self.num_layers = 1
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self.lrx = lrx
        self.lrh = lrh
        # Recurrent noise
        self._sigma_rec = np.sqrt(2*alpha) * sigma_rec

    def init_hidden(self, x):
        batch_size = x.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device))

    def recurrence(self, x, h):
        state, output, dwx, dwh = h
        total_input = self.x2h(x, dwx) + self.h2h(h, dwh)
        new_state = state * self.oneminusalpha + total_input * self.alpha + self._sigma_rec * torch.randn_like(state)
        new_output = torch.relu(new_state)
        dwx += self.lrx*torch.einsum('bi, bj->bij', new_output, x)
        dwh += self.lrx*torch.einsum('bi, bj->bij', new_output, output)
        return new_state, new_output, dwx, dwh

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x)

        output = []
        steps = range(x.size(0))
        for i in steps:
            hidden = self.recurrence(x[i], hidden)
            output.append(hidden[1])

        output = torch.stack(output, dim=0)
        return output, hidden
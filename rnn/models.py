import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from matplotlib import pyplot as plt


def _get_activation_function(func_name):
    if func_name=='relu':
        return F.relu
    elif func_name=='softplus':
        return F.softplus
    elif func_name=='retanh':
        return lambda x: torch.tanh(F.relu(x))
    elif func_name=='sigmoid':
        return torch.sigmoid
    else:
        raise RuntimeError(F"{func_name} is an invalid activation function.")

def _get_pos_function(func_name):
    if func_name=='relu':
        return F.relu
    else:
        raise RuntimeError(F"{func_name} is an invalid function enforcing positive weight.")

class EILinear(nn.Module):
    def __init__(self, input_size, output_size, remove_diag, zero_cols_prop,
                     e_prop=0.8, bias=True, pos_function='relu', init_spectral=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        assert(e_prop<=1 and e_prop>=0)
        
        self.e_prop = e_prop
        self.e_size = int(e_prop * input_size)
        self.i_size = input_size - self.e_size
        self.zero_cols = round(zero_cols_prop * input_size)

        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        sign_mask = torch.FloatTensor([1]*self.e_size+[-1]*self.i_size).reshape(1, input_size)
        exist_mask = torch.cat([torch.ones(input_size-self.zero_cols), torch.zeros(self.zero_cols)]).reshape([1, input_size])

        self.mask = (sign_mask*exist_mask).repeat([output_size, 1])
        self.pos_func = _get_pos_function(pos_function)
        if remove_diag:
            assert(input_size==output_size)
            self.mask[torch.eye(input_size)>0.5]=0.0
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters(init_spectral)

    def reset_parameters(self, init_spectral):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(2*self.input_size/(self.input_size-self.zero_cols)))
            # Scale E weight by E-I ratio
            if self.i_size!=0:
                self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)

            if init_spectral is not None:
                self.weight.data *= init_spectral / torch.linalg.eigvals(self.effective_weight(self.weight)).real.max()

            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def effective_weight(self, w=None):
        if w is None:
            return self.pos_func(self.weight) * self.mask
        else:
            return (self.pos_func(self.weight).unsqueeze(0)+w) * self.mask.unsqueeze(0)

    def forward(self, input, w=None):
        # weight is non-negative
        if w is None:
            return F.linear(input, self.effective_weight(), self.bias)
        else:
            result = torch.matmul(self.effective_weight(w), input.unsqueeze(2)).squeeze(2) 
            if self.bias is not None:
                result += self.bias
            return result

class LeakyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                attn_group_size=None, plastic=True, attention=True, activation='retanh',
                dt=0.02, tau_x=0.1, tau_w=1.0, c_plasticity=None, train_init_state=False,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, truncate_iter=None, init_spectral=1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, e_prop=1, zero_cols_prop=0, bias=False)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, e_prop=e_prop, zero_cols_prop=0, bias=True, init_spectral=init_spectral)
        self.h2o = EILinear(hidden_size, output_size, remove_diag=False, e_prop=e_prop, zero_cols_prop=1-e_prop, bias=False)
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

        if train_init_state:
            self.x0 = nn.Parameter(torch.randn(1, hidden_size))
        else:
            self.x0 = torch.zeros(1, hidden_size)
        
        self.plastic = plastic
        if plastic:
            if c_plasticity is not None:
                assert(len(c_plasticity)==3)
                self.kappa_w = torch.FloatTensor(c_plasticity).log()
            else:
                self.kappa_w = nn.Parameter(torch.zeros(3))

        self.attention = attention
        # TODO: mixed selectivity is required for the soltani et al 2016 model, what does it mean here? add separate layer?
        if attention:
            assert(attn_group_size is not None)
            self.attn_func = EILinear(hidden_size, len(attn_group_size), remove_diag=False, e_prop=e_prop, zero_cols_prop=1-e_prop)
            self.attn_group_size = torch.LongTensor(attn_group_size)
        else:
            self.attn_func = None
            self.attn_group_size = None

        self.activation = _get_activation_function(activation)
        self.truncate_iter = truncate_iter

    def init_hidden(self, x):
        batch_size = x.shape[1]
        h_init = self.x0.to(x.device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        if self.plastic:
            return (h_init, h_init.relu(),
                    torch.zeros(batch_size, self.hidden_size, self.input_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.output_size, self.hidden_size).to(x.device))
        else:
            return (h_init, h_init.relu())

    def recurrence(self, x, h, R):
        batch_size = x.shape[0]
        
        if self.plastic:
            state, output, wx, wh, wo = h
        else:
            state, output = h
        
        if self.attention:
            attn_weights = self.attn_func(output)
            attn_weights = F.softmax(attn_weights, -1)
            attn_weights = torch.repeat_interleave(attn_weights, self.attn_group_size, dim=-1)
            x = x * attn_weights
        else:
            x = x / len(self.attn_group_size)

        x = torch.relu(x+self._sigma_in * torch.randn_like(x))
        if self.plastic:
            total_input = self.x2h(x, wx) + self.h2h(output, wh)
        else:
            total_input = self.x2h(x) + self.h2h(output)
        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)

        value = torch.tanh(F.relu(self.h2o(new_output, wo)))

        if self.plastic:
            R = R.unsqueeze(-1)
            wx = wx * self.oneminusalpha_w \
                + self.kappa_w[0].exp()*R*torch.einsum('bi, bj->bij', new_output, x) \
                + self._sigma_w * torch.randn_like(wx)
            wx = torch.maximum(wx, -self.x2h.pos_func(self.x2h.weight).detach().unsqueeze(0))
            wh = wh * self.oneminusalpha_w \
                + self.kappa_w[1].exp()*R*torch.einsum('bi, bj->bij', new_output, output) \
                + self._sigma_w * torch.randn_like(wh)
            wh = torch.maximum(wh, -self.h2h.pos_func(self.h2h.weight).detach().unsqueeze(0))
            wo = wo * self.oneminusalpha_w \
                + self.kappa_w[2].exp()*((R+1)/2-value)*new_output.unsqueeze(1) \
                + self._sigma_w * torch.randn_like(wo)
            return value, (new_state, new_output, wx, wh, wo)
        else:
            return value, (new_state, new_output)

    def truncate(self, hidden):
        if self.plastic:
            return (hidden[0].detach(), hidden[1].detach(), hidden[2].detach(), hidden[3].detach(), hidden[4].detach())
        else:
            return (hidden[0].detach(), hidden[1].detach())

    def forward(self, x, Rs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x)

        hs = []
        os = []
        steps = range(x.size(0))
        for i in steps:
            out, hidden = self.recurrence(x[i], hidden, Rs[i])
            hs.append(hidden[1])
            os.append(out)
            if self.truncate_iter is not None and (i+1)%self.truncate_iter==0:
                hidden = self.truncate(hidden)
    
        hs = torch.stack(hs, dim=0)
        os = torch.stack(os, dim=0)

        return os, hs
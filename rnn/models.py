from collections import defaultdict
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
                     e_prop=0.8, bias=True, pos_function='relu', init_spectral=None, 
                     init_gain=None, balance_ei=False):
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
        
        self.reset_parameters(init_spectral, init_gain, balance_ei)

    def reset_parameters(self, init_spectral, init_gain, balance_ei):
        with torch.no_grad():
            nn.init.uniform_(self.weight, a=0, b=math.sqrt(1/(self.input_size-self.zero_cols)))
            # Scale E weight by E-I ratio
            if balance_ei and self.i_size!=0:
                self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)

            if init_gain is not None:
                self.weight.data *= init_gain
            if init_spectral is not None:
                self.weight.data *= init_spectral / torch.linalg.eigvals(self.effective_weight(self.weight)).real.max()

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def effective_weight(self, w=None):
        if w is None:
            return self.pos_func(self.weight) * self.mask
        else:
            return w * self.mask.unsqueeze(0)

    def forward(self, input, w=None):
        # weight is non-negative
        if w is None:
            return F.linear(input, self.effective_weight(), self.bias)
        else:
            result = torch.matmul(self.effective_weight(w), input.unsqueeze(2)).squeeze(2) 
            if self.bias is not None:
                result += self.bias
            return result

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_type='weight',
                attn_group_size=None, plastic=True, plastic_feedback=True, activation='retanh', 
                dt=0.02, tau_x=0.1, tau_w=1.0, weight_bound=1.0, c_plasticity=None, train_init_state=False,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, truncate_iter=None, init_spectral=None, 
                balance_ei=False, rwd_input=False, sep_lr=True, input_unit_group=None, value_est=True, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.rwd_input = rwd_input
        self.weight_bound = weight_bound
        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, pos_function='relu',
                            e_prop=1, zero_cols_prop=0, bias=False, init_gain=0.5)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='relu',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1, 
                            init_spectral=init_spectral, balance_ei=balance_ei)
        if value_est:
            self.h2o = EILinear(hidden_size, output_size, remove_diag=False, pos_function='relu',
                            e_prop=1, zero_cols_prop=1-e_prop, bias=True, init_gain=0.5)
            self.h2v = EILinear(hidden_size, 1, remove_diag=False, pos_function='relu',
                            e_prop=1, zero_cols_prop=1-e_prop, bias=True, init_gain=0.5)
        else:
            self.h2o = EILinear(hidden_size, output_size, remove_diag=False, pos_function='relu',
                            e_prop=1, zero_cols_prop=1-e_prop, bias=False, init_gain=0.5)

        self.tau_x = tau_x
        self.tau_w = tau_w
        self.dt = dt
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
        
        assert attention_type in ['none', 'weight', 'sample']
        self.attention_type = attention_type
        # TODO: mixed selectivity is required for the soltani et al 2016 model, what does it mean here? add separate layer?
        if attention_type!='none':
            assert(attn_group_size is not None)
            if attention_type=='weight':
                self.attn_func = EILinear(hidden_size, len(attn_group_size), remove_diag=False, \
                                          e_prop=e_prop, zero_cols_prop=1-e_prop, init_gain=0.5)
            elif attention_type=='sample':
                self.attn_func = EILinear(hidden_size, len(attn_group_size), remove_diag=False, \
                                          e_prop=e_prop, zero_cols_prop=1-e_prop, init_gain=0.5)
            self.attn_group_size = torch.LongTensor(attn_group_size)
        else:
            self.attn_func = None
            self.attn_group_size = None

        self.activation = _get_activation_function(activation)
        if truncate_iter is not None:
            raise NotImplementedError

        self.plastic = plastic
        self.plastic_feedback = plastic_feedback
        if plastic:
            # separate lr for different neurons
            if sep_lr:
                self.rec_coords = [[0, self.h2h.e_size, 0, self.h2h.e_size],
                                   [0, self.h2h.e_size, self.h2h.e_size, self.hidden_size],
                                   [self.h2h.e_size, self.hidden_size, 0, self.h2h.e_size],
                                   [self.h2h.e_size, self.hidden_size, self.h2h.e_size, self.hidden_size]]
                self.in_coords = []
                input_unit_group.insert(0, 0)
                group_start = np.cumsum(input_unit_group)
                for i in range(len(group_start)-1):
                    self.in_coords.append([0, self.h2h.e_size, group_start[i], group_start[i+1]])
                    self.in_coords.append([self.h2h.e_size, self.hidden_size, group_start[i], group_start[i+1]])
                self.input_unit_group = input_unit_group[1:]
                kappa_count = len(self.rec_coords) + len(self.in_coords)
            else:
                self.in_coords = None
                self.rec_coords = None
                kappa_count = 6
                
            if plastic_feedback:
                self.fb_coords = []
                group_start = [0, 3, 6, 7]
                if sep_lr:
                    for i in range(len(group_start)-1):
                        self.fb_coords.append([group_start[i], group_start[i+1], 0, self.h2h.e_size])
                    kappa_count += len(self.fb_coords)
                else:
                    self.fb_coords = None
                    kappa_count += 3

            if c_plasticity is not None:
                assert(len(c_plasticity)==kappa_count)
                self.kappa_w = torch.FloatTensor(c_plasticity)
            else:
                self.kappa_w = nn.Parameter(torch.zeros(kappa_count)+1e-8)

    def init_hidden(self, x):
        batch_size = x.shape[1]
        h_init = self.x0.to(x.device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        if self.plastic:
            if self.plastic_feedback:
                return (h_init, h_init.relu(),
                        self.x2h.pos_func(self.x2h.weight).unsqueeze(0).repeat(batch_size, 1, 1), 
                        self.h2h.pos_func(self.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1),
                        self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0).repeat(batch_size, 1, 1))
            else:
                return (h_init, h_init.relu(),
                        self.x2h.pos_func(self.x2h.weight).unsqueeze(0).repeat(batch_size, 1, 1), 
                        self.h2h.pos_func(self.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1))
        else:
            return (h_init, h_init.relu())

    def multiply_blocks(self, w, kappa_w, coords):
        if coords is None:
            return w
        w_new = torch.zeros_like(w)
        for i, c in enumerate(coords):
            w_new[:, c[0]:c[1], c[2]:c[3]] = kappa_w[i]*w[:, c[0]:c[1], c[2]:c[3]]
        return w_new

    def recurrence(self, x, h, R):
        batch_size = x.shape[0]
        
        if self.plastic:
            if self.plastic_feedback:
                state, output, wx, wh, wattn = h
            else:
                state, output, wx, wh = h
                wattn = None
        else:
            state, output = h
        
        if self.attention_type=='weight':
            attn = F.softmax(self.attn_func(output, wattn), -1)
            attn_expand = torch.repeat_interleave(attn, self.attn_group_size, dim=-1)
        elif self.attention_type=='sample':
            attn = F.gumbel_softmax(self.attn_func(output, wattn), hard=True, dim=-1)
            attn_expand = torch.repeat_interleave(attn, self.attn_group_size, dim=-1)
        else:
            attn = None
            attn_expand = None

        if self.attention_type=='weight' or self.attention_type=='sample':
            x = torch.relu(x + self._sigma_in * torch.randn_like(x)) * attn_expand
        else:
            x = torch.relu(x + self._sigma_in * torch.randn_like(x)) / len(self.attn_group_size)

        if self.rwd_input:
            x = torch.cat([x, (R!=0)*(R+1)/2 + self._sigma_in * torch.randn_like(R), (R!=0)*(1-R)/2 + self._sigma_in * torch.randn_like(R)], -1)

        if self.plastic:
            total_input = self.x2h(x, wx) + self.h2h(output, wh)
        else:
            total_input = self.x2h(x) + self.h2h(output)
        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)

        if self.plastic:
            R = R.unsqueeze(-1)
            if self.in_coords is not None and self.rec_coords is not None:
                wx = wx*self.oneminusalpha_w + self.x2h.pos_func(self.x2h.weight).unsqueeze(0)*self.alpha_w \
                    + self.dt*R*(self.multiply_blocks(torch.einsum('bi, bj->bij', new_output, x), \
                        self.kappa_w[0:2*len(self.input_unit_group)].abs(), self.in_coords))
                if self._sigma_w>0:
                    wx += self._sigma_w * torch.randn_like(wx)
                wx = torch.clamp(wx, 0, self.weight_bound)
                # wx = torch.maximum(wx, -self.x2h.pos_func(self.x2h.weight).detach().unsqueeze(0))
                # wx = torch.minimum(wx, self.weight_bound-self.x2h.pos_func(self.x2h.weight).detach().unsqueeze(0))
                wh = wh*self.oneminusalpha_w + self.h2h.pos_func(self.h2h.weight).unsqueeze(0)*self.alpha_w \
                    + self.dt*R*(self.multiply_blocks(torch.einsum('bi, bj->bij', new_output, output), \
                        self.kappa_w[2*len(self.input_unit_group):2*len(self.input_unit_group)+4].abs(), self.rec_coords))
                if self._sigma_w>0:
                    wh += self._sigma_w * torch.randn_like(wh)
                wh = torch.clamp(wh, 0, self.weight_bound)
                # wh = torch.maximum(wh, -self.h2h.pos_func(self.h2h.weight).detach().unsqueeze(0))
                # wh = torch.minimum(wh, self.weight_bound-self.h2h.pos_func(self.h2h.weight).detach().unsqueeze(0))
                if self.plastic_feedback:
                    wattn = wattn*self.oneminusalpha_w + self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0)*self.alpha_w \
                        + self.dt*R*(self.multiply_blocks(torch.einsum('bi, bj->bij', attn, output), \
                            self.kappa_w[2*len(self.input_unit_group)+4:3*len(self.input_unit_group)+4].abs(), self.fb_coords))
                    if self._sigma_w>0:
                        wattn += self._sigma_w * torch.randn_like(wattn)
                    wattn = torch.clamp(wattn, 0, self.weight_bound)
                    # wattn = torch.maximum(wattn, -self.attn_func.pos_func(self.attn_func.weight).detach().unsqueeze(0))
                    # wattn = torch.minimum(wattn, self.weight_bound-self.h2h.pos_func(self.attn_func.weight).detach().unsqueeze(0))
            else:
                wx = wx*self.oneminusalpha_w + self.x2h.pos_func(self.x2h.weight).unsqueeze(0)*self.alpha_w + self.dt*R*(
                    self.kappa_w[0]*torch.reshape(x, (batch_size, 1, self.input_size)) +
                    self.kappa_w[1]*torch.reshape(new_output, (batch_size, self.hidden_size, 1)) +
                    self.kappa_w[2]*torch.einsum('bi, bj->bij', new_output, x))
                if self._sigma_w>0:
                    wx += self._sigma_w * torch.randn_like(wx)
                wx = torch.clamp(wx, 0, self.weight_bound)
                wh = wh*self.oneminusalpha_w + self.h2h.pos_func(self.h2h.weight).unsqueeze(0)*self.alpha_w + self.dt*R*(
                    self.kappa_w[3]*torch.reshape(output, (batch_size, 1, self.hidden_size)) +
                    self.kappa_w[4]*torch.reshape(new_output, (batch_size, self.hidden_size, 1)) +
                    self.kappa_w[5]*torch.einsum('bi, bj->bij', new_output, output))
                if self._sigma_w>0:
                    wh += self._sigma_w * torch.randn_like(wh)
                wh = torch.clamp(wh, 0, self.weight_bound)
                if self.plastic_feedback:
                    wattn = wattn*self.oneminusalpha_w + self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0)*self.alpha_w + self.dt*R*(
                        self.kappa_w[6]*torch.reshape(output, (batch_size, 1, self.hidden_size)) +
                        self.kappa_w[7]*torch.reshape(attn, (batch_size, len(self.attn_group_size), 1)) +
                        self.kappa_w[8]*torch.einsum('bi, bj->bij', attn*len(self.attn_group_size), output))
                    if self._sigma_w>0:
                        wattn += self._sigma_w * torch.randn_like(wattn)
                    wattn = torch.clamp(wattn, 0, self.weight_bound)
            
            if self.plastic_feedback:
                return (new_state, new_output, wx, wh, wattn, attn)
            else:
                return (new_state, new_output, wx, wh, attn)
        else:
            return (new_state, new_output, attn)

    def truncate(self, hidden):
        if self.plastic:
            return (hidden[0].detach(), hidden[1].detach(), hidden[2].detach(), hidden[3].detach())
        else:
            return (hidden[0].detach(), hidden[1].detach())

    def print_kappa(self):
        print()
        print('Input weight kappa: ')
        if self.in_coords is None:
            print(self.kappa_w[0:3].tolist())
        else:
            print (f'Input->E: ', end='')
            print(self.kappa_w[0:2*len(self.input_unit_group):2].abs().tolist())
            print (f'Input->I: ', end='')
            print(self.kappa_w[1:2*len(self.input_unit_group):2].abs().tolist())
        
        print('Recurrent weight kappa: ')
        if self.rec_coords is None:
            print(self.kappa_w[3:6].tolist())
        else:
            s = 2*len(self.input_unit_group)
            print (f'E->E: ', end='')
            print(self.kappa_w[s].abs().item(), end=', ')
            print (f'I->E: ', end='')
            print(self.kappa_w[s+1].abs().item(), end=', ')
            print (f'E->I: ', end='')
            print(self.kappa_w[s+2].abs().item(), end=', ')
            print (f'I->I: ', end='')
            print(self.kappa_w[s+3].abs().item())

        if self.plastic_feedback:
            print('Feedback weight kappa: ')
            if self.fb_coords is None:
                print(self.kappa_w[6:9].tolist())
            else:
                s = 2*len(self.input_unit_group)+4
                print (f'E->Attn: ', end='')
                print(self.kappa_w[s:s+3].abs().tolist())
        
        print()

    def forward(self, x, Rs, hidden=None, save_weight=False, save_attn=False):
        if hidden is None:
            hidden = self.init_hidden(x)

        hs = []

        if save_weight:
            wxs = []
            whs = []
        if save_attn:
            attns = []

        steps = range(x.size(0))
        for i in steps:
            hidden = self.recurrence(x[i], hidden[:-1] if i>0 else hidden, Rs[i])
            hs.append(hidden[1])
            if save_weight:
                wxs.append(hidden[2])
                whs.append(hidden[3])
            if save_attn:
                attns.append(hidden[-1])

        hs = torch.stack(hs, dim=0)
        saved_states = defaultdict(list)
        if save_weight:
            wxs = torch.stack(wxs, dim=0)
            whs = torch.stack(whs, dim=0)
            saved_states['wxs'].append(wxs)
            saved_states['whs'].append(whs)
        if save_attn:
            attns = torch.stack(attns, dim=0)
            saved_states['attns'].append(attns)
        os = self.h2o(hs)
        if hasattr(self, 'h2v'):
            vs = self.h2v(hs)
            return (os, vs), hs, saved_states
        else:
            return os, hs, saved_states
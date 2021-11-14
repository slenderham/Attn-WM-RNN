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
            # nn.init.uniform_(self.weight, a=0, b=math.sqrt(1/(self.input_size-self.zero_cols)))
            nn.init.kaiming_normal_(self.weight)
            self.weight.abs_()
            # Scale E weight by E-I ratio
            if balance_ei and self.i_size!=0:
                self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)

            if init_gain is not None:
                self.weight.data *= init_gain
            if init_spectral is not None:
                self.weight.data *= init_spectral / torch.linalg.eigvals(self.effective_weight()).real.max()

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

class MultiChoiceRNN(nn.Module):
    def __init__(self, input_size, num_choices, hidden_size, output_size, attention_type='weight',
                attn_group_size=None, plastic=True, plastic_feedback=True, activation='retanh', 
                dt=0.02, tau_x=0.1, tau_w=1.0, weight_bound=1.0, train_init_state=False,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, init_spectral=None, 
                balance_ei=False, rwd_input=False, action_input=False, sep_lr=True, plas_rule='mult',
                value_est=True, input_unit_group=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.action_input = action_input
        self.rwd_input = rwd_input
        self.input_unit_group = torch.LongTensor(input_unit_group)
        assert(num_choices>=2)
        self.num_choices = num_choices
        self.weight_bound = weight_bound
        self.sep_lr = sep_lr
        self.plas_rule = plas_rule
        self.x2h = nn.ModuleList([EILinear(input_size, hidden_size, remove_diag=False, pos_function='relu',
                                           e_prop=1, zero_cols_prop=0, bias=False, init_gain=0.5/math.sqrt(num_choices))
                                  for _ in range(num_choices)])

        self.aux_input_size = (2 if self.rwd_input else 0) + (num_choices if self.action_input else 0)
        if self.aux_input_size>0:
            self.aux2h = EILinear(self.aux_input_size, hidden_size, remove_diag=False, pos_function='relu',
                                  e_prop=1, zero_cols_prop=0, bias=False, 
                                  init_gain=0.5*math.sqrt(self.aux_input_size/(hidden_size*num_choices)))
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='relu',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1,
                            init_spectral=init_spectral, balance_ei=balance_ei)
        if value_est:
            self.h2o = EILinear(hidden_size, output_size, remove_diag=False, pos_function='relu',
                                e_prop=1, zero_cols_prop=1-e_prop, bias=True, init_gain=0.15)
            self.h2v = EILinear(hidden_size, 1, remove_diag=False, pos_function='relu',
                                e_prop=1, zero_cols_prop=1-e_prop, bias=True, init_gain=0.15)
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
            self.x0 = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.x0 = torch.zeros(1, hidden_size)
        
        assert attention_type in ['none', 'weight', 'sample']
        self.attention_type = attention_type

        if attention_type!='none':
            assert(attn_group_size is not None)
            self.attn_func = EILinear(hidden_size, len(attn_group_size), remove_diag=False,
                                        e_prop=e_prop, zero_cols_prop=1-e_prop, init_gain=0.05)
                                        # the same attention applies to the same dimension of both stimuli
            self.attn_group_size = torch.LongTensor(attn_group_size)
            self.attn_lr_group = torch.LongTensor([3,3,1])
        else:
            self.attn_func = None
            self.attn_group_size = None
            self.attn_lr_group = None

        self.activation = _get_activation_function(activation)

        self.plastic = plastic
        self.plastic_feedback = plastic_feedback
        if plastic:
            # separate lr for different neurons
            if not sep_lr:
                kappa_count = 6
            if plastic_feedback:
                if not sep_lr:
                    kappa_count += 3
            
            if sep_lr:
                self.kappa_in = nn.ParameterList([nn.Parameter(torch.rand(1, self.hidden_size, len(input_unit_group))*1e-4) 
                                                    for _ in range(num_choices)])
                self.kappa_rec = nn.Parameter(torch.rand(1, self.hidden_size, self.hidden_size)*1e-4)
                if plastic_feedback:
                    self.kappa_fb = nn.Parameter(torch.rand(1, len(input_unit_group), self.hidden_size)*1e-4)
            else:
                self.kappa_w = nn.Parameter(torch.zeros(kappa_count)+1e-8)

    def init_hidden(self, x):
        batch_size = x.shape[1]
        h_init = self.x0.to(x.device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        if self.plastic:
            if self.plastic_feedback:
                return (h_init, h_init.relu(),
                        [x2hi.pos_func(x2hi.weight).unsqueeze(0).repeat(batch_size, 1, 1) for x2hi in self.x2h], 
                        self.h2h.pos_func(self.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1),
                        self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0).repeat(batch_size, 1, 1), None)
            else:
                return (h_init, h_init.relu(),
                        [x2hi.pos_func(x2hi.weight).unsqueeze(0).repeat(batch_size, 1, 1) for x2hi in self.x2h], 
                        self.h2h.pos_func(self.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1), None)
        else:
            return (h_init, h_init.relu())
        
    def init_state(self, hidden):
        batch_size = hidden[0].shape[1]
        h_init = self.x0.to(hidden[0].device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        return (h_init, h_init.relu(), *hidden[2:])

    def plasticity_func(self, w, baseline, R, pre, post, kappa, lb, ub):
        if self.plas_rule=='add':
            new_w = baseline*self.alpha_w + w*self.oneminusalpha_w + self.dt*R*kappa*torch.einsum('bi, bj->bij', post, pre)
            if self._sigma_w>0:
                new_w += self._sigma_w * torch.randn_like(new_w)
        elif self.plas_rule=='mult':
            expnt = 0.4
            new_w = baseline*self.alpha_w + w*self.oneminusalpha_w \
                  + self.dt*R*kappa*(w**expnt)*torch.einsum('bi, bj->bij', post**expnt, pre**expnt)
            if self._sigma_w>0:
                new_w += w * self._sigma_w * torch.randn_like(new_w)
        new_w = torch.clamp(new_w, lb, ub)
        return new_w

    def recurrence(self, x, h, R, v=None, a=None):
        batch_size = x.shape[0]
        
        if self.plastic:
            if self.plastic_feedback:
                state, output, wx, wh, wattn, _ = h
            else:
                state, output, wx, wh, _ = h
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
            x = torch.relu(x * attn_expand * len(self.attn_group_size) + self._sigma_in * torch.randn_like(x))
        else:
            x = torch.relu(x + self._sigma_in * torch.randn_like(x))

        if self.plastic:
            total_input = self.h2h(output, wh)
            for i in range(self.num_choices):
                total_input += self.x2h[i](x[:,i,:], wx[i])
        else:
            total_input = self.h2h(output)

        if self.aux_input_size>0:
            aux = torch.zeros(batch_size, self.aux_input_size)
            if self.rwd_input:
                aux[:, 0:1] = (R!=0)*(R+1)/2 + self._sigma_in * torch.randn_like(R)
                aux[:, 1:2] = (R!=0)*(1-R)/2 + self._sigma_in * torch.randn_like(R)
            if self.action_input:
                aux[:,-self.num_choices:] = a + self._sigma_in * torch.randn_like(a)
            aux = aux.relu()
            total_input += self.aux2h(aux)

        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)

        if self.plastic:
            if v is None:
                R = R.unsqueeze(-1)
                v = 0
            else:
                R = (R!=0)*(R.unsqueeze(-1)+1)/2
                v = v.unsqueeze(-1)
            if self.sep_lr:
                for i in range(self.num_choices):
                    wx[i] = self.plasticity_func(wx[i], self.x2h[i].pos_func(self.x2h[i].weight).unsqueeze(0), R-v, x[:,i], new_output, 
                                                 torch.repeat_interleave(self.kappa_in[i].relu(), self.input_unit_group, dim=2), 
                                                 0, self.weight_bound)
                wh = self.plasticity_func(wh, self.h2h.pos_func(self.h2h.weight).unsqueeze(0), R-v, output, new_output,
                                          self.kappa_rec.relu(), 0, self.weight_bound)
                if self.plastic_feedback:
                    wattn = self.plasticity_func(wattn, self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0), R-v, output, attn,
                                                 torch.repeat_interleave(self.kappa_fb.relu(), self.attn_lr_group, dim=1), 0, self.weight_bound)
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

    def forward(self, x, Rs, Vs=None, acts=None, hidden=None, save_weight=False, save_attn=False):
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
            hidden = self.recurrence(x[i], hidden, Rs[i], Vs[i] if Vs is not None else None, acts[i] if acts is not None else None)
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
            return (os, vs), hs, hidden, saved_states
        else:
            return os, hs, hidden, saved_states

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_type='weight',
                attn_group_size=None, plastic=True, plastic_feedback=True, activation='retanh', 
                dt=0.02, tau_x=0.1, tau_w=1.0, weight_bound=10.0, c_plasticity=None, train_init_state=False,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, init_spectral=None, 
                balance_ei=False, rwd_input=False, sep_lr=True, input_unit_group=None, value_est=True, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.rwd_input = rwd_input
        self.weight_bound = weight_bound
        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, pos_function='relu',
                                           e_prop=1, zero_cols_prop=0, bias=False, init_gain=1)

        self.aux_input_size = (2 if self.rwd_input else 0)
        if self.aux_input_size>0:
            self.aux2h = EILinear(self.aux_input_size, hidden_size, remove_diag=False, pos_function='relu',
                                  e_prop=1, zero_cols_prop=0, bias=False, 
                                  init_gain=math.sqrt(self.aux_input_size/(input_size)))
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='relu',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1,
                            init_spectral=init_spectral, balance_ei=balance_ei)
        if value_est:
            self.h2o = EILinear(hidden_size, output_size, remove_diag=False, pos_function='relu',
                                e_prop=1, zero_cols_prop=1-e_prop, bias=False, init_gain=0.5)
            self.h2v = EILinear(hidden_size, 1, remove_diag=False, pos_function='relu',
                                e_prop=1, zero_cols_prop=1-e_prop, bias=False, init_gain=0.5)
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
            self.x0 = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.x0 = torch.zeros(1, hidden_size)
        
        assert attention_type in ['none', 'weight', 'sample']
        self.attention_type = attention_type

        if attention_type!='none':
            assert(attn_group_size is not None)
            self.attn_func = EILinear(hidden_size, len(attn_group_size), remove_diag=False,
                                        e_prop=e_prop, zero_cols_prop=1-e_prop, init_gain=1) 
                                        # the same attention applies to the same dimension of both stimuli
            self.attn_group_size = torch.LongTensor(attn_group_size)
        else:
            self.attn_func = None
            self.attn_group_size = None

        self.activation = _get_activation_function(activation)

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
                        self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0).repeat(batch_size, 1, 1), None)
            else:
                return (h_init, h_init.relu(),
                        self.x2h.pos_func(self.x2h.weight).unsqueeze(0).repeat(batch_size, 1, 1), 
                        self.h2h.pos_func(self.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1), None)
        else:
            return (h_init, h_init.relu())

    def multiply_blocks(self, w, kappa_w, coords):
        if coords is None:
            return w
        w_new = torch.zeros_like(w)
        for i, c in enumerate(coords):
            w_new[:, c[0]:c[1], c[2]:c[3]] = kappa_w[i]*w[:, c[0]:c[1], c[2]:c[3]]
        return w_new

    def plasticity_func(self, w, baseline, R, pre, post, kappa, coords, lb, ub):
        new_w = w-(w-baseline)*self.alpha_w \
             + self.dt*R*(self.multiply_blocks(torch.einsum('bi, bj->bij', post, pre), kappa, coords))
        if self._sigma_w>0:
            new_w += self._sigma_w * torch.randn_like(new_w)
        new_w = torch.clamp(new_w, lb, ub)
        return new_w

    def recurrence(self, x, h, R, a=None):
        batch_size = x.shape[0]
        
        if self.plastic:
            if self.plastic_feedback:
                state, output, wx, wh, wattn, _ = h
            else:
                state, output, wx, wh, _ = h
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
            x = torch.relu(x * attn_expand * len(self.attn_group_size) + self._sigma_in * torch.randn_like(x))
        else:
            x = torch.relu(x + self._sigma_in * torch.randn_like(x))

        if self.plastic:
            total_input = self.x2h(x, wx) + self.h2h(output, wh)
        else:
            total_input = self.x2h(x) + self.h2h(output)

        if self.aux_input_size>0:
            aux = torch.zeros(batch_size, self.aux_input_size)
            if self.rwd_input:
                aux[:, 0:1] = (R != 0)*(R+1)/2 + self._sigma_in * torch.randn_like(R)
                aux[:, 1:2] = (R != 0)*(1-R)/2 + self._sigma_in * torch.randn_like(R)
            total_input += self.aux2h(aux)

        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)

        if self.plastic:
            R = R.unsqueeze(-1)
            if self.in_coords is not None and self.rec_coords is not None:
                wx = self.plasticity_func(wx, self.x2h.pos_func(self.x2h.weight).unsqueeze(0), R, x, new_output, 
                                          self.kappa_w[:2*len(self.input_unit_group)].abs(), self.in_coords,
                                          0, self.weight_bound)
                wh = self.plasticity_func(wh, self.h2h.pos_func(self.h2h.weight).unsqueeze(0), R, output, new_output,
                                          self.kappa_w[2*len(self.input_unit_group):2*len(self.input_unit_group)+4].abs(),
                                          self.rec_coords, 0, self.weight_bound)
                if self.plastic_feedback:
                    wattn = self.plasticity_func(wattn, self.attn_func.pos_func(self.attn_func.weight).unsqueeze(0), R, output, attn,
                                                 self.kappa_w[2*len(self.input_unit_group)+4:3*len(self.input_unit_group)+4].abs(),
                                                 self.fb_coords, 0, self.weight_bound)
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
            hidden = self.recurrence(x[i], hidden, Rs[i])
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
            return (os, vs), hs, hidden, saved_states
        else:
            return os, hs, hidden, saved_states

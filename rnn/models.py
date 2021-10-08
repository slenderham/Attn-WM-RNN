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

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                attn_group_size=None, plastic=True, attention=True, activation='retanh',
                dt=0.02, tau_x=0.1, tau_w=1.0, c_plasticity=None, train_init_state=False,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, truncate_iter=None, init_spectral=None, 
                balance_ei=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, pos_function='relu',
                            e_prop=1, zero_cols_prop=0, bias=False, init_gain=0.5)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='relu',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1, 
                            init_spectral=init_spectral, balance_ei=balance_ei)
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
        
        self.plastic = plastic
        if plastic:
            if c_plasticity is not None:
                assert(len(c_plasticity)==6)
                self.kappa_w = torch.FloatTensor(c_plasticity)
            else:
                self.kappa_w = nn.Parameter(torch.zeros(6)+self.alpha_w)

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
        if truncate_iter is not None:
            raise NotImplementedError

    def init_hidden(self, x):
        batch_size = x.shape[1]
        h_init = self.x0.to(x.device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        if self.plastic:
            return (h_init, h_init.relu(),
                    torch.zeros(batch_size, self.hidden_size, self.input_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device))
        else:
            return (h_init, h_init.relu())

    def recurrence(self, x, h, R):
        batch_size = x.shape[0]
        
        if self.plastic:
            state, output, wx, wh = h
        else:
            state, output = h
        
        if self.attention:
            attn_weights = self.attn_func(output)
            attn_weights = F.softmax(attn_weights, -1)
            attn_weights = torch.repeat_interleave(attn_weights, self.attn_group_size, dim=-1)
            x = torch.relu(x + self._sigma_in * torch.randn_like(x)) * attn_weights * len(self.attn_group_size)
        else:
            x = torch.relu(x + self._sigma_in * torch.randn_like(x))

        if self.plastic:
            total_input = self.x2h(x, wx) + self.h2h(output, wh)
        else:
            total_input = self.x2h(x) + self.h2h(output)
        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)

        if self.plastic:
            R = R.unsqueeze(-1)
            wx = wx * self.oneminusalpha_w + self.dt*R*(
                self.kappa_w[0]*torch.reshape(x, (batch_size, 1, self.input_size)) +
                self.kappa_w[1]*torch.reshape(new_output, (batch_size, self.hidden_size, 1)) +
                self.kappa_w[2]*torch.einsum('bi, bj->bij', new_output, x)) + \
                self._sigma_w * torch.randn_like(wx)
            wx = torch.maximum(wx, -self.x2h.pos_func(self.x2h.weight).detach().unsqueeze(0))
            wh = wh * self.oneminusalpha_w + self.dt*R*(
                self.kappa_w[3]*torch.reshape(output, (batch_size, 1, self.hidden_size)) +
                self.kappa_w[4]*torch.reshape(new_output, (batch_size, self.hidden_size, 1)) +
                self.kappa_w[5]*torch.einsum('bi, bj->bij', new_output, output)) + \
                self._sigma_w * torch.randn_like(wh)
            wh = torch.maximum(wh, -self.h2h.pos_func(self.h2h.weight).detach().unsqueeze(0))
            return (new_state, new_output, wx, wh)
        else:
            return (new_state, new_output)

    def truncate(self, hidden):
        if self.plastic:
            return (hidden[0].detach(), hidden[1].detach(), hidden[2].detach(), hidden[3].detach())
        else:
            return (hidden[0].detach(), hidden[1].detach())

    def forward(self, x, Rs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x)

        hs = []
        steps = range(x.size(0))
        for i in steps:
            hidden = self.recurrence(x[i], hidden, Rs[i])
            hs.append(hidden[1])

        hs = torch.stack(hs, dim=0)
        os = self.h2o(hs)

        return os, hs

class HierarchicalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_type='bias',
            attn_group_size=None, plastic=True, activation='retanh',
            dt=0.02, tau_x=0.1, tau_w=1.0, c_plasticity=None, train_init_state=False,
            e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, truncate_iter=None, init_spectral=None, 
            balance_ei=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.x2c = EILinear(input_size, hidden_size, remove_diag=False, pos_function='relu',
                            e_prop=1, zero_cols_prop=0, bias=False, init_gain=0.5)
        self.c2c = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='relu',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1, 
                            init_spectral=init_spectral, balance_ei=balance_ei)
        self.c2h = EILinear(hidden_size, hidden_size, remove_diag=False, pos_function='relu', 
                            e_prop=1, zero_cols_prop=1-e_prop, bias=False, init_gain=0.5)
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='relu',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1, 
                            init_spectral=init_spectral, balance_ei=balance_ei)
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
            self.x10 = nn.Parameter(torch.randn(1, hidden_size))
            self.x20 = nn.Parameter(torch.randn(1, hidden_size))
        else:
            self.x10 = torch.zeros(1, hidden_size)
            self.x20 = torch.zeros(1, hidden_size)
        
        self.plastic = plastic
        if plastic:
            if c_plasticity is not None:
                assert(len(c_plasticity)==12)
                self.kappa_w = torch.FloatTensor(c_plasticity)
            else:
                self.kappa_w = nn.Parameter(torch.zeros(12)+self.alpha_w)

        assert attention_type in ['none', 'bias', 'weight']
        self.attention_type = attention_type
        # TODO: mixed selectivity is required for the soltani et al 2016 model, what does it mean here? add separate layer?
        if attention_type!='none':
            assert(attn_group_size is not None)
            self.attn_func = EILinear(hidden_size, hidden_size, remove_diag=False, \
                                      e_prop=e_prop, zero_cols_prop=1-e_prop, init_gain=0.1)
            self.attn_group_size = torch.LongTensor(attn_group_size)
        else:
            self.attn_func = None
            self.attn_group_size = None

        self.activation = _get_activation_function(activation)
        if truncate_iter is not None:
            raise NotImplementedError

    def init_hidden(self, x):
        batch_size = x.shape[1]
        h_init = self.x10.to(x.device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        c_init = self.x20.to(x.device) + self._sigma_rec * torch.randn(batch_size, self.hidden_size)
        if self.plastic:
            return (c_init, c_init.relu(), h_init, h_init.relu(),
                    torch.zeros(batch_size, self.hidden_size, self.input_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(x.device))
        else:
            return (h_init, h_init.relu())

    def recurrence(self, x, h, R):
        batch_size = x.shape[0]
        
        # retrieve previous states
        if self.plastic:
            state_c, output_c, state_h, output_h, wxc, wcc, wch, whh = h
        else:
            state_c, output_c, state_h, output_h = h
        
        # calculate attention, it is prospective since the hidden state at the previous state 
        # modulates the attention at the current time step
        if self.attention_type=='bias':
            attn = self.activation(self.attn_func(output_h))
        elif self.attention_type=='weight':
            attn = F.softmax(self.attn_func(output_h), -1)

        # calculate state for c
        if self.plastic:
            total_input_c = self.x2c(x, wxc) + self.c2c(output_c, wcc)
        else:
            total_input_c = self.x2c(x) + self.c2c(output_c)

        # if use additive feedback, add attn when calculating input
        if self.attention_type=='bias':
            total_input_c += attn

        new_state_c = state_c * self.oneminusalpha_x \
                      + total_input_c * self.alpha_x \
                      + self._sigma_rec * torch.randn_like(state_c)
        new_output_c = self.activation(new_state_c)

        # if use multiplicative feedback, modulated firing rate
        if self.attention_type=='weight':
            new_output_c *= attn

        # calculate state for h
        if self.plastic:
            total_input_h = self.c2h(new_output_c, wch) + self.h2h(output_h, whh)
        else:
            total_input_h = self.c2h(new_output_c) + self.h2h(output_h)

        new_state_h = state_h * self.oneminusalpha_x \
                      + total_input_h * self.alpha_x \
                      + self._sigma_rec * torch.randn_like(state_h)
        new_output_h = self.activation(new_state_h)

        if self.plastic:
            R = R.unsqueeze(-1)
            wxc = self.plasticity_func(wxc, R, x, new_output_c, self.kappa_w[0:3], 
                                        lb=-self.x2c.pos_func(self.x2c.weight).detach().unsqueeze(0), ub=None)
            wcc = self.plasticity_func(wcc, R, output_c, new_output_c, self.kappa_w[3:6], 
                                        lb=-self.c2c.pos_func(self.c2c.weight).detach().unsqueeze(0), ub=None)
            wch = self.plasticity_func(wch, R, new_output_c, new_output_h, self.kappa_w[6:9], 
                                        lb=-self.c2h.pos_func(self.c2h.weight).detach().unsqueeze(0), ub=None)
            whh = self.plasticity_func(whh, R, output_h, new_output_h, self.kappa_w[9:12], 
                                        lb=-self.h2h.pos_func(self.h2h.weight).detach().unsqueeze(0), ub=None)
            return (new_state_c, new_output_c, new_state_h, new_output_h, wxc, wcc, wch, whh)
        else:
            return (new_state_c, new_output_c, new_state_h, new_output_h)

    def plasticity_func(self, w, R, pre, post, kappa, lb, ub):
        batch_size = w.shape[0]
        new_w = w * self.oneminusalpha_w + self.dt*R*(
                kappa[0]*pre.unsqueeze(1) +
                kappa[1]*post.unsqueeze(2) +
                kappa[2]*torch.einsum('bi, bj->bij', post, pre)) + \
                self._sigma_w * torch.randn_like(w)
        if lb is not None:
            new_w = torch.maximum(new_w, lb)
        if ub is not None:
            new_w = torch.minimum(new_w, ub)
        return new_w
    
    def truncate(self, hidden):
        return (h.detach() for h in hidden)

    def forward(self, x, Rs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x)

        hs = []
        steps = range(x.size(0))
        for i in steps:
            hidden = self.recurrence(x[i], hidden, Rs[i])
            hs.append(hidden[3])

        hs = torch.stack(hs, dim=0)
        os = self.h2o(hs)

        return os, hs
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

@torch.jit.script
def retanh(x):
    return torch.tanh(F.relu(x))

def _get_activation_function(func_name):
    if func_name=='relu':
        return F.relu
    elif func_name=='softplus':
        return lambda x: F.softplus(x-1)
    elif func_name=='softplus2':
        return lambda x: x/(1-torch.exp(-x))
    elif func_name=='retanh':
        return retanh
    elif func_name=='sigmoid':
        return lambda x: torch.tanh(x-1)+1
    else:
        raise RuntimeError(F"{func_name} is an invalid activation function.")

def _get_pos_function(func_name):
    if func_name=='relu':
        return F.relu
    if func_name=='abs':
        return torch.abs
    else:
        raise RuntimeError(F"{func_name} is an invalid function enforcing positive weight.")

def _get_connectivity_mask_rec(rec_mask, num_areas, e_hidden_units_per_area, i_hidden_units_per_area):
    conn_mask = {}
    # recurrent connection mask
    rec_mask_ee = torch.kron(rec_mask, torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie = torch.kron(rec_mask, torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ei = torch.kron(torch.eye(num_areas), torch.ones(e_hidden_units_per_area, i_hidden_units_per_area))
    rec_mask_ii = torch.kron(torch.eye(num_areas), torch.ones(i_hidden_units_per_area, i_hidden_units_per_area))
    conn_mask['rec'] = torch.cat([
                    torch.cat([rec_mask_ee, rec_mask_ei], dim=1),
                    torch.cat([rec_mask_ie, rec_mask_ii], dim=1)], dim=0)

    # within region and cross_region connectivity
    rec_mask_ee_intra = torch.kron(torch.eye(num_areas), torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie_intra = torch.kron(torch.eye(num_areas), torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    conn_mask['rec_intra'] = torch.cat([
                    torch.cat([rec_mask_ee_intra, rec_mask_ei], dim=1),
                    torch.cat([rec_mask_ie_intra, rec_mask_ii], dim=1)], dim=0)

    # feedforward connectivity
    rec_mask_ee_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),-1), torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),-1), torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    conn_mask['rec_inter_ff'] = torch.cat([
                    torch.cat([rec_mask_ee_intra_ff, rec_mask_ei*0], dim=1),
                    torch.cat([rec_mask_ie_intra_ff, rec_mask_ii*0], dim=1)], dim=0)
    
    # feedback connectivity
    rec_mask_ee_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),1), torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),1), torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    conn_mask['rec_inter_fb'] = torch.cat([
                    torch.cat([rec_mask_ee_intra_ff, rec_mask_ei*0], dim=1),
                    torch.cat([rec_mask_ie_intra_ff, rec_mask_ii*0], dim=1)], dim=0)


    conn_mask['rec_inter'] = conn_mask['rec']-conn_mask['rec_intra']
    return conn_mask

def _get_connectivity_mask_in(in_mask, input_units, e_hidden_units_per_area, i_hidden_units_per_area, key_name):
    conn_mask = {}
    in_mask_e = torch.kron(in_mask, torch.ones(e_hidden_units_per_area, input_units))
    in_mask_i = torch.kron(in_mask, torch.ones(i_hidden_units_per_area, input_units))
    conn_mask[key_name] = torch.cat([in_mask_e, in_mask_i], dim=0)
    return conn_mask

def _get_connectivity_mask_aux(aux_mask, aux_units, e_hidden_units_per_area, i_hidden_units_per_area, key_name):
    conn_mask = {}
    rwd_mask, act_mask = aux_mask
    rwd_units, act_units = aux_units
    in_mask_e = torch.cat([torch.kron(rwd_mask, torch.ones(e_hidden_units_per_area, rwd_units)), \
                           torch.kron(act_mask, torch.ones(e_hidden_units_per_area, act_units))],dim=-1)
    in_mask_i = torch.cat([torch.kron(rwd_mask, torch.ones(i_hidden_units_per_area, rwd_units)), \
                           torch.kron(act_mask, torch.ones(i_hidden_units_per_area, act_units))],dim=-1)
    conn_mask[key_name] = torch.cat([in_mask_e, in_mask_i], dim=0)
    return conn_mask

def _get_connectivity_mask(in_mask, aux_mask, rec_mask, input_units, aux_input_units, e_hidden_units_per_area, i_hidden_units_per_area):
    conn_mask = {}

    num_areas = in_mask.shape[0]
    num_objs = in_mask.shape[1]
    assert(aux_mask[0].shape==(num_areas,1) and aux_mask[1].shape==(num_areas,1))
    assert(in_mask.shape==(num_areas,num_objs))
    assert(rec_mask.shape==(num_areas,num_areas))

    conn_mask.update(_get_connectivity_mask_rec(rec_mask, num_areas, e_hidden_units_per_area, i_hidden_units_per_area))
    conn_mask.update(_get_connectivity_mask_in(in_mask, input_units, e_hidden_units_per_area, i_hidden_units_per_area, key_name='in'))
    conn_mask.update(_get_connectivity_mask_aux(aux_mask, aux_input_units, e_hidden_units_per_area, i_hidden_units_per_area, key_name='aux'))
    return conn_mask

class EILinear(nn.Module):
    def __init__(self, input_size, output_size, remove_diag, zero_cols_prop, 
                 conn_mask=None, e_prop=0.8, bias=True, pos_function='abs', 
                 init_spectral=None, init_gain=None, balance_ei=False):
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

        if conn_mask is None:
            mask = (sign_mask*exist_mask).repeat([output_size, 1])
            self.register_buffer('mask', mask)
        else:
            mask = (sign_mask*exist_mask).repeat([output_size, 1])*conn_mask
            self.register_buffer('mask', mask)
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
            # nn.init.kaiming_uniform_(self.weight, a=1)
            self.weight.data = torch.from_numpy(
                np.random.gamma(np.ones_like(self.mask.numpy()), 
                                np.sqrt(1/(np.abs(self.mask).sum(dim=1, keepdim=True).numpy()+1e-8)), 
                                size=self.weight.data.shape)).float()
            # Scale E weight by E-I ratio
            if balance_ei is not None and self.i_size!=0:
                # self.weight.data[:, :self.e_size] /= self.e_size/self.i_size
                self.weight.data[:, :self.e_size] /= balance_ei

            if init_gain is not None:
                self.weight.data *= init_gain
            if init_spectral is not None:
                self.weight.data *= init_spectral / np.abs(torch.linalg.eigvals(self.effective_weight())).max()

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
            result = torch.bmm(self.effective_weight(w), input.unsqueeze(2)).squeeze(2) 
            if self.bias is not None:
                result += self.bias
            return result

class PlasticSynapse(nn.Module):
    def __init__(self, input_size, output_size, dt_w=1.0, tau_w=100.0, weight_bound=1.0, sigma_w=0.001, **kwargs):
        '''
        w(t+1)(rwd, post, pre) = w(t)+kappa*rwd*(hebb(post, pre)+noise)
        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_bound = weight_bound
        self.lb = 0
        self.ub = weight_bound
        
        self.dt_w = dt_w
        self.tau_w = tau_w
        
        self.alpha_w = dt_w / self.tau_w

        self._sigma_w = np.sqrt(2/self.alpha_w) * sigma_w

        self.kappa = nn.Parameter(torch.ones(self.output_size, self.input_size)*self.alpha_w)

    def forward(self, w, baseline, R, pre, post):
        '''
            w: previous plastic weight 
            baseline: fixed weight
            R: rwd \in {-1, +1}
            pre: pre-synaptic firing rates
            post: post-synaptic firing rates
            kappa: learning rate
            time_decay: number of timesteps where there were no updates
            time_plas: number of timesteps where there were updates
        '''
        new_w = baseline*(self.alpha_w) + w*(1-self.alpha_w) \
            + R*self.kappa.abs()*(torch.bmm(post.unsqueeze(2), pre.unsqueeze(1))\
                                         +self._sigma_w*torch.randn_like(w))
        new_w = torch.clamp(new_w, self.lb, self.ub)
        return new_w
    

class LeakyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, aux_input_size, num_areas, 
                activation='retanh', dt_x=0.02, tau_x=0.1, train_init_state=False,
                e_prop=0.8, sigma_rec=0, sigma_in=0, init_spectral=None, 
                balance_ei=False, conn_mask={}, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aux_input_size = aux_input_size

        self.x2h = EILinear(input_size, hidden_size, remove_diag=False, pos_function='abs',
                            e_prop=1, zero_cols_prop=0, bias=False, 
                            init_gain=math.sqrt(self.input_size/(input_size+aux_input_size)/hidden_size*num_areas),
                            conn_mask=conn_mask.get('in', None))

        if self.aux_input_size>0:
            self.aux2h = EILinear(self.aux_input_size, hidden_size, remove_diag=False, pos_function='abs',
                                  e_prop=1, zero_cols_prop=0, bias=False,
                                  init_gain=math.sqrt(self.aux_input_size/(input_size+aux_input_size)/hidden_size*num_areas),
                                  conn_mask=conn_mask.get('aux', None))
        self.h2h = EILinear(hidden_size, hidden_size, remove_diag=True, pos_function='abs',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1,
                            init_spectral=init_spectral, balance_ei=balance_ei,
                            conn_mask=conn_mask['rec'])
        
        self.tau_x = tau_x
        self.dt_x = dt_x

        self.alpha_x = dt_x / self.tau_x
        self.oneminusalpha_x = 1 - self.alpha_x
        self._sigma_rec = np.sqrt(2*self.alpha_x) * sigma_rec
        self._sigma_in = np.sqrt(2/self.alpha_x) * sigma_in

        if train_init_state:
            self.x0 = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.x0 = torch.zeros(1, hidden_size)

        self.activation = _get_activation_function(activation)


    def forward(self, x, state, wh, aux_x):

        output = self.activation(state)
        x = torch.relu(x + self._sigma_in * torch.randn_like(x))

        total_input = self.h2h(output, wh) + self.x2h(x)

        if self.aux_input_size>0:
            aux = torch.relu(aux_x + self._sigma_in * torch.randn_like(aux_x))
            total_input += self.aux2h(aux)

        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)
        
        return new_state, new_output

class HierarchicalPlasticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_areas, num_options,
                inter_regional_sparsity, inter_regional_gain,
                plastic=True, activation='retanh', train_init_state=False,
                dt_x=0.02, dt_w=1.0, tau_x=0.1, tau_w=100.0, weight_bound=1.0,
                e_prop=0.8, sigma_rec=0, sigma_in=0, sigma_w=0, init_spectral=None, 
                action_input=True, rwd_input=True, balance_ei=False, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =  output_size
        self.num_options = num_options
        self.action_input = action_input
        self.rwd_input = rwd_input
        self.num_areas = num_areas
        self.e_size = int(e_prop * hidden_size)
        # aux input are reward and choice input
        # reward input is 2 units for + - reward
        # action input uses the same encoding scheme as the perceptual input
        self.aux_input_size = [2 if self.rwd_input else 0] + [output_size if self.action_input else 0]
        
        self.plastic = plastic
        self.weight_bound = weight_bound

        # specify connectivity
        in_mask = torch.FloatTensor([1, *[0]*(self.num_areas-1)]).unsqueeze(-1)
        # for _ in range(self.output_size):
        #     in_mask = torch.cat([in_mask, torch.FloatTensor([[0, 1, *[0]*(self.num_areas-2)]]).T], dim=-1)
        # rwd input goes to all areas, action only goes to first area (choice presentation), and last area (motor efference copy)
        aux_mask = [torch.FloatTensor([1, *[1]*(self.num_areas-1)]).unsqueeze(-1), \
                    torch.FloatTensor([*[0]*(self.num_areas-1), 1]).unsqueeze(-1)]
        rec_mask = torch.eye(self.num_areas) + torch.diag(torch.ones(self.num_areas-1), 1) + torch.diag(torch.ones(self.num_areas-1), -1)

        self.conn_masks = _get_connectivity_mask(in_mask=in_mask, aux_mask=aux_mask, rec_mask=rec_mask,
                                                 input_units=input_size, aux_input_units=self.aux_input_size, 
                                                 e_hidden_units_per_area=self.e_size, i_hidden_units_per_area=hidden_size-self.e_size)
        
        self.register_buffer('mask_rec_inter', self.conn_masks['rec_inter'], persistent=False)

        balance_ei = self.conn_masks['rec_intra']\
                    +inter_regional_gain[0]*inter_regional_sparsity[0]\
                        *self.conn_masks['rec_inter_ff']\
                    +inter_regional_gain[1]*inter_regional_sparsity[1]\
                        *self.conn_masks['rec_inter_fb']\
                    -torch.eye(self.hidden_size*self.num_areas)
        balance_ei = balance_ei[:,:self.e_size*self.num_areas].sum(1, keepdim=True)/balance_ei[:,self.e_size*self.num_areas:].sum(1, keepdim=True)

        self.rnn = LeakyRNNCell(input_size=self.input_size, hidden_size=hidden_size*self.num_areas, aux_input_size=sum(self.aux_input_size), plastic=plastic, 
                                activation=activation, dt_x=dt_x, tau_x=tau_x, train_init_state=train_init_state, 
                                e_prop=e_prop, sigma_rec=sigma_rec, sigma_in=sigma_in, init_spectral=init_spectral,
                                balance_ei=balance_ei, conn_mask=self.conn_masks, num_areas=num_areas)
        
        self.plasticity = PlasticSynapse(input_size=self.hidden_size*self.num_areas, output_size=self.hidden_size*self.num_areas, 
                                   dt_w=dt_w, tau_w=tau_w, weight_bound=weight_bound, sigma_w=sigma_w)
        
        # sparsify inter-regional connectivity, but not enforeced
        sparse_mask_ff = (torch.rand((self.conn_masks['rec_inter_ff'].abs().sum().long(),))<inter_regional_sparsity[0])
        sparse_mask_fb = (torch.rand((self.conn_masks['rec_inter_fb'].abs().sum().long(),))<inter_regional_sparsity[1])
        # self.rnn.h2h.weight.data[self.conn_masks['rec_intra'].abs()>0.5] *= 1.5
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_ff'].abs()>0.5] *= sparse_mask_ff*inter_regional_gain[0]
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_ff'].abs()>0.5] += 1e-8
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_fb'].abs()>0.5] *= sparse_mask_fb*inter_regional_gain[1]
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_fb'].abs()>0.5] += 1e-8
        if init_spectral is not None:
            temp_spectral, _ = torch.sort(torch.abs(torch.linalg.eigvals(self.rnn.h2h.effective_weight())), descending=True)
            temp_spectral = temp_spectral[1]
            self.rnn.h2h.weight.data *= init_spectral / temp_spectral

        # choice and value output
        self.h2o = EILinear(self.e_size, self.output_size, remove_diag=False, e_prop=1, zero_cols_prop=0, bias=False, init_gain=1)
        # self.compiled_h2o = torch.compile(self.h2o, mode='reduce-overhead', backend='aot_eager')
        # self.compiled_h2o = self.h2o

        # init state
        if train_init_state:
            self.x0 = nn.Parameter(torch.zeros(1, hidden_size*self.num_areas))
        else:
            self.register_buffer("x0", torch.zeros(1, hidden_size*self.num_areas))

    def init_hidden(self, x):
        batch_size = x.shape[0]
        h_init = self.x0 + self.rnn._sigma_rec * torch.randn(batch_size, self.hidden_size*self.num_areas, device=x.device)
        if self.plastic:
            return [h_init, self.rnn.h2h.pos_func(self.rnn.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1)]
        else:
            return [h_init]

    def reinit_act(self, x):
        batch_size = x.shape[0]
        h_init = self.x0.to(x.device) + self.rnn._sigma_rec * torch.randn(batch_size, self.hidden_size*self.num_areas)
        return [h_init, h_init.relu()]

    def forward(self, x, steps, neumann_order=5, 
                DAs=None, Rs=None, acts=None, 
                hidden=None, w_hidden=None,
                save_all_states=False):
        # initialize firing rate and fixed weight
        if hidden is None and w_hidden is None:
            hidden, w_hidden = self.init_hidden(x)
        
        if save_all_states: 
            hs = []

        # assemble choice and reward input, if any
        if sum(self.aux_input_size)>0:
            aux_x = []
            if self.rwd_input:
                aux_x.append(Rs)
            if self.action_input:
                aux_x.append(acts)
            aux_x = torch.cat(aux_x, dim=-1)
        else:
            aux_x = None

        # fixed point iterations, not keeping gradient
        for _ in range(steps-neumann_order):
            with torch.no_grad():
                hidden, output = self.rnn(x, hidden, w_hidden, aux_x)
            if save_all_states:
                hs.append(output)
        # k-order neumann series approximation
        for _ in range(min(steps, neumann_order)):
            hidden, output = self.rnn(x, hidden, w_hidden, aux_x)
            if save_all_states:
                hs.append(output)

        # if dopamine is not None, update weight
        if DAs is not None:
            w_hidden = self.plasticity(w_hidden, self.rnn.h2h.pos_func(self.rnn.h2h.weight).unsqueeze(0), DAs, output, output)
        
        if save_all_states:
            hs = torch.stack(hs, dim=0)
        else:
            hs = output

        os = self.h2o(output[...,(self.num_areas-1)*self.e_size:self.num_areas*self.e_size])
        return os, hidden, w_hidden, hs

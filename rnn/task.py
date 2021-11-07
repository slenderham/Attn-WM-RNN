import numpy as np
import torch
from matplotlib import pyplot as plt

# TODO: Add reversal functionality
# TODO: support different dimensions and stim values

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
    
    def clear(self):
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
    
    def convert2torch(self):
        return {'action': torch.stack(self.actions, dim=0),
                'logprobs': torch.stack(self.logprobs, dim=0),
                'rewards': torch.stack(self.rewards, dim=0),
                'values': torch.stack(self.values, dim=0)}
    
    def append(self, a, log_p, r, v):
        self.actions.append(a)
        self.logprobs.append(log_p)
        self.rewards.append(r)
        self.values.append(v)

class MDPRL():
    def __init__(self, times, input_type, task):
        self.prob_mdprl = np.zeros((1, 3, 3, 3))
        self.prob_mdprl[:, :, :, 0] = ([0.92, 0.75, 0.43], [0.50, 0.50, 0.50], [0.57, 0.25, 0.08])
        self.prob_mdprl[:, :, :, 1] = ([0.16, 0.75, 0.98], [0.50, 0.50, 0.50], [0.02, 0.25, 0.84])
        self.prob_mdprl[:, :, :, 2] = ([0.92, 0.75, 0.43], [0.50, 0.50, 0.50], [0.57, 0.25, 0.08])

        # generalizable probability matrix
        prob_gen = np.zeros((3, 3, 3))
        prob_gen[:, :, 0] = 0.9
        prob_gen[:, :, 1] = 0.5
        prob_gen[:, :, 2] = 0.1

        # 0.5 probability matrix
        prob_noinf = 0.5*np.ones((3, 3, 3))

        s = 1
        T = np.linspace(times['start_time']*s, times['end_time']*s, 1+int(times['total_time']*s/times['dt']))
        # when stimuli is present on the screen
        self.T_s = (T > times['stim_onset']*s) & (T <= times['stim_end']*s)
        # when dopamine is released
        self.T_da = (T > times['rwd_onset']*s) & (T <= times['rwd_end']*s)
        # when choice is read (only used for making the target)
        self.T_ch = (T > times['choice_onset']*s) & (T <= times['choice_end']*s)
        # when choice is read (used for training the network)

        self.T = T
        self.s = s
        self.times = times

        assert(task in ['value', 'off_policy_single', 'on_policy_single', 'on_policy_double'])
        self.task = task

        assert(input_type in ['feat', 'feat+conj', 'feat+conj+obj']), 'invalid input type'
        if input_type=='feat':
            self.input_indexes = np.arange(0, 9)
        elif input_type=='feat+conj':
            self.input_indexes = np.arange(0, 36)
        elif input_type=='feat+conj+obj':
            self.input_indexes = np.arange(0, 63)
        elif input_type=='feat+obj':
            self.input_indexes = np.concat([np.arange(0, 9), np.arange(36, 63)])
        else:
            raise RuntimeError

        # -----------------------------------------------------------------------------------------
        # initialization
        # -----------------------------------------------------------------------------------------
        index_pttrn = np.zeros((3, 3, 3))
        index_shp = np.zeros((3, 3, 3))
        index_clr = np.zeros((3, 3, 3))

        index_shppttrn = np.zeros((3, 3, 3))
        index_pttrnclr = np.zeros((3, 3, 3))
        index_shpclr = np.zeros((3, 3, 3))

        self.filter_s = self.T_s.astype(int)
        self.filter_da = self.T_da.astype(int).reshape((1, -1))
        self.filter_ch = self.T_ch.astype(int).reshape((1, -1))

        # -----------------------------------------------------------------------------------------
        # indexing features
        # -----------------------------------------------------------------------------------------
        for d in range(3):
            index_shp[:, :, d] = np.matrix([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
            index_pttrn[:, :, d] = np.matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
            index_clr[:, :, d] = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])*d

            index_shppttrn[:, :, d] = index_shp[:, :, d] * 3 + index_pttrn[:, :, d]
            index_pttrnclr[:, :, d] = index_pttrn[:, :, d] * 3 + index_clr[:, :, d]
            index_shpclr[:, :, d] = index_shp[:, :, d]*3 + index_clr[:, :, d]

        self.index_shp = index_shp.flatten().astype(int)
        self.index_pttrn = index_pttrn.flatten().astype(int)
        self.index_clr = index_clr.flatten().astype(int)
        self.index_shppttrn = index_shppttrn.flatten().astype(int)
        self.index_pttrnclr = index_pttrnclr.flatten().astype(int)
        self.index_shpclr = index_shpclr.flatten().astype(int)

        # -----------------------------------------------------------------------------------------
        # generate input population activity
        # -----------------------------------------------------------------------------------------
        index_s = np.arange(0, 27, 1)
        pop_o = np.zeros((len(index_s), len(self.T), 27))
        pop_s = np.zeros((len(index_s), len(self.T), 63))
        for n in range(len(index_s)):
            pop_s[n, :, self.index_shp[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 3+self.index_pttrn[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 6+self.index_clr[index_s[n]]] = self.filter_s*1

            pop_s[n, :, 9+self.index_shppttrn[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 18+self.index_pttrnclr[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 27+self.index_shpclr[index_s[n]]] = self.filter_s*1

            pop_s[n, :, 36+index_s[n]] = self.filter_s*1
            pop_o[n, :, index_s[n]] = self.filter_s*1

        self.pop_s = pop_s
        self.pop_o = pop_o

    def _generate_rand_prob(self, batch_size):
        return np.random.rand(batch_size, 3, 3, 3)

    def generateinput(self, batch_size, N_s, prob_index=None, scramble=True):
        if self.task=='value' or 'single' in self.task:
            return self.generateinput_single(batch_size, N_s, prob_index, scramble)
        else:
            return self.generateinput_double(batch_size, N_s, prob_index)

    def generateinput_double(self, batch_size, N_s, prob_index=None):
        '''
        Generate random stimuli AND choice for learning
        '''
        if prob_index is not None:
            assert(len(prob_index.shape)==4 and prob_index.shape[1:] == (3, 3, 3))
        else:
            prob_index = self._generate_rand_prob(batch_size)

        batch_size = prob_index.shape[0]

        prob_index = np.reshape(prob_index, (batch_size, 27))

        index_s = np.repeat(np.arange(0,27,1), N_s)
        index_s_1 = np.random.permutation(index_s)
        index_s_2 = np.random.permutation(index_s)

        pop_s = np.zeros((len(index_s), len(self.T), batch_size, 2, 63))
        ch_s = np.zeros((len(index_s), len(self.T), batch_size, 2))
        prob_s = np.stack([prob_index[:, index_s_1], prob_index[:, index_s_2]], axis=-1)

        for i in range(batch_size):
            pop_s[:,:,i,0,:] = self.pop_s[index_s_1,:,:]
            pop_s[:,:,i,1,:] = self.pop_s[index_s_2,:,:]
            ch_s[:,:,i,0] = self.filter_ch*prob_index[i, index_s_1].reshape((27*N_s,1))
            ch_s[:,:,i,1] = self.filter_ch*prob_index[i, index_s_2].reshape((27*N_s,1))

        DA_s = self.filter_da.reshape((len(self.T),1,1))

        output_mask = {'fixation': torch.from_numpy((self.T<0.0*self.s)).reshape(1, len(self.T), 1), \
                        'target': torch.from_numpy(self.T_ch).reshape(1, len(self.T), 1)}

        pop_s = pop_s[:,:,:,:,self.input_indexes]
        pop_s = {
            'pre_choice': torch.from_numpy(pop_s[:, self.T <= self.times['choice_end']*self.s]).float(),
            'post_choice': torch.from_numpy(pop_s[:, self.T > self.times['choice_end']*self.s]).float()
        }

        DA_s = {
            'pre_choice': torch.from_numpy(DA_s[self.T <= self.times['choice_end']*self.s]).float(),
            'post_choice': torch.from_numpy(DA_s[self.T > self.times['choice_end']*self.s]).float()
        }

        return DA_s, torch.from_numpy(ch_s).float(), pop_s, \
               torch.stack([torch.from_numpy(index_s_1), torch.from_numpy(index_s_2)], dim=-1), \
               torch.from_numpy(prob_s).transpose(0, 1), output_mask

    def generateinput_single(self, batch_size, N_s, prob_index=None, scramble=True):
        '''
        Generate random stimuli AND choice for learning
        '''
        if prob_index is not None:
            assert(len(prob_index.shape) ==4 and prob_index.shape[1:] == (3, 3, 3))
        else:
            prob_index = self._generate_rand_prob(batch_size)

        batch_size = prob_index.shape[0]

        prob_index = np.reshape(prob_index, (batch_size, 27))

        if scramble:
            index_s = np.repeat(np.arange(0,27,1), N_s)
            index_s = np.random.permutation(index_s)
        else:
            index_s = np.tile(np.arange(0,27,1), N_s)

        pop_s = np.zeros((len(index_s), len(self.T), batch_size, 63))
        ch_s = np.zeros((len(index_s), len(self.T), batch_size, 1))
        R = np.zeros((len(index_s), batch_size))

        for i in range(batch_size):
            pop_s[:,:,i,:] = self.pop_s[index_s,:,:]
            if self.task=='on_policy':
                R[:,i] = np.ones(1, prob_index[i, index_s]) 
                # if on_policy learning, return the mask and decide actual reward during simulation
            else:
                R[:,i] = np.random.binomial(1, prob_index[i, index_s]) 
            ch_s[:,:,i,0] = self.filter_ch*prob_index[i, index_s].reshape((27*N_s,1))
        
        DA_s = self.filter_da.reshape((1,len(self.T),1,1))*(2*R.reshape((len(index_s), 1, batch_size, 1))-1)
        DA_s = DA_s.reshape((len(self.T)*len(index_s), batch_size, 1))
        # ch_s = ch_s.reshape((len(self.T)*len(index_s), batch_size, 1))
        pop_s = pop_s.reshape((len(self.T)*len(index_s), batch_size, 63))

        if self.task=='value':
            output_mask = torch.from_numpy(1.5*(self.T<0.0*self.s) + self.T_ch).reshape(1, len(self.T), 1)
        else:
            output_mask = {'fixation': torch.from_numpy((self.T<0.0*self.s)).reshape(1, len(self.T), 1), \
                           'target': torch.from_numpy(self.T_ch).reshape(1, len(self.T), 1)}

        return torch.from_numpy(DA_s).float(), torch.from_numpy(ch_s).float(), \
            torch.from_numpy(pop_s[:,:,self.input_indexes]).float(), torch.from_numpy(index_s), \
            torch.from_numpy(prob_index[:, index_s]).t(), output_mask

    def generateinputfromexp(self, batch_size, test_N_s):
        return self.generateinput(batch_size, test_N_s, self.prob_mdprl, scramble=False)

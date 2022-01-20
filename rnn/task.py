import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.io as sio
import os

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
    def __init__(self, times, input_type):
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

        self.gen_levels = ['feat_1', 'feat_2', 'feat_3', 'conj', 'feat+conj', 'obj']

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
            index_shpclr[:, :, d] = index_shp[:, :, d] * 3 + index_clr[:, :, d]

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
        pop_s = np.zeros((len(index_s), len(self.T), 63))
        for n in range(len(index_s)):
            pop_s[n, :, self.index_shp[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 3+self.index_pttrn[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 6+self.index_clr[index_s[n]]] = self.filter_s*1

            pop_s[n, :, 9+self.index_shppttrn[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 18+self.index_pttrnclr[index_s[n]]] = self.filter_s*1
            pop_s[n, :, 27+self.index_shpclr[index_s[n]]] = self.filter_s*1

            pop_s[n, :, 36+index_s[n]] = self.filter_s*1

        self.pop_s = pop_s
        
        # -----------------------------------------------------------------------------------------
        # loading experimental conditions
        # -----------------------------------------------------------------------------------------
        subjects = ['aa', 'ab', 'ac', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'al',
                    'am', 'an', 'ao', 'aq', 'ar', 'as', 'at', 'av', 'aw', 'ax', 'ay']
        base_dir = '../Multi-dimensional-probablistic-learning/Behavioral task/PRLexp/inputs'
        self.test_stim_order = []
        # self.test_rwd = []
        for s in subjects:
            exp_inputs = sio.loadmat(os.path.join(base_dir, f'input_{s}.mat'))
            self.choiceMap = exp_inputs['expr']['choiceMap'][0,0]
            self.test_stim_order.append(exp_inputs['input']['inputTarget'][0,0])
            # self.test_rwd.append(exp_inputs['input']['inputReward'][0,0])
        self.test_stim_order = np.stack(self.test_stim_order, axis=0).transpose(2,0,1)-1
        # self.test_rwd = torch.from_numpy(np.stack(self.test_rwd, axis=0).transpose(2,0,1))

    def _generate_generalizable_prob(self, gen_level, jitter=0.01):
        # different level of gernalizability in terms of nonlinear terms: 0 (all linear), 1 (conjunction of two features), 2 (no regularity)
        # feat_1,2,3: all linear terms, with 2,1,0 irrelevant features
        # conj, feat+conj: a conj of two features, with a relevant or irrelevant feature
        # obj: a conj of all features
        assert gen_level in self.gen_levels
        if gen_level in ['feat_1', 'feat_2', 'feat_3']:
            irrelevant_features = 3-int(gen_level[5])
            log_odds = np.random.randn(3,3)
            log_odds[:irrelevant_features] = 0 # make certain features irrelavant
            probs = np.empty((3,3,3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        probs[i,j,k] = (log_odds[0,i]+log_odds[1,j]+log_odds[2,k])/np.sqrt(int(gen_level[5]))
        elif gen_level=='conj':
            log_odds = np.random.randn(9)
            probs = np.empty((3,3,3))
            for ipj in range(9):
                i = ipj//3
                j = ipj%3
                probs[i,j,:] = log_odds[ipj]
        elif gen_level=='feat+conj':
            feat_log_odds = np.random.randn(3)
            conj_log_odds = np.random.randn(9)
            probs = np.empty((3,3,3))
            for i in range(3):
                for jpk in range(9):
                    j = jpk//3
                    k = jpk%3
                    probs[i,j,k] = (feat_log_odds[i]+conj_log_odds[jpk])/np.sqrt(2)
        elif gen_level=='obj':
            probs = np.random.randn(3,3,3)
        else:
            raise RuntimeError

        # add jitter to break draws
        probs = 1/(1+np.exp(-(probs*np.sqrt(2)+jitter*np.random.randn(*probs.shape))))
        
        # permute axis to change the order of dimensoins with different levels of information
        probs = probs.transpose(np.random.permutation(3))
        probs = probs.reshape(1, 3, 3, 3)
        return probs

    def generateinput(self, batch_size, N_s, num_choices, gen_level='obj', prob_index=None, stim_order=None):
        '''
        Generate random stimuli AND choice for learning
        '''
        if prob_index is not None:
            assert(len(prob_index.shape)==4 and prob_index.shape[1:] == (3, 3, 3))
        else:
            if gen_level is not None:
                prob_index = self._generate_generalizable_prob(gen_level)
            else:
                prob_index = np.random.rand(batch_size,3,3,3)

        batch_size = prob_index.shape[0]

        prob_index = np.reshape(prob_index, (batch_size, 27))
        if stim_order is not None:
            len_seq = len(stim_order)
        else:
            len_seq = N_s*27

        if stim_order is None:
            index_s = np.repeat(np.arange(0,27,1), N_s)
            index_s_i = [np.random.permutation(index_s) for _ in range(num_choices)]
            index_s_i = np.stack(index_s_i, axis=1)
            # while np.any([len(np.unique(index_s_i[:,j])) != len(index_s_i[:,j]) for j in range(len(index_s))]):
            #     print(np.sum([len(np.unique(index_s_i[:,j])) != len(index_s_i[:,j]) for j in range(len(index_s))]))
            #     index_s_i = [np.random.permutation(index_s) for _ in range(num_choices)]
            #     index_s_i = np.stack(index_s_i, axis=0)
        else:
            index_s_i= stim_order

        pop_s = np.zeros((len_seq, len(self.T), batch_size, num_choices, 63))
        ch_s = np.zeros((len_seq, len(self.T), batch_size, num_choices))
        prob_s = np.stack([prob_index[:, index_s_i[:,i]] for i in range(num_choices)], axis=-1)
        prob_s += 1e-8*np.random.rand(*prob_s.shape) # for random stickbreaking

        for i in range(batch_size):
            for j in range(num_choices):
                pop_s[:,:,i,j,:] = self.pop_s[index_s_i[:,j],:,:]
                ch_s[:,:,i,j] = self.filter_ch*prob_index[i, index_s_i[:,j]].reshape((len_seq,1))

        DA_s = self.filter_da.reshape((len(self.T),1,1))

        output_mask = {'fixation': torch.from_numpy((self.T<0.0*self.s)).reshape(1, len(self.T), 1), \
                        'target': torch.from_numpy(self.T_ch).reshape(len(self.T))[self.T <= self.times['choice_end']*self.s]}

        pop_s = pop_s[:,:,:,:,self.input_indexes]
        pop_s = {
            'pre_choice': torch.from_numpy(pop_s[:, self.T <= self.times['choice_end']*self.s]).float(),
            'post_choice': torch.from_numpy(pop_s[:, self.T > self.times['choice_end']*self.s]).float()
        }

        DA_s = {
            'pre_choice': torch.from_numpy(DA_s[self.T <= self.times['choice_end']*self.s]).float(),
            'post_choice': torch.from_numpy(DA_s[self.T > self.times['choice_end']*self.s]).float()
        }

        return DA_s, torch.from_numpy(ch_s).float(), pop_s, torch.from_numpy(index_s_i), \
               torch.from_numpy(prob_s).transpose(0, 1), output_mask

    def generateinputfromexp(self, batch_size, test_N_s, num_choices, participant_num):
        return self.generateinput(batch_size, test_N_s, prob_index=self.prob_mdprl, num_choices=num_choices, stim_order=self.test_stim_order[:,participant_num])

    def value_est(self, probdata=None):
        if probdata is None:
            probdata = self.prob_mdprl

        probdata = probdata.reshape(27)

        means_shp = np.empty(3)
        means_pttrn = np.empty(3)
        means_clr = np.empty(3)
        for d in range(3):
            means_shp[d] = probdata[self.index_shp==d].mean()
            means_pttrn[d] = probdata[self.index_pttrn==d].mean()
            means_clr[d] = probdata[self.index_clr==d].mean()

        self.est_shp = means_shp[self.index_shp]
        self.est_pttrn = means_pttrn[self.index_pttrn]
        self.est_clr = means_shp[self.index_clr]

        means_shppttrn = np.empty(9)
        means_pttrnclr = np.empty(9)
        means_shpclr = np.empty(9)
        for d in range(9):
            means_shppttrn[d] = probdata[self.index_shppttrn==d].mean()
            means_pttrnclr[d] = probdata[self.index_pttrnclr==d].mean()
            means_shpclr[d] = probdata[self.index_shpclr==d].mean()

        self.est_shppttrn = means_shppttrn[self.index_shppttrn]
        self.est_pttrnclr = means_pttrnclr[self.index_pttrnclr]
        self.est_shpclr = means_shpclr[self.index_shpclr]
        
        self.est_shppttrnclr = probdata

        return [self.est_shp, self.est_pttrn, self.est_clr, 
                self.est_shppttrn, self.est_pttrnclr, self.est_shpclr, 
                self.est_shppttrnclr]

    def stim_encoding(self):
        return {'C': self.index_clr, 'P': self.index_pttrn, 'S': self.index_shp}

    def generalizability(self, probdata=None):
        raise NotImplementedError
        return
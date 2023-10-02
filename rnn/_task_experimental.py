import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.io as sio
import os

# TODO: Add reversal functionality
# TODO: support different dimensions and stim values

class MDPRL():
    def __init__(self, times, input_type):
        self.prob_mdprl = np.zeros((1, 3, 3, 3))
        self.prob_mdprl[:, :, :, 0] = ([0.92, 0.75, 0.43], [0.50, 0.50, 0.50], [0.57, 0.25, 0.08])
        self.prob_mdprl[:, :, :, 1] = ([0.16, 0.75, 0.98], [0.50, 0.50, 0.50], [0.02, 0.25, 0.84])
        self.prob_mdprl[:, :, :, 2] = ([0.92, 0.75, 0.43], [0.50, 0.50, 0.50], [0.57, 0.25, 0.08])

        s = 1
        # T = np.linspace(times['start_time']*s, times['end_time']*s, 1+int(times['total_time']*s/times['dt']))
        # first phase, fixation, nothing on screen
        self.T_fixation = int((times['stim_onset']-times['start_time'])/times['dt'])
        # second phase, stimuli, only, no choice
        # self.T_stim = (T > times['stim_onset']*s) & (T <= times['stim_end']*s)
        self.T_stim = int((times['choice_onset']-times['stim_onset'])/times['dt'])
        # third phase, choice is made
        # self.T_ch = (T > times['choice_onset']*s) & (T <= times['choice_end']*s)
        self.T_ch = int((times['rwd_onset']-times['choice_onset'])/times['dt'])
        # fourth phase, when dopamine is released
        # self.T_rwd = (T > times['rwd_onset']*s) & (T <= times['rwd_end']*s)
        # self.T_rwd = int((times['rwd_end']-times['rwd_onset'])/times['dt'])
        # when choice is read (used for training the network)
        # self.T_mask = (T > times['mask_onset']*s) & (T <= times['mask_end']*s)

        # self.T = T
        self.s = s
        self.times = times

        assert(input_type in ['feat', 'feat+conj', 'feat+obj', 'feat+conj+obj']), 'invalid input type'
        if input_type=='feat':
            self.input_indexes = np.arange(0, 9)
        elif input_type=='feat+conj':
            self.input_indexes = np.arange(0, 36)
        elif input_type=='feat+conj+obj':
            self.input_indexes = np.arange(0, 63)
        elif input_type=='feat+obj':
            self.input_indexes = np.concatenate([np.arange(0, 9), np.arange(36, 63)])
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

        # self.filter_stim = self.T_stim.astype(int)
        # self.filter_rwd = self.T_rwd.astype(int).reshape((1, -1))
        # self.filter_ch = self.T_ch.astype(int).reshape((1, -1))
        # self.filter_mask = self.T_mask.astype(int).reshape((1, -1))

        # -----------------------------------------------------------------------------------------
        # indexing features
        # -----------------------------------------------------------------------------------------
        for d in range(3):
            index_shp[d, :, :] = d
            index_pttrn[:, d, :] = d
            index_clr[:, :, d] = d
        
        index_shppttrn = index_shp * 3 + index_pttrn
        index_pttrnclr = index_pttrn * 3 + index_clr
        index_shpclr = index_shp * 3 + index_clr

        self.index_shp = index_shp.flatten().astype(int)
        self.index_pttrn = index_pttrn.flatten().astype(int)
        self.index_clr = index_clr.flatten().astype(int)
        self.index_shppttrn = index_shppttrn.flatten().astype(int)
        self.index_pttrnclr = index_pttrnclr.flatten().astype(int)
        self.index_shpclr = index_shpclr.flatten().astype(int)

        # -----------------------------------------------------------------------------------------
        # generate input population activity
        # -----------------------------------------------------------------------------------------
        index_s = np.arange(0, 27, 1) # 27 objects
        pop_stim = np.zeros((len(index_s), 63))
        for n in range(len(index_s)):
            pop_stim[n, self.index_shp[index_s[n]]] = 1
            pop_stim[n, 3+self.index_pttrn[index_s[n]]] = 1
            pop_stim[n, 6+self.index_clr[index_s[n]]] = 1

            pop_stim[n, 9+self.index_pttrnclr[index_s[n]]] = 1
            pop_stim[n, 18+self.index_shpclr[index_s[n]]] = 1
            pop_stim[n, 27+self.index_shppttrn[index_s[n]]] = 1

            pop_stim[n, 36+index_s[n]] = 1

        self.pop_stim = pop_stim # objects, input_size
        self.pop_ch = torch.eye(27) # objects, input_size, for choice encoding

        # -----------------------------------------------------------------------------------------
        # generate feasible pairs
        # -----------------------------------------------------------------------------------------
        pairs = []
        for i in range(len(index_s)):
            for j in range(len(index_s)):
                if i==j:
                    continue
                if self.index_shp[i]==self.index_shp[j] or \
                   self.index_clr[i]==self.index_clr[j] or \
                   self.index_pttrn[i]==self.index_pttrn[j]:
                    continue
                pairs.append([i,j])
        assert(len(pairs)==27*8)
        self.pairs = np.array(pairs)

        # -----------------------------------------------------------------------------------------
        # loading experimental conditions
        # -----------------------------------------------------------------------------------------
        subjects = ['AA', 'AB',
                    'AC', 'AD',
                    'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 
                    'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 
                    'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 
                    'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 
                    'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'CC', 'DD', 
                    'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 
                    'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', 
                    'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ']
        subjects = [f'inputs/input_{s.lower()}' for s in subjects]
        subjects2 = ['AA', 'AB',
                     'AC', 'AD', 
                     'AE', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 
                     'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 
                     'AW', 'AX', 'AY']
        subjects += [f'inputs2/input_{s}' for s in subjects2]
        base_dir = '../MdPRL/Behavioral task/PRLexp/inputs_all'
        
        # order to stimuli, same as in human experiment
        self.test_stim_order = []
        # randomly permute stim<->reward mapping, without changing the average feature and conjunction values 
        self.test_stim_dim_order = []
        self.test_stim_dim_order_reverse = []
        self.test_stim_val_order = []
        self.test_stim_val_order_reverse = []
        self.test_stim_order_sensory = []
        self.test_stim2sensory_idx = []
        self.test_sensory2stim_idx = []
        # actual sampled reward, same as in human experiment
        self.test_rwd = []
        
        for s in subjects:
            exp_inputs = sio.loadmat(os.path.join(base_dir, f'{s}.mat'))
            self.choiceMap = exp_inputs['expr']['choiceMap'][0,0]
            self.test_stim_order.append(exp_inputs['input']['inputTarget'][0,0]-1)
            # to test for the network's behavior without bias:
            # randomly permute correspondence between stim dim and schedule dim
            curr_stim_dim_order = np.random.permutation(3)
            self.test_stim_dim_order.append(curr_stim_dim_order)
            self.test_stim_dim_order_reverse.append(np.argsort(curr_stim_dim_order))
            # randomly permute correspondence of feature values with stim dim
            self.test_stim_val_order.append([])
            self.test_stim_val_order_reverse.append([])
            # for each dimension, randomly permute
            for d in range(3):
                curr_dim_val_order = np.random.permutation(3)
                self.test_stim_val_order[-1].append(curr_dim_val_order)
                self.test_stim_val_order_reverse[-1].append(np.argsort(curr_dim_val_order))
            self.test_stim_val_order[-1] = np.stack(self.test_stim_val_order[-1], axis=0)
            self.test_stim_val_order_reverse[-1] = np.stack(self.test_stim_val_order_reverse[-1], axis=0)
            # using the above two permutations, permute the objects to get the 
            curr_stim2sensory_idx, curr_sensory2stim_idx = self.stim_to_sensory(self.test_stim_dim_order[-1], \
                                                                                self.test_stim_val_order[-1])
            self.test_stim2sensory_idx.append(curr_stim2sensory_idx)
            self.test_sensory2stim_idx.append(curr_sensory2stim_idx)
            
            self.test_rwd.append(exp_inputs['input']['inputReward'][0,0])
        
        self.test_stim_order = np.stack(self.test_stim_order, axis=0).transpose(2,0,1)
        self.test_stim_dim_order = np.stack(self.test_stim_dim_order, axis=0)
        self.test_stim_dim_order_reverse = np.stack(self.test_stim_dim_order_reverse, axis=0)
        self.test_stim_val_order = np.stack(self.test_stim_val_order, axis=0)
        self.test_stim_val_order_reverse = np.stack(self.test_stim_val_order_reverse, axis=0)
        self.test_stim2sensory_idx = np.stack(self.test_stim2sensory_idx, axis=0)
        self.test_sensory2stim_idx = np.stack(self.test_sensory2stim_idx, axis=0)
        self.test_rwd = np.stack(self.test_rwd, axis=0).transpose(0,2,1)

    def _generate_generalizable_prob(self, gen_level, jitter=0.0, reward_mean_scale=[-1, 1], reward_range_scale=[2, 8]):
        # different level of gernalizability in terms of nonlinear terms: 0 (all linear), 1 (conjunction of two features), 2 (no regularity)
        # feat_1,2,3: all linear terms, with 2,1,0 irrelevant features
        # conj, feat+conj: a conj of two features, with a relevant or irrelevant feature
        # obj: a conj of all features
        # assert gen_level in self.gen_levels
        if gen_level in ['feat_1', 'feat_2', 'feat_3']:
            irrelevant_features = 3-int(gen_level[5])
            log_odds = np.random.randn(3,3)
            log_odds[:irrelevant_features] = 0 # make certain features irrelavant
            probs = np.empty((3,3,3))*np.nan
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        probs[i,j,k] = (log_odds[0,i]+log_odds[1,j]+log_odds[2,k])/np.sqrt(int(gen_level[5]))
        elif gen_level=='conj':
            log_odds = np.random.randn(9)
            probs = np.empty((3,3,3))*np.nan
            for ipj in range(9):
                i = ipj//3
                j = ipj%3
                probs[i,j,:] = log_odds[ipj]
        elif gen_level=='feat+conj':
            feat_log_odds = np.random.randn(3)
            conj_log_odds = np.random.randn(9)
            probs = np.empty((3,3,3))*np.nan
            for i in range(3):
                for jpk in range(9):
                    j = jpk//3
                    k = jpk%3
                    probs[i,j,k] = (feat_log_odds[i]+conj_log_odds[jpk])/np.sqrt(2)
        elif gen_level=='obj':
            probs = np.random.randn(3,3,3)
        else:
            raise RuntimeError

        # independently control the mean and std of reward
        probs = (probs-np.mean(probs))/np.ptp(probs)
        reward_mean = reward_mean_scale[0]+np.random.rand()*(reward_mean_scale[1]-reward_mean_scale[0]) # uniformly between about 0.3-0.7
        reward_range = reward_range_scale[0]+np.random.rand()*(reward_range_scale[1]-reward_range_scale[0]) # uniformly between about 0.5-1.0
        probs = probs*reward_range+reward_mean
        
        # add jitter to break draws
        probs = 1/(1+np.exp(-(probs+jitter*np.random.randn(*probs.shape))))
        
        # permute axis to change the order of dimensoins with different levels of information
        probs = probs.transpose(np.random.permutation(3))
        probs = probs.reshape(1, 3, 3, 3)
        return probs

    def generateinput(self, batch_size, N_s, num_choices, 
                      gen_level=None, rwd_schedule=None, 
                      stim_order=None, rwd_order=None, 
                      stim2sensory_idx=None):
        
        if batch_size>1:
            raise NotImplementedError
        
        '''
        sample reward schedule for the current session
        '''
        if rwd_schedule is not None:
            assert(len(rwd_schedule.shape)==4 and rwd_schedule.shape[1:] == (3, 3, 3))
        else:
            if gen_level is None:
                gen_level = np.random.choice(self.gen_levels) # sample generalization level
            rwd_schedule = self._generate_generalizable_prob(gen_level)
        batch_size = rwd_schedule.shape[0]
        rwd_schedule = np.reshape(rwd_schedule, (batch_size, 27))
        
        '''
        sample order of presentation
        '''
        if stim_order is not None:
            len_seq = len(stim_order)
        else:
            len_seq = N_s
        
        # index_s_i is index in the reward schedule matrix, not the sensory space
        if stim_order is None:
            if num_choices==2:
                index_s_i = self.pairs[np.random.choice(np.arange(len(self.pairs)), size=len_seq)] 
            elif num_choices==1:
                index_s_i = np.repeat(np.random.permutation(27), N_s)
                index_s_i = np.random.permutation(index_s_i).reshape(len_seq,1)
            else:
                raise ValueError
        else:
            index_s_i = stim_order
        # index_s_i shape is (len_seq X num_choices)

        # true reward prob for each stim
        # do this before changing to the sensory space
        prob_s = np.stack([rwd_schedule[:, index_s_i[:,i]] for i in range(num_choices)], axis=-1) 

        # mapping from the reward schedule matrix to the sensory space, only useful for testing
        if stim2sensory_idx is not None:
            index_s_i = self.permute_mapping(index_s_i, stim2sensory_idx)
        
        if rwd_order is None:
            rwd_s = (np.random.rand(*prob_s.shape)<prob_s) # sampled reward for each stim
        else:
            raise NotImplementedError
        # assert(index_s_i.shape==(len_seq, num_choices)), f"{index_s_i.shape}"
        # assert(prob_s.shape==(batch_size, len_seq, num_choices)), f"{prob_s.shape}"
        # assert(rwd_s.shape==(batch_size, len_seq, num_choices)), f"{prob_s.shape}"
        # prob_s += 1e-8*np.random.rand(*prob_s.shape)

        '''
        make the input to the network and target
        '''
        pop_s = np.zeros((len_seq, batch_size, num_choices, 63)) # input population activity
        pop_c = np.zeros((len_seq, batch_size, num_choices, 27)) # choice population activity
        target = np.zeros((len_seq, batch_size))*np.nan # initialize target array

        for i in range(batch_size):
            for j in range(num_choices):
                pop_s[:,i,j,:] = self.pop_stim[index_s_i[:,j],:] # size is num_trials X input size
                pop_c[:,i,j,:] = self.pop_ch[index_s_i[:,j],:] # size is num_trials X input size
            if num_choices==1:
                target[:,i] = prob_s[i,:].squeeze(-1) # if only one choice, the target is the reward prob
            else:
                target[:,i] = np.argmax(prob_s[i,:,:], -1) # if more than one choice, find position of more rewarding target

        # mask for which signals to supervise
        # output_mask = {'fixation': torch.from_numpy((self.T<0.0*self.s)).reshape(1, len(self.T), 1), \
                        # 'target': torch.from_numpy(self.T_mask).reshape(len(self.T))[self.T <= self.times['mask_end']*self.s]}
        
        # select input encoding dimensions (to include conj/obj input or not)
        # pop_s = pop_s[:,:,:,:,self.input_indexes]
        # pop_s = {
            # 'pre_choice': torch.from_numpy(pop_s[:, self.T <= self.times['choice_onset']*self.s]).float(),
            # 'post_choice': torch.from_numpy(pop_s[:, self.T > self.times['choice_onset']*self.s]).float()
        # }

        # mask for reward outcome presentation
        # rwd_mask = self.filter_rwd.reshape((len(self.T),1,1)) 
        # rwd_mask = {
        #     'pre_choice': torch.from_numpy(rwd_mask[self.T <= self.times['choice_onset']*self.s]).float(),
        #     'post_choice': torch.from_numpy(rwd_mask[self.T > self.times['choice_onset']*self.s]).float()
        # }

        # mask for choice presentation
        # ch_mask = self.filter_ch.reshape((len(self.T),1,1)) 
        # ch_mask = {
        #     'pre_choice': torch.from_numpy(ch_mask[self.T <= self.times['choice_onset']*self.s]).float(),
        #     'post_choice': torch.from_numpy(ch_mask[self.T > self.times['choice_onset']*self.s]).float()
        # }

        # target = {
        #     'pre_choice': torch.from_numpy(target[:, self.T <= self.times['choice_onset']*self.s]),
        #     'post_choice': torch.from_numpy(target[:, self.T > self.times['choice_onset']*self.s])
        # }

        return torch.from_numpy(pop_s).float(), torch.from_numpy(pop_c).float(), \
               torch.from_numpy(rwd_s).long().transpose(0, 1), torch.from_numpy(target).long(), \
               torch.from_numpy(index_s_i).long(), \
               torch.from_numpy(prob_s).transpose(0, 1)

    def generateinputfromexp(self, batch_size, test_N_s, num_choices, participant_num):
        return self.generateinput(batch_size, test_N_s, 
                                 rwd_schedule=self.prob_mdprl, 
                                 num_choices=num_choices, 
                                 stim_order=self.test_stim_order[:,participant_num],
                                 stim2sensory_idx=self.test_stim2sensory_idx[participant_num])

    def value_est(self, probdata=None):
        if probdata is None:
            probdata = self.prob_mdprl.copy()

        probdata = probdata.flatten()

        means_shp = np.empty(3)
        means_clr = np.empty(3)
        means_pttrn = np.empty(3)
        for d in range(3):
            means_shp[d] = probdata[self.index_shp==d].mean()
            means_pttrn[d] = probdata[self.index_pttrn==d].mean()
            means_clr[d] = probdata[self.index_clr==d].mean()

        est_shp = means_shp[self.index_shp]
        est_pttrn = means_pttrn[self.index_pttrn]
        est_clr = means_clr[self.index_clr]

        means_pttrnclr = np.empty(9)
        means_shppttrn = np.empty(9)
        means_shpclr = np.empty(9)
        for d in range(9):
            means_shppttrn[d] = probdata[self.index_shppttrn==d].mean()
            means_pttrnclr[d] = probdata[self.index_pttrnclr==d].mean()
            means_shpclr[d] = probdata[self.index_shpclr==d].mean()

        est_shppttrn = means_shppttrn[self.index_shppttrn]
        est_pttrnclr = means_pttrnclr[self.index_pttrnclr]
        est_shpclr = means_shpclr[self.index_shpclr]
        
        est_shppttrnclr = probdata

        return [est_shp, est_pttrn, est_clr,
                est_pttrnclr, est_shpclr, est_shppttrn, 
                est_shppttrnclr]

    def stim_encoding(self, encoding_type='feature_idx'):
        if encoding_type=='feature_idx':
            return {'C': self.index_clr, 'P': self.index_pttrn, 'S': self.index_shp}
        elif encoding_type=='all_onehot':
            encmat = self.pop_stim
            assert(encmat.shape==(27,63))
            return encmat

    def reversal(self, probs):
        new_order = np.random.permutation(3)
        dim_to_flip = np.random.randint(0, 3)
        new_probs = np.take(probs, new_order, dim_to_flip)
        return new_probs
    
    def stim_to_sensory(self, stim_dim_order, stim_val_order):
        # given a permutation of feature dimension permutation and feature value permutation,
        # calculate the mapping from reward schedule matrix to sensory matrix and back
        obj2sensory_idx = np.arange(3**3).reshape((3,3,3)).astype(int)
        obj2sensory_idx = np.transpose(obj2sensory_idx, stim_dim_order)
        for dim_idx in range(3):
            obj2sensory_idx = np.take(obj2sensory_idx, indices=stim_val_order[dim_idx], axis=dim_idx)
        obj2sensory_idx = obj2sensory_idx.reshape(3**3)
        # each entry of the 
        sensory2obj_idx = np.argsort(obj2sensory_idx)
        return obj2sensory_idx.astype(int), sensory2obj_idx.astype(int)
    
    def permute_mapping(self, orig, mapping):
        # apply the new mapping to the original sequence
        # each entry mapping[i]->new obj index
        new_order = np.ones_like(orig)*np.nan
        for obj_idx in range(3**3):
            new_order[orig==obj_idx] = mapping[obj_idx]
        new_order = new_order.astype(int)
        return new_order

    # def generate_test_schedule(self, gen_level, log_odds):
    #     rwd_schedule = np.zeros((1,3,3,3))
    #     if gen_level=="feat_1":
    #         # if log_odds = 2.25, roughly (0.9, 0.5, 0.1)
    #         # if log_odds = 1.5, roughly (0.8, 0.5, 0.2)
    #         # if log_odds = 0.85, roughly (0.7, 0.5, 0.3)
    #         rwd_schedule[:,0,...] = log_odds["feat_1"]
    #         rwd_schedule[:,2,...] = -log_odds["feat_1"]
    #     elif gen_level=="feat_2":
    #         # for a [[0.9, 0.8, 0.65], 
    #         #        [0.7, 0.5, 0.3],
    #         #        [0.35, 0.2, 0.1]] 
    #         # use log odds 1: 1.4, log_odds 2: 0.85
    #         rwd_schedule[:,0,:,:] += log_odds["feat_1"]
    #         rwd_schedule[:,2,:,:] += -log_odds["feat_1"]
    #         rwd_schedule[:,:,0,:] += log_odds["feat_2"]
    #         rwd_schedule[:,:,2,:] += -log_odds["feat_2"]
    #     elif gen_level=="feat_3":
    #         rwd_schedule[:,0,:,:] += log_odds["feat_1"]
    #         rwd_schedule[:,2,:,:] += -log_odds["feat_1"]
    #         rwd_schedule[:,:,0,:] += log_odds["feat_2"]
    #         rwd_schedule[:,:,2,:] += -log_odds["feat_2"]
    #         rwd_schedule[:,:,:,0] += log_odds["feat_3"]
    #         rwd_schedule[:,:,:,2] += -log_odds["feat_3"]
    #     if gen_level=="conj":
    #         rwd_schedule[:,0,0,:] += log_odds["feat_1"]
    #         rwd_schedule[:,0,2,:] += log_odds["feat_1"]
    #         rwd_schedule[:,2,0,:] += log_odds["feat_2"]
    #         rwd_schedule[:,2,2,:] += log_odds["feat_2"]

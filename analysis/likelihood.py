from multiprocessing.sharedctypes import Value
from unittest import skip
import numpy as np
from scipy.special import softmax
import scipy.io as sio
import os
from scipy.optimize import minimize
import pickle


FEATURE_DIMENSIONS = ['clr', 'shp', 'pttrn', 'shppttrn', 'pttrnclr', 'shpclr', 'obj']
CHANNEL_SIZES = [3, 3, 3, 9, 9, 9, 27]

index_pttrn = np.zeros((3, 3, 3))
index_shp = np.zeros((3, 3, 3))
index_clr = np.zeros((3, 3, 3))

index_shppttrn = np.zeros((3, 3, 3))
index_pttrnclr = np.zeros((3, 3, 3))
index_shpclr = np.zeros((3, 3, 3))
for d in range(3):
    index_shp[:, :, d] = np.matrix([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    index_pttrn[:, :, d] = np.matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    index_clr[:, :, d] = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])*d

    index_shppttrn[:, :, d] = index_shp[:, :, d] * 3 + index_pttrn[:, :, d]
    index_pttrnclr[:, :, d] = index_pttrn[:, :, d] * 3 + index_clr[:, :, d]
    index_shpclr[:, :, d] = index_shp[:, :, d]*3 + index_clr[:, :, d]

index_shp = index_shp.flatten().astype(int)
index_pttrn = index_pttrn.flatten().astype(int)
index_clr = index_clr.flatten().astype(int)
index_shppttrn = index_shppttrn.flatten().astype(int)
index_pttrnclr = index_pttrnclr.flatten().astype(int)
index_shpclr = index_shpclr.flatten().astype(int)

# print(index_shp, index_clr, index_pttrn, index_shppttrn, index_pttrnclr, index_shpclr)

def obj_to_feat_conj_ind(stim):
    # look up the value at each dimension of a stimulus
    return {
        'clr': index_clr[stim],
        'shp': index_shp[stim],
        'pttrn': index_pttrn[stim],
        'shppttrn': index_shppttrn[stim],
        'pttrnclr': index_pttrnclr[stim],
        'shpclr': index_pttrnclr[stim],
        'obj': stim
    }

def obj_to_feat_conj_values(stims, values):
    all_vals = []
    for s in stims:
        all_vals.append([])
        for k in values.keys():
            stim_k = obj_to_feat_conj_ind(s)[k]
            all_vals[-1].append(values[k][stim_k])
    all_vals = np.array(all_vals)
    assert all_vals.shape==(stims.shape[0], len(values.keys()))
    assert stims.shape[0]==2 # only work for choice among binary options
    return all_vals


class BehaviorLikelihood:
    def __init__(self, update_type, lr_type, attention_type, dims):
        assert update_type in ['uncoupled', 'coupled', 'decay']
        assert lr_type in ['uniform', 'rew_unrew', 'inf_non_inf']
        self.update_type = update_type
        self.dims = dims
        self.attn = AttentionModel(attn_type=attention_type, dims=dims)
        return

    def _init_values(self):
        # make value a dictionary for clarity
        # (dimension, values) pair, all initialized to 0.5
        values = {}
        for k, v in zip(FEATURE_DIMENSIONS, CHANNEL_SIZES):
            values[k] = 0.5*np.ones(v)
        return values

    def _single_trial_ll(self, params, values, stims, choice):
        values_of_stims = obj_to_feat_conj_values(stims, values) # get values for each dim
        attns = self.attn.get_attn_for_choice(params, values, stims) # get attn weights
        assert(attns.shape==(len(self.dims),))
        weighted_values = attns.reshape(1, len(self.dims))*values_of_stims # weight by attn
        biasL = params['biasL']
        logits = weighted_values.sum(1) # only use attn weight, no learned magnitude
        assert(logits.shape==(2,))
        logits[0] += biasL
        beta = params['beta_choice']
        pChoiceL = 1/(1+np.exp(-beta*(logits[0]-logits[1])+biasL))
        pChoiceL = np.clip(pChoiceL, a_min=1e-6, a_max=1-1e-6)
        if choice==0:
            return np.log(pChoiceL)
        else:
            return np.log(1-pChoiceL)

    def _rl_update(self, params, values, stims, choice, rewards):
        attn_for_update = self.attn.get_attn_for_learning(params, values, stims)
        stims = [obj_to_feat_conj_ind(s) for s in stims.tolist()]
        stims_ch = stims[choice[0]]
        reward_ch = rewards[choice[0]]
        if reward_ch>0.5:
            alphas = params['alphas_rew']
            for i, k in enumerate(self.dims):
                values[k][stims_ch[k]] += alphas*(1-values[k][stims_ch[k]])*attn_for_update[i]
        else:
            alphas = params['alphas_unrew']
            for i, k in enumerate(self.dims):
                values[k][stims_ch[k]] -= alphas*values[k][stims_ch[k]]*attn_for_update[i]

        if self.update_type=='coupled' or self.update_type=='decay':
            stims_unch = stims[1-choice[0]]
            reward_unch = rewards[1-choice[0]]
            if reward_unch>0.5:
                alphas = params['alphas_rew']
                for i, k in enumerate(self.dims):
                    values[k][stims_unch[k]] += alphas*(1-values[k][stims_unch[k]])*attn_for_update[i]
            else:
                alphas = params['alphas_unrew']
                for i, k in enumerate(self.dims):
                    values[k][stims_unch[k]] -= alphas*values[k][stims_unch[k]]*attn_for_update[i]

        if self.update_type=='decay':
            decay = params['decay']
            for i, k in enumerate(self.dims):
                stim_to_decay = set(list(range(CHANNEL_SIZES[i])))
                stim_to_decay.remove(stims_ch[k])
                stim_to_decay.remove(stims_unch[k])
                stim_to_decay = list(stim_to_decay)
                values[k][stim_to_decay] -= decay*(values[k][stim_to_decay]-0.5)
        return values

    def forward(self, param_list, x):
        num_trials = len(x['choices'])
        values = self._init_values()
        nll = 0
        params = self._make_param_dict(param_list)
        for i in range(num_trials):
            nll -= self._single_trial_ll(params, values, x['stims'][i], x['choices'][i])
            values = self._rl_update(params, values, x['stims'][i], x['choices'][i], x['rewards'][i])
        return nll

    def _make_param_dict(self, params):
        param_keys = ['alphas_rew', 'alphas_unrew', 'beta_choice', 'beta_attn_choice', 'beta_attn_learn', 'biasL', 'decay']
        return {k: v for k, v in zip(param_keys, params.tolist())}

    def _check_value(self, values):
        for k in self.dims:
            assert np.all(values[k]>0 & values[k]<1)


class AttentionModel:
    def __init__(self, attn_type, dims):
        attn_type_for_choice, attn_type_for_learning = attn_type
        assert attn_type_for_choice in ['constant', 'value_sum', 'value_diff', 'value_max']
        assert attn_type_for_learning in ['constant', 'value_sum', 'value_diff', 'value_max']
        self.attn_type = attn_type
        self.dims = dims

    def _get_attn(self, beta, values, stims, attn_type):
        values = obj_to_feat_conj_values(stims, values)
        if attn_type=='constant':
            return np.ones(len(self.dims))/len(self.dims)
        if attn_type=='value_sum':
            return softmax(np.sum(values, 0)*beta, axis=0)
        elif attn_type=='value_diff':
            return softmax(np.abs(values[0]-values[1])*beta, axis=0)
        elif attn_type=='value_max':
            return softmax(np.max(values, 0)*beta, axis=0)
        else:
            raise ValueError

    def get_attn_for_choice(self, params, values, stims):
        beta = params['beta_attn_choice']
        return self._get_attn(beta, values, stims, self.attn_type[0])

    def get_attn_for_learning(self, params, values, stims):
        beta = params['beta_attn_learn']
        return self._get_attn(beta, values, stims, self.attn_type[1])


if __name__=='__main__':
    # change model state space feature, feature+conj X3, feature+obj X3, object

    NREP = 1

    # subjects = ['aa', 'ab', 'ac', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'al',
    #             'am', 'an', 'ao', 'aq', 'ar', 'as', 'at', 'av', 'aw', 'ax', 'ay']
    subjects = ['aa', 'ab']
    base_dir = '../Multi-dimensional-probablistic-learning/Behavioral task/'

    all_fitting_res = []
    dim_subsets = []


    for i1, a1 in enumerate(['constant', 'value_sum', 'value_diff', 'value_max']):
        all_fitting_res.append([])
        for i2, a2 in enumerate(['constant', 'value_sum', 'value_diff', 'value_max']):
            print(f'Attention model for choice {a1}')
            print(f'Attention model for learning {a2}')
            all_fitting_res[-1].append([])
            for s in subjects:
                exp_inputs = sio.loadmat(os.path.join(base_dir, f'PRLexp/inputs/input_{s}.mat'))
                all_results = sio.loadmat(os.path.join(base_dir, f'PRLexp/SubjectData/PRL_{s}.mat'))

                stimuli = exp_inputs['input']['inputTarget'][0,0].T-1
                rewards = exp_inputs['input']['inputReward'][0,0].T
                choices = all_results['results']['choice'][0,0]-1
                x = {
                    'stims': stimuli,
                    'rewards': rewards,
                    'choices': choices
                }

                best_ll = 1e10
                best_res = None
                for i in range(NREP):
                    print(f'Fitting Subject {s}, iteration {i}')
                    bl = BehaviorLikelihood('decay', 'rew_unrew', (a1, a2), dims=FEATURE_DIMENSIONS)
                    # alpha_rew, alpha_unrew, beta_choice, beta_attn_choice, beta_attn_learn, biasL, decay
                    ll = lambda params: bl.forward(params, x)
                    results = minimize(fun=ll, x0=np.random.rand(7), bounds=[(0,1), (0,1), (0,10), (0,10), (0,10), (-np.inf, np.inf), (0,1)])
                    if results.fun < best_ll:
                        best_res = results
                        best_ll = results.fun
                    all_fitting_res[i1][i2].append(best_res)

    with open('attn_fit_results', 'wb') as f:
        pickle.dump(all_fitting_res, f)
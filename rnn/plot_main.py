from collections import defaultdict
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from torch.serialization import save
from analysis import representational_similarity_analysis
from utils import load_checkpoint
from models import SimpleRNN
from task import MDPRL
from analysis import *
import torch
import os
import json
import math
import numpy as np

def plot_mean_and_std(ax, m, sd, label):
    ax.plot(m, label=label)
    ax.fill_between(range(len(m)), m-sd, m+sd, alpha=0.1)

def plot_imag_centered_cm(ax, im):
    max_mag = im.abs().max()
    ax.imshow(im, vmax=max_mag, vmin=-max_mag, colormap='RdBu')

def plot_connectivity(x2hw, h2hw, hb, h2ow):
    maxmax = abs(max([x2hw.max().item(), h2hw.max().item(), hb.max().item(), h2ow.max().item()]))
    minmin = abs(min([x2hw.min().item(), h2hw.min().item(), hb.min().item(), h2ow.min().item()]))
    vbound = max([maxmax, minmin])
    fig, axes = plt.subplots(2, 3, \
        gridspec_kw={'width_ratios': [h2hw.shape[1], x2hw.shape[1], 1], 'height_ratios': [h2hw.shape[0], 1]})
    ims = []
    ims.append(axes[0, 0].imshow(h2hw, cmap='bwr', vmin=-vbound*0.3, vmax=vbound*0.3, interpolation='nearest'))
    ims.append(axes[0, 1].imshow(x2hw, cmap='bwr', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ims.append(axes[0, 2].imshow(hb.unsqueeze(1), cmap='bwr', vmin=-vbound*0.5, vmax=vbound*0.5, interpolation='nearest'))
    ims.append(axes[1, 0].imshow(h2ow, cmap='bwr', vmin=-vbound*0.5, vmax=vbound*0.5, interpolation='nearest'))
    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)
    for i in range(2):
        for j in range(3):
            axes[i, j].axis('off')
    plt.axis('off')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(ims[-1], cax=cbar_ax)
    # plt.tight_layout()
    plt.show()
    # plt.savefig('plots/connectivity')

def plot_learning_curve(lm, lsd):
    fig, ax = plt.subplots()
    plot_mean_and_std(ax, lm[0], lsd[0], label='Percent Better')
    plot_mean_and_std(ax, lm[1], lsd[1], label='Reward')
    ax.vlines(x=args['N_s']*args['stim_val']**args['stim_dim'], ymin=0, ymax=(lm+lsd).max()*1.05, colors='black')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Percent Correct')
    plt.tight_layout()
    plt.savefig('plots/learning_curve')

def plot_attn_entropy(attns):
    log_attns = np.log(attns+1e-6)
    ents = -(attns*log_attns).sum(axis=-1)
    ents_mean = ents.mean(1)
    ents_std = ents.std(1)
    fig, ax = plt.subplots()
    plot_mean_and_std(ax, ents_mean, ents_std)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Entropy')
    plt.tight_layout()
    plt.savefig('plots/attn_entropy')

def plot_attn_distribution(attns):
    fig, axes = plt.subplots(args['test_N_s'], 1, )
    mean_attns = attns.mean(1)
    rep_len = len(mean_attns)//args['test_N_s']
    for i in range(10):
        im = axes[i].imshow(mean_attns[i*rep_len:(i+1)*rep_len].t(), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    fig.supxlabel('Time step')
    fig.supylabel('Dimension')
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/attn_bars')

def plot_sorted_matrix(w, selectivity, e_size):
    # from https://github.com/gyyang/nn-brain/blob/master/EI_RNN.ipynb
    ind_sort = np.concatenate((np.argsort(selectivity[:e_size]), np.argsort(selectivity[e_size:])+e_size))
    w = w[ind_sort, :][:, ind_sort]
    ax = plt.subplot(111)
    plot_imag_centered_cm(ax, w)
    return w

def plot_rsa(hs, inputs, stim_order, stim_probs, splits=10):
    n_timesteps, n_batch, n_hidden = hs.shape
    n_steps_par_split = n_timesteps//splits
    probs_by_feat, probs_by_conj, probs_by_obj = stim_probs
    probs_by_feat = probs_by_feat[stim_order[0]] - probs_by_feat[stim_order[1]]
    probs_by_conj = probs_by_conj[stim_order[0]] - probs_by_conj[stim_order[1]]
    probs_by_obj = probs_by_obj[stim_order[0]] - probs_by_obj[stim_order[1]]
    rnn_sims = []
    input_sims = []
    reg_results = defaultdict(list)
    for i in range(splits-1):
        xs = np.concatenate([probs_by_feat[i*n_steps_par_split:(i+1)*n_steps_par_split],
              probs_by_conj[i*n_steps_par_split:(i+1)*n_steps_par_split],
              probs_by_obj[i*n_steps_par_split:(i+1)*n_steps_par_split]], axis=-1).transpose(1,0) # 
        rnn_sim, input_sim, reg_result = representational_similarity_analysis(xs, hs[i*n_steps_par_split:(i+1)*n_steps_par_split])
        rnn_sims.append(rnn_sim)
        input_sims.append(input_sim)
        reg_results.append(reg_result)
    
    plt.subplot(111).plot()

def run_model(args, model, task_mdprl):
    model.eval()
    accs = []
    rwds = []
    all_indices = []
    all_saved_states_pre = defaultdict(list)
    all_saved_states_post = defaultdict(list)
    with torch.no_grad():
        for i in range(10):
            print(i)
            DA_s, ch_s, pop_s, index_s, prob_s, output_mask = task_mdprl.generateinputfromexp(1, args['test_N_s'])
            if args['task_type']=='value':
                output, hs, _ = model(pop_s, DA_s)
                output = output.reshape(args['stim_val']**args['stim_dim']*args['test_N_s'], output_mask.shape[1], 1) # trial X T X batch size
                loss = (output[:, output_mask.squeeze()==1]-ch_s[:, output_mask.squeeze()==1].squeeze(-1)).pow(2).mean(1) # trial X batch size
            else:
                acc = []
                curr_rwd = []
                hidden = None
                for i in range(len(pop_s['pre_choice'])):
                    # first phase, give stimuli and no feedback
                    output, hs, hidden, ss = model(pop_s['pre_choice'][i], hidden=hidden, 
                                                Rs=0*DA_s['pre_choice'], Vs=None,
                                                acts=torch.zeros(args['batch_size'], 2)*DA_s['pre_choice'], 
                                                save_attns=True, save_weights=True)
                    for k, v in ss.items():
                        all_saved_states_pre[k].append(v)
                    all_saved_states_pre['hs'].append(hs)

                    # use output to calculate action, reward, and record loss function
                    logprob, value = output
                    m = torch.distributions.categorical.Categorical(logits=logprob[-1])
                    action = m.sample().reshape(args['batch_size'])
                    rwd = (torch.rand(args['batch_size'])<prob_s[i][range(args['batch_size']), action]).float()
                    acc.append((torch.argmax(logprob[-1], -1)==torch.argmax(prob_s[i], -1)).float())
                    curr_rwd.append(rwd)
                    # use the action (optional) and reward as feedback
                    pop_post = pop_s['post_choice'][i]
                    action_enc = torch.eye(2)[action]
                    pop_post = pop_post*action_enc.reshape(1,1,2,1)
                    action_enc = action_enc*DA_s['post_choice']
                    R = (2*rwd-1)*DA_s['post_choice']
                    if args['rpe']:
                        V = value[-1]*DA_s['post_choice']
                    else:
                        V = None
                    _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=R, Vs=V, acts=action_enc, save_attns=True, save_weights=True)
                    for k, v in ss.items():
                        all_saved_states_post[k].append(v)
                    all_saved_states_post['hs'].append(hs)
                acc = torch.stack(acc, dim=0)
                curr_rwd = torch.stack(curr_rwd, dim=0)
            accs.append(acc)
            rwds.append(curr_rwd)
            all_indices.append(index_s)
        
        accs_means = torch.cat(accs, dim=1).mean(1) # loss per trial
        accs_stds = torch.cat(accs, dim=1).std(1) # loss per trial
        rwds_means = torch.cat(rwds, dim=1).mean(1) # loss per trial
        rwds_stds = torch.cat(rwds, dim=1).std(1) # loss per trial
        all_saved_states = {}
        for k in all_saved_states_pre.keys():
            all_saved_states[k] = torch.cat([torch.cat(all_saved_states_pre[k], dim=1),
                                             torch.cat(all_saved_states_post[k], dim=1)], dim=0)
        all_indices = torch.stack(all_indices, dim=1) # trials X batch size X 2
        return [accs_means, rwds_means], [accs_stds, rwds_stds], all_saved_states, all_indices


# TODO: use FC to characterize feature and object selection neurons, as predicted by the hierarchical model
# def plot_functional_connectivity(args, hs):
    # for _ in range(args['test_N_s']*args['stim_val']**args['stim_dim']):


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Directory of trained model')
    parser.add_argument('--connectivity', action='store_true')
    parser.add_argument('--tensor_decomp', action='store_true')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--rsa', action='store_true')
    parser.add_argument('--learning_curve', action='store_true')
    parser.add_argument('--attn_entropy', action='store_true')
    parser.add_argument('--attn_distribution', action='store_true')
    plot_args = parser.parse_args()

    # load training config
    f = open(os.path.join(plot_args.exp_dir, 'args.json'), 'r')
    args = json.load(f)
    print('loaded args')
    # load model
    exp_times = {
        'start_time': -0.25,
        'end_time': 0.75,
        'stim_onset': 0.0,
        'stim_end': 0.6,
        'rwd_onset': 0.5,
        'rwd_end': 0.6,
        'choice_onset': 0.35,
        'choice_end': 0.5,
        'total_time': 1}
    exp_times['dt'] = args['dt']
    task_mdprl = MDPRL(exp_times, args['input_type'])
    print('loaded_task')

    input_size = {
        'feat': args['stim_dim']*args['stim_val'],
        'feat+obj': args['stim_dim']*args['stim_val']+args['stim_val']**args['stim_dim'], 
        'feat+conj+obj': args['stim_dim']*args['stim_val']+args['stim_dim']*args['stim_val']*args['stim_val']+args['stim_val']**args['stim_dim'],
    }[args['input_type']]

    if args['rwd_input']:
        input_size += 2

    input_unit_group = {
        'feat': [args['stim_dim']*args['stim_val']], 
        'feat+obj': [args['stim_dim']*args['stim_val'], args['stim_val']**args['stim_dim']], 
        'feat+conj+obj': [args['stim_dim']*args['stim_val'], args['stim_dim']*args['stim_val']*args['stim_val'], args['stim_val']**args['stim_dim']]
    }[args['input_type']]

    if args['attn_type']!='none':
        if args['input_type']=='feat':
            attn_group_size = [args['stim_val']]*args['stim_dim']
        elif args['input_type']=='feat+obj':
            attn_group_size = [args['stim_val']]*args['stim_dim'] + [args['stim_val']**args['stim_dim']]
        elif args['input_type']=='feat+conj+obj':
            attn_group_size = [args['stim_val']]*args['stim_dim'] + [args['stim_val']*args['stim_val']]*args['stim_dim'] + [args['stim_val']**args['stim_dim']]
    else:
        attn_group_size = [input_size]
    
    model_specs = {'input_size': input_size, 'hidden_size': args['hidden_size'], 'output_size': 1, 
            'plastic': args['plas_type']=='all', 'attention_type': 'weight', 'activation': args['activ_func'],
            'dt': args['dt'], 'tau_x': args['tau_x'], 'tau_w': args['tau_w'], 'attn_group_size': attn_group_size,
            'c_plasticity': None, 'e_prop': args['e_prop'], 'init_spectral': args['init_spectral'], 'balance_ei': args['balance_ei'],
            'sigma_rec': args['sigma_rec'], 'sigma_in': args['sigma_in'], 'sigma_w': args['sigma_w'], 'rwd_input': args.get('rwd_input', False),
            'input_unit_group': input_unit_group, 'sep_lr': args['sep_lr'], 'plastic_feedback': args['plastic_feedback']}
    model = SimpleRNN(**model_specs)
    state_dict = torch.load(os.path.join(plot_args.exp_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print('loaded model')

    losses_means, losses_stds, all_saved_states, all_indices = run_model(args, model, task_mdprl)
    print('simulation complete')
    
    # load metrics
    metrics = json.load(open(os.path.join(plot_args.exp_dir, 'metrics.json'), 'r'))

    if plot_args.connectivity:
        plot_connectivity(model.x2h.effective_weight().detach(), \
                          model.h2h.effective_weight().detach(), \
                          state_dict['h2h.bias'].detach(), \
                          model.h2o.effective_weight().detach())
    if plot_args.learning_curve:
        plot_learning_curve(losses_means, losses_stds)
    if plot_args.attn_entropy:
        plot_attn_entropy(all_saved_states['attns'])
    if plot_args.attn_distribution:
        plot_attn_distribution(all_saved_states['attns'])
    if plot_args.rsa:
        plot_rsa(all_saved_states['hs'], task_mdprl.value_est(), stim_order=all_indices, stim_probs=task_mdprl.prob_mdprl)

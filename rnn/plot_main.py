from collections import defaultdict
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from torch.serialization import save
from analysis import representational_similarity_analysis, hierarchical_clustering, linear_regression
from utils import load_checkpoint
from models import SimpleRNN, MultiChoiceRNN
from task import MDPRL
from analysis import *
import scipy.cluster.hierarchy as sch
import torch
import os
import json
import math
import numpy as np
# plt.rcParams["figure.figsize"] = (16,10)

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def plot_mean_and_std(ax, m, sd, label):
    if label is not None:
        ax.plot(m, label=label)
    else:
        ax.plot(m)
    ax.fill_between(range(len(m)), m-sd, m+sd, alpha=0.1)

def plot_imag_centered_cm(ax, im):
    max_mag = im.abs().max()*0.3
    im = ax.imshow(im, vmax=max_mag, vmin=-max_mag, cmap='bwr')
    return im

def plot_connectivity(x2hw, h2hw, hb, h2ow, e_size):
    # maxmax = abs(max([x2hw.max().item(), h2hw.max().item(), hb.max().item(), h2ow.max().item()]))
    # minmin = abs(min([x2hw.min().item(), h2hw.min().item(), hb.min().item(), h2ow.min().item()]))
    # vbound = max([maxmax, minmin])
    selectivity = h2ow[0,:e_size]-h2ow[1,:e_size]
    sort_inds = torch.argsort(selectivity)
    sort_inds = torch.cat([sort_inds, torch.arange(e_size, h2hw.shape[0])])
    vbound = 0.15
    # fig, axes = plt.subplots(2, 3, \
        # gridspec_kw={'width_ratios': [h2hw.shape[1], x2hw.shape[1], 1], 'height_ratios': [h2hw.shape[0], 1]})
    fig = plt.figure('connectivity')
    ims = []
    ax01 = fig.add_axes((0.05, 0.1, 0.04, 0.6))
    ims.append(ax01.imshow(x2hw[0][sort_inds], cmap='coolwarm', vmin=-vbound, vmax=vbound, interpolation='nearest', aspect='auto'))
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.axis('off')
    ax02 = fig.add_axes((0.11, 0.1, 0.04, 0.6))
    ims.append(ax02.imshow(x2hw[1][sort_inds], cmap='coolwarm', vmin=-vbound, vmax=vbound, interpolation='nearest', aspect='auto'))
    ax02.set_xticks([])
    ax02.set_yticks([])
    ax02.axis('off')
    ax1w = fig.add_axes((0.18, 0.1, 0.6, 0.6))
    ims.append(ax1w.imshow(h2hw[sort_inds][:,sort_inds], cmap='coolwarm', vmin=-vbound, vmax=vbound, interpolation='nearest', aspect='auto'))
    ax1w.set_xticks([])
    ax1w.set_yticks([])
    ax1w.axis('off')
    ax1b = fig.add_axes((0.80, 0.1, 0.01, 0.6))
    ims.append(ax1b.imshow(hb[sort_inds].unsqueeze(1), cmap='coolwarm', vmin=-vbound, vmax=vbound, interpolation='nearest', aspect='auto'))
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    ax1b.axis('off')
    ax2 = fig.add_axes((0.18, 0.73, 0.6, 0.01))
    ims.append(ax2.imshow(h2ow[:,sort_inds], cmap='coolwarm', vmin=-vbound, vmax=vbound, interpolation='nearest', aspect='auto'))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    # for i in range(2):
    #     for j in range(3):
    #         axes[i, j].axis('off')
    # plt.axis('off')
    # fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.6])
    fig.colorbar(ims[-1], cax=cbar_ax)
    plt.suptitle('Model Connectivity', y=0.8)
    # plt.tight_layout()
    plt.show()
    # plt.savefig('plots/connectivity')

def plot_learning_curve(all_l, lm, lsd):
    fig_all = plt.figure('perf_all')
    ax = fig_all.add_subplot()
    ax.imshow(all_l[0].squeeze(-1).t(), aspect='auto', interpolation='nearest')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Episodes')
    plt.tight_layout()
    plt.savefig('plots/performance_all')

    fig_summ = plt.figure('perf_summary')
    ax = fig_summ.add_subplot()
    window_size = 20
    plot_mean_and_std(ax, np.convolve(lm[0], np.ones(window_size)/window_size,'valid'), lsd[0].numpy()[:-window_size+1], label='Percent Better')
    plot_mean_and_std(ax, np.convolve(lm[1], np.ones(window_size)/window_size,'valid'), lsd[1].numpy()[:-window_size+1], label='Reward')
    ax.vlines(x=args['N_s']*args['stim_val']**args['stim_dim'], ymin=0.3, ymax=0.9, colors='black')
    ax.legend()
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
    fig, axes = plt.subplots(16, 1)
    mean_attns = attns.flatten(0,1).mean(1)
    rep_len = len(mean_attns)//16
    for i in range(16):
        im = axes[i].imshow(mean_attns[i*rep_len:(i+1)*rep_len].t(), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    fig.supxlabel('Time step')
    fig.supylabel('Dimension')
    # fig.colorbar(im)
    # plt.tight_layout()
    plt.show()
    # plt.savefig('plots/attn_bars')

def plot_sorted_matrix(w, e_size, w_type):
    # from https://github.com/gyyang/nn-brain/blob/master/EI_RNN.ipynb
    Z = hierarchical_clustering(w[:e_size, :e_size])
    fig = plt.figure()
    ax1 = fig.add_axes([0.09,0.22,0.2,0.48])
    Z = sch.dendrogram(Z, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ind_sort = np.concatenate((np.argsort(selectivity[:e_size]), np.argsort(selectivity[e_size:])+e_size))
    ind_sort = np.concatenate((Z['leaves'][:e_size], np.arange(e_size, w.shape[0])))
    ax2 = fig.add_axes([0.3,0.1,0.6,0.6])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axvline(x=e_size-0.5, linewidth=1, color='black')
    ax2.axhline(y=e_size-0.5, linewidth=1, color='black')
    w = w[ind_sort, :][:, ind_sort]
    if w_type=='weight':
        im = plot_imag_centered_cm(ax2, w)
        plt.title('Sorted Recurrent Connectivity')
        plt.colorbar(im)
        plt.savefig('plots/sorted_rec_w')
    elif w_type=='lr':
        im = plt.imshow(w, cmap='hot', vmax=w.max()*0.4)
        plt.title('Sorted Recurrent Learning Rates')
        plt.colorbar(im)
        plt.savefig('plots/sorted_rec_lr')
    else:
        raise ValueError

def plot_single_unit_regression(hs, stim_order, stim_probs):
    n_trials, n_timesteps, n_batch, n_hidden = hs.shape
    hs = hs.mean(1)
    stim_probs_ordered = []
    for est in stim_probs:
        stim_probs_ordered.append([])
        for i in range(n_batch):
            stim_probs_ordered[-1].append(est[stim_order[:,i,0]]-est[stim_order[:,i,1]])
        stim_probs_ordered[-1] = np.stack(stim_probs_ordered[-1], axis=-1)
    
    reg_results = []
    xs = np.stack(stim_probs_ordered, -1)
    reg_results.append([])
    for j in range(n_hidden):
        reg_results[-1].append(linear_regression(xs, hs[:,:,j].reshape(n_trials*n_batch)))

    fig = fig = plt.figure('rsa_coeffs')
    labels = ['Shape', 'Pattern', 'Color', 'Shape+Pattern', 'Pattern+Color', 'Shape+Color', 'Shape+Pattern+Color']
    ax = fig.add_subplot()
    for i in range(7):
        coeffs = [res.coef[i+1] for res in reg_results]
        ses = [res.se[i+1] for res in reg_results]
        ax.plot(coeffs, alpha=0.1)
        ax.errorbar(range(n_hidden), coeffs, ses, capsize=3)
    
def plot_rsa(hs, stim_order, stim_probs, splits=8):
    n_trials, n_timesteps, n_batch, n_hidden = hs.shape
    hs = hs.mean(1)
    n_steps_par_split = n_trials//splits
    stim_probs_ordered = []
    for est in stim_probs:
        stim_probs_ordered.append([])
        for i in range(n_batch):
            stim_probs_ordered[-1].append(est[stim_order[:,i,0]]-est[stim_order[:,i,1]])
        stim_probs_ordered[-1] = np.stack(stim_probs_ordered[-1], axis=-1)
    rnn_sims = []
    input_sims = []
    reg_results = []
    for i in range(splits-1):
        xs = [est[i*n_steps_par_split:(i+1)*n_steps_par_split] for est in stim_probs_ordered]
        rnn_sim, input_sim, reg_result = representational_similarity_analysis(xs, hs[i*n_steps_par_split:(i+1)*n_steps_par_split])
        rnn_sims.append(rnn_sim)
        input_sims.append(input_sim)
        reg_results.append(reg_result)
    
    fig = plt.figure('rsa_coeffs')
    labels = ['Shape', 'Pattern', 'Color', 'Shape+Pattern', 'Pattern+Color', 'Shape+Color', 'Shape+Pattern+Color']
    ax = fig.add_subplot()
    for i in range(7):
        coeffs = [res.coef[i+1] for res in reg_results]
        ses = [res.se[i+1] for res in reg_results]
        plot_mean_and_std(ax, np.array(coeffs), np.array(ses), labels[i])
    ax.legend()
    ax.set_xlabel('Time Segment')
    ax.set_ylabel('Regression Coefficient of RDM')
    plt.tight_layout()
    # plt.savefig('plots/rsa_coeffs')
    plt.show()

def run_model(args, model, task_mdprl):
    model.eval()
    accs = []
    rwds = []
    all_indices = []
    all_saved_states_pre = defaultdict(list)
    all_saved_states_post = defaultdict(list)
    n_samples = 21
    with torch.no_grad():
        for batch_idx in range(n_samples):
            print(batch_idx)
            DA_s, ch_s, pop_s, index_s, prob_s, output_mask = task_mdprl.generateinputfromexp(1, args['test_N_s'], batch_idx)
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
                                                acts=torch.zeros(1, 2)*DA_s['pre_choice'], 
                                                save_attns=True, save_weight=False)
                    for k, v in ss.items():
                        all_saved_states_pre[k].append(v)
                    all_saved_states_pre['hs'].append(hs)

                    # use output to calculate action, reward, and record loss function
                    logprob, value = output
                    m = torch.distributions.categorical.Categorical(logits=logprob[-1])
                    action = m.sample().reshape(1)
                    rwd = (torch.rand(1)<prob_s[i,0,action]).float()
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
                    _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=R, Vs=V, acts=action_enc, save_attns=True, save_weight=False)
                    for k, v in ss.items():
                        all_saved_states_post[k].append(v)
                    all_saved_states_post['hs'].append(hs)
                acc = torch.stack(acc, dim=0)
                curr_rwd = torch.stack(curr_rwd, dim=0)
            accs.append(acc)
            rwds.append(curr_rwd)
            all_indices.append(index_s)
        accs = torch.cat(accs, dim=1)
        rwds = torch.cat(rwds, dim=1)
        accs_means = accs.mean(1) # loss per trial
        accs_stds = accs.std(1)/math.sqrt(n_samples) # loss per trial
        rwds_means = rwds.mean(1) # loss per trial
        rwds_stds = rwds.std(1)/math.sqrt(n_samples) # loss per trial
        all_saved_states = {}
        print(rwds_means.mean(), accs_means.mean())
        for k in all_saved_states_pre.keys():
            all_saved_states[k] = torch.cat([torch.cat(all_saved_states_pre[k], dim=1),
                                             torch.cat(all_saved_states_post[k], dim=1)], dim=0)
            trial_len, batch_times_trials, *hidden_sizes = all_saved_states[k].shape
            all_saved_states[k] = all_saved_states[k].reshape(trial_len, n_samples, len(index_s), *hidden_sizes).transpose(1,2).transpose(0,1)
        all_indices = torch.stack(all_indices, dim=1) # trials X batch size X 2
        return [accs, rwds], [accs_means, rwds_means], [accs_stds, rwds_stds], all_saved_states, all_indices


# TODO: use FC to characterize feature and object selection neurons, as predicted by the hierarchical model
# def plot_functional_connectivity(args, hs):
    # for _ in range(args['test_N_s']*args['stim_val']**args['stim_dim']):


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Directory of trained model')
    parser.add_argument('--connectivity', action='store_true')
    parser.add_argument('--learning_rates', action='store_true')
    parser.add_argument('--sort_rec_w', action='store_true')
    parser.add_argument('--sort_rec_lr', action='store_true')
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
    task_mdprl = MDPRL(exp_times, args['input_type'], args['task_type'])
    print('loaded task')

    input_size = {
        'feat': args['stim_dim']*args['stim_val'],
        'feat+obj': args['stim_dim']*args['stim_val']+args['stim_val']**args['stim_dim'], 
        'feat+conj+obj': args['stim_dim']*args['stim_val']+args['stim_dim']*args['stim_val']*args['stim_val']+args['stim_val']**args['stim_dim'],
    }[args['input_type']]

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
    
    model_specs = {'input_size': input_size, 'hidden_size': args['hidden_size'], 'output_size': 2 if 'double' in args['task_type'] else 1, 
            'plastic': args['plas_type']=='all', 'attention_type': 'weight', 'activation': args['activ_func'],
            'dt': args['dt'], 'tau_x': args['tau_x'], 'tau_w': args['tau_w'], 'attn_group_size': attn_group_size,
            'c_plasticity': None, 'e_prop': args['e_prop'], 'init_spectral': args['init_spectral'], 'balance_ei': args['balance_ei'],
            'sigma_rec': args['sigma_rec'], 'sigma_in': args['sigma_in'], 'sigma_w': args['sigma_w'], 
            'rwd_input': args.get('rwd_input', False), 'action_input': args['action_input'], 
            'input_unit_group': input_unit_group, 'sep_lr': args['sep_lr'], 'plastic_feedback': args['plastic_feedback'],
            'value_est': 'policy' in args['task_type'], 'num_choices': 2 if 'double' in args['task_type'] else 1}
    if 'double' in args['task_type']:
        model = MultiChoiceRNN(**model_specs)
    else:
        model = SimpleRNN(**model_specs)
    state_dict = torch.load(os.path.join(plot_args.exp_dir, 'checkpoint_best.pth.tar'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print('loaded model')


    if plot_args.sort_rec_w:
        plot_sorted_matrix(model.h2h.effective_weight().detach(), int(args['e_prop']*args['hidden_size']), 'weight')
    if plot_args.sort_rec_lr:
        plot_sorted_matrix(model.kappa_rec.relu().squeeze().detach(), int(args['e_prop']*args['hidden_size']), 'lr')
    if plot_args.connectivity:
        plot_connectivity([mx2h.effective_weight().detach() for mx2h in model.x2h], \
                          model.h2h.effective_weight().detach(), \
                          state_dict['h2h.bias'].detach(), \
                          model.h2o.effective_weight().detach(), e_size=int(args['e_prop']*args['hidden_size']))
    if plot_args.learning_rates:
        plot_connectivity([ki.relu().detach()[0] for ki in model.kappa_in], \
                    model.kappa_rec.relu()[0].detach(), \
                    0*state_dict['h2h.bias'].detach(), \
                    model.h2o.effective_weight().detach(), e_size=int(args['e_prop']*args['hidden_size']))

    losses, losses_means, losses_stds, all_saved_states, all_indices = run_model(args, model, task_mdprl)
    print('simulation complete')
    
    # load metrics
    metrics = json.load(open(os.path.join(plot_args.exp_dir, 'metrics.json'), 'r'))

    if plot_args.learning_curve:
        plot_learning_curve(losses, losses_means, losses_stds)
    if plot_args.attn_entropy:
        plot_attn_entropy(all_saved_states['attns'])
    if plot_args.attn_distribution:
        plot_attn_distribution(all_saved_states['attns'])
    if plot_args.rsa:
        plot_rsa(all_saved_states['hs'], stim_probs=task_mdprl.value_est(), stim_order=all_indices)
from collections import defaultdict

import numpy as np
from scipy.signal import convolve2d
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import tqdm
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from plot_utils import *
from utils import load_checkpoint
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import norm, mannwhitneyu, spearmanr, pearsonr


def run_model(args, model_list, task_mdprl, n_samples=None):
    """
    Runs the provided models on the given task and collects all relevant states and outputs.

    Args:
        args (dict): Dictionary of model and task parameters.
        model_list (list): List of PyTorch models to evaluate.
        task_mdprl: Task object with methods for generating input and test stimulus order.
        n_samples (int, optional): Number of samples to run. If None, uses the number of test stimuli.

    Returns:
        dict: Dictionary containing all saved states, outputs, and metadata from the model runs.
    """
    all_saved_states = defaultdict(list)
    output_size = args['output_size']    

    for model in model_list:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False # disable gradient calculation for parameters
    if n_samples is None:
        n_samples = task_mdprl.test_stim_order.shape[1]
        
    n_models = len(model_list)
    model_assignment = np.concatenate([
        np.repeat(np.arange(n_models), n_samples//n_models),
        np.random.choice(np.arange(n_models), size=n_samples%n_models, replace=False)])
    model_assignment = np.random.permutation(model_assignment)
    
    print(np.unique(model_assignment, return_counts=True))
    
    for batch_idx in tqdm.tqdm(range(n_samples)):
        model = model_list[model_assignment[batch_idx]]
        pop_s, rwd_s, _, index_s, prob_s, _ = task_mdprl.generateinputfromexp(
            batch_size=1, test_N_s=args['test_N_s'], num_choices=args['num_options'], participant_num=batch_idx)
        
        # add empty list for the current episode
        all_saved_states['whs'].append([])

        all_saved_states['stimuli'].append(torch.from_numpy(np.expand_dims(index_s, axis=(1,2)))) # num trials X time_steps(1) X batch_size(1) X num_choices
        all_saved_states['reward_probs'].append(torch.from_numpy(np.expand_dims(prob_s, axis=(1,)))) # num_trials X time_steps(1) X batch_size X num_choices
        
        all_saved_states['logits'].append([])
        all_saved_states['choices'].append([])
        all_saved_states['foregone'].append([])
        all_saved_states['rewards'].append([])
        all_saved_states['choose_better'].append([])
        all_saved_states['hs'].append([])
        all_saved_states['rwd_sensitivity'].append([])
#         all_saved_states['ch_sensitivity'].append([])

        # reinitialize hidden layer activity
        hidden = None
        w_hidden = None

        for i in range(len(pop_s)):
            
            curr_trial_hs = []
            
            # first phase, give stimuli and no feedback
            all_x = {
                'stim': torch.zeros_like(pop_s[i].sum(1)),
                'action': torch.zeros(1, output_size),
            }
            _, hidden, w_hidden, hs = model(all_x, steps=task_mdprl.T_fixation, neumann_order = args['neumann_order'],
                                        hidden=hidden, w_hidden=w_hidden, DAs=None, save_all_states=True)
            
            curr_trial_hs.append(hs.detach())

            # second phase, give stimuli and no feedback
            all_x = {
                'stim': pop_s[i].sum(1),
                'action': torch.zeros(1, output_size),
            }
            output, hidden, w_hidden, hs = model(all_x, steps=task_mdprl.T_stim, neumann_order = args['neumann_order'],
                                        hidden=hidden, w_hidden=w_hidden, DAs=None, save_all_states=True)
            curr_trial_hs.append(hs.detach())

            if args['task_type']=='on_policy_double':
                # use output to calculate action, reward, and record loss function
                if args['decision_space']=='action':
                    raise NotImplementedError
                elif args['decision_space']=='good':
                    # action_valid = torch.argmax(output[:,index_s[i]], -1) # the object that can be chosen (0~1), (batch size, )
                    action_valid = torch.multinomial(output['action'][:,index_s[i]].softmax(-1), num_samples=1).squeeze(-1)
                    all_saved_states['logits'][-1].append(
                        (output['action'][:,index_s[i,1]]-output['action'][:,index_s[i,0]]).detach()[None,...])
                    # backpropagate from choice to previous reward
                    if i>0:
                        (output['action'][:,index_s[i,1]]-output['action'][:,index_s[i,0]]).squeeze().backward()
                        all_saved_states['rwd_sensitivity'][-1].append(DAs.grad[None])
#                         all_saved_states['ch_sensitivity'][-1].append(action_enc.grad[None])
                        DAs.grad=None
                    else:
                        all_saved_states['rwd_sensitivity'][-1].append(torch.zeros(1, 1))
#                         all_saved_states['ch_sensitivity'][-1].append(torch.zeros(1, 1, 27))
                    
                    action = index_s[i, action_valid] # (batch size, )
                    nonaction = index_s[i, 1-action_valid] # (batch size, )
                    # rwd = (torch.rand(args['batch_size'])<prob_s[i][range(args['batch_size']), action_valid]).long()
                    rwd = rwd_s[i][0, action_valid]
                    all_saved_states['choose_better'][-1].append((action_valid==torch.argmax(prob_s[i], -1)).float()[None,...]) 
                all_saved_states['rewards'][-1].append(rwd.float()[None,...])
                all_saved_states['choices'][-1].append(action[None,...])
                all_saved_states['foregone'][-1].append(nonaction[None,...])
            elif args['task_type'] == 'value':
                raise NotImplementedError
            
            if args['task_type']=='on_policy_double':
                all_x = {
                    'stim': pop_s[i].sum(1),
                    'action': F.one_hot(action, num_classes=output_size).float(),
                }
                
                DAs = (2*rwd.float()-1)
                DAs = DAs.requires_grad_()
                
                hidden = hidden.detach()
                w_hidden = w_hidden.detach()

                _, hidden, w_hidden, hs = model(all_x, steps=task_mdprl.T_ch, neumann_order = args['neumann_order'],
                                                hidden=hidden, w_hidden=w_hidden, DAs=DAs, save_all_states=True)
                curr_trial_hs.append(hs.detach())

            elif args['task_type'] == 'value':
                raise NotImplementedError
            
            all_saved_states['hs'][-1].append(torch.cat(curr_trial_hs)) 
            # [num_sessions, [num_trials, [time, num_batch_size, hidden_size]]]
            all_saved_states['whs'][-1].append(w_hidden.detach()[None]) 
            # [num_sessions, [num_trials, [1, num_batch_size, hidden_size, hidden_size]]]

        # stack to create a trial dimension for each session
        for k in all_saved_states.keys():
            if isinstance(all_saved_states[k][-1], list):
                all_saved_states[k][-1] = torch.stack(all_saved_states[k][-1], axis=0)
        # [num_sessions, [num_trials, time_steps, num_batch_size, ...]]

    # concatenate all saved states along the batch dimension
    for k in all_saved_states.keys():
        all_saved_states[k] = torch.cat(all_saved_states[k], axis=2) # [num_trials, time_steps, num_sessions, ...]

    # concatenate all accuracies and rewards
    print(all_saved_states['rewards'].mean(), all_saved_states['choose_better'].mean())
    
    for k, v in all_saved_states.items():
        print(k, v.shape)
        
    all_saved_states['model_assignment'] = model_assignment
    all_saved_states['test_stim_dim_order'] = task_mdprl.test_stim_dim_order
    all_saved_states['test_stim_dim_order_reverse'] = task_mdprl.test_stim_dim_order_reverse
    all_saved_states['test_stim_val_order'] = task_mdprl.test_stim_val_order
    all_saved_states['test_stim_val_order_reverse'] = task_mdprl.test_stim_val_order_reverse
    all_saved_states['test_stim2sensory_idx'] = task_mdprl.test_stim2sensory_idx
    all_saved_states['test_sensory2stim_idx'] = task_mdprl.test_sensory2stim_idx
    
    return all_saved_states


def get_input_encodings(wxs, stim_enc_mat):
    """
    Computes input encodings for each stimulus using the provided weight matrix and stimulus encoding matrix.

    Args:
        wxs (np.ndarray): Weight matrix of shape (hidden_size, input_size).
        stim_enc_mat (np.ndarray): Stimulus encoding matrix of shape (27, input_size).

    Returns:
        tuple: Tuple containing the following:
            - Encoded stimuli (np.ndarray)
            - Global average (np.ndarray)
            - Feature averages (np.ndarray)
            - Conjunction averages (np.ndarray)
            - Object averages (np.ndarray)
    """
    # wxs: hidden_size X input_size
    # stim_enc_mat: stim_nums X input_size
    # hypothesis: input pattern for each input ~ avg + feature + conj + obj
    hidden_size, input_size = wxs.shape
    assert(stim_enc_mat.shape==(27, input_size))
    stims = wxs@stim_enc_mat.T # hidden_size X stim_nums
    global_avg = stims.mean(1)
    stims = stims-global_avg[:,None]
    ft_avg = np.empty((9, hidden_size))
    for i in range(9):
        ft_avg[i,:] = stims@stim_enc_mat[:,i].squeeze()/sum(stim_enc_mat[:,i].squeeze()) # hidden_size X stim_nums @ stim_nums
    
    stims = stims-ft_avg.T@stim_enc_mat[:,:9].T
    conj_avg = np.empty((27, hidden_size))
    for i in range(27):
        conj_avg[i,:] = stims@stim_enc_mat[:,9+i].squeeze()/sum(stim_enc_mat[:,9+i].squeeze()) # hidden_size X stim_nums @ stim_nums

    stims = stims-conj_avg.T@stim_enc_mat[:,9:36].T
    obj_avg = stims.T

    return wxs@stim_enc_mat.T, global_avg, ft_avg, conj_avg, obj_avg


def plot_connectivity_lr(sort_inds, x2hw, h2hw, hb, h2ow, aux2h, h2aux, kappa_rec, e_size, mdl_idx, args):
    """
    Plots the connectivity matrices and learning rates of the model.

    Args:
        sort_inds (np.ndarray): Indices for sorting neurons.
        x2hw (np.ndarray): Input-to-hidden weights.
        h2hw (np.ndarray): Hidden-to-hidden weights.
        hb (np.ndarray): Hidden biases.
        h2ow (np.ndarray): Hidden-to-output weights.
        aux2h (np.ndarray): Auxiliary-to-hidden weights.
        h2aux (np.ndarray): Hidden-to-auxiliary weights.
        kappa_rec (np.ndarray): Recurrent learning rates.
        e_size (int): Number of excitatory units.
        mdl_idx (int): Index of the model.
        args (dict): Additional arguments for plotting and saving.
    """
    fig = plt.figure('connectivity', (10, 10))
    ims = []
    hidden_size = h2hw.shape[0]
    PLOT_W = 0.6/hidden_size
    hidden_size = h2hw.shape[0]*PLOT_W
    input_size = x2hw.shape[1]*PLOT_W
    output_size = h2ow.shape[0]*PLOT_W
    MARGIN = 0.01
    LEFT = (1-(input_size+hidden_size+MARGIN*2+PLOT_W))/2
    BOTTOM = 0.1
    
    vbound = np.percentile(x2hw.abs(), 97)
    ax01 = fig.add_axes((LEFT, BOTTOM+2*output_size+MARGIN*2, input_size, hidden_size))
    ims.append(ax01.imshow(x2hw[sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.axis('off')
    ax01.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    vbound = np.percentile(h2hw.abs(), 97)
    ax1w = fig.add_axes((LEFT+input_size+MARGIN, BOTTOM+2*output_size+MARGIN*2, hidden_size, hidden_size))
    ims.append(ax1w.imshow(h2hw[sort_inds][:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax1w.set_xticks([])
    ax1w.set_yticks([])
    ax1w.axis('off')
    ax1w.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    ax1w.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    vbound = np.percentile(hb.abs(), 97)
    ax1b = fig.add_axes((LEFT+input_size+hidden_size+MARGIN*2, BOTTOM+2*output_size+MARGIN*2, PLOT_W, hidden_size))
    ims.append(ax1b.imshow(hb[sort_inds].unsqueeze(1), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    ax1b.axis('off')
    ax1b.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    vbound = np.percentile(h2aux.abs(), 97)
    axoutputaux = fig.add_axes((LEFT+input_size+MARGIN, BOTTOM+output_size+MARGIN, hidden_size, output_size))
    ims.append(axoutputaux.imshow(h2aux[:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    axoutputaux.set_xticks([])
    axoutputaux.set_yticks([])
    axoutputaux.axis('off')
    axoutputaux.axvline(x=e_size-0.5, color='grey', linewidth=0.5)

    vbound = np.percentile(h2ow.abs(), 97)
    axoutputw = fig.add_axes((LEFT+input_size+MARGIN, BOTTOM, hidden_size, output_size))
    ims.append(axoutputw.imshow(h2ow[:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    axoutputw.set_xticks([])
    axoutputw.set_yticks([])
    axoutputw.axis('off')
    axoutputw.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    
    plt.suptitle(f'Model Connectivity {mdl_idx}', y=0.95)
    plt.show()
    with PdfPages(f'plots/{args["plot_save_dir"]}/connectivity{mdl_idx}.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{args["plot_save_dir"]}/connectivity{mdl_idx}.pdf')


    fig = plt.figure('learning_rates', (10, 10))
    ims = []
    
    LEFT = (1-hidden_size)/2
    vbound = np.percentile(kappa_rec.abs(), 97)
    ax1w = fig.add_axes((LEFT, BOTTOM+output_size+MARGIN, hidden_size, hidden_size))
    ims.append(ax1w.imshow(kappa_rec[sort_inds][:,sort_inds].squeeze(), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax1w.set_xticks([])
    ax1w.set_yticks([])
    ax1w.axis('off')
    ax1w.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    ax1w.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    

    plt.suptitle(f'Model Learning Rates {mdl_idx}', y=0.95)
    plt.show()
    with PdfPages(f'plots/{args["plot_save_dir"]}/learning_rates_{mdl_idx}.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{args["plot_save_dir"]}/learning_rates_{mdl_idx}.pdf')
    

def plot_weight_summary(args, ws):
    """
    Plots summary statistics of weight matrices over trials, including update norms, weight norms, and variability.

    Args:
        args (dict): Dictionary of model and task parameters.
        ws (np.ndarray): Weight matrices of shape (trials, timesteps, batch_size, post_dim, pre_dim).
    """
    trials, timesteps, batch_size, post_dim, pre_dim = ws.shape
    assert(timesteps==1)

    all_submats = get_sub_mats(ws, args['num_areas'], 
                               round(args['hidden_size']*args['e_prop']), 
                               round(args['hidden_size']*(1-args['e_prop'])))
    
    submat_keys = [
        ["rec_intra_ee_0", "rec_inter_fb_ee_1_0",],
        ["rec_inter_ff_ee_0_1", "rec_intra_ee_1"],
        ["rec_intra_ie_0", "rec_inter_fb_ie_1_0"],
        ["rec_inter_ff_ie_0_1", "rec_intra_ie_1"],
    ]

    submat_names = [
        [r"EE 0 $\to$ 0", r"EE 1 $\to$ 0"],
        [r"EE 0 $\to$ 1", r"EE 1 $\to$ 1"],
        [r"IE 0 $\to$ 0", r"IE 1 $\to$ 0"],
        [r"IE 0 $\to$ 1", r"IE 1 $\to$ 1"],
    ]

    for k in all_submats.keys():
        all_submats[k] = all_submats[k].squeeze()

    # norm of update
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    for i in range(4):
        for j in range(2):
            sub_w = all_submats[submat_keys[i][j]]
            diff_ws = ((sub_w[1:]-sub_w[:-1])**2).mean([-1, -2])
            plot_mean_and_std(axes[i][j], diff_ws.mean(1), diff_ws.std(1)/np.sqrt(batch_size), 
                              None, color='salmon' if j<=1 else 'skyblue')
            axes[i][j].set_title(submat_names[i][j])
            axes[i][j].tick_params()
    fig.supxlabel('Trial')
    fig.supylabel(r'$|\Delta W|_2$')
    plt.tight_layout()
    sns.despine()
    fig.show()
    print('Finished calculating norm of update')

    # norm of weights
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    for i in range(4):
        for j in range(2):
            sub_w = all_submats[submat_keys[i][j]]
            norm_ws = (sub_w**2).mean([-1, -2])
            plot_mean_and_std(axes[i][j], norm_ws.mean(1), norm_ws.std(1)/np.sqrt(batch_size), 
                              None, color='salmon' if j<=1 else 'skyblue')
            axes[i][j].set_title(submat_names[i][j])
            axes[i][j].tick_params()
    fig.supxlabel('Trial')
    fig.supylabel(r'$|W|_2$')
    plt.tight_layout()
    sns.despine()
    fig.show()
    print('Finished calculating weight norms')

    # variance of entries across trials
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    for i in range(4):
        for j in range(2):
            sub_w = all_submats[submat_keys[i][j]]
            mean_ws = sub_w.mean(1, keepdims=True)
            std_ws = ((sub_w-mean_ws)**2).mean([-1, -2])
            plot_mean_and_std(axes[i][j], std_ws.mean(1), std_ws.std(1)/np.sqrt(batch_size), 
                              None, color='salmon' if j<=1 else 'skyblue')
            axes[i][j].set_title(submat_names[i][j])
            axes[i][j].tick_params()
    fig.supxlabel('Trial')
    fig.supylabel('Cross session variability')
    plt.tight_layout()
    sns.despine()
    fig.show()
    print('Finished calculating variability')
    

def plot_learning_curve(args, all_rewards, all_choose_betters, plot_save_dir):
    """
    Plots the learning curve for model performance, showing reward and percent better over trials.

    Args:
        args (dict): Dictionary of model and task parameters.
        all_rewards (np.ndarray): Array of rewards per trial.
        all_choose_betters (np.ndarray): Array of 'choose better' metrics per trial.
        plot_save_dir (str): Directory to save the plot PDF.
    """
    window_size = 20
    all_choose_betters = convolve2d(all_choose_betters.squeeze(), np.ones((window_size, 1))/window_size, mode='valid')
    all_rewards = convolve2d(all_rewards.squeeze(), np.ones((window_size, 1))/window_size, mode='valid')

    fig_summ = plt.figure('perf_summary', figsize=(6.4,4.8))
    ax = fig_summ.add_subplot()
    plot_mean_and_std(ax, 
                      all_choose_betters.mean(axis=1), 
                      all_choose_betters.std(axis=1)/np.sqrt(all_choose_betters.squeeze().shape[1]), 
                      label='Percent Better', color='grey')
    plot_mean_and_std(ax, 
                      all_rewards.mean(axis=1), 
                      all_rewards.std(axis=1)/np.sqrt(all_rewards.squeeze().shape[1]), 
                      label='Reward', color='black')
    
    ax.vlines(x=args['N_s'], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='black', linestyle='--')
    ax.legend(loc='lower right')
    ax.set_xlabel('Trials')
    plt.tight_layout()
    sns.despine()
    plt.show()
    
    with PdfPages(f'plots/{plot_save_dir}/learning_curve.pdf') as pdf:
        pdf.savefig(fig_summ)
        print(f'Figure saved at plots/{plot_save_dir}/learning_curve.pdf')


def plot_psth_geometry(all_model_dpca, dim_labels,axes, title):
    """
    Plots the geometry of PSTH (peri-stimulus time histogram) features using dPCA results for all models.

    Args:
        all_model_dpca (list): List of dPCA result objects for each model.
        axes (list): List of matplotlib axes to plot on.
        title (str): Title for the plot.
    """
    # Initialize a list to store average weights for all models
    all_model_dpca_psth = []
    
    # Create a hue mapping for plotting, grouping dimensions by feature type
    dim_hue = [0]*3+[1]*3+[2]*3+[3]*9+[4]*9+[5]*9+[6]*27
    dim_hue = np.tile(np.array(dim_hue).reshape(1, 63), len(all_model_dpca['marginalized_psth'])).flatten()

    # Loop through each model's dPCA result to extract and concatenate marginalized PSTHs
    for curr_model_psth in all_model_dpca['marginalized_psth']:
        hidden_size = curr_model_psth['s'].shape[0]
        curr_model_all_psth = np.concatenate([curr_model_psth['s'].squeeze(), 
                                              curr_model_psth['p'].squeeze(), 
                                              curr_model_psth['c'].squeeze(), 
                                              curr_model_psth['pc'].squeeze().reshape((hidden_size, 9)), 
                                              curr_model_psth['sc'].squeeze().reshape((hidden_size, 9)), 
                                              curr_model_psth['sp'].squeeze().reshape((hidden_size, 9)), 
                                              curr_model_psth['spc'].squeeze().reshape((hidden_size, 27))], 
                                              axis=1)
        all_model_dpca_psth.append(curr_model_all_psth.T)
        
    all_model_dpca_psth = np.stack(all_model_dpca_psth)
    
    # Plot cosine similarity heatmap between all features, averaged across models
    all_model_dpca_psth_similarity = batch_cosine_similarity(all_model_dpca_psth, all_model_dpca_psth).mean(0)-np.eye(63)
    cmap_scale = all_model_dpca_psth_similarity.max()*1.1
    
    sns.heatmap(all_model_dpca_psth_similarity, cmap='RdBu_r', vmin=-cmap_scale, vmax=cmap_scale, 
                ax=axes[0], square=True, cbar=False, 
                annot_kws={'fontdict':{'fontsize':10}}, cbar_kws={"shrink": 0.8})
    axes[0].set_title(title)
    
    # Define block boundaries and corresponding tick positions for the heatmap
    block_boundaries = [3, 6, 9, 18, 27, 36]
    ticks = [1.5, 4.5, 7.5, 13.5, 22.5, 31.5, 49.5]
    
    
    # Add block boundary and tick label comments for the heatmap
    for bb in block_boundaries:
        axes[0].axvline(x=bb,color='grey',lw=1)
        axes[0].axhline(y=bb,color='grey',lw=1)
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(dim_labels, fontsize=12, rotation=0)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(dim_labels, fontsize=12)

    xxx_for_plot = np.tile(np.arange(63).reshape(1, 63), len(all_model_dpca['marginalized_psth'])).flatten()

    sns.stripplot(ax=axes[1], x=xxx_for_plot, y=np.linalg.norm(all_model_dpca_psth, axis=2).flatten(), 
                  color='k', linewidth=1, size=1, legend=False, alpha=0.2)
    sns.barplot(ax=axes[1], x=xxx_for_plot, y=np.linalg.norm(all_model_dpca_psth, axis=2).flatten(), 
                hue=dim_hue, errorbar=None, palette='tab10', legend=False)
    axes[1].set_xticks(np.array(ticks)-0.5)
    axes[1].set_xticklabels(dim_labels, fontsize=12, rotation=0)
    axes[1].tick_params(axis='y', labelsize=12)
    
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)


def plot_weight_exp_vars(n_components_for_dpca, all_model_dpca, axes, ylabel):
    """
    Plots the explained variance for each dPCA component across all models.

    Args:
        n_components_for_dpca (dict): Number of components for each dPCA key.
        all_model_dpca (list): List of dPCA result objects for each model.
        axes (matplotlib.axes.Axes): Axes to plot on.
        ylabel (str): Y-axis label for the plot.
    """
    key_plot_order = ['s', 'p', 'c', 'pc', 'sc', 'sp', 'spc']
    
    # Create x-coordinates for plotting: repeat 0-7 for each model and each key
    xxx_for_plot = np.tile(np.arange(8).reshape(1,1,8), [len(all_model_dpca['total_explained_var']),7,1])
    # Create hue values for color coding: repeat 0-6 for each model and each component
    hue_for_plot = np.tile(np.arange(7).reshape(1,7,1), [len(all_model_dpca['total_explained_var']),1,8])

    # Initialize list to store explained variance data for all models
    all_model_dpca_exp_var = []
    
    # Loop through each model's dPCA results
    for curr_model_exp_vars in all_model_dpca['total_explained_var']:
        # Initialize array to store explained variance ratios for all keys and components
        all_dpca_exp_var = np.zeros((7, 8))
        for k_idx, k in enumerate(key_plot_order):
            # Extract explained variance ratios for the current key, up to the specified number of components
            all_dpca_exp_var[k_idx][...,:n_components_for_dpca[k]] = \
                np.array(curr_model_exp_vars[k])
        all_model_dpca_exp_var.append(all_dpca_exp_var)
            
    all_model_dpca_exp_var = np.stack(all_model_dpca_exp_var)
    # Replace very small values (< 1e-6) with NaN to avoid plotting noise
    all_model_dpca_exp_var[all_model_dpca_exp_var<1e-6] = np.nan
    
    # Create a strip plot showing individual data points for each model
    sns.stripplot(ax=axes, x=xxx_for_plot.flatten(), y=all_model_dpca_exp_var.flatten(),
                  hue=hue_for_plot.flatten(), palette='tab10', size=4,
                  legend=False, linewidth=1, dodge=True, alpha=0.25)
    # Create a bar plot showing the mean values, overlaid on the strip plot
    bb = sns.barplot(ax=axes, x=xxx_for_plot.flatten(), y=all_model_dpca_exp_var.flatten(),
                 hue=hue_for_plot.flatten(), palette='tab10', dodge=True, errorbar=None)
    # Remove the legend from the bar plot to avoid duplication
    bb.legend_.remove()
    
    # Set x-axis ticks and labels
    axes.set_xticks(np.arange(0,8,1))
    axes.set_xticklabels([])  # No x-axis labels
    axes.set_title(ylabel)


def test_dpca_overlap(all_dpca_results, n_components_for_dpca, overlap_scale, label, axes):
    """
    Tests and visualizes the overlap between dPCA axes and low-dimensional hidden states across models.

    Args:
        all_dpca_results (list): List of dPCA result objects for each model.
        n_components_for_dpca (dict): Number of components for each dPCA key.
        overlap_scale (float): Scale for significance thresholding.
        label (str): Title for the plot.
        axes (list): List of matplotlib axes to plot on.

    Returns:
        np.ndarray: Array of concatenated dPCA axes for all models.
    """
    keys = list(n_components_for_dpca.keys())
    
    all_model_axes_overlap = []
    all_model_flat_overlap = []
    all_model_low_hs_corr_val = []
    all_model_pvals = []

    num_models = len(all_dpca_results['encoding_axes'])
        
    for mdl_idx in range(num_models):
        all_dpca_axes = np.concatenate([all_dpca_results['encoding_axes'][mdl_idx][k] for k in keys], axis=1) # concat all axes
        all_dpca_low_hs = np.concatenate([all_dpca_results['low_hs'][mdl_idx][k].reshape((all_dpca_results['low_hs'][mdl_idx][k].shape[0],-1)) 
                                          for k in keys], axis=0) # concat all axes
        low_hs_corr_val = np.corrcoef(all_dpca_low_hs)
        axes_overlap = all_dpca_axes.T@all_dpca_axes # dot product similarity
        all_overlaps = []
        all_pvals = []
        sig_thresh = np.abs(norm.ppf(0.001))*overlap_scale

        for k_idx1 in range(len(keys)):
            for k_idx2 in range(k_idx1+1, len(keys)):
                pair_overlaps = (all_dpca_results['encoding_axes'][mdl_idx][keys[k_idx1]].T@\
                                 all_dpca_results['encoding_axes'][mdl_idx][keys[k_idx2]]).flatten()
                all_overlaps.append(pair_overlaps)
                all_pvals.append(norm.cdf(-np.abs(pair_overlaps), loc=0, scale=overlap_scale)+ \
                    norm.sf(np.abs(pair_overlaps), loc=0, scale=overlap_scale))
                
        all_overlaps = np.concatenate(all_overlaps)
        all_pvals = np.concatenate(all_pvals)
        _, all_corrected_pvals = fdrcorrection(all_pvals)

        all_model_axes_overlap.append(axes_overlap)
        all_model_pvals.append(all_corrected_pvals)
        all_model_flat_overlap.append(all_overlaps)
        all_model_low_hs_corr_val.append(low_hs_corr_val)
        
    all_model_axes_overlap = np.stack(all_model_axes_overlap)
    all_model_pvals = np.stack(all_model_pvals)
    all_model_flat_overlap = np.stack(all_model_flat_overlap)
    all_model_low_hs_corr_val = np.stack(all_model_low_hs_corr_val)
        
    tril_mask = np.zeros_like(all_model_axes_overlap.mean(0))
    tril_mask[np.tril_indices(axes_overlap.shape[0], k=-1)] = 1

    triu_mask = np.zeros_like(all_model_axes_overlap.mean(0))
    triu_mask[np.triu_indices(axes_overlap.shape[0], k=0)] = 1
    
    # plot overlap values
    im = sns.heatmap(all_model_axes_overlap.mean(0)*triu_mask+\
                   all_model_low_hs_corr_val.mean(0)*tril_mask, \
                   cmap='RdBu_r', vmin=-1, vmax=1, ax=axes[1], square=True,
                    annot_kws={'fontdict':{'fontsize':10}}, cbar_kws={"shrink": 0.6})
    
    txs, tys = np.meshgrid(np.arange(axes_overlap.shape[0]),np.arange(axes_overlap.shape[0]))
    txs = txs[(np.abs(all_model_axes_overlap.mean(0))>sig_thresh)]
    tys = tys[(np.abs(all_model_axes_overlap.mean(0))>sig_thresh)]
    
    block_boundaries = np.cumsum(list(n_components_for_dpca.values()))[:-1]
    for i in block_boundaries:
        axes[1].axvline(x=i,color='grey',linewidth=0.2)
        axes[1].axhline(y=i,color='grey',linewidth=0.2)
        
    tick_locs = np.cumsum([0, *n_components_for_dpca.values()])[:-1]+\
                np.array(list(n_components_for_dpca.values()))//2-0.5

    axes[1].set_xticks(tick_locs, [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$'], size=15, rotation=0)
    axes[1].set_yticks(tick_locs, [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$'], size=18)
    
    for (x,y) in zip(txs, tys):
        if x<=y:
            continue
        else:
            axes[0].text(x-0.4, y+0.7, '*', {'size': 16})
    
    sns.histplot(x=all_model_flat_overlap.flatten(), color='purple', ax=axes[0], stat='probability', 
                 kde=True, bins=np.linspace(-0.5,0.5,21), linewidth=0.1, alpha=0.2)
    axes[0].set_xlabel('Overlap')
    axes[0].axvline(sig_thresh, color='grey', linestyle=':')
    axes[0].axvline(all_overlaps.mean(), color='black', linestyle='--')
    axes[0].axvline(-sig_thresh, color='grey', linestyle=':')
    
    axes[0].set_title(label, fontsize=24)
    
    axes[0].spines[['right', 'top']].set_visible(False)
    
    print(f"{(all_model_pvals.flatten()<0.05).sum()} out of {np.prod(all_model_pvals.shape)} comparisons were significant")
    
    return


def plot_selectivity_clusters(all_dpca_results, keys, ideal_centroids, E_SIZE, I_SIZE, label, axes):
    
    num_models = len(all_dpca_results['unitwise_explained_var'])
    all_mdl_exp_vars = []

    # Loop through each model's dPCA results and extract the unitwise explained variance ratios
    for dpca_unitwise_exp_vars in all_dpca_results['unitwise_explained_var']:
        curr_mdl_exp_vars = np.stack([np.sum(dpca_unitwise_exp_vars[k], 0) for k in keys])
        all_mdl_exp_vars.append(curr_mdl_exp_vars)

    # Stack the unitwise explained variance ratios for all models
    all_mdl_exp_vars = np.stack([exp_vars.T for exp_vars in all_mdl_exp_vars]) # (num_models, num_units, num_keys)
    
    # Extract the excitatory unitwise explained variance ratios and cluster them
    concat_exp_vars_exc = all_mdl_exp_vars[:,:E_SIZE].reshape(-1,len(keys))
    
    for num_clus_test in range(2, 11):
        kmeans_mdl_exc = SpectralClustering(n_clusters=num_clus_test, assign_labels="kmeans", n_init=20,
                            affinity='cosine', kernel_params={'gamma': 1/len(keys)}).fit(concat_exp_vars_exc)
        print(num_clus_test, np.mean(silhouette_samples(concat_exp_vars_exc, kmeans_mdl_exc.labels_)))
    print("-"*50)
    
    num_clus_exc = len(ideal_centroids[0])
    kmeans_mdl_exc = SpectralClustering(n_clusters=num_clus_exc, assign_labels="kmeans", n_init=20,
                            affinity='cosine', kernel_params={'gamma': 1/len(keys)}).fit(concat_exp_vars_exc)
    

    # Compute the centroids of the excitatory clusters and match them to the ideal centroids
    exp_vars_centroids_exc = []
    for clus in range(num_clus_exc):
        exp_vars_centroids_exc.append(concat_exp_vars_exc[kmeans_mdl_exc.labels_==clus].mean(0))
    exp_vars_centroids_exc = np.stack(exp_vars_centroids_exc)

    # Match the excitatory centroids to the ideal centroids, for each ideal centroid find the centroid that is most correlated
    # ideal_to_clus: index of ideal centroid -> index of excitatory centroid
    # Compute correlation between each cluster centroid and each ideal centroid
    corr_matrix = np.zeros((num_clus_exc, num_clus_exc))
    for i in range(num_clus_exc):
        for j in range(num_clus_exc):
            corr_matrix[i, j] = np.corrcoef(ideal_centroids[0][i], exp_vars_centroids_exc[j])[0, 1]
    _, ideal_to_clus = linear_sum_assignment(-corr_matrix)
    exp_vars_centroids_exc = exp_vars_centroids_exc[ideal_to_clus]

    # change the cluster labels to match the ideal centroids
    clus_to_ideal = np.argsort(ideal_to_clus) # index of excitatory centroid -> index of ideal centroid
    exc_clusters = clus_to_ideal[kmeans_mdl_exc.labels_].reshape((num_models, E_SIZE))

    
    # Extract the inhibitory unitwise explained variance ratios and cluster them
    concat_exp_vars_inh = all_mdl_exp_vars[:,E_SIZE:].reshape(-1,len(keys))
    num_clus_inh = len(ideal_centroids[1])
    
    if not np.isnan(concat_exp_vars_inh).any():
        for num_clus_test in range(2, 11):
            kmeans_mdl_inh = SpectralClustering(n_clusters=num_clus_test, assign_labels="kmeans", n_init=20,
                                affinity='cosine', kernel_params={'gamma': 1/len(keys)}).fit(concat_exp_vars_inh)
            print(num_clus_test, np.mean(silhouette_samples(concat_exp_vars_inh, kmeans_mdl_inh.labels_)))
        print("-"*50)

        kmeans_mdl_inh = SpectralClustering(n_clusters=num_clus_inh, assign_labels="kmeans", n_init=20,
                                    affinity='cosine', kernel_params={'gamma': 1/len(keys)}).fit(concat_exp_vars_inh)
        inh_clusters = kmeans_mdl_inh.labels_
        exp_vars_centroids_inh = []
        for clus in range(num_clus_inh):
            exp_vars_centroids_inh.append(concat_exp_vars_inh[kmeans_mdl_inh.labels_==clus].mean(0))
        exp_vars_centroids_inh = np.stack(exp_vars_centroids_inh)   
        
        # match the inhibitory centroids to the ideal centroids
        # Compute correlation between each cluster centroid and each ideal centroid
        corr_matrix = np.zeros((num_clus_inh, num_clus_inh))
        for i in range(num_clus_inh):
            for j in range(num_clus_inh):
                corr_matrix[i, j] = np.corrcoef(ideal_centroids[1][i], exp_vars_centroids_inh[j])[0, 1]
        _, ideal_to_clus = linear_sum_assignment(-corr_matrix)
        exp_vars_centroids_inh = exp_vars_centroids_inh[ideal_to_clus]
        
        # change the cluster labels to match the ideal centroids
        clus_to_ideal = np.argsort(ideal_to_clus) # index of inhibitory centroid -> index of ideal centroid
        inh_clusters = clus_to_ideal[inh_clusters].reshape((num_models, I_SIZE))
    else:
        concat_exp_vars_inh = np.nan_to_num(concat_exp_vars_inh)
        exp_vars_centroids_inh = np.zeros((num_clus_inh, len(keys)))
        inh_clusters = np.zeros((num_models, I_SIZE))
    
    # concatenate the excitatory and inhibitory centroids
    exp_vars_centroids = np.concatenate([exp_vars_centroids_exc, exp_vars_centroids_inh])

    cmap_scale = min(np.nanmax(np.abs(exp_vars_centroids))*0.9,0.4)
    
    sns.heatmap(exp_vars_centroids, \
                   cmap='Purples', vmin=0, vmax=cmap_scale, ax=axes[0], 
                    annot_kws={'fontdict':{'fontsize':12}}, cbar_kws={"shrink": 0.8})
    axes[0].set_xticks(np.arange(7)+0.5, [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$'])
    axes[0].set_yticks(np.arange(num_clus_exc+num_clus_inh)+0.5, 
                       [f'E{i+1}' for i in range(num_clus_exc)]+[f'I{i+1}' for i in range(num_clus_inh)], 
                       rotation=0)
    axes[0].axhline(num_clus_exc, c='k', lw=2)
    
    
    all_model_exp_var_corr = np.stack([spearmanr(exp_vars, nan_policy='omit').statistic-np.eye(len(keys))
                                              for exp_vars in all_mdl_exp_vars])
    
    cmap_scale = np.nanmax(np.abs(all_model_exp_var_corr.mean(0)))*1.1
    
    sns.heatmap(all_model_exp_var_corr.mean(0).T, \
                   cmap='RdBu_r', vmin=-cmap_scale, vmax=cmap_scale, ax=axes[1], 
                    annot_kws={'fontdict':{'fontsize':12}}, cbar_kws={"shrink": 0.8})
    
    axes[1].set_xticks(np.arange(7)+0.5, [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$'])
    axes[1].set_yticks(np.arange(7)+0.5, [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$'], 
                       rotation=0)

    axes[0].set_title(label)
    
    return [exc_clusters, inh_clusters]


# TODO: align all low dim dpca outputs to the orthogonal contrasts and 
#       apply the dpca axes to the input and output weights to get the reparameterized weights -> get the reparameterized lrs
# TODO: turn the sensitivity analysis into a function
# TODO: turn the dpca analyses of the firing rates into a function
# TODO: plot the connectivity and learning rates by clusters


def plot_input_output_overlap(all_model_dpca_in, all_model_dpca_out, n_components_for_dpca, dim_labels,axes):

    '''
    Plots the overlap between input and output axes. Orthogonalize the input axes from the output axes.

    Args:
    all_model_dpca_in: dict, result of get_all_dpca_results, for the input weights
    all_model_dpca_out: dict, result of get_all_dpca_results, for the output weights
    n_components_for_dpca: dict, number of components for each dimension
    dim_labels: list, labels for each dimension
    axes: matplotlib axes, axes to plot on
    '''

    all_model_output_choice_overlap = []

    # compute the overlap between input and output axes
    num_models = len(all_model_dpca_in['encoding_axes'])
    for idx_model in range(num_models):
        input_output_overlap = np.nan*np.empty((7,7))
        for k_out_idx, k_out in enumerate(n_components_for_dpca.keys()):
            for k_in_idx, k_in in enumerate(n_components_for_dpca.keys()):
                input_output_overlap[k_out_idx, k_in_idx] = \
                    np.sum((all_model_dpca_out['encoding_axes'][idx_model][k_out].T@
                            all_model_dpca_in['encoding_axes'][idx_model][k_in])**2)/\
                        all_model_dpca_out['encoding_axes'][idx_model][k_out].shape[1]

        all_model_output_choice_overlap.append(input_output_overlap)
    all_model_output_choice_overlap = np.stack(all_model_output_choice_overlap)

    cmap_scale = all_model_output_choice_overlap.mean(0).max()
    sns.heatmap(all_model_output_choice_overlap.mean(0), vmin=0, vmax=cmap_scale, ax=axes, 
                square=True, cbar_kws={"shrink": 0.8}, cmap='rocket')
    axes.set_xticks(np.arange(7), dim_labels)
    axes.set_yticks(np.arange(7), dim_labels)
    axes.set_xlabel("Input axes")
    axes.set_ylabel("Output axes")
    
    # orthogonalize the input axes from the output axes
    all_model_ortho_input_axes = [] # list of dicts, each dict is a model's ortho input axes
    for idx_model in range(num_models):
        curr_mdl_ortho_input_axes = dict()
        for k_name in n_components_for_dpca.keys():
            input_axes = all_model_dpca_in['encoding_axes'][idx_model][k_name]
            output_axes = all_model_dpca_out['encoding_axes'][idx_model][k_name]
            curr_dim_ortho_input_axes = []
            for axes_idx in range(input_axes.shape[1]):
                curr_input_axes = input_axes[:, axes_idx:axes_idx+1]
                q, _  = np.linalg.qr(np.concatenate([output_axes, curr_input_axes], axis=1))
                curr_dim_ortho_input_axes.append(q[:, -1])
            curr_mdl_ortho_input_axes[k_name] = np.stack(curr_dim_ortho_input_axes, axis=1)
                
        all_model_ortho_input_axes.append(curr_mdl_ortho_input_axes)

    return all_model_ortho_input_axes


def plot_reparam_weights(all_model_dpca, all_model_rec, n_components_for_dpca, axes_heatmap, axes_violin):
    '''
    Plots the reparameterized weights for each model.

    Args:
        all_model_dpca: dict of {dpca_dict_name: dpca_result}, where dpca_dict_name is the name of the dpca dictionary
        all_model_rec: dict of {(area_in, area_out): recurrent weight matrices}
        n_components_for_dpca: dict, number of components for each dimension
        axes_heatmap: matplotlib axes, axes to plot on for the heatmap
        axes_violin: matplotlib axes, axes to plot on for the violin plot
    '''

    num_models = len(all_model_rec[0][0])
    num_areas = len(all_model_rec)

    all_pos_reparam_weights = [[[] for _ in range(num_areas)] for _ in range(num_areas)] # each weight has shape (component, component)
    all_pos_within_weights = [[[] for _ in range(num_areas)] for _ in range(num_areas)] # each weight has shape (component,)
    all_pos_between_weights = [[[] for _ in range(num_areas)] for _ in range(num_areas)] # each weight has shape (component*(component-1))

    for row_idx in range(num_areas): # area out
        for col_idx in range(num_areas): # area in
            curr_pos_dpca_axes_in = all_model_dpca[col_idx]['encoding_axes']
            curr_pos_dpca_axes_out = all_model_dpca[row_idx]['encoding_axes']
            curr_pos_raw_weights = all_model_rec[row_idx][col_idx]
            for mdl_idx in range(num_models):
                curr_mdl_dpca_axes_in = np.concatenate([curr_pos_dpca_axes_in[mdl_idx][k] for k in n_components_for_dpca.keys()], axis=1)
                curr_mdl_dpca_axes_out = np.concatenate([curr_pos_dpca_axes_out[mdl_idx][k] for k in n_components_for_dpca.keys()], axis=1)
                num_components = curr_mdl_dpca_axes_in.shape[1]
                rec_current = curr_pos_raw_weights[mdl_idx].detach().numpy()@curr_mdl_dpca_axes_in
                # rec_current = rec_current/np.linalg.norm(rec_current, axis=0)
                curr_mdl_reparam_weights = curr_mdl_dpca_axes_out.T@rec_current
                all_pos_reparam_weights[row_idx][col_idx].append(curr_mdl_reparam_weights)
                all_pos_within_weights[row_idx][col_idx].append(np.diag(curr_mdl_reparam_weights))
                all_pos_between_weights[row_idx][col_idx].append(curr_mdl_reparam_weights[np.where(~np.eye(num_components,dtype=bool))])
            
            all_pos_reparam_weights[row_idx][col_idx] = np.stack(all_pos_reparam_weights[row_idx][col_idx]) 
            all_pos_within_weights[row_idx][col_idx] = np.stack(all_pos_within_weights[row_idx][col_idx])
            all_pos_between_weights[row_idx][col_idx] = np.stack(all_pos_between_weights[row_idx][col_idx])

    
    concat_reparam_weights = np.concatenate([np.concatenate(all_pos_reparam_weights[row_idx], axis=2) 
                                             for row_idx in range(num_areas)], axis=1)
    all_pos_within_weights = np.array(all_pos_within_weights) 
    all_pos_between_weights = np.array(all_pos_between_weights) 

    cmap_scale = concat_reparam_weights.mean(0).max()*0.9
    sns.heatmap(concat_reparam_weights.mean(0), ax=axes_heatmap, vmin=-cmap_scale, vmax=cmap_scale, cmap='RdBu_r', 
                     square=True, annot_kws={'fontdict':{'fontsize':10}}, cbar_kws={"shrink": 0.8})

    
    violin_xxx = ['Ftr']*(num_models*6)\
                +['Cnj']*(num_models*12)\
                +['Obj']*(num_models*8)\
                +['Btw']*(all_pos_between_weights.size//(num_areas**2))
    for row_idx in range(num_areas):
        for col_idx in range(num_areas):
            violin_scale = all_pos_within_weights[row_idx,col_idx].max()*1.1
            sns.violinplot(ax=axes_violin[row_idx][col_idx], 
                           x=violin_xxx, hue=violin_xxx,
                           y=np.concatenate([all_pos_within_weights[row_idx,col_idx][:,:6].flatten(), 
                                             all_pos_within_weights[row_idx,col_idx][:,6:18].flatten(), 
                                             all_pos_within_weights[row_idx,col_idx][:,18:27].flatten(), 
                                             all_pos_between_weights[row_idx,col_idx].flatten()]),
                           palette=sns.color_palette('Purples_r', 4), cut=0, legend=False)
            
            # plot the p-value for the between-weights vs. within-weights
            unit_len = violin_scale/8

            temp_stats = mannwhitneyu(all_pos_within_weights[row_idx,col_idx][:,:6].flatten(), 
                            all_pos_between_weights[row_idx,col_idx].flatten())        
            bar_bottom = violin_scale+unit_len*3
            bar_top = violin_scale+unit_len*3.2
            axes_violin[row_idx][col_idx].plot([0, 0, 3, 3], [bar_bottom, bar_top, bar_top, bar_bottom], lw=1, c='k')
            axes_violin[row_idx][col_idx].text(0.5, bar_top*1.02, 
                                               convert_pvalue_to_asterisks(temp_stats.pvalue), 
                                               ha='center', va='center', c='k', fontsize=14)
            
            bar_bottom = violin_scale+unit_len*2
            bar_top = violin_scale+unit_len*2.2
            axes_violin[row_idx][col_idx].plot([1, 1, 3, 3], [bar_bottom, bar_top, bar_top, bar_bottom], lw=1, c='k')
            temp_stats = mannwhitneyu(all_pos_within_weights[row_idx,col_idx][:,6:18].flatten(), 
                            all_pos_between_weights[row_idx,col_idx].flatten())
            axes_violin[row_idx][col_idx].text(1.5, bar_top*1.02, 
                                               convert_pvalue_to_asterisks(temp_stats.pvalue), 
                                               ha='center', va='center', c='k', fontsize=14)
            
            bar_bottom = violin_scale+unit_len*1
            bar_top = violin_scale+unit_len*1.2
            axes_violin[row_idx][col_idx].plot([2, 2, 3, 3], [bar_bottom, bar_top, bar_top, bar_bottom], lw=1, c='k')
            temp_stats = mannwhitneyu(all_pos_within_weights[row_idx,col_idx][:,18:27].flatten(), 
                            all_pos_between_weights[row_idx,col_idx].flatten())
            axes_violin[row_idx][col_idx].text(2.5, bar_top*1.02, 
                                               convert_pvalue_to_asterisks(temp_stats.pvalue), 
                                               ha='center', va='center', c='k', fontsize=14)

            axes_violin[row_idx][col_idx].axhline(0, linestyle='--', color='k')
            axes_violin[row_idx][col_idx].set_xlim([-0.5, 3.5])
            axes_violin[row_idx][col_idx].set_ylim([-violin_scale, violin_scale*(2.0)])
            
            sns.despine(ax=axes_violin[row_idx][col_idx])


def plot_reparam_lrs(all_model_dpca, all_model_kappa, n_components_for_dpca, axes_heatmap, axes_violin):
    """
    Plots the overlap between Hebbian memory matrices and dPCA axes for encoding and retrieval phases.

    Args:
        all_model_dpca: list of size (num_areas-1, 2), 
                        each element is a dpca result object, separately for the input and output subspaces
        all_model_kappa: list of size (num_areas-1, 2), 
                        each element is a learning rate matrix, separately for the feedforward and feedback learning rates
        n_components_for_dpca: dict, number of components for each dimension
        axes_heatmap: matplotlib axes, axes to plot on for the heatmap
        axes_violin: matplotlib axes, axes to plot on for the violin plot
        title: str, title for the plot
    """
    num_models = len(all_model_dpca[0][0]['encoding_axes'])
    num_areas = len(all_model_kappa)+1
    num_axis = sum(list(n_components_for_dpca.values()))

    all_pos_reparam_lrs = [[np.zeros((num_models, num_axis, num_axis)) for _ in range(num_areas)] for _ in range(num_areas)] # initialize to zero

    # only inter-area connections are plastic
    all_pos_mem_overlaps = {'same_pre_post': [[[], []] for _ in range(num_areas-1)],  
                          # (num_models, num_axis*num_axis)
                          'same_pre': [[[], []] for _ in range(num_areas-1)], 
                          # (num_models, num_axis*num_axis*(num_axis-1))
                          'same_post': [[[], []] for _ in range(num_areas-1)], 
                          # (num_models, num_axis*num_axis*(num_axis-1))
                          'diff': [[[], []] for _ in range(num_areas-1)]} 
                          # (num_models, num_axis*num_axis*(num_axis-1)*(num_axis-1))}

    for area_idx in range(num_areas-1): # area out
        for ff_fb_idx in range(2):
            
            if ff_fb_idx == 0:
                curr_pos_dpca_axes_in = all_model_dpca[area_idx][0]['encoding_axes']
                curr_pos_dpca_axes_out = all_model_dpca[area_idx][1]['encoding_axes']
            else:
                curr_pos_dpca_axes_in = all_model_dpca[area_idx][1]['encoding_axes']
                curr_pos_dpca_axes_out = all_model_dpca[area_idx][0]['encoding_axes']
            
            curr_pos_raw_lrs = all_model_kappa[area_idx][ff_fb_idx]

            pos_row_idx = area_idx+1 if ff_fb_idx == 0 else area_idx
            pos_col_idx = area_idx if ff_fb_idx == 0 else area_idx+1

            for mdl_idx in tqdm.tqdm(range(num_models), desc=f'Calculating reparameterized learning rates for {pos_col_idx} -> {pos_row_idx}'):
                curr_mdl_dpca_axes_in = np.concatenate([curr_pos_dpca_axes_in[mdl_idx][k] for k in n_components_for_dpca.keys()], axis=1)
                curr_mdl_dpca_axes_out = np.concatenate([curr_pos_dpca_axes_out[mdl_idx][k] for k in n_components_for_dpca.keys()], axis=1)
                curr_mdl_raw_lrs = curr_pos_raw_lrs[mdl_idx].detach().numpy()

                for i in range(num_axis):
                    for j in range(num_axis):
                        mem_mat = curr_mdl_raw_lrs*(curr_mdl_dpca_axes_out[:,i:i+1]@curr_mdl_dpca_axes_in[:,j:j+1].T)

                        for k in range(num_axis):
                            for l in range(num_axis):
                                curr_overlap = (curr_mdl_dpca_axes_out[:,k:k+1].T@mem_mat@curr_mdl_dpca_axes_in[:,l:l+1]).squeeze()
                                if i==k and j==l:
                                    all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx].append(curr_overlap)
                                    all_pos_reparam_lrs[pos_row_idx][pos_col_idx][mdl_idx,i,j] = curr_overlap
                                elif j==l:
                                    all_pos_mem_overlaps['same_pre'][area_idx][ff_fb_idx].append(curr_overlap)
                                elif i==k:
                                    all_pos_mem_overlaps['same_post'][area_idx][ff_fb_idx].append(curr_overlap)
                                else:
                                    all_pos_mem_overlaps['diff'][area_idx][ff_fb_idx].append(curr_overlap)

            all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx] = np.array(all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx])
            all_pos_mem_overlaps['same_pre'][area_idx][ff_fb_idx] = np.array(all_pos_mem_overlaps['same_pre'][area_idx][ff_fb_idx])
            all_pos_mem_overlaps['same_post'][area_idx][ff_fb_idx] = np.array(all_pos_mem_overlaps['same_post'][area_idx][ff_fb_idx])
            all_pos_mem_overlaps['diff'][area_idx][ff_fb_idx] = np.array(all_pos_mem_overlaps['diff'][area_idx][ff_fb_idx])
    
    all_pos_reparam_lrs = np.concatenate([np.concatenate(all_pos_reparam_lrs[pos_row_idx], axis=2) 
                                                for pos_row_idx in range(num_areas)], axis=1)

    # plot the average reparameterized learning rates
    cmap_scale_max = all_pos_reparam_lrs.mean(0).max()
    cmap_scale_min = all_pos_reparam_lrs.mean(0).min()

    print('Plotting average reparameterized learning rates')
    sns.heatmap(all_pos_reparam_lrs.mean(0), ax=axes_heatmap, cmap='RdBu_r', center=0, vmin=cmap_scale_min, vmax=cmap_scale_max,
                square=True, cbar_kws={"shrink": 0.7})
    axes_heatmap.set_xlabel("Pre")
    axes_heatmap.set_ylabel("Post")
    print('Finished heat map')


    # plot all the memory overlaps in violin plots
    cmap_scale_max = all_pos_reparam_lrs.max()
    cmap_scale_min = -0.01

    print('Plotting violin plots of memory overlaps')
    for area_idx in range(num_areas-1):
        for ff_fb_idx in range(2):
            print(area_idx, ff_fb_idx)
            if num_areas == 2:
                curr_ax = axes_violin[ff_fb_idx]
            elif num_areas >= 3:
                curr_ax = axes_violin[area_idx][ff_fb_idx]
            else:
                raise ValueError(f'num_areas must be 2 or greater, but got {num_areas}')
            
            sns.violinplot(ax=curr_ax,
                        x=['Same']*np.prod(all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx].shape)+\
                        ['Diff O']*np.prod(all_pos_mem_overlaps['same_pre'][area_idx][ff_fb_idx].shape)+
                        ['Diff I']*np.prod(all_pos_mem_overlaps['same_post'][area_idx][ff_fb_idx].shape)+\
                        ['Diff IO']*np.prod(all_pos_mem_overlaps['diff'][area_idx][ff_fb_idx].shape),
                        y=np.concatenate([all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx].flatten(),
                                        all_pos_mem_overlaps['same_pre'][area_idx][ff_fb_idx].flatten(),
                                        all_pos_mem_overlaps['same_post'][area_idx][ff_fb_idx].flatten(),
                                        all_pos_mem_overlaps['diff'][area_idx][ff_fb_idx].flatten()]),
                        hue=['Same']*np.prod(all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx].shape)+\
                            ['Diff O']*np.prod(all_pos_mem_overlaps['same_pre'][area_idx][ff_fb_idx].shape)+
                            ['Diff I']*np.prod(all_pos_mem_overlaps['same_post'][area_idx][ff_fb_idx].shape)+\
                            ['Diff IO']*np.prod(all_pos_mem_overlaps['diff'][area_idx][ff_fb_idx].shape),
                        palette=sns.color_palette('pastel', 4), cut=0, legend=False)
            
            curr_ax.set_xticks(np.arange(4))
            curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=30)
            curr_ax.axhline(0, linestyle = '--', color='k', linewidth=0.5)
            curr_ax.set_ylim([cmap_scale_min*1.2, cmap_scale_max*1.7])

            for key_idx, key in enumerate(['same_pre', 'same_post', 'diff']):
                temp_stats = mannwhitneyu(all_pos_mem_overlaps['same_pre_post'][area_idx][ff_fb_idx].flatten(), 
                                                all_pos_mem_overlaps[key][area_idx][ff_fb_idx].flatten())
                
                unit_len = cmap_scale_max/8
                
                bar_bottom = cmap_scale_max+unit_len*(key_idx+1)
                bar_top = cmap_scale_max+unit_len*(key_idx+1.2)
                
                curr_ax.plot([0, 0, key_idx+1, key_idx+1], 
                            [bar_bottom, bar_top, bar_top, bar_bottom], lw=1, c='k')
                curr_ax.text(0.5+key_idx, bar_top*1.02, 
                            convert_pvalue_to_asterisks(temp_stats.pvalue), 
                            ha='center', va='center', c='k', fontsize=12)

            sns.despine(ax=curr_ax)
    print('Finished violin plots')


def plot_conn_lr_by_clusters(all_model_clusters, all_model_rec, axes):
    '''
    Plot the average connectivity/learning rate within and between clusters.

    Args:
        all_model_clusters: list of size (num_areas,), each element is a (num_models, num_units) cluster index matrix
        all_model_rec: list of size (num_areas, num_areas), 
                        each element is the stacked recurrent weight matrices for all models
        label: str, label for the plot
        axes: matplotlib axes, axes to plot on
    '''

    # The following code is based on the structure of plot_reparam_weights and plot_reparam_lrs,
    # adapted for plotting average connectivity/learning rate within and between clusters.

    # Assume all_model_clusters is a list of (num_areas,) where each element is a (num_models, num_units) array
    # all_model_rec is a list of (num_areas, num_areas), each element is the stacked recurrent weight matrices for all models

    # For each area pair, plot the mean connectivity/learning rate between clusters

    # We'll assume for each area, clusters[area][model_idx] gives the cluster assignment for each unit in that area for that model

    num_areas = len(all_model_clusters)
    num_models = all_model_clusters[0].shape[0]

    for area_pre in range(num_areas):
        for area_post in range(num_areas):
            # Get all cluster assignments for pre and post areas
            clusters_pre = all_model_clusters[area_pre]  # shape (num_models, num_units_pre)
            clusters_post = all_model_clusters[area_post]  # shape (num_models, num_units_post)
            # Get all recurrent weights/learning rates for this area pair
            rec_mats = all_model_rec[area_post][area_pre]  # shape (num_models, num_units_post, num_units_pre)

            # Determine number of clusters in pre and post
            n_clus_pre = int(np.max(clusters_pre) + 1)
            n_clus_post = int(np.max(clusters_post) + 1)

            # Prepare matrix to hold mean connectivity between clusters for each model
            clus_conn_mat = np.zeros((num_models, n_clus_post, n_clus_pre))

            for mdl_idx in range(num_models):
                for clus_i in range(n_clus_post):
                    for clus_j in range(n_clus_pre):
                        mask_post = clusters_post[mdl_idx] == clus_i
                        mask_pre = clusters_pre[mdl_idx] == clus_j
                        if np.any(mask_post) and np.any(mask_pre):
                            clus_conn_mat[mdl_idx, clus_i, clus_j] = rec_mats[mdl_idx][np.ix_(mask_post, mask_pre)].mean()
                        else:
                            clus_conn_mat[mdl_idx, clus_i, clus_j] = np.nan

            # Plot the mean across models for this area pair
            ax = axes[area_post, area_pre] if axes.ndim == 2 else axes
            cmap_scale = np.nanmax(np.abs(np.nanmean(clus_conn_mat, axis=0)))
            sns.heatmap(np.nanmean(clus_conn_mat, axis=0), ax=ax, cmap='RdBu_r',
                        norm=mpl.colors.CenteredNorm(halfrange=cmap_scale))
            ax.set_xticks(np.arange(n_clus_pre) + 0.5)
            ax.set_xticklabels([f'C{j+1}' for j in range(n_clus_pre)], rotation=0)
            ax.set_yticks(np.arange(n_clus_post) + 0.5)
            ax.set_yticklabels([f'C{i+1}' for i in range(n_clus_post)], rotation=0)
            ax.set_xlabel(f'Pre Clusters (Area {area_pre})')
            ax.set_ylabel(f'Post Clusters (Area {area_post})')
            ax.set_title(f'Area {area_post} $\leftarrow$ Area {area_pre}')
    
    num_clusters_exc_pre = np.unique(all_model_clusters_pre[0])[-1]+1
    if all_model_clusters_pre[1] is not None:
        num_clusters_inh_pre = np.unique(all_model_clusters_pre[1])[-1]+1
    else:
        num_clusters_inh_pre = 0

    num_clusters_exc_post = np.unique(all_model_clusters_post[0])[-1]+1
    if all_model_clusters_post[1] is not None:
        num_clusters_inh_post = np.unique(all_model_clusters_post[1])[-1]+1
    else:
        num_clusters_inh_post = 0
    
    clus_conn_mat = np.zeros((len(all_model_rec), num_clusters_exc_post+num_clusters_inh_post, num_clusters_exc_pre+num_clusters_inh_pre ))

    
    for mdl_idx, mdl_rec in enumerate(all_model_rec):
        for clus_i in range(num_clusters_exc_post):
            for clus_j in range(num_clusters_exc_pre):
                clus_mask_i = all_model_clusters_post[0][mdl_idx]==clus_i
                clus_mask_j = all_model_clusters_pre[0][mdl_idx]==clus_j
                clus_conn_mat[mdl_idx, clus_i, clus_j] = mdl_rec[:E_SIZE,:E_SIZE][clus_mask_i][:,clus_mask_j].mean()

        for clus_i in range(num_clusters_exc_post):
            for clus_j in range(num_clusters_inh_pre):
                clus_mask_i = all_model_clusters_post[0][mdl_idx]==clus_i
                clus_mask_j = all_model_clusters_pre[1][mdl_idx]==clus_j
                clus_conn_mat[mdl_idx, clus_i, num_clusters_exc_pre+clus_j] = mdl_rec[:E_SIZE,E_SIZE:][clus_mask_i][:,clus_mask_j].mean()

        for clus_i in range(num_clusters_inh_post):
            for clus_j in range(num_clusters_exc_pre):
                clus_mask_i = all_model_clusters_post[1][mdl_idx]==clus_i
                clus_mask_j = all_model_clusters_pre[0][mdl_idx]==clus_j
                clus_conn_mat[mdl_idx, num_clusters_exc_pre+clus_i, clus_j] = mdl_rec[E_SIZE:,:E_SIZE][clus_mask_i][:,clus_mask_j].mean()

        for clus_i in range(num_clusters_inh_post):
            for clus_j in range(num_clusters_inh_pre):
                clus_mask_i = all_model_clusters_post[1][mdl_idx]==clus_i
                clus_mask_j = all_model_clusters_pre[1][mdl_idx]==clus_j
                clus_conn_mat[mdl_idx, num_clusters_exc_pre+clus_i, num_clusters_exc_pre+clus_j] = mdl_rec[E_SIZE:,E_SIZE:][clus_mask_i][:,clus_mask_j].mean()

    cmap_scale = max(np.abs(np.nanmean(clus_conn_mat, 0)).max(), 0)
    sns.heatmap(np.nanmean(clus_conn_mat, 0), ax=axes, cmap='RdBu_r', norm=mpl.colors.CenteredNorm(halfrange=cmap_scale))
    axes.set_xticks(np.arange(0,num_clusters_exc_pre+num_clusters_inh_pre)+0.5)
    axes.set_xticklabels([f'E{i+1}' for i in range(num_clusters_exc_pre)]+[f'I{i+1}' for i in range(num_clusters_inh_pre)], 
                       rotation=0)
    axes.set_yticks(np.arange(0,num_clusters_exc_post+num_clusters_inh_post)+0.5)
    axes.set_yticklabels([f'E{i+1}' for i in range(num_clusters_exc_post)]+[f'I{i+1}' for i in range(num_clusters_inh_post)], 
                       rotation=0)
    axes.set_xlabel('Cluster')
    axes.set_ylabel('Cluster')
    axes.set_title(label)

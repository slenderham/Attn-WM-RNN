import numpy as np
import os
import pickle
from scipy.stats import binom
from scipy.signal import convolve
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
import itertools
import statsmodels.api as sm
from analysis import convert_pvalue_to_asterisks
from utils import create_colormap
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def obj_to_ft_conj(obj, task_mdprl):
    F1 = task_mdprl.index_shp[obj]
    F2 = task_mdprl.index_pttrn[obj]
    F3 = task_mdprl.index_clr[obj]

    C1 = task_mdprl.index_pttrnclr[obj]
    C2 = task_mdprl.index_shpclr[obj]
    C3 = task_mdprl.index_shppttrn[obj]

    return np.stack([F1, F2, F3, C1, C2, C3, obj], axis=-1)

def obj_to_value_est(obj, task_mdprl):
    pF1, pF2, pF3, pC1, pC2, pC3, pO = task_mdprl.value_est()

    return pF1[obj], pC1[obj], pO[obj]

def steady_state_choice_analysis(all_saved_states, task_mdprl, plot_save_dir, start_trial=216):
    num_trials = all_saved_states['rewards'].shape[0]
    num_trials_to_fit = np.arange(432-start_trial, num_trials)
    num_subj = all_saved_states['rewards'].shape[2]

    all_Xs = []
    all_Ys = []

    for idx_subj in range(num_subj):
        # stim in sensory space, ntrials X 2
        stims = all_saved_states['stimuli'][num_trials_to_fit,0,idx_subj,:] 
        # stim back to reward schedule space, ntrials X 2
        stims_rwd_mat = task_mdprl.permute_mapping(stims, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        pF1, pC1, pO = obj_to_value_est(stims_rwd_mat, task_mdprl) # ntrials X 2 for each
        # choices are in sensory space, map back to schedule space
        choices = all_saved_states['choices'][num_trials_to_fit,0,idx_subj]==stims[:,1]

        all_Xs.append(np.stack([np.log(pF1[:,1]/pF1[:,0]), 
                                np.log(pC1[:,1]/pC1[:,0]),
                                np.log(pO[:,1]/pO[:,0]),], axis=1))
        all_Ys.append(choices)
        
    all_Xs = np.concatenate(all_Xs, 0)
    all_Ys = np.concatenate(all_Ys, 0)[:, None]
    
    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=['pFinf', 'pCinf', 'pO', 'choice']).fillna(0)
    data_group_by_F_m = all_data.groupby('pFinf').mean().apply(np.array).loc[:,'choice']
    data_group_by_F_se = all_data.groupby('pFinf').sem().apply(np.array).loc[:,'choice']
    data_group_by_C_m = all_data.groupby('pCinf').mean().apply(np.array).loc[:,'choice']
    data_group_by_C_se = all_data.groupby('pCinf').sem().apply(np.array).loc[:,'choice']

    fig, axes = plt.subplots()
    fig.set_size_inches((6.4, 4))
    axes.errorbar(data_group_by_F_m.index.values, 
                 data_group_by_F_m.values, 
                 data_group_by_F_se.values,
                 c=mpl.colormaps['tab10']([0]),
                 ls='', marker='o', mfc='white')
    axes.errorbar(data_group_by_C_m.index.values, 
                 data_group_by_C_m.values, 
                 data_group_by_C_se.values,
                 c=mpl.colormaps['tab10']([3]),
                 ls='', marker='o', mfc='white')
    
    mdl = smf.glm('choice~pFinf+pCinf+pO', all_data, missing='drop', family=sm.families.Binomial())
    mdlf = mdl.fit()
    print(mdlf.summary())
    all_coeffs = mdlf.params[1:]
    all_ses = mdlf.bse[1:]
    all_ps = mdlf.pvalues[1:]

    all_coeffs = np.stack(all_coeffs)
    all_ses = np.stack(all_ses)
    all_ps = np.stack(all_ps)

    var_names = [r'$F_{I}$', r'$C_{I}$', 'O']
    
    xs = np.linspace(-1., 1., 100)
    
    axes.plot(xs, 1/(1+np.exp(-all_coeffs[0]*xs)), color=mpl.colormaps['tab10']([0]), label=r'$F_{I}$')
    axes.plot(xs, 1/(1+np.exp(-all_coeffs[1]*xs)), color=mpl.colormaps['tab10']([3]), label=r'$C_{I}$')
    axes.legend()
    
    axin = axes.inset_axes([0.65, 0.1, 0.3, 0.3])
    axin.bar(np.arange(1, len(var_names)+1), all_coeffs, color=mpl.colormaps['tab10']([0, 3, 7]))
    axin.errorbar(np.arange(1, len(var_names)+1), all_coeffs, all_ses, linestyle="", color='k')
    axin.text(1, all_coeffs[0]+all_ses[0]+0.05, convert_pvalue_to_asterisks(all_ps[0]), 
            verticalalignment='center', horizontalalignment='center', fontsize=16)
    axin.text(2, all_coeffs[1]+all_ses[1]+0.05, convert_pvalue_to_asterisks(all_ps[1]), 
            verticalalignment='center', horizontalalignment='center', fontsize=16)
    axin.text(3, all_coeffs[2]+np.sign(all_coeffs[2])*(all_ses[2]+0.05), convert_pvalue_to_asterisks(all_ps[2]), 
            verticalalignment='center', horizontalalignment='center', fontsize=16)
    axin.set_xticks(range(1,4), labels=var_names, fontsize=12)
    axin.set_yticks(range(0,3), labels=range(0,3), fontsize=12)
    axin.set_ylim(np.array(axin.get_ylim())*1.2)
    axin.set_ylabel('Slopes', fontsize=16)
    axes.set_xlabel('Log odd of reward')
    axes.set_ylabel('Choice probability')
    plt.tight_layout()
    sns.despine()
    sns.despine(ax=axin)
    plt.savefig(os.path.join('plots/', plot_save_dir, "choice_curves_slopes.pdf"))
    print(f'Figure saved at plots/{plot_save_dir}/choice_curves_slopes.pdf')
    plt.show()
    plt.close()
    return

def credit_assignment(all_saved_states, task_mdprl, plot_save_dir, end_trial=216):
    # find chosen feedback, unchosen feedback
    # stimuli torch.Size([432, 1, 92, 2])
    # reward_probs torch.Size([432, 1, 92, 2])
    # choices torch.Size([432, 1, 92])
    # rewards torch.Size([432, 1, 92])
    # choose_better torch.Size([432, 1, 92])
    num_trials = all_saved_states['rewards'].shape[0]
    num_trials_to_fit = np.arange(0, end_trial-1)
    num_subj = all_saved_states['rewards'].shape[2]
    all_Xs = []
    all_Ys = []

    for idx_subj in range(num_subj):
        
        stims_post_perceptual = all_saved_states['stimuli'][num_trials_to_fit+1,0,idx_subj,:] # ntrials X 2
        stims_post = task_mdprl.permute_mapping(stims_post_perceptual, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        stims_pre_chosen_perceptual = all_saved_states['choices'][num_trials_to_fit,0,idx_subj] # ntrials
        stims_pre_chosen = task_mdprl.permute_mapping(stims_pre_chosen_perceptual, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        choices = all_saved_states['choices'][num_trials_to_fit+1,0,idx_subj]
        choices = task_mdprl.permute_mapping(choices, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        choices = choices==stims_post[:,1]
        rwd_pre = 2*all_saved_states['rewards'][num_trials_to_fit,0,idx_subj]-1 # ntrials
        
        stimsFCO_pre_chosen = obj_to_ft_conj(stims_pre_chosen, task_mdprl) # ntrials X 7
        stimsFCO_post = obj_to_ft_conj(stims_post, task_mdprl) # ntrials X 2 X 7
        
        # predictors are inf dim R chosen, inf dim C chosen, noninf dim R chosen, noninf dim C chosen, 
        subj_Xs = np.concatenate([rwd_pre[:,None]*(stimsFCO_pre_chosen==stimsFCO_post[:,1,:])-
                                  rwd_pre[:,None]*(stimsFCO_pre_chosen==stimsFCO_post[:,0,:]),\
                                  1.0*(stimsFCO_pre_chosen==stimsFCO_post[:,1,:])-
                                  1.0*(stimsFCO_pre_chosen==stimsFCO_post[:,0,:])], axis=-1)
        
        all_Xs.append(subj_Xs)
        all_Ys.append(choices)

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_Ys = np.concatenate(all_Ys, axis=0)[:,None]
    
    var_names = ['F0', 'F1', 'F2', 'C0', 'C1', 'C2', 'O']

    rw_ch = ['R', 'C']

    all_var_names = ['_'.join([s, 'R']) for s in var_names] + ['_'.join([s, 'C']) for s in var_names]

    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=[*all_var_names, 'choice'])

    mdl = smf.glm('choice~'+'+'.join(all_var_names), data=all_data, family=sm.families.Binomial())
    mdlf = mdl.fit()
    print(mdlf.summary())
    all_coeffs = mdlf.params[1:].to_numpy()
    all_ses = mdlf.bse[1:].to_numpy()
    all_ps = mdlf.pvalues[1:].to_numpy()

    all_xlabels = [r'$F_{I}$', r'$F_{N1}$', r'$F_{N2}$', r'$C_{I}$', r'$C_{N1}$', r'$C_{N2}$', 'O']

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    
    bar_rw = axes[0].bar(x=np.arange(1,len(var_names)+1), \
                height=all_coeffs[:len(var_names)], \
                yerr=all_ses[:len(var_names)], \
                color=mpl.colormaps['tab10'](np.arange(0,7)), 
                capsize=5)
    bar_ch = axes[1].bar(x=np.arange(1,len(var_names)+1), \
                height=all_coeffs[len(var_names):], \
                yerr=all_ses[len(var_names):], \
                color=mpl.colormaps['tab10'](np.arange(0,7)), 
                capsize=5)
    axes[0].set_ylabel('Regression weights')
    axes[0].set_xlabel('Win-stay lose-switch')
    axes[1].set_xlabel('Choice autocorrelation')
    axes[0].set_ylim(np.array(axes[0].get_ylim())*1.1)
    axes[1].set_ylim(np.array(axes[1].get_ylim())*1.1)
    axes[0].set_xticks(np.arange(1, len(var_names)+1), labels=all_xlabels)
    axes[1].set_xticks(np.arange(1, len(var_names)+1), labels=all_xlabels)
    for i in range(len(var_names)):
        axes[0].text(i+1, all_coeffs[i]+np.sign(all_coeffs[i])*(all_ses[i]+0.05)-0.01, 
                     convert_pvalue_to_asterisks(all_ps[i]), 
                     verticalalignment='center', horizontalalignment='center')

    for i in range(len(var_names)):
        axes[1].text(i+1, all_coeffs[i+len(var_names)]+np.sign(all_coeffs[i+len(var_names)])*(all_ses[i+len(var_names)]+0.05)-0.01, 
                     convert_pvalue_to_asterisks(all_ps[i+len(var_names)]), 
                     verticalalignment='center', horizontalalignment='center')
    plt.tight_layout()
    sns.despine()
    with PdfPages(f'plots/{plot_save_dir}/credit_assignment.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/credit_assignment.pdf')
    plt.show()
    plt.close()
    return

def steady_state_choice_logit_analysis(all_saved_states, task_mdprl, plot_save_dir, start_trial=432//2):
    num_trials = all_saved_states['rewards'].shape[0]
    num_trials_to_fit = np.arange(432-start_trial, num_trials)
    num_subj = all_saved_states['rewards'].shape[2]

    all_Xs = []
    all_Ys = []

    for idx_subj in range(num_subj):
        # stim in sensory space, ntrials X 2
        stims = all_saved_states['stimuli'][num_trials_to_fit,0,idx_subj,:] 
        # stim back to reward schedule space, ntrials X 2
        stims_rwd_mat = task_mdprl.permute_mapping(stims, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        pF1, pC1, pO = obj_to_value_est(stims_rwd_mat, task_mdprl) # ntrials X 2 for each
        # choices are in sensory space, map back to schedule space
        logits = all_saved_states['logits'][num_trials_to_fit,0,idx_subj]

        all_Xs.append(np.stack([np.log(pF1[:,1]/pF1[:,0]), 
                                np.log(pC1[:,1]/pC1[:,0]),
                                np.log(pO[:,1]/pO[:,0]),
                                np.ones_like(logits)*idx_subj], axis=1))
        all_Ys.append(logits)
        
    all_Xs = np.concatenate(all_Xs, 0)
    all_Ys = np.concatenate(all_Ys, 0)[:, None]
    
    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=['pFinf', 'pCinf', 'pO', 'idx_subj', 'logits']).fillna(0)

    fig, axes = plt.subplots(figsize=(7,5))
    
    sns.stripplot(data=all_data, x='pFinf', y='logits', jitter=True, native_scale=True,
                  color=mpl.colormaps['tab10']([0]), alpha=0.02)
    sns.stripplot(data=all_data, x='pCinf', y='logits', jitter=True, native_scale=True,
                  color=mpl.colormaps['tab10']([3]), alpha=0.02)

    mdl = smf.mixedlm('logits~pFinf+pCinf+pO', all_data, missing='drop', groups=all_data['idx_subj'],
                      re_formula='~pFinf+pCinf+pO')
    free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(4), np.eye(4))
    
    mdlf = mdl.fit(free=free, method=['lbfgs'])
    print(mdlf.summary())
    all_coeffs = mdlf.params[1:]
    all_ses = mdlf.bse[1:]
    all_ps = mdlf.pvalues[1:]

    all_coeffs = np.stack(all_coeffs)
    all_ses = np.stack(all_ses)
    all_ps = np.stack(all_ps)

    var_names = [r'$F_{I}$', r'$C_{I}$', 'O']
    
    xs = np.linspace(-1., 1., 100)
    
    axes.plot(xs, all_coeffs[0]*xs, color=mpl.colormaps['tab10']([0]), label=r'$F_{I}$')
    axes.plot(xs, all_coeffs[1]*xs, color=mpl.colormaps['tab10']([3]), label=r'$C_{I}$')
    axes.legend()
    axes.set_ylim(np.array(axes.get_ylim())+[0, 2])
    axes.set_xlim([-1.1, 1.1])
    
    axes.text(axes.get_xlim()[1], all_coeffs[0], convert_pvalue_to_asterisks(all_ps[0]), 
              verticalalignment='center', horizontalalignment='left', fontsize=16,
              color=mpl.colormaps['tab10'](0))
    axes.text(axes.get_xlim()[1], all_coeffs[1], convert_pvalue_to_asterisks(all_ps[1]), 
              verticalalignment='center', horizontalalignment='left', fontsize=16,
              color=mpl.colormaps['tab10'](3))

    
    axes.set_xlabel('Log odd of reward')
    axes.set_ylabel('Choice logits')
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join('plots/', plot_save_dir, "choice_logit_curves_slopes.pdf"))
    print(f'Figure saved at plots/{plot_save_dir}/choice_logit_curves_slopes.pdf')
    plt.show()
    plt.close()
    return mdlf

def credit_assignment_logit(all_saved_states, task_mdprl, plot_save_dir, end_trial=432//2):
    # find chosen feedback, unchosen feedback
    # stimuli torch.Size([432, 1, 92, 2])
    # reward_probs torch.Size([432, 1, 92, 2])
    # choices torch.Size([432, 1, 92])
    # rewards torch.Size([432, 1, 92])
    # choose_better torch.Size([432, 1, 92])
    num_trials = all_saved_states['rewards'].shape[0]
    num_trials_to_fit = np.arange(0, end_trial-1)
    num_subj = all_saved_states['rewards'].shape[2]
    all_Xs = []
    all_Ys = []

    for idx_subj in range(num_subj):
        
        stims_post_perceptual = all_saved_states['stimuli'][num_trials_to_fit+1,0,idx_subj,:] # ntrials X 2
        stims_post = task_mdprl.permute_mapping(stims_post_perceptual, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        stims_pre_chosen_perceptual = all_saved_states['choices'][num_trials_to_fit,0,idx_subj] # ntrials
        stims_pre_chosen = task_mdprl.permute_mapping(stims_pre_chosen_perceptual, all_saved_states['test_sensory2stim_idx'][idx_subj]) 
        logits = all_saved_states['logits'][num_trials_to_fit+1,0,idx_subj]
        rwd_pre = 2*all_saved_states['rewards'][num_trials_to_fit,0,idx_subj]-1 # ntrials
        
        stimsFCO_pre_chosen = obj_to_ft_conj(stims_pre_chosen, task_mdprl) # ntrials X 7
        stimsFCO_post = obj_to_ft_conj(stims_post, task_mdprl) # ntrials X 2 X 7
        
        
        # predictors are inf dim R chosen, inf dim C chosen, noninf dim R chosen, noninf dim C chosen, 
        subj_Xs = np.concatenate([rwd_pre[:,None]*(stimsFCO_pre_chosen==stimsFCO_post[:,1,:])-
                                  rwd_pre[:,None]*(stimsFCO_pre_chosen==stimsFCO_post[:,0,:]),\
                                  1.0*(stimsFCO_pre_chosen==stimsFCO_post[:,1,:])-
                                  1.0*(stimsFCO_pre_chosen==stimsFCO_post[:,0,:]),
                                  np.ones_like(logits)[:,None]*idx_subj], axis=-1)
        
        all_Xs.append(subj_Xs)
        all_Ys.append(logits)

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_Ys = np.concatenate(all_Ys, axis=0)[:,None]
    
    var_names = ['F0', 'F1', 'F2', 'C0', 'C1', 'C2', 'O']
    rw_ch = ['R', 'C']

    all_var_names = ['_'.join([s, 'R']) for s in var_names] + ['_'.join([s, 'C']) for s in var_names]

    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=[*all_var_names, 'idx_subj', 'logits'])

    mdl = smf.mixedlm('logits~'+'+'.join(all_var_names), data=all_data, groups=all_data['idx_subj'],
                     re_formula='~'+'+'.join(all_var_names))
    free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(15), np.eye(15))
    
    mdlf = mdl.fit(free=free, method=['lbfgs'])
    print(mdlf.summary())
    all_coeffs = mdlf.params[1:15].to_numpy()
    all_ses = mdlf.bse[1:15].to_numpy()
    all_ps = mdlf.pvalues[1:15].to_numpy()

    all_xlabels = [r'$F_{I}$', r'$F_{N1}$', r'$F_{N2}$', r'$C_{I}$', r'$C_{N1}$', r'$C_{N2}$', 'O']

    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches((10, 5))
    
    bar_rw = axes[0].bar(x=np.arange(1,len(var_names)+1), \
                height=all_coeffs[:len(var_names)], \
                yerr=all_ses[:len(var_names)], \
                color=mpl.colormaps['tab10'](np.arange(0,7)), 
                capsize=5)
    bar_ch = axes[1].bar(x=np.arange(1,len(var_names)+1), \
                height=all_coeffs[len(var_names):], \
                yerr=all_ses[len(var_names):], \
                color=mpl.colormaps['tab10'](np.arange(0,7)), 
                capsize=5)
    axes[0].set_ylabel('Regression weights')
    axes[0].set_ylim(np.array(axes[0].get_ylim())*1.1)
    axes[1].set_ylim(np.array(axes[1].get_ylim())*1.1)
    axes[0].set_xlabel('Win-stay lose-switch')
    axes[1].set_xlabel('Choice autocorrelation')
    axes[0].set_xticks(np.arange(1, len(var_names)+1), labels=all_xlabels)
    axes[1].set_xticks(np.arange(1, len(var_names)+1), labels=all_xlabels)
    for i in range(len(var_names)):
        axes[0].text(i+1, all_coeffs[i]+np.sign(all_coeffs[i])*(all_ses[i]+0.05)-0.01, 
                     convert_pvalue_to_asterisks(all_ps[i]), 
                     verticalalignment='center', horizontalalignment='center')

    for i in range(len(var_names)):
        axes[1].text(i+1, all_coeffs[i+len(var_names)]+np.sign(all_coeffs[i+len(var_names)])*(all_ses[i+len(var_names)]+0.05)-0.01, 
                     convert_pvalue_to_asterisks(all_ps[i+len(var_names)]), 
                     verticalalignment='center', horizontalalignment='center')
    plt.tight_layout()
    sns.despine()
    with PdfPages(f'plots/{plot_save_dir}/credit_assignment_logit.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/credit_assignment_logit.pdf')
    plt.show()
    plt.close()
        
    return mdlf
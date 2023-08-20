import json
import math
import os
from collections import defaultdict
import itertools

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.signal import convolve2d
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import cluster
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from torch.nn.functional import interpolate
from torch.serialization import save
import tqdm
from analysis import *
from models import HierarchicalRNN
from analysis import targeted_dimensionality_reduction
from analysis import participation_ratio, run_svd_time_varying_w, get_dpca
from task import MDPRL
from utils import load_checkpoint

# plt.rcParams["figure.figsize"] = (16,10)

def get_sub_mats(ws, num_areas, e_hidden_size, i_hidden_size, separate_ei=True):
    trials, timesteps, batch_size, post_dim, pre_dim = ws.shape
    assert((e_hidden_size+i_hidden_size)*num_areas==pre_dim and (e_hidden_size+i_hidden_size)*num_areas==post_dim)
    total_e_size = e_hidden_size*num_areas
    submats = {}
    if not separate_ei:
        for i in range(num_areas):
            submats[f"rec_intra_{i}"] = ws[:,:,:,list(range(i*e_hidden_size, (i+1)*e_hidden_size))+\
                                                list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))]\
                                        [:,:,:,:,list(range(i*e_hidden_size, (i+1)*e_hidden_size))+\
                                                list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))]

        for i in range(num_areas-1):
            submats[f"rec_inter_ff_{i}_{i+1}"] = ws[:,:,:,list(range((i+1)*e_hidden_size, (i+2)*e_hidden_size))+\
                                                    list(range(total_e_size+(i+1)*i_hidden_size, total_e_size+(i+2)*i_hidden_size))]\
                                            [:,:,:,:,list(range(i*e_hidden_size, (i+1)*e_hidden_size))]
            submats[f"rec_inter_fb_{i+1}_{i}"] = ws[:,:,:,list(range(i*e_hidden_size, i*e_hidden_size))+\
                                                    list(range(total_e_size+(i+1)*i_hidden_size, total_e_size+(i+2)*i_hidden_size))]\
                                            [:,:,:,:,list(range((i+1)*e_hidden_size, (i+2)*e_hidden_size))]
        return submats
    else:
        for i in range(num_areas):
            e_indices = list(range(i*e_hidden_size, (i+1)*e_hidden_size))
            i_indices = list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))

            submats[f"rec_intra_ee_{i}"] = ws[...,e_indices,:][:,:,:,:,e_indices]
            submats[f"rec_intra_ie_{i}"] = ws[...,i_indices,:][:,:,:,:,e_indices]
            submats[f"rec_intra_ei_{i}"] = ws[...,e_indices,:][:,:,:,:,i_indices]
            submats[f"rec_intra_ii_{i}"] = ws[...,i_indices,:][:,:,:,:,i_indices]

        for i in range(num_areas-1):
            e_hi_indices = list(range((i+1)*e_hidden_size, (i+2)*e_hidden_size))
            e_lo_indices = list(range(i*e_hidden_size, (i+1)*e_hidden_size))
            i_hi_indices = list(range(total_e_size+(i+1)*i_hidden_size, total_e_size+(i+2)*i_hidden_size))
            i_lo_indices = list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))

            submats[f"rec_inter_ff_ee_{i}_{i+1}"] = ws[:,:,:,e_hi_indices,:][:,:,:,:,e_lo_indices]
            submats[f"rec_inter_ff_ie_{i}_{i+1}"] = ws[:,:,:,i_hi_indices,:][:,:,:,:,e_lo_indices]
            submats[f"rec_inter_fb_ee_{i+1}_{i}"] = ws[:,:,:,e_lo_indices,:][:,:,:,:,e_hi_indices]
            submats[f"rec_inter_fb_ie_{i+1}_{i}"] = ws[:,:,:,i_lo_indices,:][:,:,:,:,e_hi_indices]
        
        return submats

def get_input_encodings(wxs, stim_enc_mat):
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

def plot_mean_and_std(ax, m, sd, label, color, alpha=1):
    if label is not None:
        ax.plot(m, alpha=alpha, label=label, c=color)
    else:
        ax.plot(m, alpha=alpha, c=color)
    ax.fill_between(range(len(m)), m-sd, m+sd, color=color, alpha=0.1)

def plot_imag_centered_cm(ax, im):
    max_mag = im.abs().max()*0.3
    im = ax.imshow(im, vmax=max_mag, vmin=-max_mag, cmap='RdBu_r')
    return im

def plot_connectivity_lr(sort_inds, x2hw, h2hw, hb, h2ow, aux2h, kappa_rec, e_size, args):
    # maxmax = abs(max([x2hw.max().item(), h2hw.max().item(), hb.max().item(), h2ow.max().item()]))
    # minmin = abs(min([x2hw.min().item(), h2hw.min().item(), hb.min().item(), h2ow.min().item()]))
    # vbound = max([maxmax, minmin])
    # selectivity = h2ow[0,:e_size]-h2ow[1,:e_size]
    # sort_inds = torch.argsort(selectivity)
    # sort_inds = torch.cat([sort_inds, torch.arange(e_size, h2hw.shape[0])])

    # fig, axes = plt.subplots(2, 3, \
        # gridspec_kw={'width_ratios': [h2hw.shape[1], x2hw.shape[1], 1], 'height_ratios': [h2hw.shape[0], 1]})
    fig = plt.figure('connectivity', (10, 10))
    ims = []
    hidden_size = h2hw.shape[0]
    PLOT_W = 0.6/hidden_size
    hidden_size = h2hw.shape[0]*PLOT_W
    input_size = x2hw.shape[1]*PLOT_W
    output_size = h2ow.shape[0]*PLOT_W
    # attn_size = h2attnw.shape[0]*PLOT_W
    aux_size = aux2h.shape[1]*PLOT_W
    # value_size = h2vw.shape[0]*PLOT_W
    MARGIN = 0.01
    LEFT = (1-(input_size+aux_size+hidden_size+MARGIN*3+PLOT_W))/2
    BOTTOM = 0.1
    
    vbound = np.percentile(x2hw.abs(), 95)
    ax01 = fig.add_axes((LEFT, BOTTOM+output_size+MARGIN, input_size, hidden_size))
    ims.append(ax01.imshow(x2hw[sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.axis('off')
    ax01.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    # ax02 = fig.add_axes((LEFT+input_size+MARGIN, BOTTOM+output_size+MARGIN*3, input_size, hidden_size))
    # ims.append(ax02.imshow(x2hw[1][sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # ax02.set_xticks([])
    # ax02.set_yticks([])
    # ax02.axis('off')
    # ax02.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    vbound = np.percentile(aux2h.abs(), 95)
    axaux = fig.add_axes((LEFT+input_size+MARGIN, BOTTOM+output_size+MARGIN, aux_size, hidden_size))
    ims.append(axaux.imshow(aux2h[sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    axaux.set_xticks([])
    axaux.set_yticks([])
    axaux.axis('off')
    axaux.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    vbound = np.percentile(h2hw.abs(), 95)
    ax1w = fig.add_axes((LEFT+input_size+aux_size+MARGIN*2, BOTTOM+output_size+MARGIN, hidden_size, hidden_size))
    ims.append(ax1w.imshow(h2hw[sort_inds][:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax1w.set_xticks([])
    ax1w.set_yticks([])
    ax1w.axis('off')
    ax1w.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    ax1w.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    vbound = np.percentile(hb.abs(), 95)
    ax1b = fig.add_axes((LEFT+input_size+aux_size+hidden_size+MARGIN*3, BOTTOM+output_size+MARGIN, PLOT_W, hidden_size))
    ims.append(ax1b.imshow(hb[sort_inds].unsqueeze(1), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    ax1b.axis('off')
    ax1b.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    # axattnw = fig.add_axes((LEFT+input_size*2+aux_size+MARGIN*3, BOTTOM+value_size+output_size+MARGIN*2, hidden_size, attn_size))
    # ims.append(axattnw.imshow(h2attnw[:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # axattnw.set_xticks([])
    # axattnw.set_yticks([])
    # axattnw.axis('off')
    # axattnw.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    
    # axattnb = fig.add_axes((LEFT+input_size*2+aux_size+hidden_size+MARGIN*4, BOTTOM+value_size+output_size+MARGIN*2, PLOT_W, attn_size))
    # ims.append(axattnb.imshow(h2attnb.unsqueeze(1), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # axattnb.set_xticks([])
    # axattnb.set_yticks([])
    # axattnb.axis('off')
    
    vbound = np.percentile(h2ow.abs(), 95)
    axoutputw = fig.add_axes((LEFT+input_size+aux_size+MARGIN*2, BOTTOM, hidden_size, output_size))
    ims.append(axoutputw.imshow(h2ow[:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    axoutputw.set_xticks([])
    axoutputw.set_yticks([])
    axoutputw.axis('off')
    axoutputw.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    
    # axoutputb = fig.add_axes((LEFT+input_size+aux_size+hidden_size+MARGIN*3, BOTTOM, PLOT_W, output_size))
    # ims.append(axoutputb.imshow(h2ob.unsqueeze(1), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # axoutputb.set_xticks([])
    # axoutputb.set_yticks([])
    # axoutputb.axis('off')
    
    # axvaluew = fig.add_axes((LEFT+input_size*2+aux_size+MARGIN*3, BOTTOM, hidden_size, value_size))
    # ims.append(axvaluew.imshow(h2vw[:,sort_inds], cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # axvaluew.set_xticks([])
    # axvaluew.set_yticks([])
    # axvaluew.axis('off')
    # axvaluew.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    
    # axvalueb = fig.add_axes((LEFT+input_size*2+aux_size+hidden_size+MARGIN*4, BOTTOM, PLOT_W, value_size))
    # ims.append(axvalueb.imshow(h2vb.unsqueeze(1), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # axvalueb.set_xticks([])
    # axvalueb.set_yticks([])
    # axvalueb.axis('off')
    # for i in range(2):
    #     for j in range(3):
    #         axes[i, j].axis('off')
    # plt.axis('off')
    
    # fig.subplots_adjust(right=0.7)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.6])
    # fig.colorbar(ims[-1], cax=cbar_ax)
    plt.suptitle('Model Connectivity', y=0.85)
    # plt.tight_layout()
    plt.show()
    # plt.savefig(f'plots/{args["exp_dir"]}/connectivity.jpg')
    with PdfPages(f'plots/{args["exp_dir"]}/connectivity.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{args["exp_dir"]}/connectivity.pdf')


    fig = plt.figure('learning_rates', (10, 10))
    ims = []
    # ax01 = fig.add_axes((LEFT, BOTTOM+attn_size+MARGIN, input_size, hidden_size))
    # ims.append(ax01.imshow(kappa_in[0][sort_inds].squeeze(), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # ax01.set_xticks([])
    # ax01.set_yticks([])
    # ax01.axis('off')
    # ax01.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    # ax02 = fig.add_axes((LEFT+input_size+MARGIN, BOTTOM+attn_size+MARGIN*1, input_size, hidden_size))
    # ims.append(ax02.imshow(kappa_in[1][sort_inds].squeeze(), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # ax02.set_xticks([])
    # ax02.set_yticks([])
    # ax02.axis('off')
    # ax02.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    LEFT = (1-hidden_size)/2
    vbound = np.percentile(kappa_rec.abs(), 95)
    ax1w = fig.add_axes((LEFT, BOTTOM+output_size+MARGIN, hidden_size, hidden_size))
    ims.append(ax1w.imshow(kappa_rec[sort_inds][:,sort_inds].squeeze(), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    ax1w.set_xticks([])
    ax1w.set_yticks([])
    ax1w.axis('off')
    ax1w.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    ax1w.axhline(y=e_size-0.5, color='grey', linewidth=0.5)
    
    # axfb = fig.add_axes((LEFT+input_size*2+MARGIN*2, BOTTOM, hidden_size, attn_size))
    # ims.append(axfb.imshow(kappa_fb[:,sort_inds].squeeze(), cmap='RdBu_r', vmin=-vbound, vmax=vbound, interpolation='nearest'))
    # axfb.set_xticks([])
    # axfb.set_yticks([])
    # axfb.axis('off')
    # axfb.axvline(x=e_size-0.5, color='grey', linewidth=0.5)
    
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.6])
    # fig.colorbar(ims[-1], cax=cbar_ax)
    plt.suptitle('Model Learning Rates', y=0.85)
    # plt.tight_layout()
    plt.show()
    with PdfPages(f'plots/{args["exp_dir"]}/learning_rates.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{args["exp_dir"]}/learning_rates.pdf')
    
def plot_weight_summary(args, ws, w_baseline):
    trials, timesteps, batch_size, post_dim, pre_dim = ws.shape
    assert(timesteps==1)

    all_submats = get_sub_mats(ws, args['num_areas'], 
                               round(args['hidden_size']*args['e_prop']), 
                               round(args['hidden_size']*(1-args['e_prop'])))
    all_submat_baselines = get_sub_mats(w_baseline[None,None,None,...], args['num_areas'],
                               round(args['hidden_size']*args['e_prop']), 
                               round(args['hidden_size']*(1-args['e_prop'])))
    
    submat_keys = [
        ["rec_intra_ee_0", "rec_inter_fb_ee_1_0", "rec_intra_ei_0"],
        ["rec_inter_ff_ee_0_1", "rec_intra_ee_1", "rec_intra_ei_1"],
        ["rec_intra_ie_0", "rec_inter_fb_ie_1_0", "rec_intra_ii_0"],
        ["rec_inter_ff_ie_0_1", "rec_intra_ie_1", "rec_intra_ii_1"],
    ]

    submat_names = [
        [r"EE 0 $\to$ 0", r"EE 1 $\to$ 0", r"EI 0 $\to$ 0"],
        [r"EE 0 $\to$ 1", r"EE 1 $\to$ 1", r"EI 1 $\to$ 1"],
        [r"IE 0 $\to$ 0", r"IE 1 $\to$ 0", r"II 0 $\to$ 0"],
        [r"IE 0 $\to$ 1", r"IE 1 $\to$ 1", r"II 1 $\to$ 1"],
    ]

    for k in all_submats.keys():
        all_submats[k] = all_submats[k].squeeze()
        all_submat_baselines[k] = all_submat_baselines[k].squeeze()

    # norm of update
    fig, axes = plt.subplots(4, 3)
    for i in range(4):
        for j in range(3):
            sub_w = all_submats[submat_keys[i][j]]
            sub_w_baseline = all_submat_baselines[submat_keys[i][j]]
            diff_ws = ((sub_w[1:]-sub_w[:-1])**2).sum([-1, -2])/(sub_w_baseline**2).sum([-1, -2])
            plot_mean_and_std(axes[i][j], diff_ws.mean(1), diff_ws.std(1)/np.sqrt(batch_size), 
                              None, color='salmon' if j<=1 else 'skyblue')
            axes[i][j].set_title(submat_names[i][j], fontsize=11)
            axes[i][j].tick_params(labelsize=11)
    fig.supxlabel('Trial', fontsize=11)
    fig.supylabel(r'$|\Delta W|_2$', fontsize=11)
    plt.tight_layout()
    fig.show()
    print('Finished calculating norm of update')

    # norm of weights
    fig, axes = plt.subplots(4, 3)
    for i in range(4):
        for j in range(3):
            sub_w = all_submats[submat_keys[i][j]]
            sub_w_baseline = all_submat_baselines[submat_keys[i][j]]
            norm_ws = (sub_w**2).sum([-1, -2])/(sub_w_baseline**2).sum([-1, -2])
            plot_mean_and_std(axes[i][j], norm_ws.mean(1), norm_ws.std(1)/np.sqrt(batch_size), 
                              None, color='salmon' if j<=1 else 'skyblue')
            axes[i][j].set_title(submat_names[i][j], fontsize=11)
            axes[i][j].tick_params(labelsize=11)
    fig.supxlabel('Trial', fontsize=11)
    fig.supylabel(r'$|W|_2$', fontsize=11)
    plt.tight_layout()
    fig.show()
    print('Finished calculating weight norms')

    # variance of entries across trials
    fig, axes = plt.subplots(4, 3)
    for i in range(4):
        for j in range(3):
            sub_w = all_submats[submat_keys[i][j]]
            sub_w_baseline = all_submat_baselines[submat_keys[i][j]]
            mean_ws = sub_w.mean(1, keepdims=True)
            std_ws = ((sub_w-mean_ws)**2).sum([-1, -2])/(sub_w_baseline**2).sum([-1, -2])
            plot_mean_and_std(axes[i][j], std_ws.mean(1), std_ws.std(1)/np.sqrt(batch_size), 
                              None, color='salmon' if j<=1 else 'skyblue')
            axes[i][j].set_title(submat_names[i][j], fontsize=11)
            axes[i][j].tick_params(labelsize=11)
    fig.supxlabel('Trial', fontsize=11)
    fig.supylabel('Cross session variability', fontsize=11)
    plt.tight_layout()
    fig.show()
    print('Finished calculating variability')

def plot_learning_curve(args, all_rewards, all_choose_betters):
    # fig_all = plt.figure('perf_all')
    # ax = fig_all.add_subplot()
    # ax.imshow(all_l[0].squeeze(-1).t(), interpolation='nearest')
    # ax.set_xlabel('Trials')
    # ax.set_ylabel('Episodes')
    # plt.tight_layout()
    # # plt.savefig(f'plots/{plot_args.exp_dir}/performance_all')
    # plt.show()
    
    window_size = 20
    all_choose_betters = convolve2d(all_choose_betters.squeeze(), np.ones((window_size, 1))/window_size, mode='valid')
    all_rewards = convolve2d(all_rewards.squeeze(), np.ones((window_size, 1))/window_size, mode='valid')

    fig_summ = plt.figure('perf_summary')
    ax = fig_summ.add_subplot()
    plot_mean_and_std(ax, 
                      all_choose_betters.mean(axis=1), 
                      all_choose_betters.std(axis=1)/np.sqrt(all_choose_betters.squeeze().shape[1]), 
                      label='Percent Better', color='grey')
    plot_mean_and_std(ax, 
                      all_rewards.mean(axis=1), 
                      all_rewards.std(axis=1)/np.sqrt(all_rewards.squeeze().shape[1]), 
                      label='Reward', color='black')
    
    ax.vlines(x=args['N_s'], ymin=0.3, ymax=0.9, colors='black', linestyle='--')
    ax.legend()
    ax.set_xlabel('Trials')
    ax.set_ylim([0.45, 0.85])
    plt.tight_layout()
    plt.savefig(f'plots/{args["exp_dir"]}/learning_curve.pdf')
    # plt.show()

def plot_sorted_matrix(w, e_size, w_type, plot_args):
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
    ax2.axvline(x=e_size-0.5, linewidth=1, color='grey')
    ax2.axhline(y=e_size-0.5, linewidth=1, color='grey')
    w = w[ind_sort, :][:, ind_sort]
    if w_type=='weight':
        im = plot_imag_centered_cm(ax2, w)
        plt.title('Sorted Recurrent Connectivity')
        plt.colorbar(im)
        plt.savefig(f'plots/{plot_args.exp_dir}/sorted_rec_w')
    elif w_type=='lr':
        im = plt.imshow(w, cmap='hot', vmax=w.max()*0.4)
        plt.title('Sorted Recurrent Learning Rates')
        plt.colorbar(im)
        plt.savefig(f'plots/{plot_args.exp_dir}/sorted_rec_lr')
    else:
        raise ValueError

def plot_subspace(ws, num_areas, e_hidden_size, i_hidden_size):
    submats = get_sub_mats(ws, num_areas=num_areas, e_hidden_size=e_hidden_size, i_hidden_size=i_hidden_size)
    all_us = {}
    all_ss = {}
    all_vhs = {}
    for w_name, w_vals in submats.items():
        us, ss, vhs = run_svd_time_varying_w(w_vals)
        all_us[w_name] = us
        all_ss[w_name] = ss
        all_vhs[w_name] = vhs

    # plot participation ratios for change of dimensionality through training
    fig, axes = plt.subplots(num_areas, num_areas)
    for i in range(num_areas):
        pr_rec = participation_ratio(all_ss[f"rec_intra_{i}"])
        plot_mean_and_std(axes[i, i], pr_rec.mean(0), pr_rec.std(0)/np.sqrt(pr_rec.shape[0]))
        axes[i,i].set_title(fr"Area {i} Recurrent")

    for i in range(num_areas-1):
        pr_ff = participation_ratio(all_ss[f"rec_inter_ff_{i}_{i+1}"])
        plot_mean_and_std(axes[i, i+1], pr_ff.mean(0), pr_ff.std(0)/np.sqrt(pr_ff.shape[0]))
        pr_fb = participation_ratio(all_ss[f"rec_inter_fb_{i}_{i+1}"])
        plot_mean_and_std(axes[i+1, i], pr_fb.mean(0), pr_fb.std(0)/np.sqrt(pr_fb.shape[0]))
        axes[i,i+1].set_title(fr"Area {i} \to Area {i+1} FF")
        axes[i+1,i].set_title(fr"Area {i+1} \to Area {i} FB")

    fig.supylabel('Participation Ratios')
    fig.supxlabel('Trials')

    # plot dimensional alignment?
    # SVD(W) = U S VT
    # for each recurrent matrix's left singular vector (columns of V), see how much it is accepted by its left singular vector (columns of U)
    # each column of U is a dimension of the output space, match it with rows of V^T through V^TU
    # then each column of V^TU is the match between all of V^T and one column of U
    # pre-multiplying with S weighs each 
    # size (trials, batch_size, post_dim, pre_dim)

    '''
    for each area, m-intra, n-intra, I-fb, w-ff
        mn - recurrence if overlap
        
        mI - direct feedforward if no overlap
        nI - feedforward to recurrence
        
        mw - recurrence to readout
        Iw - feedforward to readout

        nw - ? not important
    '''

    fig, axes = plt.subplots(num_areas, num_areas)
    mn_cov = []
    for i in range(num_areas):
        w_name = f"rec_intra_{i}"
        mn_cov.append(all_vhs[w_name]@(all_us[w_name] * all_ss[w_name]))
    
    mff_cov = []
    mfb_cov = []
    nff_cov = []
    nfb_cov = []
    for i in range(num_areas-1):
        ff_name = f"rec_inter_ff_{i}_{i+1}"
        fb_name = f"rec_inter_fb_{i}_{i+1}"
        w_name = f"rec_intra_{i}"

        mff_cov.append((all_ss[ff_name]*all_vhs[ff_name])@all_us[w_name])
        mfb_cov.append(all_vhs[w_name]@all_us[w_name])
        
def unit_selectivity(hs, target, e_size):
    n_trials, n_timesteps, n_batch, n_hidden = hs.shape
    hs = hs.mean(1) # n_trials, n_batch, n_hidden
    selectivity = []
    # grand_mean_hs = hs.mean(axis=0)
    grand_var_hs = hs.std(axis=0)
    for i in range(np.unique(target).max()):
        sse_i = (hs-hs[target==i].mean(axis=0))**2
        sse_not_i = (hs-hs[target!=i].mean(axis=0))**2
        selectivity.append(1-(sse_i+sse_not_i)/(grand_var_hs+1e-8)) # sum sq explained by target
    selectivity = torch.stack(selectivity) # n_targets, n_batch, n_hidden
    plt.hist(selectivity.flatten().numpy(), bins=20)
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')
    plt.show()
    sort_inds = np.concatenate((np.argsort(selectivity[:e_size]), np.argsort(selectivity[e_size:])+e_size))
    cluster_labels = cluster(selectivity.reshape(n_hidden, 1))
    cluster_labels = cluster_labels[sort_inds]
    sorted_cluster_labels = cluster_labels.copy()
    sorted_cluster_labels[cluster_labels==cluster_labels[0]] = 0
    sorted_cluster_labels[cluster_labels==cluster_labels[-1]] = 2
    sorted_cluster_labels[(cluster_labels!=cluster_labels[0]) & (cluster_labels!=cluster_labels[-1])] = 1
    print(sorted_cluster_labels)
    return selectivity, sort_inds, sorted_cluster_labels

def plot_dpca(all_saved_states, task_mdprl, args):
    '''
    (1) regress hs activity with
            previous trial choice shape, color, pattern (3x3x3), 
            previous trial reward (2),
            current trial stimuli shape, color, pattern pairs (3x3x3), 
            current trial choice shape, color, pattern (3x3x3), 
            current trial reward (2),
    (2) get beta weights which is a mixture of value and stimulus intensity: h ~ Xw. 
        This will give beta weights timepoints X trials X hidden X latent variables,
        calculate cpd gives timepoints X trials X hidden X latent variables CPD values
    (3) compare w with marginal reward probability? see which it dimension it corresponds to the best
    '''

    n_trials, n_timesteps, n_sessions, n_hidden = all_saved_states['hs'].shape
    n_areas = args['num_areas'] 

    print("Calculating PSTH")

    '''
    organize by previous trial outcome
    '''
    hs_by_prev = np.zeros((n_hidden//n_areas, n_timesteps, 2, 3, 3, 3)) # sort data by previous trial choices and outcomes

    flat_hs_post = all_saved_states['hs'].numpy()[1:,...].transpose((2,0,1,3)).reshape((n_sessions*(n_trials-1), n_timesteps, n_hidden//n_areas))
    flat_rwds_pre = all_saved_states['rewards'].numpy()[1:,...].transpose((2,0,1,3)).reshape((n_sessions*(n_trials-1)))
    flat_acts_pre = all_saved_states['choices'].numpy()[1:,...].transpose((2,0,1,3)).reshape((n_sessions*(n_trials-1)))

    # the prev_f{}_vals are IN TERMS OF THE REWARD SCHEDULE, NOT THE PERCEPTUAL DIMENSIONS

    for prev_rwd_val in range(2):
        for prev_f1_val in range(3): 
            for prev_f2_val in range(3):
                for prev_f3_val in range(3):
                    # n_trials, 1, n_sessions, ...
                    act_f1_val = task_mdprl.index_shp[flat_acts_pre]
                    act_f2_val = task_mdprl.index_pttrn[flat_acts_pre]
                    act_f3_val = task_mdprl.index_clr[flat_acts_pre]

                    where_trial = (flat_rwds_pre==prev_rwd_val) & \
                                  (act_f1_val==prev_f1_val) & \
                                  (act_f2_val==prev_f2_val) & \
                                  (act_f3_val==prev_f3_val)
                    hs_by_prev[:, :, prev_rwd_val, prev_f1_val, prev_f2_val, prev_f3_val] = flat_hs_post[where_trial,...].mean(0)

    del flat_hs_post
    del flat_rwds_pre
    del flat_acts_pre
    
    '''
    organize by current trial stimuli
    '''
    hs_by_curr_stim = np.zeros((n_hidden, n_timesteps, 6, 6, 6)) # sort data by current trial choices and outcomes

    flat_hs_curr = all_saved_states['hs'].numpy().transpose((2,0,1,3)).reshape((n_sessions*n_trials, n_timesteps, n_hidden))
    flat_stims = all_saved_states['stimuli'].numpy().transpose((2,0,1,3)).reshape((n_sessions*n_trials, 2))

    for curr_f1_val in range(6):
        for curr_f2_val in range(6):
            for curr_f3_val in range(6):
                # n_trials, 1, n_sessions, ...
                stim_f1_val = task_mdprl.index_shp[flat_stims[:,0]]*2+task_mdprl.index_shp[flat_stims[:,1]]
                stim_f2_val = task_mdprl.index_pttrn[flat_stims[:,0]]*2+task_mdprl.index_pttrn[flat_stims[:,1]]
                stim_f3_val = task_mdprl.index_clr[flat_stims[:,0]]*2+task_mdprl.index_clr[flat_stims[:,1]]
                where_trial = (stim_f1_val==curr_f1_val) & \
                              (stim_f2_val==curr_f2_val) & \
                              (stim_f3_val==curr_f3_val)
                hs_by_prev[:, :, curr_f1_val, curr_f2_val, curr_f3_val] = flat_hs_post[where_trial,...].mean(0)

    del flat_stims

    '''
    organize by current trial outcome
    '''

    hs_by_curr_outcome = np.zeros((n_hidden, n_timesteps, 2, 3, 3, 3)) # sort data by previous trial choices and outcomes
    flat_rwds_curr = all_saved_states['rewards'].numpy().transpose((2,0,1,3)).reshape((n_sessions*n_trials))
    flat_acts_curr = all_saved_states['choices'].numpy().transpose((2,0,1,3)).reshape((n_sessions*n_trials))

    for curr_rew_val in range(2):
        for curr_f1_val in range(3):
            for curr_f2_val in range(3):
                for curr_f3_val in range(3):
                    # n_trials, 1, n_sessions, ...
                    act_f1_val = task_mdprl.index_shp[flat_acts_curr]
                    act_f2_val = task_mdprl.index_pttrn[flat_acts_curr]
                    act_f3_val = task_mdprl.index_clr[flat_acts_curr]
                    where_trial = (flat_rwds_curr==curr_rew_val) & \
                                  (act_f1_val==curr_f1_val) & \
                                  (act_f2_val==curr_f2_val) & \
                                  (act_f3_val==curr_f3_val)
                    hs_by_curr_outcome[:, :, curr_rew_val, curr_f1_val, curr_f2_val, curr_f3_val] = flat_hs_curr[where_trial,...].mean(0)
         
    del flat_hs_curr
    del flat_rwds_curr
    del flat_acts_curr


    print('Calculating DPCA')
    low_hs_by_prev, all_axes_by_prev, all_explained_vars_by_prev, all_labels_by_prev = \
        get_dpca(hs_by_prev, "rscp", n_components=10)
    low_hs_by_curr_stim, all_axes_by_curr_stim, all_explained_vars_by_curr_stim, all_labels_by_curr_stim = \
        get_dpca(hs_by_curr_stim, "scp", n_components=10)
    low_hs_by_curr_outcome, all_axes_by_curr_outcome, all_explained_vars_by_curr_outcome, all_labels_by_curr_outcome = \
        get_dpca(hs_by_curr_outcome, "rscp", n_components=10)

#     fig, axes = plt.subplots(2, 2)
#     for i in range(4):
#         plot_mean_and_std(axes[i//2, i%2], all_cpds[i].mean([0, 2]), 
#                           all_cpds[i].std([0, 2])/np.sqrt(n_trials//4*n_batch), label=['F1', 'F2', 'F3', 'C1', 'C2', 'C3', 'O'])
    return all_lrs, all_cpds, all_betas

def plot_rsa(hs, stim_order, stim_probs, cluster_label, e_size, splits=8):
    n_trials, n_timesteps, n_batch, n_hidden = hs.shape
    hs = hs.mean(1)
    ehs = hs[:,:,:e_size]
    ihs = hs[:,:,e_size:]
    n_steps_par_split = n_trials//splits
    stim_probs_ordered = order_stim_probs(stim_order, stim_probs) # 7, n_trials, n_batch, n_choice
    rnn_sims = []
    input_sims = []
    reg_results = []
    for i in range(splits-1):
        xs = [est[i*n_steps_par_split:(i+1)*n_steps_par_split, :, 0]-est[i*n_steps_par_split:(i+1)*n_steps_par_split, :, 1] 
              for est in stim_probs_ordered]
        rnn_sims.append([])
        input_sims.append([])
        reg_results.append([])
        for l in range(cluster_label.max()+1):
            rnn_sim, input_sim, reg_result = representational_similarity_analysis(xs, ehs[i*n_steps_par_split:(i+1)*n_steps_par_split,:,cluster_label[:e_size]==l])
            rnn_sims[-1].append(rnn_sim)
            input_sims[-1].append(input_sim)
            reg_results[-1].append(reg_result)
            rnn_sim, input_sim, reg_result = representational_similarity_analysis(xs, ihs[i*n_steps_par_split:(i+1)*n_steps_par_split,:,cluster_label[e_size:]==l])
            rnn_sims[-1].append(rnn_sim)
            input_sims[-1].append(input_sim)
            reg_results[-1].append(reg_result)

    fig = plt.figure('rsa_coeffs')
    labels = ['Shape', 'Pattern', 'Color', 'Shape+Pattern', 'Pattern+Color', 'Shape+Color', 'Shape+Pattern+Color']
    titles = ['Exc Left', 'Inh Left', 'Exc Nonsel', 'Inh Nonsel', 'Exc Right', 'Inh Right']
    for j in range(2*(cluster_label.max()+1)):
        ax = fig.add_subplot(320+j+1)
        for i in range(7):
            coeffs = [res[j].coef[i+1] for res in reg_results]
            ses = [res[j].se[i+1] for res in reg_results]
            plot_mean_and_std(ax, np.array(coeffs), np.array(ses), labels[i])
        ax.set_title(titles[j])
    fig.supxlabel('Time Segment')
    fig.supylabel('Regression Coefficient of RDM')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    # plt.savefig(f'plots/{plot_args.exp_dir}/rsa_coeffs')
    plt.show()

def run_model(args, model, task_mdprl, n_samples=None):
    model.eval()
    all_saved_states = defaultdict(list)
    output_size = args['output_size']

#     all_dim_orders = list(itertools.permutations(range(1, 4)))
#     all_dim_orders = [list(dim_order) for dim_order in all_dim_orders]

    for p in model.parameters():
        p.requires_grad = False # disable gradient calculation for parameters

    if n_samples is None:
        n_samples = task_mdprl.test_stim_order.shape[1]
    for batch_idx in tqdm.tqdm(range(n_samples)):
        # sample random order to permute reward schedule dimensions
#             curr_dim_order = all_dim_orders[np.random.choice(len(all_dim_orders))]
        pop_s, target_valid, output_mask, rwd_mask, ch_mask, index_s, prob_s = task_mdprl.generateinputfromexp(
            batch_size=1, test_N_s=args['test_N_s'], num_choices=args['num_options'], participant_num=batch_idx)
        
#             all_saved_states['dim_orders'].append(torch.from_numpy(np.expand_dims(np.array(curr_dim_order), axis=(0,1,2)))) # num trials (1) X time_steps(1) X batch_size(1) X num_dims
        # add empty list for the current episode
        all_saved_states['whs_final'].append([])

        all_saved_states['stimuli'].append(torch.from_numpy(np.expand_dims(index_s, axis=(1,2)))) # num trials X time_steps(1) X batch_size(1) X num_choices
        all_saved_states['reward_probs'].append(torch.from_numpy(np.expand_dims(prob_s, axis=(1,)))) # num_trials X time_steps(1) X batch_size X num_choices

        all_saved_states['choices'].append([])
        all_saved_states['foregone'].append([])
        all_saved_states['rewards'].append([])
        all_saved_states['choose_better'].append([])
        
        all_saved_states['hs_pre'].append([])
        all_saved_states['hs_post'].append([])

        all_saved_states['sensitivity'].append([])

        # reinitialize hidden layer activity
        hidden = None

        for i in range(len(pop_s['pre_choice'])):
            # first phase, give stimuli and no feedback
            output, hs, hidden, ss = model(pop_s['pre_choice'][i], hidden=hidden, 
                                            DAs=torch.zeros(1, args['batch_size'], 1)*rwd_mask['pre_choice'],
                                            Rs=torch.zeros(1, args['batch_size'], 2)*rwd_mask['pre_choice'],
                                            acts=torch.zeros(1, args['batch_size'], output_size)*ch_mask['pre_choice'],
                                            save_weights=True)

            # save activities before reward
            all_saved_states['hs_pre'][-1].append(hs) # [num_sessions, [num_trials, [time_pre, num_batch_size, hidden_size]]]

            if args['task_type']=='on_policy_double':
                # use output to calculate action, reward, and record loss function
                if args['decision_space']=='action':
                    action = torch.argmax(output[-1,:,:], -1) # batch size
                    rwd = (torch.rand(args['batch_size'])<prob_s[i][range(args['batch_size']), action]).float()
                    all_saved_states['choose_better'][-1].append((action==torch.argmax(prob_s[i], -1)).float().squeeze())
                elif args['decision_space']=='good':
                    # action_valid = torch.argmax(output[-1,:,index_s[i]], -1) # the object that can be chosen (0~1), (batch size, )
                    action_valid = torch.multinomial(output[-1,:,index_s[i]].softmax(-1), num_samples=1).squeeze(-1)
                    # backpropagate from choice to previous reward
                    (output[-1,:,index_s[i,1]]-output[-1,:,index_s[i,0]]).backward()
                    if i>0:
                        all_saved_states['sensitivity'][-1].append(rwd.grad)
                    else:
                        all_saved_states['sensitivity'][-1].append(torch.zeros_like(action_valid))
                    
                    action = index_s[i, action_valid] # (batch size, )
                    nonaction = index_s[i, 1-action_valid] # (batch size, )
                    rwd = (torch.rand(args['batch_size'])<prob_s[i][range(args['batch_size']), action_valid]).long()
                    rwd.requires_gradient = True ### for sensitivity analysis!
                    all_saved_states['choose_better'][-1].append((action_valid==torch.argmax(prob_s[i], -1)).float()[None,...]) 
                all_saved_states['rewards'][-1].append(rwd.float()[None,...])
                all_saved_states['choices'][-1].append(action[None,...])
                all_saved_states['foregone'][-1].append(nonaction[None,...])
            elif args['task_type'] == 'value':
                raise NotImplementedError
                rwd = (torch.rand(1)<prob_s[i]).float()
                output = output.reshape(output_mask['target'].shape[0], 1, output_size)
                acc.append(((output-target_valid['pre_choice'][i])*output_mask['target'].float().unsqueeze(-1)).pow(2).mean(0)/output_mask['target'].float().mean())
                curr_rwd.append(rwd)
            
            if args['task_type']=='on_policy_double':
                # use the action (optional) and reward as feedback
                pop_post = pop_s['post_choice'][i]
                action_enc = torch.eye(output_size)[action]
                rwd_enc = torch.eye(2)[rwd]
                action_enc = action_enc*ch_mask['post_choice']
                rwd_enc = rwd_enc*rwd_mask['post_choice']
                DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=action_enc, DAs=DAs, save_weights=False)
            elif args['task_type'] == 'value':
                raise NotImplementedError
                pop_post = pop_s['post_choice'][i]
                rwd_enc = torch.eye(2)[rwd]
                DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=None, DAs=DAs, save_weights=False)
            
            all_saved_states['hs_post'][-1].append(hs) # [num_sessions, [num_trials, [time_post, num_batch_size, hidden_size]]]
            all_saved_states['whs_final'][-1].append(hidden[2][None,...]) 
                # [num_sessions, [num_trials, [1, num_batch_size, hidden_size, hidden_size]]]

        # stack to create a trial dimension for each session
        for k in all_saved_states.keys():
            if isinstance(all_saved_states[k][-1], list):
                all_saved_states[k][-1] = torch.stack(all_saved_states[k][-1], axis=0)
        # [num_sessions, [num_trials, time_steps, num_batch_size, ...]]

    # concatenate all saved states along the batch dimension
    for k in all_saved_states.keys():
        all_saved_states[k] = torch.cat(all_saved_states[k], axis=2) # [num_trials, time_steps, num_sessions, ...]
        
    # all saved states of the form [num_trials, time_steps, num_sessions, ...]

    # concatenate activities pre- and post-feedback
    all_saved_states['hs'] = torch.cat([all_saved_states['hs_pre'], all_saved_states['hs_post']], dim=1)

    # concatenate all accuracies and rewards
    print(all_saved_states['rewards'].mean(), all_saved_states['choose_better'].mean())
    
    for k, v in all_saved_states.items():
        print(k, v.shape)
    
    return all_saved_states

def run_model_all_pairs_with_hidden_init(args, model, task_mdprl, n_samples=60, hidden_init=None):
    model.eval()
    all_indices = []
    all_probs = []
    all_saved_states_pre = defaultdict(list) # each entry has value of size (num pairs X num_samples X ...)
    all_saved_states_post = defaultdict(list)
    all_saved_states = defaultdict(list)
    output_size = args['output_size']
    with torch.no_grad():
        for pair_idx in tqdm.tqdm(range(task_mdprl.pairs.shape[0])):
            for batch_idx in range(n_samples):
                pop_s, target_valid, output_mask, rwd_mask, ch_mask, index_s, prob_s = \
                    task_mdprl.generateinput(batch_size=1, N_s=0, num_choices=args['num_options'], rwd_schedule=task_mdprl.prob_mdprl, \
                                             stim_order=task_mdprl.pairs[pair_idx:pair_idx+1,:])

                hidden = hidden_init
                all_saved_states_pre
                    # first phase, give stimuli and no feedback
                output, hs, hidden, ss = model(pop_s['pre_choice'][0], hidden=hidden, 
                                                DAs=torch.zeros(1, 1, 1)*rwd_mask['pre_choice'],
                                                Rs=torch.zeros(1, 1, 2)*rwd_mask['pre_choice'],
                                                acts=torch.zeros(1, 1, output_size)*ch_mask['pre_choice'],
                                                save_weights=False)

                # add empty list for the current episode
                if batch_idx==0:
                    for k in ss.keys():
                        all_saved_states_pre[k].append([])
                        all_saved_states_post[k].append([])
#                     all_saved_states['whs_final'].append([])
                    all_saved_states_pre['hs'].append([])
                    all_saved_states_post['hs'].append([])

                # save pre-feedback states
                for k, v in ss.items():
                    all_saved_states_pre[k][-1].append(v)
                all_saved_states_pre['hs'][-1].append(hs) # [num_pairs, [num_samples_per_pair, [time_pre, 1, hidden_size]]]

                # use output to calculate action, reward, and record loss function
                action_valid = torch.argmax(output[-1,:,index_s[0]], -1)
                action = index_s[0, action_valid] # (batch size, )
                rwd = (torch.rand(args['batch_size'])<prob_s[0][range(args['batch_size']), action_valid]).long()
                
                # use the action (optional) and reward as feedback
                pop_post = pop_s['post_choice'][0]
                action_enc = torch.eye(output_size)[action]
                rwd_enc = torch.eye(2)[rwd]
                action_enc = action_enc*ch_mask['post_choice']
                rwd_enc = rwd_enc*rwd_mask['post_choice']
                DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=action_enc, DAs=DAs, save_weights=False)
                
                # save the post-feedback states
                for k, v in ss.items():
                    all_saved_states_post[k][-1].append(v)
                all_saved_states_post['hs'][-1].append(hs) # [num_pairs, [num_samples_per_pair, [time_post, 1, hidden_size]]]

#             print(len(all_saved_states_pre['hs'][0][0]))


            # stack trials for each session
            for k in all_saved_states_pre.keys():
                all_saved_states_pre[k][-1] = torch.stack(all_saved_states_pre[k][-1], axis=0)
            for k in all_saved_states_post.keys():
                all_saved_states_post[k][-1] = torch.stack(all_saved_states_post[k][-1], axis=0)
            for k in all_saved_states.keys():
                all_saved_states[k][-1] = torch.stack(all_saved_states[k][-1], axis=0)
            # [num_pairs, [num_samples_per_pair, time_steps, 1, hidden_size]]
            
            # save accuracies and reward
            all_indices.append(index_s)
            all_probs.append(prob_s)

        # concatenate all saved states
        for k in all_saved_states_pre.keys():
            all_saved_states_pre[k] = torch.stack(all_saved_states_pre[k], axis=0)
        for k in all_saved_states_post.keys():
            all_saved_states_post[k] = torch.stack(all_saved_states_post[k], axis=0)
        for k in all_saved_states.keys():
            all_saved_states[k] = torch.stack(all_saved_states[k], axis=0)
        # [num_pairs, num_samples_per_pair, time_steps, 1, hidden_size]

        # merge pre and post if necessary
        for k in all_saved_states_pre.keys():
            all_saved_states[k] = torch.cat([all_saved_states_pre[k], all_saved_states_post[k]], dim=2)

        for k, v in all_saved_states.items():
            print(k, v.shape)
        
        all_indices = torch.stack(all_indices, dim=1) # trials X batch size X 2
        all_probs = torch.stack(all_probs, dim=1)
        return all_saved_states
    
def order_stim_probs(stim_order, stim_probs):
    n_trials, n_batch, n_choices = stim_order.shape
    assert len(stim_probs)==7 and len(stim_probs[0])==27
    stim_probs_ordered = []
    for est in stim_probs:
        stim_probs_ordered.append(est[stim_order]) # output[t, b, c] = stim_probs[stim_order[t, b, c]]
    return stim_probs_ordered

# if __name__=='__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--exp_dir', type=str, help='Directory of trained model')
#     parser.add_argument('--connectivity_and_lr', action='store_true')
#     parser.add_argument('--n_samples', type=int, default=21)
#     parser.add_argument('--n_runs_per_sample', type=int, default=1)
#     parser.add_argument('--learning_rates', action='store_true')
#     parser.add_argument('--sort_rec_w', action='store_true')
#     parser.add_argument('--sort_rec_lr', action='store_true')
#     parser.add_argument('--tca', action='store_true')
#     parser.add_argument('--pca', action='store_true')
#     parser.add_argument('--rsa', action='store_true')
#     parser.add_argument('--tdr', action='store_true')
#     parser.add_argument('--learning_curve', action='store_true')
#     parser.add_argument('--attn_entropy', action='store_true')
#     parser.add_argument('--attn_distribution', action='store_true')
#     plot_args = parser.parse_args()

#     # load training config
#     f = open(os.path.join(plot_args.exp_dir, 'args.json'), 'r')
#     args = json.load(f)
#     print('loaded args')
#     # load model
#     exp_times = {
#         'start_time': -0.25,
#         'end_time': 0.75,
#         'stim_onset': 0.0,
#         'stim_end': 0.6,
#         'rwd_onset': 0.5,
#         'rwd_end': 0.6,
#         'choice_onset': 0.35,
#         'choice_end': 0.5,
#         'total_time': 1}
#     exp_times['dt'] = args['dt']
#     task_mdprl = MDPRL(exp_times, args['input_type'])
#     print('loaded task')

#     input_size = {
#         'feat': args['stim_dim']*args['stim_val'],
#         'feat+obj': args['stim_dim']*args['stim_val']+args['stim_val']**args['stim_dim'], 
#         'feat+conj+obj': args['stim_dim']*args['stim_val']+args['stim_dim']*args['stim_val']*args['stim_val']+args['stim_val']**args['stim_dim'],
#     }[args['input_type']]

#     input_unit_group = {
#         'feat': [args['stim_dim']*args['stim_val']], 
#         'feat+obj': [args['stim_dim']*args['stim_val'], args['stim_val']**args['stim_dim']], 
#         'feat+conj+obj': [args['stim_dim']*args['stim_val'], args['stim_dim']*args['stim_val']*args['stim_val'], args['stim_val']**args['stim_dim']]
#     }[args['input_type']]

#     if args['attn_type']!='none':
#         if args['input_type']=='feat':
#             channel_group_size = [args['stim_val']]*args['stim_dim']
#         elif args['input_type']=='feat+obj':
#             channel_group_size = [args['stim_val']]*args['stim_dim'] + [args['stim_val']**args['stim_dim']]
#         elif args['input_type']=='feat+conj+obj':
#             channel_group_size = [args['stim_val']]*args['stim_dim'] + [args['stim_val']*args['stim_val']]*args['stim_dim'] + [args['stim_val']**args['stim_dim']]
#     else:
#         channel_group_size = [input_size]

#     output_size = 1 if args['task_type']=='value' else 2
#     model_specs = {'input_size': input_size, 'hidden_size': args['hidden_size'], 'output_size': output_size, 
#             'plastic': args['plas_type']=='all', 'attention_type': args['attn_type'], 'activation': args['activ_func'],
#             'dt': args['dt'], 'tau_x': args['tau_x'], 'tau_w': args['tau_w'], 'channel_group_size': channel_group_size,
#             'c_plasticity': None, 'e_prop': args['e_prop'], 'init_spectral': args['init_spectral'], 'balance_ei': args['balance_ei'],
#             'sigma_rec': args['sigma_rec'], 'sigma_in': args['sigma_in'], 'sigma_w': args['sigma_w'], 
#             'rwd_input': args.get('rwd_input', False), 'action_input': args['action_input'], 
#             'input_unit_group': input_unit_group, 'sep_lr': args['sep_lr'], 'plastic_feedback': args['plastic_feedback'],
#             'value_est': 'policy' in args['task_type'], 'num_choices': 2 if 'double' in args['task_type'] else 1}
#     if 'double' in args['task_type']:
#         model = MultiChoiceRNN(**model_specs)
#     else:
#         model = SimpleRNN(**model_specs)
#     state_dict = torch.load(os.path.join(plot_args.exp_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))['model_state_dict']
#     model.load_state_dict(state_dict)
#     print('loaded model')


#     # if plot_args.sort_rec_w:
#         # plot_sorted_matrix(model.h2h.effective_weight().detach(), int(args['e_prop']*args['hidden_size']), 'weight')
#     # if plot_args.sort_rec_lr:
#         # plot_sorted_matrix(model.kappa_rec.relu().squeeze().detach(), int(args['e_prop']*args['hidden_size']), 'lr')
    
#     losses, losses_means, losses_stds, all_saved_states, all_indices = run_model(args, model, task_mdprl)
#     print('simulation complete')

#     stim_probs_ordered = []
#     for i in range(plot_args.n_samples):
#         stim_probs_ordered.append(np.array([task_mdprl.prob_mdprl.reshape(27)[all_indices[:,i,0]], task_mdprl.prob_mdprl.reshape(27)[all_indices[:,i,1]]]).T)
#     stim_probs_ordered = np.stack(stim_probs_ordered, axis=1)

#     selectivity, sort_inds, cluster_label = unit_selectivity(all_saved_states['hs'], np.argmax(stim_probs_ordered, axis=-1), 
#                                                         e_size=int(args['e_prop']*args['hidden_size']))
    
#     # load metrics
#     metrics = json.load(open(os.path.join(plot_args.exp_dir, 'metrics.json'), 'r'))
#     if plot_args.connectivity_and_lr:
#         plot_connectivity_lr(sort_inds, x2hw=[mx2h.effective_weight().detach() for mx2h in model.x2h],
#                              h2hw=model.h2h.effective_weight().detach(),
#                              hb=state_dict['h2h.bias'].detach(),
#                              h2ow=model.h2o.effective_weight().detach(),
#                              h2ob=state_dict['h2o.bias'].detach(),
#                              h2vw=model.h2v.effective_weight().detach(),
#                              h2vb=state_dict['h2o.bias'].detach(),
#                              h2attnw=model.attn_func.effective_weight().detach(),
#                              h2attnb=torch.zeros(model.attn_func.weight.shape[0]),
#                              aux2h=model.aux2h.effective_weight().detach(),
#                              kappa_in=[ki.squeeze().abs().detach()*model.x2h[0].mask for ki in model.kappa_in],
#                              kappa_rec=model.kappa_rec.squeeze().abs().detach()*model.h2h.mask,
#                              kappa_fb=model.kappa_fb.squeeze().abs().detach()*model.attn_func.mask,
#                              e_size=int(args['e_prop']*args['hidden_size']))
#     if plot_args.learning_curve:
#         plot_learning_curve(losses, losses_means, losses_stds)
#     if plot_args.attn_entropy:
#         plot_attn_entropy(all_saved_states['attns'])
#     if plot_args.attn_distribution:
#         plot_attn_distribution(all_saved_states['attns'])
#     if plot_args.rsa:
#         print(cluster_label*0)
#         plot_rsa(all_saved_states['hs'], stim_probs=task_mdprl.value_est(), stim_order=all_indices, cluster_label=cluster_label*0, e_size=model.h2h.e_size)
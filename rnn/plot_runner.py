import json
import os

from plot_utils import *
from plot_main import *
from choice_analysis import *
import seaborn as sns
import scipy.stats as stats
from mne.stats import permutation_cluster_test
from dPCA import dPCA

plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams.update({'font.size': 20})
plt.rcParams['image.interpolation']='nearest'

plt.rc('axes', linewidth=1)
plt.rc('xtick.major', width=2, size=8)
plt.rc('ytick.major', width=2, size=8)
plt.rc('xtick.minor', width=1, size=4)
plt.rc('ytick.minor', width=1, size=4)
plt.rc('mathtext', default='regular')


from models import HierarchicalPlasticRNN
from task import MDPRL


import random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def plot_all_connectivities(all_models, args):
    for model_vis_idx in range(len(all_models)):
        plot_connectivity_lr(torch.arange(args['hidden_size']*args['num_areas']), 
                                x2hw=all_models[model_vis_idx].rnn.x2h['stim'].effective_weight().detach(),
                                h2hw=all_models[model_vis_idx].rnn.h2h.effective_weight().detach(),
                                hb=all_models[model_vis_idx].rnn.h2h.bias.detach(),
                                h2ow=all_models[model_vis_idx].h2o['action'].effective_weight().detach(),
                                aux2h=torch.zeros(160, 27),
                                h2aux=all_models[model_vis_idx].h2o['chosen_obj'].effective_weight().detach(),
                                kappa_rec=all_models[model_vis_idx].plasticity.effective_lr().detach(),
                                e_size=int(args['e_prop']*args['hidden_size'])*args['num_areas'], args=args, mdl_idx=model_vis_idx)


def plot_wh_spectrum(all_models, args):
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    axes[0].plot(np.cos(np.linspace(-2*np.pi,2*np.pi,100)),\
                    np.sin(np.linspace(-2*np.pi,2*np.pi,100)), \
                    'k:', linewidth=0.5)
    axes[1].axhline(1, linestyle='--', color='k')

    all_eig_plots = []

    for model_idx, model in enumerate(all_models):
        h2h = model.rnn.h2h.effective_weight().detach()
        axes[0].scatter(torch.linalg.eig(h2h).eigenvalues.real, 
                    torch.linalg.eig(h2h).eigenvalues.imag,
                    alpha=max(1/len(all_models), 0.4))
        all_eig_plots.append(axes[1].plot(torch.sort(torch.linalg.eig(h2h).eigenvalues.abs(), descending=True)[0], '-',
                    alpha=max(1/len(all_models), 0.4), label=model_idx))


    axes[0].set_xlabel(r'$\Re(\lambda)$')
    axes[0].set_ylabel(r'$\Im(\lambda)$')
    axes[1].set_xlabel('Eigenvalue rank')
    axes[1].set_ylabel(r'$|\lambda|$')

    fig.legend(fontsize=16)

    sns.despine()
    plt.tight_layout()

    with PdfPages(f'plots/{plot_save_dir}/wh_spectrum.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/wh_spectrum.pdf')


def get_weights_by_area(all_models, args):
    all_model_rec_intra = []
    all_model_rec_inter_ff = []
    all_model_rec_inter_fb = []

    for model in all_models:
        h2h = model.rnn.h2h.effective_weight().detach()

        rec_intra = []
        for i in range(NUM_AREAS):
            rec_intra.append(h2h[list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))]
                            [:,list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))])
        
        all_model_rec_intra.append(rec_intra)

        rec_inter_ff = []
        rec_inter_fb = []

        for i in range(NUM_AREAS-1):
            rec_inter_ff.append(h2h[list(range((i+1)*E_SIZE, (i+2)*E_SIZE))+\
                                    list(range(2*E_SIZE+(i+1)*I_SIZE, 2*E_SIZE+(i+2)*I_SIZE))]
                                [:,list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                    list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))])
            rec_inter_fb.append(h2h[list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                    list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))]
                                [:,list(range((i+1)*E_SIZE, (i+2)*E_SIZE))+\
                                    list(range(2*E_SIZE+(i+1)*I_SIZE, 2*E_SIZE+(i+2)*I_SIZE))])
            
        all_model_rec_inter_ff.append(rec_inter_ff)
        all_model_rec_inter_fb.append(rec_inter_fb)

    return all_model_rec_intra, all_model_rec_inter_ff, all_model_rec_inter_fb


def get_lrs_by_area(all_models, args):
    all_model_kappa_rec_intra = []
    all_model_kappa_inter_ff = []
    all_model_kappa_inter_fb = []

    for model in all_models:

        kappa_rec = model.plasticity.effective_lr().detach()

        kappa_rec_intra = []
        for i in range(NUM_AREAS):
            kappa_rec_intra.append(kappa_rec[list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                    list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))]
                                    [:,list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                        list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))])

        all_model_kappa_rec_intra.append(kappa_rec_intra)
        
        
        kappa_inter_ff = []
        kappa_inter_fb = []

        for i in range(NUM_AREAS-1):
            kappa_inter_ff.append(kappa_rec[list(range((i+1)*E_SIZE, (i+2)*E_SIZE))+\
                                        list(range(2*E_SIZE+(i+1)*I_SIZE, 2*E_SIZE+(i+2)*I_SIZE))]
                                    [:,list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                        list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))])
            kappa_inter_fb.append(kappa_rec[list(range(i*E_SIZE, (i+1)*E_SIZE))+\
                                        list(range(2*E_SIZE+i*I_SIZE, 2*E_SIZE+(i+1)*I_SIZE))]
                                        [:,list(range((i+1)*E_SIZE, (i+2)*E_SIZE))+\
                                        list(range(2*E_SIZE+(i+1)*I_SIZE, 2*E_SIZE+(i+2)*I_SIZE))])
            
        all_model_kappa_inter_ff.append(kappa_inter_ff)
        all_model_kappa_inter_fb.append(kappa_inter_fb)

    return all_model_kappa_rec_intra, all_model_kappa_inter_ff, all_model_kappa_inter_fb


def plot_weights_lrs_by_area(all_model_rec_intra, all_model_rec_inter_ff, all_model_rec_inter_fb, 
                             all_model_kappa_inter_ff, all_model_kappa_inter_fb, args):
    fig, axes = plt.subplots(4,3, figsize=(10, 8), sharex=True)

    flat_rec_w = np.array(all_model_rec_intra)
    flat_rec_w[np.abs(flat_rec_w)<np.finfo(float).eps] = np.nan

    eps_for_hist = 1e-8
    binwidth = 0.2

    sns.histplot(x=np.log10(flat_rec_w[:,0,:E_SIZE,:E_SIZE].flatten()+eps_for_hist), 
                ax=axes[0,0], color='salmon', binwidth=binwidth)

    sns.histplot(x=np.log10(flat_rec_w[:,0,E_SIZE:,:E_SIZE].flatten()+eps_for_hist), 
                ax=axes[2,0], color='salmon', binwidth=binwidth)

    sns.histplot(x=np.log10(-flat_rec_w[:,0,:E_SIZE,E_SIZE:].flatten()+eps_for_hist), 
                    ax=axes[0,2], color='deepskyblue', binwidth=binwidth)

    sns.histplot(x=np.log10(-flat_rec_w[:,0,E_SIZE:,E_SIZE:].flatten()+eps_for_hist), 
                    ax=axes[2,2], color='deepskyblue', binwidth=binwidth)


    sns.histplot(x=np.log10(flat_rec_w[:,1,:E_SIZE,:E_SIZE].flatten()+eps_for_hist), 
                    ax=axes[1,1], color='salmon', binwidth=binwidth)

    sns.histplot(x=np.log10(flat_rec_w[:,1,E_SIZE:,:E_SIZE].flatten()+eps_for_hist), 
                    ax=axes[3,1], color='salmon', binwidth=binwidth)

    sns.histplot(x=np.log10(-flat_rec_w[:,1,:E_SIZE,E_SIZE:].flatten()+eps_for_hist), 
                    ax=axes[1,2], color='deepskyblue', binwidth=binwidth)

    sns.histplot(x=np.log10(-flat_rec_w[:,1,E_SIZE:,E_SIZE:].flatten()+eps_for_hist), 
                    ax=axes[3,2], color='deepskyblue', binwidth=binwidth)


    flat_inter_w = np.array(all_model_rec_inter_ff).squeeze()
    flat_inter_kappa = np.array(all_model_kappa_inter_ff).squeeze()

    sns.histplot(x=np.log10(flat_inter_w[:,:E_SIZE,:E_SIZE].flatten()+eps_for_hist), 
                y=np.log10(flat_inter_kappa[:,:E_SIZE,:E_SIZE].flatten()+eps_for_hist),
                ax=axes[1,0], color='salmon', binwidth=binwidth)

    sns.histplot(x=np.log10(flat_inter_w[:,E_SIZE:,:E_SIZE].flatten()+eps_for_hist), 
                y=np.log10(flat_inter_kappa[:,E_SIZE:,:E_SIZE].flatten()+eps_for_hist),
                ax=axes[3,0], color='salmon', binwidth=binwidth)


    flat_inter_w = np.array(all_model_rec_inter_fb).squeeze()
    flat_inter_kappa = np.array(all_model_kappa_inter_fb).squeeze()

    sns.histplot(x=np.log10(flat_inter_w[:,:E_SIZE,:E_SIZE].flatten()+eps_for_hist), 
                y=np.log10(flat_inter_kappa[:,:E_SIZE,:E_SIZE].flatten()+eps_for_hist),
                    ax=axes[0,1], color='salmon', binwidth=binwidth)

    sns.histplot(x=np.log10(flat_inter_w[:,E_SIZE:,:E_SIZE].flatten()+eps_for_hist), 
                y=np.log10(flat_inter_kappa[:,E_SIZE:,:E_SIZE].flatten()+eps_for_hist),
                    ax=axes[2,1], color='salmon', binwidth=binwidth)

    for i in range(4):
        axes[i,2].set_ylabel("")
    fig.supxlabel('Initial weights')
    fig.supylabel('Learning rates')
    sns.despine()
    fig.tight_layout()

    with PdfPages(f'plots/{args["plot_save_dir"]}/weights_lrs_by_area.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{args["plot_save_dir"]}/weights_lrs_by_area.pdf')


def get_dpca_results_by_weights(all_model_weights, n_components_for_dpca):

    dpca_results = defaultdict(list)

    for model_idx, model_weights in enumerate(all_model_weights):
        dpca_mdl = dPCA.dPCA('spc', n_components=n_components_for_dpca, regularizer=None)
        low_hs = dpca_mdl.fit_transform(model_weights.reshape((-1,3,3,3)))

        dpca_results['low_hs'].append(low_hs)
        dpca_results['encoding_axes'].append(dpca_mdl.P)
        dpca_results['unitwise_explained_var'].append(dpca_mdl.unitwise_explained_variance_ratio_)
        dpca_results['total_explained_var'].append(dpca_mdl.explained_variance_ratio_)
        dpca_results['marginalized_psth'].append(dpca_mdl.marginalized_psth)

    dpca_results = dict(dpca_results)

    return dpca_results

def align_dpca_axes(all_model_dpca_to_align, all_model_dpca_target, n_components_for_dpca, args):
    # use the procrustes alignment to calculate the transformation matrix across low_hs
    # use the transformation matrix to align the encoding axes
    # full_matrix (num) = low_hs_orig [num_obj, num_components] @ encoding_axes_orig [num_components, num_units]
    #                   = (low_hs_target [num_obj, num_components] @ R [num_components, num_components]) @ encoding_axes_orig [num_components, num_units]
    #                   = low_hs_target [num_obj, num_components] @ (R [num_components, num_components] @ encoding_axes_orig [num_components, num_units])
    # so the new encoding_axes = R @ encoding_axes_orig, where R is the transformation matrix such that low_hs_target @ R = low_hs_orig

    all_model_dpca_to_align['original_encoding_axes'] = all_model_dpca_to_align['encoding_axes']
    # calculate the transformation matrix
    for mdl_idx in range(len(all_model_dpca_to_align['low_hs'])):
        for k_name in n_components_for_dpca.keys():
            low_hs_to_align = all_model_dpca_to_align['low_hs'][mdl_idx][k_name]
            low_hs_to_align = low_hs_to_align.reshape((n_components_for_dpca[k_name],-1)) # (num_dims, 27)

            low_hs_target = all_model_dpca_target['low_hs'][mdl_idx][k_name]
            low_hs_target = low_hs_target.reshape((n_components_for_dpca[k_name],-1)) # (num_dims, 27)

            # calculate the transformation matrix ensured to have determinant 1
            u, _, vh = np.linalg.svd(low_hs_target@low_hs_to_align.T)
            transformation_matrix = u@vh

            all_model_dpca_to_align['encoding_axes'][mdl_idx][k_name] = \
                (transformation_matrix@all_model_dpca_to_align['encoding_axes'][mdl_idx][k_name].T).T
    

def get_all_dpca_results(all_models, task_mdprl, n_components_for_dpca, args):
    all_model_stims_in = []
    all_model_choice_out = []
    all_model_choice_in = []
    all_model_stim_out = []

    # run the dpca for input and output weights from each area
    for model in all_models:
        stim_in = model.rnn.x2h['stim'].effective_weight()[input_weight_inds].detach().numpy().copy()@task_mdprl.stim_encoding('all_onehot').T
        all_model_stims_in.append(stim_in.reshape((args['hidden_size'],3,3,3)))

        choice_out = model.h2o['action'].effective_weight().detach().numpy().copy()[:,output_weight_inds]
        all_model_choice_out.append(choice_out.T.reshape((args['hidden_size'],3,3,3)))

        # choice_in = model.rnn.x2h['action'].effective_weight().detach().numpy().copy()[output_weight_inds,:]
        # all_model_choice_in.append(choice_in.reshape((args['hidden_size'],3,3,3)))
        
        stim_out = model.h2o['chosen_obj'].effective_weight().detach().numpy().copy()[:,input_weight_inds]
        all_model_stim_out.append(stim_out.T.reshape((args['hidden_size'],3,3,3)))

    all_model_dpca_stim_in = get_dpca_results_by_weights(all_model_stims_in, n_components_for_dpca)
    all_model_dpca_choice_out = get_dpca_results_by_weights(all_model_choice_out, n_components_for_dpca)
    # all_model_dpca_choice_in = get_dpca_results_by_weights(all_model_choice_in, n_components_for_dpca)
    all_model_dpca_stim_out = get_dpca_results_by_weights(all_model_stim_out, n_components_for_dpca)

    # all_model_dpca_stim_in_exc = get_dpca_results_by_weights(all_model_stims_in[:E_SIZE], n_components_for_dpca)
    # all_model_dpca_stim_in_inh = get_dpca_results_by_weights(all_model_stims_in[E_SIZE:], n_components_for_dpca)
    # all_model_dpca_choice_in_exc = get_dpca_results_by_weights(all_model_choice_in[:E_SIZE], n_components_for_dpca)
    # all_model_dpca_choice_in_inh = get_dpca_results_by_weights(all_model_choice_in[E_SIZE:], n_components_for_dpca)

    # align the encoding axes across weights and areas by using the stimulus input as reference
    # this makes each encoding axes the same for all models
    align_dpca_axes(all_model_dpca_stim_out, all_model_dpca_stim_in, n_components_for_dpca, args)
    align_dpca_axes(all_model_dpca_choice_out, all_model_dpca_stim_in, n_components_for_dpca, args)
    # align_dpca_axes(all_model_dpca_choice_in, all_model_dpca_stim_in, n_components_for_dpca, args)

    return all_model_dpca_stim_in, all_model_dpca_choice_out, all_model_dpca_stim_out
    

def run_plot_psth_geometry(all_model_dpca_stim_in, all_model_dpca_choice_out, args):

    fig, axes = plt.subplots(2,2, figsize=(10, 7), height_ratios=(6,1))

    plot_psth_geometry(all_model_dpca_stim_in, dim_labels, axes[:,0], "Stimuli input")
    plot_psth_geometry(all_model_dpca_choice_out, dim_labels, axes[:,1], "Choice output")
    # plot_psth_geometry(all_model_dpca_choice_in, dim_labels, axes[:,2], "Choice input")
    # plot_psth_geometry(all_model_dpca_stim_out, dim_labels, axes[:,3], "Stimuli output")

    axes[0,0].set_ylabel('Cosine similarity')
    axes[1,0].set_ylabel('Norm')

    plt.tight_layout()
    plt.show()

    with PdfPages(f'plots/{plot_save_dir}/weight_psth_geometry.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/weight_psth_geometry.pdf')


def run_plot_weight_exp_vars(all_model_dpca_stim_in, all_model_dpca_choice_out, args):
    
    fig, axes = plt.subplots(2,1, figsize=(10,5), sharex=True)

    plot_weight_exp_vars(n_components_for_dpca, all_model_dpca_stim_in, axes[0], "Stimuli input")
    plot_weight_exp_vars(n_components_for_dpca, all_model_dpca_choice_out, axes[1], "Choice output")
    # plot_weight_exp_vars(n_components_for_dpca, all_model_dpca_choice_in, axes[2], "Choice input")
    # plot_weight_exp_vars(n_components_for_dpca, all_model_dpca_stim_out, axes[3], "Stimuli output")

    labels = dim_labels

    axes[0].set_xlabel('Components')
    axes[0].set_xticklabels(np.arange(1,9,1))
    fig.supylabel('Explained Variance Ratio')
    plt.tight_layout()
    handles, _ = axes[0].get_legend_handles_labels()

    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    fig.legend(handles, labels, loc='center right')
    sns.despine()
    fig.show()

    with PdfPages(f'plots/{plot_save_dir}/input_output_weight_variance.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/input_output_weight_variance.pdf')


def run_plot_output_choice_overlap(all_model_dpca_stim_in, all_model_dpca_stim_out, args):

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    stim_ortho_input_axes = plot_input_output_overlap(all_model_dpca_stim_in, all_model_dpca_stim_out, n_components_for_dpca, dim_labels, axes)
    # stim_ortho_output_axes = plot_input_output_overlap(all_model_dpca_choice_in, all_model_dpca_choice_out, n_components_for_dpca, dim_labels, axes[1])

    fig.tight_layout()

    with PdfPages(f'plots/{plot_save_dir}/out_ch_overlap.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/out_ch_overlap.pdf')

    return stim_ortho_input_axes, #stim_ortho_output_axes


def run_plot_dpca_axes_overlap(all_model_dpca_stim_in, all_model_dpca_choice_out, args):
    fig, axes = plt.subplots(2, 2, height_ratios=(1, 3), figsize=(10, 6))
    
    all_model_dpca_axes_stim_in = test_dpca_overlap(all_model_dpca_stim_in, n_components_for_dpca, 
                                1/np.sqrt(args['hidden_size']), 
                                "Stimuli input", axes[:,0])
    all_model_dpca_axes_ch_out = test_dpca_overlap(all_model_dpca_choice_out, n_components_for_dpca, 
                                1/np.sqrt(int(args['hidden_size']*args['e_prop'])), 
                                "Choice output", axes[:,1])
    # all_model_dpca_axes_ch_in = test_dpca_overlap(all_model_dpca_choice_in, n_components_for_dpca, 
    #                             1/np.sqrt(args['hidden_size']), 
    #                             "Choice input", axes[:,2])
    # all_model_dpca_axes_stim_out = test_dpca_overlap(all_model_dpca_stim_out, n_components_for_dpca, 
    #                             1/np.sqrt(args['hidden_size']), 
    #                             "Stimuli output", axes[:,3])
    axes[0,1].set_ylabel(" ")
    axes[1,0].set_ylabel('Overlap', labelpad=20)

    fig.tight_layout()

    with PdfPages(f'plots/{plot_save_dir}/fixed_weight_axis_overlap.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/fixed_weight_axis_overlap.pdf')

    return all_model_dpca_axes_stim_in, all_model_dpca_axes_ch_out


def run_plot_selectivity_clusters(all_model_dpca_stim_in, all_model_dpca_choice_out, args):
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), height_ratios=(2, 1))

    # ideal_centroids = [np.concatenate([
    #                         np.concatenate([np.eye(7), np.eye(7)], axis=1)[:6],
    #                         np.concatenate([np.eye(7), np.zeros((7,7))], axis=1)]), 
    #                     np.concatenate([
    #                         np.concatenate([np.eye(7), np.eye(7)], axis=1),
    #                         np.concatenate([np.zeros((7,7)), np.eye(7)], axis=1)[3:6]]),
    #                     np.concatenate([np.eye(7), np.zeros((7,7))], axis=1)]

    # cluster units in stim and choice areas based on both the input and output weights
    all_model_selectivity_clusters_stim = \
                        plot_selectivity_clusters(all_model_dpca_stim_in, ['s','p','c','pc','sc','sp','spc'], 
                            [np.eye(7), np.eye(7)], E_SIZE, I_SIZE, "Stimuli input", axes[:,0])
    all_model_selectivity_clusters_choice = \
                        plot_selectivity_clusters(all_model_dpca_choice_out, ['s','p','c','pc','sc','sp','spc'], 
                            [np.eye(7), np.eye(7)], E_SIZE, I_SIZE, "Choice output", axes[:,1])

    axes[0,0].set_ylabel('Cluster centroids', labelpad=20)
    axes[1,0].set_ylabel("Rank corr.", labelpad=20)

    fig.tight_layout()
    with PdfPages(f'plots/{plot_save_dir}/fixed_weight_selectivity_clusters.pdf') as pdf:
        pdf.savefig(fig)
        print(f'Figure saved at plots/{plot_save_dir}/fixed_weight_selectivity_clusters.pdf')
    
    return all_model_selectivity_clusters_stim, all_model_selectivity_clusters_choice

def run_plot_reparam_weights(all_model_dpca_stim_in, all_model_dpca_choice_out, 
                             all_model_rec_intra, all_model_rec_inter_ff, all_model_rec_inter_fb, args):
    fig_heatmap, axes_heatmap = plt.subplots(1, 1, figsize=(10, 10))
    fig_violin, axes_violin = plt.subplots(2, 2, figsize=(10, 10))

    dpca_dict = [all_model_dpca_stim_in, all_model_dpca_choice_out]

    rec_weight_list = [
        [[rec_intra[0] for rec_intra in all_model_rec_intra], 
        [rec_inter_fb[0] for rec_inter_fb in all_model_rec_inter_fb]], 
        [[rec_inter_ff[0] for rec_inter_ff in all_model_rec_inter_ff],
        [rec_intra[1] for rec_intra in all_model_rec_intra]]
    ]

    plot_reparam_weights(dpca_dict, rec_weight_list, n_components_for_dpca, axes_heatmap, axes_violin)

    total_components = np.sum(list(n_components_for_dpca.values()))

    for j in range(2):
        block_boundaries = np.cumsum(list(n_components_for_dpca.values()))[:-1]
        for i in block_boundaries:
            axes_heatmap.axvline(x=total_components*j+i,color='k',linewidth=0.2)
            axes_heatmap.axhline(y=total_components*j+i,color='k',linewidth=0.2)
    axes_heatmap.axvline(x=total_components,color='k',linewidth=0.2)
    axes_heatmap.axhline(y=total_components,color='k',linewidth=0.2)
    
    tick_locs = np.cumsum([0, *n_components_for_dpca.values()])[:-1]+np.array(list(n_components_for_dpca.values()))//2
    axes_heatmap.set_xticks(np.concatenate([tick_locs, total_components+tick_locs]), 
                            [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$']*2, size=14, rotation=0)
    axes_heatmap.set_yticks(np.concatenate([tick_locs, total_components+tick_locs]), 
                            [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$']*2, size=16)

    axes_violin[0,0].set_title('Input -> Input')
    axes_violin[0,1].set_title('Input -> Output')
    axes_violin[1,0].set_title('Output -> Input')
    axes_violin[1,1].set_title('Output -> Output')
    
    fig_heatmap.tight_layout()
    with PdfPages(f'plots/{plot_save_dir}/fixed_weight_reparam_weights.pdf') as pdf:
        pdf.savefig(fig_heatmap)
        print(f'Figure saved at plots/{plot_save_dir}/fixed_weight_reparam_weights.pdf')


    fig_violin.tight_layout()
    with PdfPages(f'plots/{plot_save_dir}/fixed_weight_reparam_weights_violin.pdf') as pdf:
        pdf.savefig(fig_violin)
        print(f'Figure saved at plots/{plot_save_dir}/fixed_weight_reparam_weights_violin.pdf')

def run_plot_reparam_lrs(all_model_dpca_stim_in, all_model_dpca_choice_out, 
                         all_model_kappa_rec_intra, all_model_kappa_inter_ff, all_model_kappa_inter_fb, args):
    fig_heatmap, axes_heatmap = plt.subplots(1, 1, figsize=(10, 10))
    fig_violin, axes_violin = plt.subplots(2, 1, figsize=(8, 10))

    dpca_dict = [[all_model_dpca_stim_in, all_model_dpca_choice_out]]
    kappa_dict = [
        [[kappa_inter_ff[0] for kappa_inter_ff in all_model_kappa_inter_ff],
         [kappa_inter_fb[0] for kappa_inter_fb in all_model_kappa_inter_fb]],
    ]

    plot_reparam_lrs(dpca_dict, kappa_dict, n_components_for_dpca, axes_heatmap, axes_violin)

    total_components = np.sum(list(n_components_for_dpca.values()))
    for j in range(2):
        block_boundaries = np.cumsum(list(n_components_for_dpca.values()))[:-1]
        for i in block_boundaries:
            axes_heatmap.axvline(x=total_components*j+i,color='k',linewidth=0.2)
            axes_heatmap.axhline(y=total_components*j+i,color='k',linewidth=0.2)
    axes_heatmap.axvline(x=total_components,color='k',linewidth=0.2)
    axes_heatmap.axhline(y=total_components,color='k',linewidth=0.2)
    
    tick_locs = np.cumsum([0, *n_components_for_dpca.values()])[:-1]+np.array(list(n_components_for_dpca.values()))//2
    axes_heatmap.set_xticks(np.concatenate([tick_locs, total_components+tick_locs]), 
                            [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$']*2, size=14, rotation=0)
    axes_heatmap.set_yticks(np.concatenate([tick_locs, total_components+tick_locs]), 
                            [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$']*2, size=16)

    axes_violin[0].set_title('Input -> Output')
    axes_violin[1].set_title('Output -> Input')

    fig_heatmap.tight_layout()
    with PdfPages(f'plots/{plot_save_dir}/learning_rates_reparam_heatmap.pdf') as pdf:
        pdf.savefig(fig_heatmap)
        print(f'Figure saved at plots/{plot_save_dir}/learning_rates_reparam_heatmap.pdf')
    
    fig_violin.tight_layout()
    with PdfPages(f'plots/{plot_save_dir}/learning_rates_reparam_violin.pdf') as pdf:
        pdf.savefig(fig_violin)
        print(f'Figure saved at plots/{plot_save_dir}/learning_rates_reparam_violin.pdf')

    

def run_plot_connectivity_by_clusters(all_model_rec, all_model_clusters_pre, all_model_clusters_post, label, axes):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    plot_connectivity_by_clusters([rec_intra[0] for rec_intra in all_model_rec_intra], 
                                all_model_selectivity_clusters_in, 
                                all_model_selectivity_clusters_in, 
                                "Stimulus input", axes[0])
    plot_connectivity_by_clusters([rec_intra[1] for rec_intra in all_model_rec_intra], 
                                all_model_selectivity_clusters_out, 
                                all_model_selectivity_clusters_out, 
                                "Choice output", axes[1])
    plot_connectivity_by_clusters([rec_intra[1] for rec_intra in all_model_rec_intra], 
                                all_model_selectivity_clusters_ch,
                                all_model_selectivity_clusters_ch,
                                "Choice feedback", axes[2])

    fig.tight_layout()



if __name__ == '__main__':
    exp_dir = '/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/Attn-WM-RNN/exp/'
    plot_save_dir = 'fp_approx'
    os.chdir('/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/Attn-WM-RNN')

    # model_array_dir = [f'two_losses_{i}' for i in range(1,9)]

    model_array_dir = [f'test{i}' for i in range(1,9)]

    f = open(os.path.join(exp_dir, model_array_dir[0], 'args.json'), 'r')
    args = json.load(f)
    print('loaded args')
    # load model
    # experiment timeline [0.75 fixation, 2.5 stimulus, 0.5 action presentation, 1.0 reward presentation]
    # 2021 paper          [0.5          , 0.7         , 0.3                    , 0.2                   ]
    # here                [0.4          , 0.8         , 0.6                    , 0.02                  ]

    exp_times = {
        'fixation': 0.4,
        'stimulus_presentation': 0.8,
        'choice_presentation': 0.6,
        'total_time': 1.8,
        'dt': args['dt']}

    task_mdprl = MDPRL(exp_times, args['input_type'], args['decision_space'])

    input_size = {
        'feat': args['stim_dim']*args['stim_val'],
        'feat+obj': args['stim_dim']*args['stim_val']+args['stim_val']**args['stim_dim'], 
        'feat+conj+obj': args['stim_dim']*args['stim_val']+args['stim_dim']*args['stim_val']*args['stim_val']+args['stim_val']**args['stim_dim'],
    }[args['input_type']]
    output_size = args['stim_val']**args['stim_dim']

    input_config = {
        'stim': (input_size, [0]),
        # 'action': (output_size, [1]),
    }

    output_config = {
        'action': (output_size, [1]),
        'chosen_obj': (output_size, [0]),
    }

    args['num_options'] = 2 if 'double' in args['task_type'] else 1

    num_options = 1 if args['task_type']=='value' else 2
    if args['decision_space']=='action':
        output_size = num_options
    elif args['decision_space']=='good':
        output_size = args['stim_val']**args['stim_dim']
    else:
        raise ValueError('Invalid decision space')
    args['output_size'] = output_size


    model_specs = {'input_config': input_config, 'hidden_size': args['hidden_size'], 'output_config': output_config,
                    'plastic': args['plas_type']=='all', 'activation': args['activ_func'],
                    'dt_x': args['dt'], 'dt_w': exp_times['total_time'],  'tau_x': args['tau_x'], 'tau_w': args['tau_w'], 
                    'e_prop': args['e_prop'], 'init_spectral': args['init_spectral'], 'balance_ei': args['balance_ei'],
                    'sigma_rec': args['sigma_rec'], 'sigma_in': args['sigma_in'], 'sigma_w': args['sigma_w'], 
                    'num_areas': args['num_areas'],
                    'inter_regional_sparsity': (1, 1), 'inter_regional_gain': (1, 1)}

    global E_SIZE, I_SIZE, NUM_AREAS, input_weight_inds, output_weight_inds
    E_SIZE = round(args['hidden_size']*args['e_prop'])
    I_SIZE = round(args['hidden_size']*(1-args['e_prop']))
    NUM_AREAS = args['num_areas']
    input_weight_inds = list(range(E_SIZE)) + list(range(E_SIZE*args['num_areas'], E_SIZE*args['num_areas']+I_SIZE))
    output_weight_inds = list(range(E_SIZE, 2*E_SIZE)) + list(range(E_SIZE*args['num_areas']+I_SIZE, E_SIZE*args['num_areas']+2*I_SIZE))

    all_models = []
    for model_dir in model_array_dir:
        model = HierarchicalPlasticRNN(**model_specs)
        state_dict = torch.load(os.path.join(exp_dir, model_dir, 'checkpoint.pth.tar'), 
                                map_location=torch.device('cpu'))['model_state_dict']
        print(model.load_state_dict(state_dict))
        all_models.append(model)
        print(f'model at {model_dir} loaded successfully')

    args['plot_save_dir'] = plot_save_dir
    
    '''plot connections and learning rates'''
    plot_all_connectivities(all_models, args)
    plot_wh_spectrum(all_models, args)

    '''get weights and learning rates by area'''
    all_model_rec_intra, all_model_rec_inter_ff, all_model_rec_inter_fb = get_weights_by_area(all_models, args)
    all_model_kappa_rec_intra, all_model_kappa_inter_ff, all_model_kappa_inter_fb = get_lrs_by_area(all_models, args)

    # plot_weights_lrs_by_area(all_model_rec_intra, all_model_rec_inter_ff, all_model_rec_inter_fb, 
    #                          all_model_kappa_inter_ff, all_model_kappa_inter_fb, args)

    '''get dpca results'''
    global n_components_for_dpca, dim_labels
    n_components_for_dpca = {'s':2, 'p':2, 'c':2, 'sc':4, 'sp':4, 'pc':4, 'spc': 8}
    dim_labels = [r'$F_1$', r'$F_2$', r'$F_3$', r'$C_1$', r'$C_2$', r'$C_3$', r'$O$']
    all_model_dpca_stim_in, all_model_dpca_choice_out, all_model_dpca_stim_out = \
        get_all_dpca_results(all_models, task_mdprl, n_components_for_dpca, args)

    '''plot psth geometry'''
    # run_plot_psth_geometry(all_model_dpca_stim_in, all_model_dpca_choice_out, args)

    '''plot weight exp vars'''
    # run_plot_weight_exp_vars(all_model_dpca_stim_in, all_model_dpca_choice_out, args)

    '''plot output choice overlap'''
    # _ = run_plot_output_choice_overlap(all_model_dpca_stim_in, all_model_dpca_stim_out, args)
    # all_model_dpca_stim_in['pre_ortho_encoding_axes'] = all_model_dpca_stim_in['encoding_axes']
    # all_model_dpca_stim_in['encoding_axes'] = ortho_stim_input_axes
    # all_model_dpca_choice_in['pre_ortho_encoding_axes'] = all_model_dpca_choice_in['encoding_axes']
    # all_model_dpca_choice_in['encoding_axes'] = ortho_choice_input_axes         

    '''plot overlap between dpca axes'''
    # run_plot_dpca_axes_overlap(all_model_dpca_stim_in, all_model_dpca_choice_out, args)

    '''plot selectivity clusters'''
    # all_model_selectivity_clusters_stim, all_model_selectivity_clusters_choice = \
    #         run_plot_selectivity_clusters(all_model_dpca_stim_in, all_model_dpca_choice_out, args)
    
    '''plot reparameterized weights'''
    # run_plot_reparam_weights(all_model_dpca_stim_in, all_model_dpca_choice_out, 
    #                          all_model_rec_intra, all_model_rec_inter_ff, all_model_rec_inter_fb, args)
    
    '''plot reparameterized learning rates'''
    # run_plot_reparam_lrs(all_model_dpca_stim_in, all_model_dpca_choice_out, 
    #                      all_model_kappa_rec_intra, all_model_kappa_inter_ff, all_model_kappa_inter_fb, args)

    '''plot connectivity by clusters'''
    # run_plot_connectivity_by_clusters(all_model_rec_intra, all_model_selectivity_clusters_in, all_model_selectivity_clusters_out, args)


    '''plot learning rates by clusters'''
    # run_plot_learning_rates_by_clusters(all_model_kappa_rec_intra, all_model_selectivity_clusters_in, all_model_selectivity_clusters_out, args)

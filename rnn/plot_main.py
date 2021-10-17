from collections import defaultdict
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from torch.serialization import save
from utils import load_checkpoint
from models import SimpleRNN
from task import MDPRL
import torch
import os
import json
import math
import numpy as np

def plot_mean_and_std(ax, m, sd):
    ax.plot(m)
    ax.fill_between(range(len(m)), m-sd, m+sd, alpha=0.1)

def plot_connectivity(x2hw, h2hw, hb, h2ow):
    maxmax = abs(max([x2hw.max().item(), h2hw.max().item(), hb.max().item(), h2ow.max().item()]))
    minmin = abs(min([x2hw.min().item(), h2hw.min().item(), hb.min().item(), h2ow.min().item()]))
    vbound = max([maxmax, minmin])
    fig, axes = plt.subplots(2, 3)
    axes[0, 2].imshow(h2hw, cmap='bwr', vmin=-vbound, vmax=vbound)
    axes[0, 1].imshow(hb.unsqueeze(1), cmap='bwr', vmin=-vbound, vmax=vbound)
    axes[0, 0].imshow(h2ow.T, cmap='bwr', vmin=-vbound, vmax=vbound)
    axes[1, 0].imshow(x2hw, cmap='bwr', vmin=-vbound, vmax=vbound)
    fig.colorbar()
    plt.tight_layout()
    plt.savefig('plots/connectivity')

def plot_learning_curve(lm, lsd):
    fig, ax = plt.subplots()
    plot_mean_and_std(ax, lm, lsd)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Loss')
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
    fig, ax = plt.subplots()
    mean_attns = attns.mean(1)
    im = ax.imshow(mean_attns.t(), vmax=0.3, aspect='auto', interpolation='nearest')
    fig.colorbar(im)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Attention Distribution')
    plt.tight_layout()
    plt.savefig('plots/attn_bars')

def run_model(args, model, task_mdprl):
    model.eval()
    losses = []
    all_saved_states = defaultdict(list)
    with torch.no_grad():
        for i in range(10):
            print(i)
            DA_s, ch_s, pop_s, index_s, output_mask = task_mdprl.generateinputfromexp(args['batch_size'], 10)
            total_input = pop_s
            output, hs, saved_states = model(total_input, DA_s, save_attn=True)
            for k, v in saved_states.items():
                all_saved_states[k].append(v)
            all_saved_states['hs'].append(hs)
            
            output = output.reshape(args['stim_val']**args['stim_dim']*10, output_mask.shape[1], args['batch_size']) # trial X T X batch size
            loss = (output[:, output_mask.squeeze()==1]-ch_s[:, output_mask.squeeze()==1].squeeze(-1)).pow(2).mean(1) # trial X batch size
            losses.append(loss)
        
        losses_means = torch.cat(losses, dim=1).mean(1) # loss per trial
        losses_stds = torch.cat(losses, dim=1).std(1) # loss per trial
        for k, v in saved_states.items():
            all_saved_states[k] = torch.cat(v, dim=1)
        return losses_means, losses_stds, all_saved_states


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
            'sigma_rec': args['sigma_rec'], 'sigma_in': args['sigma_in'], 'sigma_w': args['sigma_w'], 'rwd_input': args['rwd_input'],
            'input_unit_group': input_unit_group, 'sep_lr_in': args['sep_lr_in'], 'sep_lr_rec': args['sep_lr_rec']}
    model = SimpleRNN(**model_specs)
    state_dict = torch.load(os.path.join(plot_args.exp_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print('loaded model')

    losses_means, losses_stds, all_saved_states = run_model(args, model, task_mdprl)
    print('simulation complete')
    
    # load metrics
    metrics = json.load(open(os.path.join(plot_args.exp_dir, 'metrics.json'), 'r'))

    if plot_args.connectivity:
        plot_connectivity(state_dict['x2h.weight'], state_dict['h2h.weight'], state_dict['h2h.bias'], state_dict['h2o.weight'])
    if plot_args.learning_curve:
        plot_learning_curve(losses_means, losses_stds)
    if plot_args.attn_entropy:
        plot_attn_entropy(all_saved_states['attns'])
    if plot_args.attn_distribution:
        plot_attn_distribution(all_saved_states['attns'])

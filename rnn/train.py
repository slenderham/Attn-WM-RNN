import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
    save_list_to_fs,
    save_checkpoint,
    load_checkpoint,
    load_list_from_fs
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from models import LeakyRNN
from task import MDPRL

from matplotlib import pyplot as plt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training Iterations')
    parser.add_argument('--hidden_size', type=int, default=100, help='Size of recurrent layer')
    parser.add_argument('--stim_dim', type=int, default=3, choices=[2, 3], help='Number of features')
    parser.add_argument('--stim_val', type=int, default=3, help='Possible values of features')
    parser.add_argument('--N_s', type=int, default=10, help='Number of times to repeat the entire stim set')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.01, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.0, help='Std for weight noise')
    parser.add_argument('--init_spectral', type=float, default=1, help='Initial spectral radius for the recurrent weights')
    parser.add_argument('--tau_x', type=float, default=0.1, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=20, help='Time constant for weight modification')
    parser.add_argument('--kappa_w', type=float, default=0.1, help='Learning rate for output weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.0, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--plas_type', type=str, choices=['all', 'half', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--input_type', type=str, choices=['feat', 'feat+obj', 'feat+conj+obj'], default='feat', help='Input coding')
    parser.add_argument('--add_attn', action='store_true', help='Whether to add attention')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'retanh', 'sigmoid'], 
                        default='retanh', help='Activation function for recurrent units')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load the trained model')
    parser.add_argument('--truncate', action='store_true', help='Truncate gradient for neuronal state (not weight) between trials')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    # TODO: add all plasticity
    if args.plas_type=='half':
        raise NotImplementedError

    assert args.l1w==0 and args.l2w==0, \
        "Weight regularization not implemented due to unknown interaction with plasticity"

    print(f"Parameters saved to {os.path.join(args.exp_dir, 'args.json')}")
    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))

    exp_times = {
        'start_time': -0.25,
        'end_time': 0.75,
        'stim_onset': 0.0,
        'stim_end': 0.6,
        'rwd_onset': 0.5,
        'rwd_end': 0.6,
        'choice_onset': 0.35,
        'choice_end': 0.5}
    exp_times['dt'] = args.dt
    log_interval = 1
    grad_accumulation_step = 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    task_mdprl = MDPRL(exp_times, args.N_s, args.input_type)

    input_size = {
        'feat': args.stim_dim*args.stim_val,
        'feat+conj': args.stim_dim*args.stim_val+args.stim_dim*args.stim_val*args.stim_val+args.stim_val**args.stim_dim,
        'feat+conj+obj': args.stim_dim*args.stim_val+args.stim_val**args.stim_dim
    }[args.input_type]

    assert args.input_type=='feat', "Only support feature input for now, since only feature-based attn is supported"
    attn_group_size = [args.stim_val]*args.stim_dim
    
    model = LeakyRNN(input_size=input_size, hidden_size=args.hidden_size, output_size=1, 
                plastic=args.plas_type=='all', attention=args.add_attn, activation=args.activ_func,
                dt=args.dt, tau_x=args.tau_x, tau_w=args.tau_w, kappa_w=args.kappa_w, c_plasticity=np.array([0,0,1,0,0,1]), attn_group_size=attn_group_size,
                e_prop=args.e_prop, sigma_rec=args.sigma_rec, sigma_in=args.sigma_in, sigma_w=args.sigma_w, 
                truncate_iter=1+2*int(1/exp_times['dt']) if args.truncate else None, init_spectral=args.init_spectral)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.load_checkpoint:
        load_checkpoint(model, device, folder=args.exp_dir, filename='checkpoint.pth.tar')

    def train(iters):
        losses = []
        model.train()
        pbar = tqdm(total=iters)
        optimizer.zero_grad()
        for batch_idx in range(iters):
            DA_s, ch_s, pop_s, _, output_mask = task_mdprl.generateinput(args.batch_size)
            output, hs = model(pop_s, DA_s)
            loss = (output.reshape(args.stim_val**args.stim_dim*args.N_s, output_mask.shape[1], args.batch_size, 1)*output_mask.unsqueeze(-1)-ch_s).pow(2).mean()/output_mask.mean() \
                    + args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
            loss.backward()

            if (batch_idx+1) % grad_accumulation_step==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx+1) % log_interval == 0:
                if torch.isnan(loss):
                    quit()
                losses.append(loss.item())
                save_checkpoint(model.state_dict(), folder=args.exp_dir, filename='checkpoint.pth.tar')
                save_list_to_fs(losses, os.path.join(args.exp_dir, 'metrics.txt'))
                pbar.set_description('Iteration {} Loss: {:.6f}'.format(
                    batch_idx, loss.item()))
                pbar.refresh()
            pbar.update()
        pbar.close()
        return loss.item()

    final_training_loss = train(args.iters)
    if args.save_checkpoint:
        save_checkpoint(model.state_dict(), folder=args.exp_dir, filename='checkpoint.pth.tar')
    print('====> DONE')
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
    save_checkpoint
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from models import LeakyRNN
from task import MDPRL

exp_times = {
    'start_time': -0.5,
    'end_time': 1.5,
    'stim_onset': 0.0,
    'stim_end': 1.2,
    'rwd_onset': 1.0,
    'rwd_end': 1.2,
    'choice_onset': 0.7,
    'choice_end': 1.0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training Iterations')
    parser.add_argument('--hidden_size', type=int, default=100, help='Size of recurrent layer')
    parser.add_argument('--stim_dim', type=int, default=3, choices=[2, 3], help='Number of features')
    parser.add_argument('--stim_val', type=int, default=3, help='Possible values of features')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.01, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.0, help='Std for weight noise')
    parser.add_argument('--tau_x', type=float, default=100, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=100, help='Time constant for weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.01, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--plas_type', type=str, choices=['all', 'half', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--input_type', type=str, choices=['feat', 'feat+obj', 'feat+conj+obj'], default='feat', help='Input coding')
    parser.add_argument('--add_attn', action='store_true', help='Whether to add attention')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'retanh', 'sigmoid'], 
                        default='relu', help='Activation function for recurrent units')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    # TODO: add all plasticity
    if args.plas_type=='half':
        raise NotImplementedError

    print(f"Parameters saved to {os.path.join(args.exp_dir, 'args.json')}")
    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))

    exp_times['dt'] = args.dt
    N_s = 10
    log_interval = 100

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')


    task_mdprl = MDPRL(exp_times, 10, 'all')

    input_size = {
        'feat': args.stim_dim*args.stim_val,
        'feat+conj': args.stim_dim*args.stim_val+args.stim_dim*args.stim_val*args.stim_val+args.stim_val**args.stim_dim,
        'feat+conj+obj': args.stim_dim*args.stim_val+args.stim_val**args.stim_dim
    }[args.input_type]

    if args.add_attn:
        assert args.input_type=='feat', "Only support feature-based attention for now"
        attn_group_size = [args.stim_val]*args.stim_dim
    
    model = LeakyRNN(input_size=input_size, hidden_size=args.hidden_size, output_size=1, 
                plastic=args.plas_type=='all', attention=args.add_attn, activation=args.activ_func,
                dt=args.dt, tau_x=args.tau_x, tau_w=args.tau_w, c_plasticity=None, attn_group_size=attn_group_size,
                e_prop=args.e_prop, sigma_rec=args.sigma_rec, sigma_in=args.sigma_in, sigma_w=args.sigma_w)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    def train(iters):
        model.train()

        pbar = tqdm(total=iters)
        for batch_idx in range(iters):
            DA_s, ch_s, pop_s, _ = task_mdprl.generateinput(args.batch_size)
            output, hs = model(pop_s, DA_s)
            loss = (output-ch_s).pow(2).mean() + args.l2r*hs.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                pbar.set_description('Iteration {} Loss: {:.6f}'.format(
                    batch_idx, loss.item()))
                pbar.refresh()
        
        return loss.item()

    final_training_loss = train(args.iters)
    if args.save_checkpoint:
        save_checkpoint(model.state_dict(), folder=args.exp_dir, filename='checkpoint_finetuned.pth.tar')
    print('====> DONE')
    


    

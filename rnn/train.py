from argparse import Action
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score
import math

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

from models import SimpleRNN
from task import MDPRL

from matplotlib import pyplot as plt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training iterations')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--hidden_size', type=int, default=150, help='Size of recurrent layer')
    parser.add_argument('--stim_dim', type=int, default=3, choices=[2, 3], help='Number of features')
    parser.add_argument('--stim_val', type=int, default=3, help='Possible values of features')
    parser.add_argument('--N_s', type=int, default=6, help='Number of times to repeat the entire stim set')
    parser.add_argument('--test_N_s', type=int, default=10, help='Number of times to repeat the entire stim set during eval')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Steps of gradient accumulation.')
    parser.add_argument('--eval_samples', type=int, default=60, help='Number of samples to use for evaluation.')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.01, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.0, help='Std for weight noise')
    parser.add_argument('--init_spectral', type=float, default=1, help='Initial spectral radius for the recurrent weights')
    parser.add_argument('--balance_ei', action='store_true', help='Make mean of E and I recurrent weights equal')
    parser.add_argument('--tau_x', type=float, default=0.1, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=600, help='Time constant for weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.0, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--beta_v', type=float, default=0.5, help='Weight for value estimation loss')
    parser.add_argument('--beta_entropy', type=float, default=0.01, help='Weight for entropy regularization')
    parser.add_argument('--plas_type', type=str, choices=['all', 'half', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--input_type', type=str, choices=['feat', 'feat+obj', 'feat+conj+obj'], default='feat', help='Input coding')
    parser.add_argument('--attn_type', type=str, choices=['none', 'bias', 'weight', 'sample'], 
                        default='weight', help='Type of attn. None, additive feedback, multiplicative weighing, gumbel-max sample')
    parser.add_argument('--sep_lr', action='store_true', help='Use different lr between diff type of units')
    parser.add_argument('--plastic_feedback', action='store_true', help='Plastic feedback weights')
    parser.add_argument('--task_type', type=str, choices=['value', 'off_policy', 'on_policy'],
                        help='Learn reward prob or RL. On policy if decision determines. On policy if decision determines rwd. Off policy if rwd sampled from random policy.')
    parser.add_argument('--rwd_input', action='store_true', help='Whether to use reward as input')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'retanh', 'sigmoid'], 
                        default='retanh', help='Activation function for recurrent units')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load the trained model')
    parser.add_argument('--truncate', action='store_true', help='Truncate gradient for neuronal state (not weight) between trials')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    # TODO: add all plasticity
    if args.plas_type=='half':
        raise NotImplementedError
    if args.task_type=='on_policy':
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
        'choice_end': 0.5,
        'total_time': 1}
    exp_times['dt'] = args.dt
    log_interval = 1

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    task_mdprl = MDPRL(exp_times, args.input_type, args.task_type)

    input_size = {
        'feat': args.stim_dim*args.stim_val,
        'feat+obj': args.stim_dim*args.stim_val+args.stim_val**args.stim_dim, 
        'feat+conj+obj': args.stim_dim*args.stim_val+args.stim_dim*args.stim_val*args.stim_val+args.stim_val**args.stim_dim,
    }[args.input_type]

    if args.rwd_input:
        input_size += 2

    input_unit_group = {
        'feat': [args.stim_dim*args.stim_val], 
        'feat+obj': [args.stim_dim*args.stim_val, args.stim_val**args.stim_dim], 
        'feat+conj+obj': [args.stim_dim*args.stim_val, args.stim_dim*args.stim_val*args.stim_val, args.stim_val**args.stim_dim]
    }[args.input_type]

    if args.attn_type!='none':
        if args.input_type=='feat':
            attn_group_size = [args.stim_val]*args.stim_dim
        elif args.input_type=='feat+obj':
            attn_group_size = [args.stim_val]*args.stim_dim + [args.stim_val**args.stim_dim]
        elif args.input_type=='feat+conj+obj':
            attn_group_size = [args.stim_val]*args.stim_dim + [args.stim_val*args.stim_val]*args.stim_dim + [args.stim_val**args.stim_dim]
    else:
        attn_group_size = [input_size]
    
    if not args.rwd_input:
        assert(sum(attn_group_size)==input_size)
    else:
        assert(sum(attn_group_size)==input_size-2)

    model_specs = {'input_size': input_size, 'hidden_size': args.hidden_size, 'output_size': 1 if args.task_type=='value' else 2, 
                   'plastic': args.plas_type=='all', 'attention_type': args.attn_type, 'activation': args.activ_func,
                   'dt': args.dt, 'tau_x': args.tau_x, 'tau_w': args.tau_w, 'attn_group_size': attn_group_size,
                   'c_plasticity': None, 'e_prop': args.e_prop, 'init_spectral': args.init_spectral, 'balance_ei': args.balance_ei,
                   'sigma_rec': args.sigma_rec, 'sigma_in': args.sigma_in, 'sigma_w': args.sigma_w, 'rwd_input': args.rwd_input,
                   'input_unit_group': input_unit_group, 'sep_lr': args.sep_lr, 'plastic_feedback': args.plastic_feedback,
                   'value_est': 'policy' in args.task_type}
    
    model = SimpleRNN(**model_specs)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(model)
    for n, p in model.named_parameters():
        print(n, p.numel())
    print(optimizer)

    if args.load_checkpoint:
        load_checkpoint(model, device, folder=args.exp_dir, filename='checkpoint.pth.tar')
        print('Model loaded successfully')

    def train(iters):
        model.train()
        pbar = tqdm(total=iters)
        optimizer.zero_grad()
        for batch_idx in range(iters):
            DA_s, ch_s, pop_s, index_s, prob_s, output_mask = task_mdprl.generateinput(args.batch_size, args.N_s)
            output, hs, _ = model(pop_s, DA_s)
            if args.task_type=='value':
                loss = ((output.reshape(args.stim_val**args.stim_dim*args.N_s, output_mask.shape[1], args.batch_size, 1)-ch_s)*output_mask.unsqueeze(-1)).pow(2).mean()
            elif args.task_type=='off_policy':
                log_p, value = output
                conj_mask = output_mask['target'].squeeze()==1
                log_p = log_p.reshape(args.stim_val**args.stim_dim*args.N_s, output_mask['target'].shape[1], args.batch_size, 2)
                log_p = log_p[:, conj_mask][:, -1]
                value = value.reshape(args.stim_val**args.stim_dim*args.N_s, output_mask['target'].shape[1], args.batch_size)
                value = value[:, conj_mask]
                # m = torch.distributions.categorical.Categorical(logits=log_p)
                # action = m.sample().reshape(args.stim_val**args.stim_dim*args.N_s, conj_mask.sum().int(), args.batch_size)
                # rwd_go = (torch.rand_like(prob_s)<prob_s).reshape(args.stim_val**args.stim_dim*args.N_s, 1, args.batch_size).int()
                # rwd = ((rwd_go==action).float())
                # advantage = (rwd-value).detach()
                # value_loss = (rwd-value).pow(2)
                # entropy_loss = m.entropy()
                # # advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                # loss = - (m.log_prob(action)*advantage).mean() + args.beta_v*value_loss.mean() - args.beta_entropy*entropy_loss.mean()
                loss = F.cross_entropy(log_p, (prob_s>0.5).int().squeeze(-1))
            
            loss += args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
            (loss/args.grad_accumulation_steps).backward()
            if (batch_idx+1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx+1) % log_interval == 0:
                if torch.isnan(loss):
                    quit()
                pbar.set_description('Iteration {} Loss: {:.6f}'.format(
                    batch_idx, loss.item()))
                pbar.refresh()
                
            pbar.update()
        pbar.close()
        return loss.item()

    def eval(epoch):
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(args.eval_samples):
                DA_s, ch_s, pop_s, index_s, prob_s, output_mask = task_mdprl.generateinputfromexp(1, args.test_N_s)
                output, hs, _ = model(pop_s, DA_s)
                if args.task_type=='value':
                    output = output.reshape(args.stim_val**args.stim_dim*args.test_N_s, output_mask.shape[1], 1) # trial X T X batch size
                    loss = (output[:, output_mask.squeeze()==1]-ch_s[:, output_mask.squeeze()==1].squeeze(-1)).pow(2).mean(1) # trial X batch size
                else:
                    log_p, _ = output
                    log_p = log_p.reshape(args.stim_val**args.stim_dim*args.test_N_s, output_mask['target'].shape[1], 1, 2)
                    log_p = log_p[:, output_mask['target'].squeeze()==1][:, -1]
                    action = torch.argmax(log_p, dim=-1)
                    rwd_go = (prob_s>0.5).reshape(args.stim_val**args.stim_dim*args.test_N_s, 1).int()
                    loss = (rwd_go==action).float().mean(1)
                losses.append(loss)
            losses_means = torch.cat(losses, dim=1).mean(1) # loss per trial
            losses_stds = torch.cat(losses, dim=1).std(1) # loss per trial
            print('====> Epoch {} Eval Loss: {:.4f}'.format(epoch, losses_means[-args.stim_val**args.stim_dim:].mean()))
            model.print_kappa()
            return losses_means, losses_stds

    metrics = defaultdict(list)
    for i in range(args.epochs):
        training_loss = train(args.iters)
        eval_loss_means, eval_loss_stds = eval(i)
        metrics['eval_losses_mean'].append(eval_loss_means.tolist())
        metrics['eval_losses_std'].append(eval_loss_means.tolist())
        
        save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))
        if args.save_checkpoint:
            save_checkpoint(model.state_dict(), folder=args.exp_dir, filename='checkpoint.pth.tar')
    
    print('====> DONE')
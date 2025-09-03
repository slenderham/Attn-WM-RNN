import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import time

from models import HierarchicalPlasticRNN
from task import MDPRL
from utils import (AverageMeter, load_checkpoint, load_list_from_fs,
                   save_checkpoint, save_defaultdict_to_fs, save_list_to_fs)

import wandb

def train(model, iters):
    model.train()
    optimizer.zero_grad()
    pbar = tqdm(total=iters)
    total_acc = 0
    total_loss = 0
    for batch_idx in range(iters):
        pop_s, rwd_s, target_s, index_s, prob_s, _ = task_mdprl.generateinput(
            batch_size=args.batch_size, N_s=args.N_s, num_choices=num_options)    
        index_s = index_s.to(device)
        prob_s = prob_s.to(device)
        rwd_s = rwd_s.to(device)
        pop_s = pop_s.to(device)
        target_s = target_s.to(device)
        
        loss = 0
        hidden = None
        w_hidden = None

        for i in range(len(pop_s)):
            ''' first phase, give nothing '''
            all_x = {
                'stim': torch.zeros_like(pop_s[i]),
                # 'action': torch.zeros(args.batch_size, output_size, device=device),
            }
            _, hidden, w_hidden, hs = model(all_x, steps=task_mdprl.T_fixation, 
                                            neumann_order=args.neumann_order,
                                            hidden=hidden, w_hidden=w_hidden, 
                                            DAs=None)
            loss += args.l2r*hs.pow(2).mean()/2

            ''' second phase, give stimuli and no feedback '''
            all_x = {
                'stim': pop_s[i],
                # 'action': torch.zeros(args.batch_size, output_size, device=device),
            }
            output, hidden, w_hidden, hs = model(all_x, steps=task_mdprl.T_stim, 
                                                neumann_order=args.neumann_order,
                                                hidden=hidden, w_hidden=w_hidden, 
                                                DAs=None)
            loss += args.l2r*hs.pow(2).mean()/2

            ''' use output to calculate action, reward, and record loss function '''
            if args.task_type=='on_policy_double':
                if args.decision_space=='action':
                    action_loc = torch.argmax(output['action'], dim=-1)
                    action_stim = index_s[i, action_loc]
                    rwd = rwd_s[i][torch.arange(args.batch_size), action_loc]
                    action = action_loc
                elif args.decision_space=='good':
                    action_loc = torch.argmax(output['action'][:,index_s[i]], dim=-1)
                    action_stim = index_s[i, action_loc] # (batch size)
                    rwd = rwd_s[i][torch.arange(args.batch_size), action_loc]
                    action = action_stim
                    # assert(action.shape==(args.batch_size,))
                    # assert(target.shape==(output.shape[0],args.batch_size))
                    # assert(rwd.shape==(args.batch_size,))
                action_logits = output['action'].flatten(end_dim=-2)
                target = target_s[i].flatten()
                loss += F.cross_entropy(action_logits, target)
                total_loss += F.cross_entropy(action_logits.detach(), target).detach().item()/len(pop_s)
                total_acc += (action==target).float().item()/len(pop_s)    

                chosen_obj = output['chosen_obj'].flatten(end_dim=-2) # (batch size, output_size)
                loss += F.cross_entropy(chosen_obj, action_stim.flatten())
                total_loss += F.cross_entropy(chosen_obj.detach(), action_stim.flatten()).detach().item()/len(pop_s)
                
            elif args.task_type == 'value':
                raise NotImplementedError
            
            if args.task_type=='on_policy_double':
                '''third phase, give stimuli and choice, and update weights'''
                DAs = (2*rwd.float()-1)
                _, hidden, w_hidden, _ = model(None, steps=0, 
                                                neumann_order=args.neumann_order,
                                                hidden=hidden, w_hidden=w_hidden, 
                                                DAs=DAs)

                loss += args.l2w*w_hidden.pow(2).sum(dim=(-2, -1)).mean()
                if args.num_areas>1:
                    loss += args.l1w*(model.mask_rec_inter*w_hidden).abs().sum(dim=(-2,-1)).mean()

            elif args.task_type == 'value':
                raise NotImplementedError

        # add weight decay for static weights
        loss /= len(pop_s)

        for input_w in model.rnn.x2h.values():
            loss += args.l2w*(input_w.effective_weight().pow(2).sum())
            loss += args.l1w*(input_w.effective_weight().abs().sum())
        for output_w in model.h2o.values():
            loss += args.l2w*(output_w.effective_weight().pow(2).sum())
            loss += args.l1w*(output_w.effective_weight().abs().sum())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            model.rnn.h2h.weight.data.clamp_(-model.plasticity.weight_bound, model.plasticity.weight_bound)

        if (batch_idx+1) % log_interval == 0:
            if torch.isnan(loss):
                quit()
            pbar.set_description('Iteration {} Loss: {:.4f} Acc: {:.4f}'.format(batch_idx+1, total_loss/(batch_idx+1), total_acc/(batch_idx+1)))
            # pbar.refresh()
            pbar.update(log_interval)
    pbar.close()
    total_acc = total_acc/iters
    total_loss = total_loss/iters
    print(f'Training Loss: {total_loss:.4f}, Training Acc: {total_acc:.4f}')
    wandb.log({'Training Loss': total_loss, 'Training Acc': total_acc})
    return total_loss


def eval(model, epoch):
    model.eval()
    losses_means_by_gen = []
    losses_stds_by_gen = []
    with torch.no_grad():
        for curr_gen_level in task_mdprl.gen_levels:
            losses = []
            for _ in range(args.eval_samples):
                pop_s, rwd_s, target_s, index_s, prob_s, _ = task_mdprl.generateinput(
                        batch_size=args.batch_size, N_s=args.test_N_s, num_choices=num_options, gen_level=curr_gen_level)
                index_s = index_s.to(device)
                prob_s = prob_s.to(device)
                rwd_s = rwd_s.to(device)
                pop_s = pop_s.to(device)
                target_s = target_s.to(device)
                
                loss = []
                hidden = None
                w_hidden = None
                for i in range(len(pop_s)):
                    # first phase, give nothing
                    all_x = {
                        'stim': torch.zeros_like(pop_s[i]),
                        # 'action': torch.zeros(args.batch_size, output_size, device=device),
                    }
                    _, hidden, w_hidden, _ = model(all_x, steps=task_mdprl.T_fixation, neumann_order = 0,
                                                hidden=hidden, w_hidden=w_hidden, DAs=None)
                    # second phase, give stimuli and no feedback
                    all_x = {
                        'stim': pop_s[i],
                        # 'action': torch.zeros(args.batch_size, output_size, device=device),
                    }
                    output, hidden, w_hidden, _ = model(all_x, steps=task_mdprl.T_stim, neumann_order = 0,
                                                hidden=hidden, w_hidden=w_hidden, DAs=None)
                    if args.task_type=='on_policy_double':
                        # use output to calculate action, reward, and record loss function
                        if args.decision_space=='action':
                            action_loc = torch.argmax(output['action'], dim=-1)
                            action_stim = index_s[i, action_loc]
                            rwd = rwd_s[i][torch.arange(args.batch_size), action_loc]
                            action = action_loc
                            loss.append((action==target_s[i]).float())
                        elif args.decision_space=='good':
                            action_loc = torch.argmax(output['action'][:,index_s[i]], dim=-1)
                            action_stim = index_s[i, action_loc] # (batch size)
                            # assert(action.shape==(args.batch_size,))
                            rwd = rwd_s[i][range(args.batch_size), action_loc]
                            # assert(rwd.shape==(args.batch_size,))
                            action = action_stim
                            loss.append((action==target_s[i]).float())
                    elif args.task_type == 'value':
                        raise NotImplementedError
                    
                    if args.task_type=='on_policy_double':
                        '''third phase, give stimuli and choice, and update weights'''
                        DAs = (2*rwd.float()-1)
                        _, hidden, w_hidden, _ = model(None, steps=0, neumann_order = 0,
                                                    hidden=hidden, w_hidden=w_hidden, DAs=DAs)

                    elif args.task_type == 'value':
                        raise NotImplementedError

                loss = torch.stack(loss, dim=0)
                losses.append(loss)
            losses_means = torch.cat(losses, dim=1).mean(1) # loss per trial
            losses_stds = torch.cat(losses, dim=1).std(1) # loss per trial
            losses_means_by_gen.append(losses_means.tolist())
            losses_stds_by_gen.append(losses_stds.tolist())
            print('====> Epoch {} Gen Level: {} Eval Loss: {:.4f}'.format(epoch+1, curr_gen_level, losses_means.mean()))
            wandb.log({'Eval loss': losses_means.mean()})
        return losses_means_by_gen, losses_stds_by_gen

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training iterations')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--hidden_size', type=int, default=80, help='Size of recurrent layer')
    parser.add_argument('--num_areas', type=int, default=2, help='Number of recurrent areas')
    parser.add_argument('--stim_dim', type=int, default=3, choices=[2, 3], help='Number of features')
    parser.add_argument('--stim_val', type=int, default=3, help='Possible values of features')
    parser.add_argument('--N_s', type=int, default=135, help='Number of times to repeat the entire stim set')
    parser.add_argument('--test_N_s', type=int, default=432, help='Number of times to repeat the entire stim set during eval')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--neumann_order', type=int, default=10, help='Timestep for unrolling for neumann approximation')
    parser.add_argument('--eval_samples', type=int, default=10, help='Number of samples to use for evaluation.')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.1, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.01, help='Std for weight noise')
    parser.add_argument('--init_spectral', type=float, default=1.0, help='Initial spectral radius for the recurrent weights')
    parser.add_argument('--balance_ei', action='store_true', help='Make mean of E and I recurrent weights equal')
    parser.add_argument('--tau_x', type=float, default=0.1, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=180, help='Time constant for weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.0, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--plas_type', type=str, choices=['all', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--input_type', type=str, choices=['feat', 'feat+conj+obj', 'feat+obj'], default='feat+conj+obj', help='Input coding')
    parser.add_argument('--decision_space', type=str, choices=['good', 'action'], help='Supervise with good-based or action-based decision making')
    parser.add_argument('--task_type', type=str, choices=['value', 'on_policy_double'],
                        help='Learn reward prob or RL. On policy if decision determines. On policy if decision determines rwd. Off policy if rwd sampled from random policy.')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'softplus2', 'retanh', 'sigmoid'], 
                        default='retanh', help='Activation function for recurrent units')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load the trained model')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    if args.save_checkpoint:
        print(f"Parameters saved to {os.path.join(args.exp_dir, 'args.json')}")
        save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    
    # experiment timeline [0.75 fixation, 2.5 stimulus, 0.5 action presentation]
    # 2021 paper          [0.5          , 0.7         , 0.3                    ]
    # here                [0.4          , 0.8         , 0.02                   ]
    
    exp_times = {
        'fixation': 0.4,
        'stimulus_presentation': 0.8,
        'choice_presentation': args.dt,
        'total_time': 1.8,
        'dt': args.dt}
    log_interval = 100
    # args.exp_times = exp_times

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    task_mdprl = MDPRL(exp_times, args.input_type, args.decision_space)    

    stim_size = {
        'feat': args.stim_dim*args.stim_val,
        'feat+obj': args.stim_dim*args.stim_val+args.stim_val**args.stim_dim, 
        'feat+conj+obj': args.stim_dim*args.stim_val+args.stim_dim*args.stim_val*args.stim_val+args.stim_val**args.stim_dim,
    }[args.input_type]
    
    if args.decision_space=='good':
        input_size = stim_size
        output_size = args.stim_val**args.stim_dim
    elif args.decision_space=='action':
        input_size = stim_size*2
        output_size = 2
    else:
        raise ValueError

    input_config = {
        'stim': (input_size, [0]),
        # 'action': (output_size, [0]),
    }

    output_config = {
        'action': (output_size, [1]),
        'chosen_obj': (args.stim_val**args.stim_dim, [0]),
    }

    num_options = 1 if args.task_type=='value' else 2

    model_specs = {'input_config': input_config, 'hidden_size': args.hidden_size, 'output_config': output_config,
                   'num_areas': args.num_areas, 'plastic': args.plas_type=='all', 'activation': args.activ_func,
                   'dt_x': args.dt, 'dt_w': exp_times['total_time'], 'tau_x': args.tau_x, 'tau_w': args.tau_w, 
                   'e_prop': args.e_prop, 'init_spectral': args.init_spectral, 'balance_ei': args.balance_ei,
                   'sigma_rec': args.sigma_rec, 'sigma_in': args.sigma_in, 'sigma_w': args.sigma_w, 
                   'inter_regional_sparsity': (1, 1), 'inter_regional_gain': (1, 1)}
    
    model = HierarchicalPlasticRNN(**model_specs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    print(model)
    for n, p in model.named_parameters():
        print(n, p.numel())
    print(optimizer)

    if args.load_checkpoint:
        load_checkpoint(model, optimizer, device, folder=args.exp_dir, filename='checkpoint.pth.tar')
        print('Model loaded successfully')

    wandb.init(project="attn-rnn_3x3x3", config=args)

    metrics = defaultdict(list)
    best_eval_loss = 0
    for i in range(args.epochs):
        training_loss = train(model, args.iters)
        eval_loss_means, eval_loss_stds = eval(model, i)
        # lr_scheduler.step()
        metrics['eval_losses_mean'].append(eval_loss_means)
        metrics['eval_losses_std'].append(eval_loss_stds)
        metrics = dict(metrics)
        # save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))
        if args.save_checkpoint:
            if sum([np.mean(v) for v in eval_loss_means]) > best_eval_loss:
                is_best_epoch = True
                best_eval_loss = sum([np.mean(v).item() for v in eval_loss_means])
                metrics['best_epoch'] = i
                metrics['best_eval_loss'] = best_eval_loss
            else:
                is_best_epoch = False
            save_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            is_best=is_best_epoch, folder=args.exp_dir, filename='checkpoint.pth.tar', 
                            best_filename='checkpoint_best.pth.tar')
    
    print('====> DONE')
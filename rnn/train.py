from asyncio import log
import math
import os
from argparse import Action
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm import tqdm

from models import MultiChoiceRNN, SimpleRNN, MultiAreaRNN, HierarchicalRNN
from task import MDPRL, RolloutBuffer
from utils import (AverageMeter, load_checkpoint, load_list_from_fs,
                   save_checkpoint, save_defaultdict_to_fs, save_list_to_fs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training iterations')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--hidden_size', type=int, default=150, help='Size of recurrent layer')
    parser.add_argument('--num_areas', type=int, default=6, help='Number of recurrent areas')
    parser.add_argument('--stim_dim', type=int, default=3, choices=[2, 3], help='Number of features')
    parser.add_argument('--stim_val', type=int, default=3, help='Possible values of features')
    parser.add_argument('--N_s', type=int, default=6, help='Number of times to repeat the entire stim set')
    parser.add_argument('--N_stim_train', type=int, default=27, help='Number of stimuli to train the network on each episode')
    parser.add_argument('--test_N_s', type=int, default=10, help='Number of times to repeat the entire stim set during eval')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Steps of gradient accumulation.')
    parser.add_argument('--eval_samples', type=int, default=21, help='Number of samples to use for evaluation.')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.1, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.0, help='Std for weight noise')
    parser.add_argument('--init_spectral', type=float, default=None, help='Initial spectral radius for the recurrent weights')
    parser.add_argument('--balance_ei', action='store_true', help='Make mean of E and I recurrent weights equal')
    parser.add_argument('--tau_x', type=float, default=0.1, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=600, help='Time constant for weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.0, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--attn_ent_reg', type=float, default=0.0, help='Entropy regularization for attention')
    parser.add_argument('--beta_v', type=float, default=0.5, help='Weight for value estimation loss')
    parser.add_argument('--beta_entropy', type=float, default=0.01, help='Weight for entropy regularization')
    parser.add_argument('--beta_attn_chosen', type=float, default=0.0, help='Weight for forcing attention on chosen stimuli')
    parser.add_argument('--plas_type', type=str, choices=['all', 'half', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--plas_rule', type=str, choices=['add', 'mult'], default='add', help='Plasticity rule')
    parser.add_argument('--input_type', type=str, choices=['feat', 'feat+obj', 'feat+conj+obj'], default='feat', help='Input coding')
    parser.add_argument('--attn_type', type=str, choices=['none', 'bias', 'weight', 'sample'], 
                        default='weight', help='Type of attn. None, additive feedback, multiplicative weighing, gumbel-max sample')
    parser.add_argument('--spatial_attn', action='store_true', help='Whether to add trainable spatial attenion')
    parser.add_argument('--feature_attn', action='store_true', help='Whether to add trainable feature attenion')
    parser.add_argument('--spatial_attn_agg', type=str, choices=['concat', 'avg'], default='avg', help='How to aggregate input objects after spatial attn.')
    parser.add_argument('--sep_lr', action='store_true', help='Use different lr between diff type of units')
    parser.add_argument('--plastic_feedback', action='store_true', help='Plastic feedback weights')
    parser.add_argument('--task_type', type=str, choices=['value', 'off_policy_single', 'on_policy_double'],
                        help='Learn reward prob or RL. On policy if decision determines. On policy if decision determines rwd. Off policy if rwd sampled from random policy.')
    parser.add_argument('--rwd_input', action='store_true', help='Whether to use reward as input')
    parser.add_argument('--action_input', action='store_true', help='Whether to use action as input')
    parser.add_argument('--rpe', action='store_true', help='Whether to use reward prediction error as modulation')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'retanh', 'sigmoid'], 
                        default='retanh', help='Activation function for recurrent units')
    parser.add_argument('--structured_conn', action='store_true', help='Whether to use restricted connectivity')
    parser.add_argument('--reversal_every', type=int, default=100000, help='Number of trials between reversals')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load the trained model')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    # TODO: add all plasticity
    if args.plas_type=='half':
        raise NotImplementedError
    if args.task_type=='on_policy':
        raise NotImplementedError

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
        'total_time': 1,
        'dt': args.dt}
    log_interval = 1

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    task_mdprl = MDPRL(exp_times, args.input_type)

    input_size = {
        'feat': args.stim_dim*args.stim_val,
        'feat+obj': args.stim_dim*args.stim_val+args.stim_val**args.stim_dim, 
        'feat+conj+obj': args.stim_dim*args.stim_val+args.stim_dim*args.stim_val*args.stim_val+args.stim_val**args.stim_dim,
    }[args.input_type]

    input_unit_group = {
        'feat': [args.stim_dim*args.stim_val], 
        'feat+obj': [args.stim_dim*args.stim_val, args.stim_val**args.stim_dim], 
        'feat+conj+obj': [args.stim_dim*args.stim_val, args.stim_dim*args.stim_val*args.stim_val, args.stim_val**args.stim_dim]
    }[args.input_type]

    if args.attn_type!='none':
        if args.input_type=='feat':
            channel_group_size = [args.stim_val]*args.stim_dim
        elif args.input_type=='feat+obj':
            channel_group_size = [args.stim_val]*args.stim_dim
        elif args.input_type=='feat+conj+obj':
            channel_group_size = [args.stim_val]*args.stim_dim + [args.stim_val*args.stim_val]*args.stim_dim + [args.stim_val**args.stim_dim]
    else:
        channel_group_size = [input_size]

    output_size = 1 if args.task_type=='value' else 2

    model_specs = {'input_size': input_size, 'hidden_size': args.hidden_size, 'output_size': output_size, 
                   'plastic': args.plas_type=='all', 'attention_type': args.attn_type, 'activation': args.activ_func,
                   'dt': args.dt, 'tau_x': args.tau_x, 'tau_w': args.tau_w, 'channel_group_size': channel_group_size,
                   'e_prop': args.e_prop, 'init_spectral': args.init_spectral, 'balance_ei': args.balance_ei,
                   'sigma_rec': args.sigma_rec, 'sigma_in': args.sigma_in, 'sigma_w': args.sigma_w, 
                   'rwd_input': args.rwd_input, 'action_input': args.action_input, 'plas_rule': args.plas_rule,
                   'input_unit_group': input_unit_group, 'sep_lr': args.sep_lr, 'plastic_feedback': args.plastic_feedback,
                   'value_est': 'policy' in args.task_type, 'num_choices': 2 if 'double' in args.task_type else 1,
                   'structured_conn': args.structured_conn, 'spatial_attn_agg': args.spatial_attn_agg}
    
    # if args.num_areas>1:
    #     model_specs['num_areas'] = args.num_areas
    #     model_specs['loc_input'] = args.spatial_attn
    #     model_specs['inter_regional_sparsity'] = 0.1
    #     model_specs['add_sa'] = args.spatial_attn
    #     model_specs['add_fa'] = args.feature_attn
    #     model = MultiAreaRNN(**model_specs)
    if 'double' in args.task_type:
        # model = MultiChoiceRNN(**model_specs)
        model_specs['num_areas'] = args.num_areas
        model_specs['inter_regional_sparsity'] = (0.5, 0.5)
        model = HierarchicalRNN(**model_specs)
    else:
        model = SimpleRNN(**model_specs)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(model)
    for n, p in model.named_parameters():
        print(n, p.numel())
    print(optimizer)

    if args.load_checkpoint:
        load_checkpoint(model, optimizer, device, folder=args.exp_dir, filename='checkpoint.pth.tar')
        print('Model loaded successfully')

    def train(iters):
        model.train()
        pbar = tqdm(total=iters)
        optimizer.zero_grad()
        total_acc = 0
        for batch_idx in range(iters):
            curr_gen_lvl = np.random.choice(task_mdprl.gen_levels)
            DA_s, ch_s, pop_s, index_s, prob_s, output_mask = task_mdprl.generateinput(
                batch_size=args.batch_size, N_s=args.N_s, num_choices=output_size, gen_level=curr_gen_lvl, subsample_stims=args.N_stim_train)
            if args.task_type == 'value':
                output, hs, _, _ = model(pop_s, DA_s)
                loss = ((output.reshape(args.stim_val**args.stim_dim*args.N_s, output_mask.shape[1], args.batch_size, 1)-ch_s)*output_mask.unsqueeze(-1)).pow(2).mean() \
                        + args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
            elif args.task_type=='on_policy_double':
                loss = 0
                hidden = None
                # plt.imshow(model.rnn.x2h.effective_weight().detach())
                # plt.colorbar()
                # plt.show()           
                # plt.imshow(model.rnn.aux2h.effective_weight().detach())
                # plt.colorbar()
                # plt.show()                
                # plt.imshow(model.rnn.h2h.effective_weight().detach(), vmax=0.1, vmin=-0.1, cmap='seismic')
                # plt.colorbar()
                # plt.show()
                # plt.imshow(model.h2o.effective_weight().detach())
                # plt.colorbar()
                # plt.show() 
                # plt.imshow(model.h2sa.effective_weight().detach())
                # plt.colorbar()
                # plt.show()   
                # plt.imshow(model.h2fa.effective_weight().detach())
                # plt.colorbar()
                # plt.show()  
                # v = torch.linalg.eigvals(model.rnn.h2h.effective_weight()).detach()
                # plt.scatter(v.real,v.imag)
                # plt.show()
                for i in range(len(pop_s['pre_choice'])):
                    if (i+1)%args.reversal_every==0:
                        probs = task_mdprl.reversal(probs)
                    # first phase, give stimuli and no feedback
                    output, hs, hidden, ss = model(pop_s['pre_choice'][i], hidden=hidden, 
                                                  Rs=0*DA_s['pre_choice'], Vs=None,
                                                  acts=torch.zeros(args.batch_size, output_size)*DA_s['pre_choice'],
                                                  save_weights=True)
                    # plt.plot(output.squeeze().detach())
                    # plt.show()
                    # use output to calculate action, reward, and record loss function
                    action = torch.argmax(output[-1], -1)
                    output = output.reshape(output_mask['target'].shape[0], args.batch_size, output_size)
                    output = (output*output_mask['target'].reshape(-1, 1, 1)).flatten(1)
                    target = torch.argmax(prob_s[i], -1).reshape(1, args.batch_size)
                    target = torch.repeat_interleave(target, output_mask['target'].shape[0], dim=0).flatten()
                    loss += F.multi_margin_loss(output, target, p=2)
                    rwd = (torch.rand(args.batch_size)<prob_s[i][range(args.batch_size), action]).float()
                    total_acc += (action==torch.argmax(prob_s[i], -1)).float().item()
                    # plt.imshow(hs.squeeze().detach().t(), aspect='auto')
                    # plt.imshow(ss['wxs'][-1,0].detach(), aspect='auto', interpolation='nearest')
                    # plt.colorbar()
                    # plt.ylabel('choice prob')
                    # print(hs.shape)
                    # plt.plot((hs[1:]-hs[:-1]).pow(2).sum([-1,-2]).detach())
                    # plt.show()

                    reg = args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
                    if args.plastic_feedback:
                        reg += args.l2w*(ss['wxs'].pow(2).sum(dim=(-2,-1)).mean()\
                                        +ss['whs'].pow(2).sum(dim=(-2,-1)).mean()\
                                        +ss['wfbs'].pow(2).sum(dim=(-2,-1)).mean())
                    else:
                        reg += args.l2w*(ss['wxs'].pow(2).sum(dim=(-2,-1)).mean()\
                                        +ss['whs'].pow(2).sum(dim=(-2,-1)).mean())
                    if args.num_areas>1:
                        reg += args.l1w*(model.conn_masks['rec_inter']*ss['whs'].abs()).sum(dim=(-2,-1)).mean()
                    # if args.attn_type=='weight':
                    #     if args.num_areas>1:
                    #         if args.spatial_attn:
                    #             sas = F.softmax(ss['sas'], -1).mean([0,1])
                    #             reg += args.attn_ent_reg*(sas*torch.log(sas)).sum()
                    #         if args.feature_attn:
                    #             fas = F.softmax(ss['fas'], -1).mean([0,1])
                    #             reg += args.attn_ent_reg*(fas*torch.log(fas)).sum()

                    loss += reg*len(pop_s['pre_choice'][i])/(len(pop_s['pre_choice'][i])+len(pop_s['post_choice'][i]))
                    
                    # use the action (optional) and reward as feedback
                    pop_post = pop_s['post_choice'][i]
                    action_enc = torch.eye(output_size)[action]
                    if args.num_areas==1 or not args.spatial_attn:
                        pop_post = pop_post*action_enc.reshape(1,1,2,1)
                    action_enc = action_enc*DA_s['post_choice']
                    R = (2*rwd-1)*DA_s['post_choice']
                    _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=R, Vs=None, acts=action_enc, save_weights=True)

                    # plt.imshow(hs.squeeze().detach().t(), aspect='auto')
                    # plt.colorbar()                    
                    # plt.plot(ss['sas'].squeeze().detach().softmax(-1))
                    # plt.plot(ss['fas'].squeeze().detach().softmax(-1))
                    # plt.show()
                    # plt.plot((hs[1:]-hs[:-1]).pow(2).sum([-1,-2]).detach())
                    # plt.show()

                    reg = args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
                    if args.plastic_feedback:
                        reg += args.l2w*(ss['wxs'].pow(2).sum(dim=(-2, -1)).mean()\
                                        +ss['whs'].pow(2).sum(dim=(-2, -1)).mean()\
                                        +ss['wfbs'].pow(2).sum(dim=(-2, -1)).mean())
                    else:
                        reg += args.l2w*(ss['wxs'].pow(2).sum(dim=(-2, -1)).mean()\
                                        +ss['whs'].pow(2).sum(dim=(-2, -1)).mean())
                    if args.num_areas>1:
                        reg += args.l1w*(model.conn_masks['rec_inter']*ss['whs'].abs()).sum(dim=(-2,-1)).mean()
                    # if args.attn_type=='weight':
                    #     if args.num_areas>1:
                    #         if args.spatial_attn:
                    #             sas = F.softmax(ss['sas'], -1).mean([0,1])
                    #             reg += args.attn_ent_reg*(sas*torch.log(sas)).sum()
                    #             reshaped_action = action.reshape(1, args.batch_size).repeat(ss['sas'].shape[0], 1).flatten()
                    #             reshaped_gaze = ss['sas'][(DA_s['post_choice']>0.5).squeeze()]
                    #             reshaped_action = action.reshape(1, args.batch_size).repeat(reshaped_gaze.shape[0], 1).flatten()
                    #             reg += args.beta_attn_chosen*F.cross_entropy(input=(reshaped_gaze).flatten(-2), target=reshaped_action)
                    #         if args.feature_attn:
                    #             fas = F.softmax(ss['fas'], -1).mean([0,1])
                    #             reg += args.attn_ent_reg*(fas*torch.log(fas)).sum()

                    loss += reg*len(pop_s['post_choice'][i])/(len(pop_s['pre_choice'][i])+len(pop_s['post_choice'][i]))
            
            (loss/args.grad_accumulation_steps/len(pop_s['pre_choice'])).backward()
            if (batch_idx+1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                # for n, p in model.named_parameters():
                    # print(n, p.grad.pow(2).sum())
                # plt.imshow(model.rnn.h2h.weight.grad)
                # plt.colorbar()
                # plt.show()
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx+1) % log_interval == 0:
                if torch.isnan(loss):
                    quit()
                pbar.set_description('Iteration {} Loss: {:.4f}'.format(
                    batch_idx, loss.item() if 'policy' not in args.task_type else total_acc/len(pop_s['pre_choice'])/(batch_idx+1)))
                pbar.refresh()
                
            pbar.update()
        pbar.close()
        total_acc = total_acc/len(pop_s['pre_choice'])/iters
        print(f'Training Acc: {total_acc}')
        return loss.item()

    def eval(epoch):
        model.eval()
        losses_means_by_gen = {}
        losses_stds_by_gen = {}
        with torch.no_grad():
            for curr_gen_level in task_mdprl.gen_levels:
                losses = []
                for batch_idx in range(args.eval_samples):
                    DA_s, ch_s, pop_s, index_s, prob_s, output_mask = task_mdprl.generateinput(
                        args.batch_size, args.test_N_s, num_choices=output_size, gen_level=curr_gen_level)
                    if args.task_type == 'value':
                        output, hs, _ = model(pop_s, DA_s)
                        output = output.reshape(args.stim_val**args.stim_dim*args.test_N_s, output_mask.shape[1], 1) # trial X T X batch size
                        loss = (output[:, output_mask.squeeze()==1]-ch_s[:, output_mask.squeeze()==1].squeeze(-1)).pow(2).mean(1) # trial X batch size
                    else:
                        loss = []
                        hidden = None
                        for i in range(len(pop_s['pre_choice'])):
                            # first phase, give stimuli and no feedback
                            output, hs, hidden, _ = model(pop_s['pre_choice'][i], hidden=hidden, 
                                                        Rs=0*DA_s['pre_choice'], Vs=None,
                                                        acts=torch.zeros(1, output_size)*DA_s['pre_choice'])
                            # use output to calculate action, reward, and record loss function
                            action = torch.argmax(output[-1], -1)
                            rwd = (torch.rand(args.batch_size)<prob_s[i][range(args.batch_size), action]).float()
                            loss.append((action==torch.argmax(prob_s[i], -1)).float())
                            # use the action (optional) and reward as feedback
                            pop_post = pop_s['post_choice'][i]
                            action_enc = torch.eye(output_size)[action]
                            if args.num_areas==1:
                                pop_post = pop_post*action_enc.reshape(1,1,2,1)                            
                            action_enc = action_enc*DA_s['post_choice']
                            R = (2*rwd-1)*DA_s['post_choice']
                            V = None
                            _, hs, hidden, _ = model(pop_post, hidden=hidden, Rs=R, Vs=V, acts=action_enc)
                        loss = torch.stack(loss, dim=0)
                    losses.append(loss)
                losses_means = torch.cat(losses, dim=1).mean(1) # loss per trial
                losses_stds = torch.cat(losses, dim=1).std(1) # loss per trial
                losses_means_by_gen[curr_gen_level] = losses_means.tolist()
                losses_stds_by_gen[curr_gen_level] = losses_stds.tolist()
                print('====> Epoch {} Gen Level: {} Eval Loss: {:.4f}'.format(epoch, curr_gen_level, losses_means.mean()))
            return losses_means_by_gen, losses_stds_by_gen

    metrics = defaultdict(list)
    best_eval_loss = 0
    for i in range(args.epochs):
        training_loss = train(args.iters)
        eval_loss_means, eval_loss_stds = eval(i)
        metrics['eval_losses_mean'].append(eval_loss_means)
        metrics['eval_losses_std'].append(eval_loss_stds)
        metrics = dict(metrics)
        save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))
        if args.save_checkpoint:
            if sum([np.mean(v) for v in eval_loss_means.values()]) > best_eval_loss:
                is_best_epoch = True
                best_eval_loss = sum([np.mean(v).item() for v in eval_loss_means.values()])
                metrics['best_epoch'] = i
                metrics['best_eval_loss'] = best_eval_loss
            else:
                is_best_epoch = False
            save_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            is_best=is_best_epoch, folder=args.exp_dir, filename='checkpoint.pth.tar', 
                            best_filename='checkpoint_best.pth.tar')
    
    print('====> DONE')

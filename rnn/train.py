import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm import tqdm

from models import HierarchicalRNN
from task import MDPRL
from utils import (AverageMeter, load_checkpoint, load_list_from_fs,
                   save_checkpoint, save_defaultdict_to_fs, save_list_to_fs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training iterations')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--hidden_size', type=int, default=80, help='Size of recurrent layer')
    parser.add_argument('--num_areas', type=int, default=6, help='Number of recurrent areas')
    parser.add_argument('--stim_dim', type=int, default=3, choices=[2, 3], help='Number of features')
    parser.add_argument('--stim_val', type=int, default=3, help='Possible values of features')
    parser.add_argument('--N_s_min', type=int, default=135, help='Number of times to repeat the entire stim set')
    parser.add_argument('--N_s_max', type=int, default=135, help='Number of times to repeat the entire stim set')
    parser.add_argument('--N_stim_train', type=int, default=27, help='Number of stimuli to train the network on each episode')
    parser.add_argument('--test_N_s', type=int, default=432, help='Number of times to repeat the entire stim set during eval')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Steps of gradient accumulation.')
    parser.add_argument('--eval_samples', type=int, default=21, help='Number of samples to use for evaluation.')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.05, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.0001, help='Std for weight noise')
    parser.add_argument('--init_spectral', type=float, default=None, help='Initial spectral radius for the recurrent weights')
    parser.add_argument('--balance_ei', action='store_true', help='Make mean of E and I recurrent weights equal')
    parser.add_argument('--tau_x', type=float, default=0.1, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=100, help='Time constant for weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.0, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--plas_type', type=str, choices=['all', 'half', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--plas_rule', type=str, choices=['add', 'mult'], default='add', help='Plasticity rule')
    parser.add_argument('--input_plas_off', action='store_true', help='Disable input plasticity')
    parser.add_argument('--input_type', type=str, choices=['feat', 'feat+obj', 'feat+conj+obj'], default='feat+conj+obj', help='Input coding')
    parser.add_argument('--decision_space', type=str, choices=['good', 'good_feat', 'good_feat_conj_obj', 'action'], help='Supervise with good-based or action-based decision making')
    parser.add_argument('--sep_lr', action='store_true', help='Use different lr between diff type of units')
    parser.add_argument('--task_type', type=str, choices=['value', 'off_policy_single', 'on_policy_double'],
                        help='Learn reward prob or RL. On policy if decision determines. On policy if decision determines rwd. Off policy if rwd sampled from random policy.')
    parser.add_argument('--rwd_input', action='store_true', help='Whether to use reward as input')
    parser.add_argument('--action_input', action='store_true', help='Whether to use action as input')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'softplus2', 'retanh', 'sigmoid'], 
                        default='retanh', help='Activation function for recurrent units')
    parser.add_argument('--structured_conn', action='store_true', help='Whether to use restricted connectivity')
    parser.add_argument('--reversal_every', type=int, default=100000, help='Number of trials between reversals')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load the trained model')
    parser.add_argument('--debug', action='store_true', help='Debug mode for plotting')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    # TODO: add all plasticity
    if args.plas_type=='half':
        raise NotImplementedError
    if args.task_type=='on_policy':
        raise NotImplementedError

    if args.save_checkpoint:
        print(f"Parameters saved to {os.path.join(args.exp_dir, 'args.json')}")
        save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))

    ITI = 0.25
    choice_start = 0.6
    rwd_start = 0.75
    stim_end = 0.9
    mask_onset = 0.4
    # experiment timeline [0.75 fixation, 2.5 stimulus, 0.5 action presentation, 1.0 reward presentation]
    # 2021 paper          [0.5          , 0.7         , 0.3                    , 0.2                   ]
    # here                [0.25         , 0.6         , 0.15                   , 0.15                  ]
    
    exp_times = {
        'start_time': -ITI,
        'end_time': stim_end,
        'stim_onset': 0.0,
        'stim_end': stim_end,
        'mask_onset': mask_onset,
        'mask_end': choice_start,
        'choice_onset': choice_start,
        'choice_end': stim_end,
        'rwd_onset': rwd_start,
        'rwd_end': stim_end,
        'total_time': ITI+stim_end,
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

    num_options = 1 if args.task_type=='value' else 2
    if args.decision_space=='action':
        output_size = num_options
    elif args.decision_space=='good':
        output_size = args.stim_val**args.stim_dim
    else:
        raise ValueError('Invalid decision space')

    model_specs = {'input_size': input_size, 'hidden_size': args.hidden_size, 'output_size': output_size, 'num_options': num_options,
                   'plastic': args.plas_type=='all', 'activation': args.activ_func,
                   'dt': args.dt, 'tau_x': args.tau_x, 'tau_w': args.tau_w, 
                   'e_prop': args.e_prop, 'init_spectral': args.init_spectral, 'balance_ei': args.balance_ei,
                   'sigma_rec': args.sigma_rec, 'sigma_in': args.sigma_in, 'sigma_w': args.sigma_w, 
                   'rwd_input': args.rwd_input, 'action_input': args.action_input, 'plas_rule': args.plas_rule,
                   'sep_lr': args.sep_lr, 'num_choices': 2 if 'double' in args.task_type else 1,
                   'structured_conn': args.structured_conn, 'num_areas': args.num_areas, 
                   'inter_regional_sparsity': (1, 1), 'inter_regional_gain': (0.5, 0.5),
                   'input_plastic': not args.input_plas_off}
    
    model = HierarchicalRNN(**model_specs)
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
        total_loss = 0
        for batch_idx in range(iters):
            pop_s, target_valid, output_mask, rwd_mask, ch_mask, index_s, prob_s = task_mdprl.generateinput(
                batch_size=args.batch_size, N_s=np.random.randint(args.N_s_min, args.N_s_max+1), num_choices=num_options, subsample_stims=args.N_stim_train)    
            loss = 0
            hidden = None
            if args.debug:
                plt.imshow(model.rnn.x2h.effective_weight().detach())
                plt.colorbar()
                plt.show()
                # plt.imshow(model.rnn.kappa_in.abs().detach())
                # plt.colorbar()
                # plt.show()
                plt.imshow(model.rnn.aux2h.effective_weight().detach())
                plt.colorbar()
                plt.show()                
                wlim = np.percentile(model.rnn.h2h.effective_weight().detach().abs(), 90)
                plt.imshow(model.rnn.h2h.effective_weight().detach(), vmax=wlim, vmin=-wlim, cmap='RdBu_r')
                plt.colorbar()
                plt.show()
                wlim = np.percentile(model.rnn.kappa_rec.abs().detach(), 90)
                plt.imshow(model.rnn.h2h.effective_weight(model.rnn.kappa_rec.abs()).detach().squeeze(), vmax=wlim, vmin=-wlim, cmap='RdBu_r')
                plt.colorbar()
                plt.show()
                v = torch.linalg.eigvals(model.rnn.h2h.effective_weight()).detach()
                plt.scatter(v.real,v.imag)
                plt.show()
            
            for i in range(len(pop_s['pre_choice'])):
                if (i+1)%args.reversal_every==0:
                    probs = task_mdprl.reversal(probs)
                # first phase, give stimuli and no feedback
                output, hs, hidden, ss = model(pop_s['pre_choice'][i], hidden=hidden, 
                                                DAs=torch.zeros(1, args.batch_size, 1)*rwd_mask['pre_choice'],
                                                Rs=torch.zeros(1, args.batch_size, 2)*rwd_mask['pre_choice'],
                                                acts=torch.zeros(1, args.batch_size, output_size)*ch_mask['pre_choice'],
                                                save_weights=True)

                if args.debug:
                    plt.plot(output.softmax(-1)[:,:,index_s[i, torch.sort(prob_s[i])[1]]].squeeze().detach())
                    plt.plot(output.softmax(-1).squeeze().detach(), c='k', alpha=0.1)
                    # plt.plot(ch_s['pre_choice'][i].squeeze())
                    plt.ylim([-0.1, 1.1])
                    plt.show()
    
                # use output to calculate action, reward, and record loss function
                if args.task_type=='on_policy_double':
                    if args.decision_space=='action':
                        # action = torch.argmax(output[-1,:,:], -1) # batch size
                        action = torch.multinomial(output[-1,:,:], 1)
                        rwd = (torch.rand(args.batch_size)<prob_s[i][range(args.batch_size), action]).long()
                    elif args.decision_space=='good':
                        # action_valid = torch.argmax(output[-1,:,index_s[i]], -1) # size = (batch_size)
                        action_valid = torch.multinomial(output[-1,:,index_s[i]].softmax(-1), num_samples=1).squeeze(-1)
                        action = index_s[i, action_valid] # (batch size)
                        assert(action.shape==(args.batch_size,))
                        target = index_s[i, target_valid['pre_choice'][i].long()] # (batch size)
                        assert(target.shape==(output.shape[0],args.batch_size))
                        rwd = (torch.rand(args.batch_size)<prob_s[i][range(args.batch_size), action_valid]).long() #(batch_size)
                        assert(rwd.shape==(args.batch_size,))
                    output = output[output_mask['target'].squeeze()>0.5,:,:].flatten(1)
                    target = target[output_mask['target'].squeeze()>0.5,:].flatten()
                    loss += F.cross_entropy(output, target)
                    total_loss += F.cross_entropy(output, target).detach().item()/len(pop_s['pre_choice'])
                    total_acc += (action_valid==target_valid['pre_choice'][i,-1]).float().item()/len(pop_s['pre_choice'])
                elif args.task_type == 'value':
                    rwd = (torch.rand(args.batch_size)<prob_s[i]).float()
                    output = output.reshape(output_mask['target'].shape[0], args.batch_size, output_size)
                    loss += ((output-target['pre_choice'][i])*output_mask['target'].unsqueeze(-1)).pow(2).mean()/output_mask['target'].float().mean()
                    total_acc += ((output-target['pre_choice'][i])*output_mask['target'].unsqueeze(-1)).pow(2).mean().item()/output_mask['target'].float().mean()
                
                if args.debug:
                    # plt.imshow(model.rnn.h2h.effective_weight(ss['whs'][-1,0]).detach().squeeze(), aspect='auto', interpolation='nearest', cmap='seismic', vmin=-0.1, vmax=0.1)
                    # plt.imshow(hs.squeeze().detach().t(), aspect='auto', cmap='hot', interpolation='nearest')
                    # plt.colorbar()
                    # plt.show()
                    hs_pre = hs
                    # plt.ylabel('choice prob')
                    # print(hs.shape)
                    # plt.plot((hs[1:]-hs[:-1]).pow(2).sum([-1,-2]).detach())

                reg = args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
                reg += args.l2w*(ss['wxs'].pow(2).sum(dim=(-2,-1)).mean()\
                                +ss['whs'].pow(2).sum(dim=(-2,-1)).mean())
                if args.num_areas>1:
                    reg += args.l1w*(model.conn_masks['rec_inter']*ss['whs'].abs()).sum(dim=(-2,-1)).mean()

                loss += reg*len(pop_s['pre_choice'][i])/(len(pop_s['pre_choice'][i])+len(pop_s['post_choice'][i]))
                
                if args.task_type=='on_policy_double':
                    # use the action (optional) and reward as feedback
                    pop_post = pop_s['post_choice'][i]
                    action_enc = torch.eye(output_size)[action]
                    # action_enc = torch.from_numpy(task_mdprl.stim_encoding('all_onehot'))[action,:].float()
                    rwd_enc = torch.eye(2)[rwd]
                    # if args.decision_space=='good':
                    #     action_valid_enc = torch.eye(num_options)[action_valid]
                    #     pop_post = pop_post*action_valid_enc.reshape(1,1,num_options,1)
                    # elif args.decision_space=='action':
                    #     pop_post = pop_post*action_enc.reshape(1,1,output_size,1)
                    action_enc = action_enc*ch_mask['post_choice']
                    assert(action_enc.shape==(pop_post.shape[0],args.batch_size,output_size))
                    rwd_enc = rwd_enc*rwd_mask['post_choice']
                    assert(rwd_enc.shape==(pop_post.shape[0],args.batch_size,2))
                    DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                    assert(DAs.shape==(pop_post.shape[0],args.batch_size,1))
                    _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=action_enc, DAs=DAs, save_weights=True)
                elif args.task_type == 'value':
                    pop_post = pop_s['post_choice'][i]
                    rwd_enc = torch.eye(2)[rwd]
                    DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                    _, hs, hidden, ss = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=None, DAs=DAs, save_weights=True)

                if args.debug:
                    plt.imshow(torch.cat([hs_pre, hs], dim=0).squeeze().detach().t(), aspect='auto', cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    plt.show()
                    v = torch.linalg.eigvals(model.rnn.h2h.effective_weight(hidden[3])).detach()
                    plt.scatter(v.real,v.imag)
                    plt.show()
                # plt.plot(ss['sas'].squeeze().detach().softmax(-1))
                # plt.plot(ss['fas'].squeeze().detach().softmax(-1))
                # plt.show()
                # plt.plot((hs[1:]-hs[:-1]).pow(2).sum([-1,-2]).detach())
                # plt.show()

                reg = args.l2r*hs.pow(2).mean() + args.l1r*hs.abs().mean()
                reg += args.l2w*(ss['wxs'].pow(2).sum(dim=(-2, -1)).mean()\
                                +ss['whs'].pow(2).sum(dim=(-2, -1)).mean())
                if args.num_areas>1:
                    reg += args.l1w*(model.conn_masks['rec_inter']*ss['whs'].abs()).sum(dim=(-2,-1)).mean()

                loss += reg*len(pop_s['post_choice'][i])/(len(pop_s['pre_choice'][i])+len(pop_s['post_choice'][i]))
            
            # add weight decay for static weights
            loss /= len(pop_s['pre_choice'])
            loss += args.l2w*(model.rnn.aux2h.effective_weight().pow(2).sum()+model.h2o.effective_weight().pow(2).sum())

            (loss/args.grad_accumulation_steps).backward()
            if (batch_idx+1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                # for n, p in model.named_parameters():
                #     print(n, p.grad.pow(2).sum())
                # plt.imshow(model.rnn.x2h.weight.grad)
                # plt.colorbar()
                # plt.show()
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx+1) % log_interval == 0:
                if torch.isnan(loss):
                    quit()
                pbar.set_description('Iteration {} Loss: {:.4f}'.format(batch_idx+1, total_loss/(batch_idx+1)))
                pbar.refresh()
                
            pbar.update()
        pbar.close()
        total_acc = total_acc/iters
        total_loss = total_loss/iters
        print(f'Training Loss: {total_loss:.4f}, Training Acc: {total_acc:.4f}')
        return loss.item()

    def eval(epoch):
        model.eval()
        losses_means_by_gen = {}
        losses_stds_by_gen = {}
        with torch.no_grad():
            for curr_gen_level in task_mdprl.gen_levels:
                losses = []
                for batch_idx in range(args.eval_samples):
                    pop_s, target_valid, output_mask, rwd_mask, ch_mask, index_s, prob_s = task_mdprl.generateinput(
                        args.batch_size, args.test_N_s, num_choices=num_options, gen_level=curr_gen_level)
                    loss = []
                    hidden = None
                    for i in range(len(pop_s['pre_choice'])):
                        # first phase, give stimuli and no feedback
                        output, _, hidden, _ = model(pop_s['pre_choice'][i], hidden=hidden, 
                                                        DAs=torch.zeros(1, args.batch_size, 1)*rwd_mask['pre_choice'],
                                                        Rs=torch.zeros(1, args.batch_size, 2)*rwd_mask['pre_choice'],
                                                        acts=torch.zeros(1, args.batch_size, output_size)*ch_mask['pre_choice'],
                                                        save_weights=False)

                        if args.task_type=='on_policy_double':
                            # use output to calculate action, reward, and record loss function
                            if args.decision_space=='action':
                                action = torch.argmax(output[-1,:,:], -1)
                                rwd = (torch.rand(args.batch_size)<prob_s[i][range(args.batch_size), action]).float()
                                loss.append((action==torch.argmax(prob_s[i], -1)).float())
                            elif args.decision_space=='good':
                                # action_valid = torch.argmax(output[-1,:,index_s[i]], -1) # the object that can be chosen (0~1)
                                action_valid = torch.multinomial(output[-1,:,index_s[i]].softmax(-1), num_samples=1).squeeze(-1)
                                action = index_s[i, action_valid] # the object chosen (0~26), but only the valid one
                                rwd = (torch.rand(args.batch_size)<prob_s[i][range(args.batch_size), action_valid]).long() #(batch_size)
                                loss.append((action_valid==target_valid['pre_choice'][i,-1]).float())
                        elif args.task_type == 'value':
                            rwd = (torch.rand(args.batch_size)<prob_s[i]).float()
                            output = output.reshape(output_mask['target'].shape[0], args.batch_size, output_size)
                            loss.append(((output-target_valid['pre_choice'][i])*output_mask['target'].unsqueeze(-1)).pow(2).mean(0)/output_mask['target'].float().mean())
                        
                        if args.task_type=='on_policy_double':
                            # use the action (optional) and reward as feedback
                            pop_post = pop_s['post_choice'][i]
                            action_enc = torch.eye(output_size)[action]
                            # action_enc = torch.from_numpy(task_mdprl.stim_encoding('all_onehot'))[action,:].float()
                            rwd_enc = torch.eye(2)[rwd]
                            # if args.decision_space=='good':
                            #     action_valid_enc = torch.eye(num_options)[action_valid]
                            #     pop_post = pop_post*action_valid_enc.reshape(1,1,num_options,1)
                            # elif args.decision_space=='action':
                            #     pop_post = pop_post*action_enc.reshape(1,1,output_size,1)
                            action_enc = action_enc*ch_mask['post_choice']
                            rwd_enc = rwd_enc*rwd_mask['post_choice']
                            DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                            _, _, hidden, _ = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=action_enc, DAs=DAs, save_weights=False)
                        elif args.task_type == 'value':
                            pop_post = pop_s['post_choice'][i]
                            rwd_enc = torch.eye(2)[rwd]
                            DAs = (2*rwd.float()-1)*rwd_mask['post_choice']
                            _, _, hidden, _ = model(pop_post, hidden=hidden, Rs=rwd_enc, acts=None, DAs=DAs, save_weights=False)
                    loss = torch.stack(loss, dim=0)
                    losses.append(loss)
                losses_means = torch.cat(losses, dim=1).mean(1) # loss per trial
                losses_stds = torch.cat(losses, dim=1).std(1) # loss per trial
                losses_means_by_gen[curr_gen_level] = losses_means.tolist()
                losses_stds_by_gen[curr_gen_level] = losses_stds.tolist()
                print('====> Epoch {} Gen Level: {} Eval Loss: {:.4f}'.format(epoch+1, curr_gen_level, losses_means.mean()))
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
#!/bin/bash

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 100\
        --epochs 10000\
        --eval_samples 5\
        --batch_size 1\
        --hidden_size 80\
        --num_areas 3\
        --N_s 6\
        --N_stim_train 27\
        --input_type feat+conj+obj\
        --attn_type weight\
        --l2r 1e-5\
        --l2w 1e-8\
        --l1w 1e-8\
        --init_spectral 1.0\
        --sigma_rec 0.05\
        --sep_lr\
        --balance_ei\
        --rwd_input\
        --action_input\
        --learning_rate 1e-3\
        --task_type on_policy_double\
        --activ_func retanh\
        --spatial_attn_agg concat\
        --exp_dir exp/dual_hierarchical

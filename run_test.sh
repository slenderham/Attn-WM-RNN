#!/bin/bash

python rnn/train.py --cuda\
        --debug\
        --iters 100\
        --epochs 10000\
        --eval_samples 5\
        --batch_size 1\
        --hidden_size 80\
        --num_areas 3\
        --N_s 135\
        --N_stim_train 27\
        --input_type feat+conj+obj\
        --decision_space good\
        --l2r 1e-4\
        --l2w 1e-8\
        --l1w 1e-8\
        --init_spectral 1.0\
        --sigma_rec 0.1\
        --sep_lr\
        --balance_ei\
        --rwd_input\
        --action_input\
        --input_plas_off\
        --learning_rate 1e-3\
        --task_type on_policy_double\
        --activ_func retanh\
        --exp_dir exp/test
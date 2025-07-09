#!/bin/bash

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 100\
        --epochs 10000\
        --eval_samples 5\
        --batch_size 1\
        --hidden_size 80\
        --num_areas 3\
        --init_spectral 1.0\
        --N_s 6\
        --N_stim_train 27\
        --input_type feat+conj+obj\
        --attn_type weight\
        --l2r 1e-3\
        --l2w 1e-6\
        --l1w 1e-6\
        --sep_lr\
        --balance_ei\
        --rwd_input\
        --input_plas_off\
        --learning_rate 1e-3\
        --task_type value\
        --activ_func retanh\
        --spatial_attn_agg concat\
        --exp_dir exp/hierarchical

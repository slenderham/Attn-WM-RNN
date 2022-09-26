#!/bin/bash

python rnn/train.py --cuda\
        --load_checkpoint\
        --iters 1\
        --epochs 10000\
        --eval_samples 2\
        --batch_size 1\
        --hidden_size 160\
        --num_areas 1\
        --N_s 6\
        --N_stim_train 27\
        --input_type feat+conj+obj\
        --attn_type weight\
        --l2r 0\
        --l2w 0\
        --l1w 0\
        --attn_ent_reg 0.001\
        --sep_lr\
        --balance_ei\
        --rwd_input\
        --rpe\
        --input_plas_off\
        --learning_rate 1e-3\
        --task_type value\
        --spatial_attn_agg concat\
        --activ_func retanh\
        --input_plas_off\
        --exp_dir exp/hierarchical\
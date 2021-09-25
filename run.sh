#!/bin/bash

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 10000\
        --batch_size 1\
        --stim_dim 3\
        --stim_val 3\
        --N_s 6\
        --init_spectral 1\
	--learning_rate 1e-4\
        --add_attn\
        --activ_func retanh\
        --exp_dir exp/test

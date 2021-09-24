#!/bin/bash

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 10000\
        --batch_size 32\
        --stim_dim 3\
        --stim_val 3\
        --add_attn\
        --activ_func relu\
        --exp_dir exp/test
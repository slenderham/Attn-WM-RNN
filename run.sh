#!/bin/bash

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 100\
        --batch_size 1\
        --exp_dir exp/test
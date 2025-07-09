#!/bin/bash

#SBATCH --job-name=rnn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=240:00:00
#SBATCH --mem=32GB
#SBATCH --output=exp/test%a/slurm_%j.txt

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 1000\
        --epochs 28\
        --hidden_size 80\
        --eval_samples 10\
        --num_areas 2\
        --decision_space good\
        --l2r 1e-1\
        --l2w 1e-5\
        --l1w 1e-7\
        --init_spectral 1.\
        --sep_lr\
        --balance_ei\
        --action_input\
        --learning_rate 1e-3\
        --task_type on_policy_double\
        --exp_dir exp/test$SLURM_ARRAY_TASK_ID
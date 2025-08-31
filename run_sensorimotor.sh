#!/bin/bash

#SBATCH --job-name=fp_approx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH --output=exp/test%a/slurm_%j.txt

python rnn/train.py --cuda\
        --save_checkpoint\
        --iters 1000\
        --epochs 40\
        --hidden_size 160\
        --eval_samples 20\
        --num_areas 1\
        --decision_space action\
        --l2r 1e-1\
        --l2w 1e-5\
        --l1w 1e-5\
        --init_spectral 1.0\
        --balance_ei\
        --learning_rate 1e-3\
        --task_type on_policy_double\
        --exp_dir exp/test$SLURM_ARRAY_TASK_ID
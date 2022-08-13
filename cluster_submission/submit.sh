#!/bin/bash

#SBATCH --job-name=rnn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_%j.txt

cd /dartfs-hpc/rc/home/d/f005d7d/attn-rnn/Attn-WM-RNN
./run_good_based.sh

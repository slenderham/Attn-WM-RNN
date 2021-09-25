#!/bin/bash

#SBATCH --job-name=rnn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=08:00:00
#SBATCH --mem=8GB
#SBATCH --mail-type=BEGIN,END,FAIL

cd /dartfs-hpc/rc/home/d/f005d7d/attn-rnn/Attn-WM-RNN
../run.sh
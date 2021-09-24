#!/bin/bash

# Name of the job
#SBATCH --job-name=gpu_job

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:1

# Walltime (job duration)
#SBATCH --time=02:00:00

nvidia-smi
echo $CUDA_VISIBLE_DEVICES
hostname
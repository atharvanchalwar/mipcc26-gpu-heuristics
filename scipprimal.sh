#!/bin/bash
#SBATCH --job-name=mipcc_gpu_batch
#SBATCH --partition=a30_normal_q     # Falcon A30 queue
#SBATCH --gres=gpu:1                 # request 1 A30 GPU
#SBATCH --cpus-per-task=8            # number of CPU cores
#SBATCH --mem=32G                    # memory
#SBATCH --time=24:00:00              # max walltime
#SBATCH --output=logs/sciprun_%j.out
#SBATCH --error=logs/sciprun_%j.err
#SBATCH --account=mofdesign

# -------------------------------
# Environment setup
# -------------------------------
module load Miniconda3
source activate mipcc26   # or whatever env has PySCIPOpt installed

# -------------------------------
# Run SCIP primal batch script
# -------------------------------
python scipprimal.py

#!/bin/bash
#
#SBATCH --job-name=emotion_vit
#SBATCH --output=logs/emotion_vit_%j.out
#SBATCH --error=logs/emotion_vit_%j.err
#
#SBATCH --partition=gpucluster        # GPU partition on your cluster
#SBATCH --gres=gpu:1                  # number of GPUs
#SBATCH --cpus-per-task=4             # CPU cores per task
#SBATCH --time=08:00:00               # max run time (hh:mm:ss)

echo "Starting job $SLURM_JOB_ID on $HOSTNAME"
date

# ---- 1) (Optional) modules ----
# If your cluster uses modules and sets CUDA/Python by default, you can often skip this.
# If things fail, check with `module avail` and add what's needed.
# module purge
# module load cuda/12.4
# module load python/3.10

# ---- 2) Activate your virtualenv ----
# Adjust this path if your venv is elsewhere
source ~/dl_trans/.venv/bin/activate

echo "Python path: $(which python)"
python --version || echo "Python not found"
nvidia-smi || echo "nvidia-smi not available"

# ---- 3) Change to your project directory ----
cd ~/dl_trans

# ---- 4) Run your training script ----
python train_emotion_vit.py

echo "Job finished"
date


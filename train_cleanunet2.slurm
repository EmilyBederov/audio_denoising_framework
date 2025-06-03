#!/bin/bash
#SBATCH --job-name=cleanunet2
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

echo "=== JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Date: $(date)"

# Navigate to your project
cd /home/emilybederov/work/audio_denoising_framework-main

# Activate your conda environment using full path
source /home/emilybederov/miniconda3/etc/profile.d/conda.sh
conda activate denoise_env

echo "=== ENVIRONMENT CHECK ==="
which python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi

echo "=== STARTING TRAINING ==="
python run.py \
    --config configs/configs-cleanunet2/cleanunet2-config.yaml \
    --model cleanunet2 \
    --mode train \
    --batch_size 4

echo "=== TRAINING COMPLETED ==="
echo "Date: $(date)"
echo "Models saved in:"
ls -la outputs/cleanunet2/ || echo "No outputs directory found"

#!/bin/bash
#SBATCH --job-name=cleanunet2_train
#SBATCH --output=logs/train_cleanunet2_%j.out
#SBATCH --error=logs/train_cleanunet2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda environment
source /home/emilybederov/miniconda3/etc/profile.d/conda.sh
conda activate denoise_env

# Navigate to project directory
cd /home/emilybederov/work/audio_denoising_framework-main

# Print environment info
echo "Starting CleanUNet2 training at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU Info:"
nvidia-smi

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Training command
echo "Starting training..."
python run.py \
    --config configs/configs-cleanunet2/cleanunet2-config.yaml \
    --model cleanunet2 \
    --mode train \
    --device cuda

echo "Training completed at: $(date)"

# Check if model was saved
if [ -d "outputs" ]; then
    echo "Training outputs:"
    find outputs -name "*.pth" -o -name "*.pkl" | head -10
else
    echo "No outputs directory found"
fi

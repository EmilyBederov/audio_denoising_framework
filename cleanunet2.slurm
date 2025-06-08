#!/bin/bash
#SBATCH --job-name=cleanunet2_resume_90
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "=== JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Date: $(date)"
start_time=$(date +%s)

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to your project
cd /home/emilybederov/work/audio_denoising_framework-main

# Activate your conda environment using full path
source /home/emilybederov/miniconda3/etc/profile.d/conda.sh
conda activate denoise_env

echo "=== ENVIRONMENT CHECK ==="
which python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
nvidia-smi

echo "=== PROJECT CHECK ==="
echo "Current directory: $(pwd)"
echo "Config file exists: $(test -f configs/configs-cleanunet2/cleanunet2-config.yaml && echo 'YES' || echo 'NO')"
echo "Training CSV exists: $(test -f data/training_pairs_16khz.csv && echo 'YES' || echo 'NO')"
echo "Validation CSV exists: $(test -f data/evaluation_pairs_16khz.csv && echo 'YES' || echo 'NO')"

echo "=== CHECKPOINT VERIFICATION ==="
CHECKPOINT_PATH="outputs/cleanunet2/cleanunet2_epoch_150.pth"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint found: $CHECKPOINT_PATH"
    ls -lh "$CHECKPOINT_PATH"
    echo "Checkpoint size: $(du -h "$CHECKPOINT_PATH" | cut -f1)"
else
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la outputs/cleanunet2/*.pth 2>/dev/null || echo "No .pth files found"
    exit 1
fi

# Set Python path for proper imports
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=== RESUMING CLEANUNET2 TRAINING FROM EPOCH 150 ==="
echo "Resuming from: $CHECKPOINT_PATH"
echo "Training epochs 151-300 (150 remaining epochs)"
echo "Estimated duration: 14-16 hours for remaining epochs"
echo "Expected final model size: ~58M parameters"
echo "Target performance: PESQ > 3.0, STOI > 0.95"

python run.py \
    --config configs/configs-cleanunet2/cleanunet2-config.yaml \
    --model cleanunet2 \
    --mode train \
    --checkpoint "$CHECKPOINT_PATH" \
    --device cuda \
    --batch_size 4

echo "=== TRAINING COMPLETED ==="
echo "Date: $(date)"
echo "Training duration: $(($(date +%s) - start_time)) seconds"

echo "=== FINAL RESULTS ==="
echo "Models saved in:"
ls -la outputs/cleanunet2/ || echo "No outputs directory found"

echo "=== MODEL FILES ==="
find outputs -name "*.pth" -exec ls -lh {} \; 2>/dev/null || echo "No model files found"

echo "=== TRAINING SUMMARY ==="
echo "Started from epoch: 90"
echo "Completed epochs: 91-300"
echo "Total training time: $(($(date +%s) - start_time)) seconds"

echo "=== READY FOR INFERENCE ==="
echo "Test your model with:"
echo "python inference_realtime.py --model outputs/cleanunet2/cleanunet2_best.pth --config configs/configs-cleanunet2/cleanunet2-config.yaml"

echo "=== JOB FINISHED ==="

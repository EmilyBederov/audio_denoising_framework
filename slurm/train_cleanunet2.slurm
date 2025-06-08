#!/bin/bash
#SBATCH --job-name=cleanunet2_final
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "=== JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Date: $(date)"

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

# Set Python path for proper imports
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=== STARTING CLEANUNET2 TRAINING ==="
echo "Training will take ~8-12 hours for 300 epochs"
echo "Expected final model size: ~58M parameters"
echo "Target performance: PESQ > 3.0, STOI > 0.95"

python run.py \
    --config configs/configs-cleanunet2/cleanunet2-config.yaml \
    --model cleanunet2 \
    --mode train \
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

echo "=== READY FOR INFERENCE ==="
echo "Test your model with:"
echo "python test_realtime.py --model outputs/cleanunet2/cleanunet2_best.pth --input test.wav"

echo "=== JOB FINISHED ==="

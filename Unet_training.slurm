#!/bin/bash
#SBATCH --job-name=unet_isolated           # DIFFERENT job name
#SBATCH --output=logs/unet_isolated_%j.out # DIFFERENT log prefix
#SBATCH --error=logs/unet_isolated_%j.err  # DIFFERENT error log
#SBATCH --partition=gpu                    # Adjust to your GPU partition name
#SBATCH --gres=gpu:A100:1                 # Request 1 A100 GPU
#SBATCH --cpus-per-task=8                 # 8 CPU cores
#SBATCH --mem=32G                         # 32GB RAM
#SBATCH --time=24:00:00                   # 24 hours max runtime
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks-per-node=1              # One task per node

# ISOLATION: Exclude nodes that might be running other training
# (Optional - uncomment if you want to avoid specific nodes)
##SBATCH --exclude=node1,node2

# Email notifications (optional - update with your email)
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@domain.com

# Job information
echo "=================================================="
echo " ISOLATED UNET TRAINING JOB"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=================================================="

# Check for conflicts with other jobs
echo " Checking for other running jobs..."
OTHER_JOBS=$(squeue -u $USER -h -o "%j" | grep -v "unet_isolated" | wc -l)
if [ $OTHER_JOBS -gt 0 ]; then
    echo "  Found $OTHER_JOBS other running jobs:"
    squeue -u $USER -o "%.8i %.15j %.8T %.10M %.6D %R"
    echo " UNet will run isolated on this node: $(hostname)"
else
    echo "No other jobs detected - full resources available"
fi

# Print GPU information
echo "  GPU Information:"
nvidia-smi
echo "=================================================="

# Create isolated directories
echo " Setting up isolated directories..."
mkdir -p logs/unet
mkdir -p outputs/unet_isolated      # DIFFERENT output directory
mkdir -p checkpoints/unet_isolated  # DIFFERENT checkpoint directory

# Load conda environment
echo " Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate denoise_env

# Verify environment
echo " Environment check:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=================================================="

# Navigate to project directory
cd ~/work/audio_denoising_framework-main

# Create ISOLATED config file (doesn't modify original)
echo "  Creating isolated UNet config..."
ISOLATED_CONFIG="configs/configs-unet/unet_isolated.yaml"
cp configs/configs-unet/unet.yaml $ISOLATED_CONFIG

# Update paths to be completely separate
sed -i 's|save_path: "outputs/unet"|save_path: "outputs/unet_isolated"|' $ISOLATED_CONFIG
sed -i 's|epochs: 30|epochs: 30  # ISOLATED TRAINING|' $ISOLATED_CONFIG

echo "Isolated config created: $ISOLATED_CONFIG"
echo " Output directory: outputs/unet_isolated/"
echo " Will NOT interfere with other model training!"

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Training command with ISOLATED config
echo " Starting ISOLATED UNet training..."
echo "Command: python run.py --config $ISOLATED_CONFIG --model unet --mode train"
echo "=================================================="

# Run the training with error handling
python run.py --config $ISOLATED_CONFIG --model unet --mode train

# Capture exit code
EXIT_CODE=$?

echo "=================================================="
echo " ISOLATED UNet training completed with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

# Show what was created (without affecting other models)
echo " UNet files created in ISOLATED directories:"
echo "Outputs:"
ls -la outputs/unet_isolated/ 2>/dev/null || echo "  No outputs yet"
echo "Logs:"
ls -la logs/unet_isolated_*.* 2>/dev/null || echo "  Check logs/unet_isolated_${SLURM_JOB_ID}.*"

# Print final GPU memory usage
echo "  Final GPU status:"
nvidia-smi

# Check other jobs are still running (they should be unaffected)
echo " Other jobs status (should be unaffected):"
squeue -u $USER

# If training failed, show last few lines of any error logs
if [ $EXIT_CODE -ne 0 ]; then
    echo " Training failed! Last 20 lines of output:"
    tail -20 logs/unet_isolated_${SLURM_JOB_ID}.out
    echo "=================================================="
else
    echo " UNet training completed successfully!"
    echo " Results in: outputs/unet_isolated/"
fi

echo " Isolated UNet Job ID $SLURM_JOB_ID completed."
echo " Check logs at: logs/unet_isolated_${SLURM_JOB_ID}.out"
echo " No other models were affected!"
echo "=================================================="

exit $EXIT_CODE

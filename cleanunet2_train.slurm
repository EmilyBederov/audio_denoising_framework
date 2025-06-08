#!/bin/bash
#SBATCH --job-name=cleanunet2_resume_epoch90
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_resume_%j.txt
#SBATCH --error=logs/error_resume_%j.txt
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "=== RESUME TRAINING JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
start_time=$(date +%s)

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to your project
cd /home/emilybederov/work/audio_denoising_framework-main

# Activate conda environment
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

echo "=== CHECKPOINT CHECK ==="
CHECKPOINT_PATH=""
if [ -f "outputs/cleanunet2/cleanunet2_epoch_90.pth" ]; then
    CHECKPOINT_PATH="outputs/cleanunet2/cleanunet2_epoch_90.pth"
else
    echo "ERROR: No epoch 90 checkpoint found!"
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT_PATH"
ls -lh "$CHECKPOINT_PATH"

export PYTHONPATH=$PWD:$PYTHONPATH

echo "=== RESUMING CLEANUNET2 TRAINING FROM EPOCH 90 ==="
echo "Training will stop at epoch 300"

# Create the monitor script (fixed)
cat > monitor_training.sh << 'EOF'
#!/bin/bash
LOG_FILE=$1
TARGET_EPOCH=$2
TRAIN_PID=$3

echo "Monitoring training progress..."
echo "Will stop at epoch $TARGET_EPOCH"

while kill -0 $TRAIN_PID 2> /dev/null; do
    if [ -f "$LOG_FILE" ]; then
        if grep -q "Epoch $TARGET_EPOCH/" "$LOG_FILE" 2>/dev/null; then
            echo "$(date): Reached target epoch $TARGET_EPOCH. Waiting for epoch to finish..."
            sleep 300
            echo "Stopping training..."
            kill -TERM $TRAIN_PID
            exit 0
        fi

        current_epoch=$(grep -oE "Epoch [0-9]+/" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+" || echo "0")
        if [ "$current_epoch" -gt 0 ]; then
            echo "$(date): Current progress - Displayed: Epoch $current_epoch/300"
        fi
    fi
    sleep 120
done
echo "Training process ended"
EOF

chmod +x monitor_training.sh

# Start training in background
python run.py \
    --config configs/configs-cleanunet2/cleanunet2-config.yaml \
    --model cleanunet2 \
    --mode train \
    --device cuda \
    --batch_size 4 \
    --checkpoint "$CHECKPOINT_PATH" &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

# Start monitor
./monitor_training.sh "logs/output_resume_${SLURM_JOB_ID}.txt" 300 $TRAIN_PID &
MONITOR_PID=$!

wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

kill $MONITOR_PID 2>/dev/null

echo "=== TRAINING COMPLETED ==="
echo "Date: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "Duration: $(($(date +%s) - start_time)) seconds"

final_epoch=$(grep -oE "Epoch [0-9]+/" logs/output_resume_${SLURM_JOB_ID}.txt | tail -1 | grep -oE "[0-9]+" || echo "0")
echo "Final displayed epoch: $final_epoch"

echo "=== MODEL CHECK ==="
ls -lh outputs/cleanunet2/*epoch*.pth 2>/dev/null || echo "No model files found"
if [ -f "outputs/cleanunet2/cleanunet2_best.pth" ]; then
    echo "Best model:"
    ls -lh outputs/cleanunet2/cleanunet2_best.pth
fi

rm -f monitor_training.sh

echo "=== READY FOR INFERENCE ==="
echo "Use:"
echo "python test_realtime.py --model outputs/cleanunet2/cleanunet2_best.pth --test_dir data/test/noisy"

echo "=== JOB FINISHED ==="

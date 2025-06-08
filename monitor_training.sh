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

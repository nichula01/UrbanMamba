#!/bin/bash

# MambaVision-NSST Training from Scratch
# Configuration:
# - Epochs: 50
# - Batch Size: 6
# - LR: 1e-5 (stable)
# - AMP: DISABLED (Full FP32)
# - Gradient Clip: 0.05
# - Best model saved to: models_trained/

echo "=========================================="
echo "MambaVision-NSST Training"
echo "=========================================="
echo "Training from scratch"
echo "Epochs: 50"
echo "Batch Size: 6"
echo "GPU: 1"
echo "Model save: models_trained/"
echo "=========================================="

# Setup environment
cd /storage2/ChangeDetection/SemanticSegmentation/Mambavision-NSST-fusion/semantic_segmentation/urban_mamba
export CUDA_VISIBLE_DEVICES=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mamba_new

# Create log directory
LOG_DIR="/storage2/ChangeDetection/SemanticSegmentation/Logs/MambaVision-NSST"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mambavision_training_${TIMESTAMP}.log"

echo "Log file: ${LOG_FILE}"
echo "Starting training from scratch..."
echo ""

# Run training from beginning (no --resume)
python train_mambavision.py 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Training completed or stopped."
echo "Log saved to: ${LOG_FILE}"

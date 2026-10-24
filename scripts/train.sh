#!/usr/bin/env bash
# ==============================================================================
# Training script for MLPMC-Net
# ==============================================================================

# Set environment variables for nnU-Net
export nnUNet_raw="${nnUNet_raw:-./data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-./data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-./data/nnUNet_results}"

DATASET_ID=702
CONFIGURATION="2d"
FOLD=0
TRAINER="nnUNetTrainercls_mlpmc_final"

# ------------------------------------------------------------------------------
# Option 1: Standalone Training Launcher (Recommended, works immediately)
# ------------------------------------------------------------------------------
echo "Starting Single-GPU Training on Fold ${FOLD} using scripts/run_training.py..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_training.py ${DATASET_ID} ${CONFIGURATION} ${FOLD}

# ------------------------------------------------------------------------------
# Option 2: Standard nnU-Net CLI (Requires trainer in nnunetv2 variants folder)
# ------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} -tr ${TRAINER}

# ------------------------------------------------------------------------------
# Option 3: Multi-GPU Training (Distributed Data Parallel across 4 GPUs)
# ------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     scripts/run_training.py \
#     ${DATASET_ID} ${CONFIGURATION} ${FOLD}

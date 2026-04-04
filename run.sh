#!/bin/bash

# --- Configuration Section ---
# Model architecture: cnn, mlp, resnet, vit
MODEL="cnn"
# Dataset name: mnist, cifar10
DATASET="mnist"
# Number of training epochs.
EPOCHS=500
# Number of processes per node.
NPROC=1
# Path to pretrained model and dataset directory.
PRETRAINED="./output/final_model.pth"
DATASET_DIR="/media/horizon/Database/robotic_datasets/visual_learning"
# Training options
DIST_OPTS="--enable_distributed"
AMP_OPTS="--use_amp --amp_dtype bfloat16"

# --- Training Command ---
echo "Starting training: Model=$MODEL, Dataset=$DATASET, Epochs=$EPOCHS"
torchrun --nproc_per_node=$NPROC ./src/train_model.py \
    $DIST_OPTS \
    $AMP_OPTS \
    --model "$MODEL" \
    --dataset_name "$DATASET" \
    --max_epochs $EPOCHS \
    --pretrained_model_path "$PRETRAINED" \
    --dataset_dir "$DATASET_DIR"

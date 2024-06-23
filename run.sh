#!/bin/bash

# Set the number of threads for OpenMP
export OMP_NUM_THREADS=4

# Variables
MODEL_NAME="vitb_100k"
FIRST_SUFFIX="_head"
SECOND_SUFFIX="_full"
FIRST_RUN_NAME="${MODEL_NAME}${FIRST_SUFFIX}"
SECOND_RUN_NAME="${MODEL_NAME}${SECOND_SUFFIX}"
MODEL_DIR="saved_models"
PRETRAIN_DIR="saved_models_pretrain"
BEST_MODEL_NAME="best.pth"
FIRST_RUN_MODEL_PATH="${PRETRAIN_DIR}/${FIRST_RUN_NAME}.pth"
FINAL_MODEL_PATH="${MODEL_DIR}/${SECOND_RUN_NAME}.pth"

# Train the model with frozen backbone
torchrun --nproc_per_node=8 train.py -n $FIRST_RUN_NAME -b 512 -l 5e-3 --freeze_backbone

# Check if the first training was successful
if [ $? -ne 0 ]; then
  echo "First training run failed"
  exit 1
fi

# Copy the best model from the first run to a new location
cp ${MODEL_DIR}/${BEST_MODEL_NAME} ${FIRST_RUN_MODEL_PATH}

# Check if the copy was successful
if [ $? -ne 0 ]; then
  echo "Failed to copy the best model"
  exit 1
fi

# Train the model again, loading the state dictionary
torchrun --nproc_per_node=8 train.py -n $SECOND_RUN_NAME -b 96 -l 1e-4 -s ${FIRST_RUN_MODEL_PATH}

# Check if the second training was successful
if [ $? -ne 0 ]; then
  echo "Second training run failed"
  exit 1
fi

# Copy the best model to the final model path
cp ${MODEL_DIR}/${BEST_MODEL_NAME} ${FINAL_MODEL_PATH}

# Check if the final copy was successful
if [ $? -ne 0 ]; then
  echo "Failed to copy the final model"
  exit 1
fi

echo "Both training runs and final model copy completed successfully"

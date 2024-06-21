#!/bin/bash

# Set the number of threads for OpenMP
export OMP_NUM_THREADS=4

# Train the model with frozen backbone
torchrun --nproc_per_node=8 train.py -n head -b 512 -l 5e-3 --freeze_backbone

# Check if the first training was successful
if [ $? -ne 0 ]; then
  echo "First training run failed"
  exit 1
fi

# Copy the best model from the first run to a new location
cp saved_models/best.pth saved_models_pretrain/vitb_head.pth

# Check if the copy was successful
if [ $? -ne 0 ]; then
  echo "Failed to copy the best model"
  exit 1
fi

# Train the model again, loading the state dictionary
torchrun --nproc_per_node=8 train.py -n full -b 96 -l 1e-4 -s saved_models_pretrain/vitb_head.pth

# Check if the second training was successful
if [ $? -ne 0 ]; then
  echo "Second training run failed"
  exit 1
fi

echo "Both training runs completed successfully"

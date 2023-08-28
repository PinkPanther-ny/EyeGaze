#!/bin/bash

torchrun --nproc_per_node=3 train.py -n resnet101_20k_224_b64_aug_use_freeze_pretrain
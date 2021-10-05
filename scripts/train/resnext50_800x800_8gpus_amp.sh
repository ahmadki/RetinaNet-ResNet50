#!/usr/bin/env bash

nvidia-smi
env
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --backbone resnext50_32x4d_fpn \
    --batch-size 2 \
    --dataset coco \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor -1 \
    --lr 0.01 \
    --fixed-size 800 800 \
    --amp

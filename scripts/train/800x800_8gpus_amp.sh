#!/usr/bin/env bash

nvidia-smi
env
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --dataset coco \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.01 \
    --fixed-size 800 800 \
    --amp

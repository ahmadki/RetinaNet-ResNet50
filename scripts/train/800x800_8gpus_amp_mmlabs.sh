#!/usr/bin/env bash

nvidia-smi
env
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --dataset coco \
    --batch-size 1 \
    --epochs 13 \
    --lr 0.01 \
    --fixed-size 800 800 \
    --amp

#!/usr/bin/env bash

nvidia-smi
env
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --dataset coco \
    --model retinanet_resnet50_fpn \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.01 \
    --batch-size 4 \
    --min-size=800 \
    --max-size=800

#!/usr/bin/env bash

export TRAIN_CMD="train.py --batch-size 32 --dataset coco --epochs 52 --lr-steps 30 44 --aspect-ratio-group-factor 3 --lr 0.1 --fixed-size 800 800"


## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

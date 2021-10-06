#!/usr/bin/env bash

export COCO_FOLDER=/raid/datasets/coco/coco-2017/coco2017

srun \
    --mpi=none \
    --ntasks="$(( DGXNNODES * DGXNGPU ))" \
    --ntasks-per-node="${DGXNGPU}" \
    --container-image="${1}" \
    --container-workdir=/ssd \
    --container-mounts=$(pwd):/ssd,${COCO_FOLDER}:/datasets/coco2017 \
    ./run_and_time.sh

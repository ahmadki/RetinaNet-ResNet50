#!/usr/bin/env bash
source ./scripts/docker/config.sh

DATA_DIR=$1
RESULTS_DIR=$2

docker run --init -it --rm \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --gpus=all \
  --ipc=host \
  --workdir="/ssd" \
  -v $PWD:/ssd \
  -v "$DATA_DIR":/datasets/coco2017 \
  -v "$RESULTS_DIR":/results \
  -v /datasets/backbones:/root/.cache/torch/hub/checkpoints/ \
  $target_docker_image bash

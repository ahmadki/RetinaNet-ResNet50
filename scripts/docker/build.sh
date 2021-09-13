#!/usr/bin/env bash
source ./scripts/docker/config.sh

docker build . --rm --no-cache -t $target_docker_image --build-arg FROM_IMAGE_NAME=$source_docker_image

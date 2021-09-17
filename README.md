
# Retinanet-ResNet50
This is an experimental repository that is based on torchvision's RetinaNet and is used to test and evaluate the model as the new object detection benchmark for MLPerf training and inference.
[Torchvision RetinaNet](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html)
[MLPerf-SSD](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd)
[MLCommons](https://mlcommons.org/en/)

## Usage instructions

build and launch the container:
```bash
./scripts/docker/build.sh
./scripts/docker/launch_local.sh  # you might want to change the dataset mount location
```
If necessary, Download MS-COCO dataset
```bash
./scripts/download_dataset.sh
```

To train the model, use any of the training scripts in `scripts/train`

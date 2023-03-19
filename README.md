
# Deprecation notice

This repo has been merged with upstream MLCommons benchmark here: https://github.com/mlcommons/training

-----------------------------------
# Retinanet-ResNet50
This is an experimental repository that is based on torchvision's RetinaNet and is used to test and evaluate the model as the new object detection benchmark for MLPerf training and inference.

[Torchvision RetinaNet](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html)

[MLPerf-SSD](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd)

[MLCommons](https://mlcommons.org/en/)

## Usage instructions

*This README is a WIP. The build and run scripts and instructions will be improved in the coming weeks*

### Download the dataset
If necessary, Download MS-COCO dataset
```bash
./scripts/download_dataset.sh
```

### Build the container
build and launch the container:
```bash
./scripts/docker/build.sh
```
You will want to `change target_docker_image` in `./scripts/docker/config.sh`

### Local training / Single node training
Run the built container
```bash
./scripts/docker/launch_local.sh
```
In addition to changing `target_docker_image` in `./scripts/docker/config.sh`, you'll also want to change the
dataset and results paths in the script.

To train the model, use any of the training scripts in `scripts/train`

### SLURM
The repository includes template scripts to train the model on a SLURM cluster.
This training method can be used for both single-node and multi-node.

Your SLURM cluster will need to be prepared with:
* [Pyxis](https://github.com/NVIDIA/pyxis)
* [Enroot](https://github.com/NVIDIA/enroot) (multi-node)

After you built your image, submit a SLURM job with:

```bash
./scripts/launchers/selene.sh -c <config_name> -d <continaer_name>
```

You will want to change:
* --account in `./scripts/launchers/slurm.sh`
* --partition `./scripts/launchers/slurm.sh`
* --job-name `./scripts/launchers/slurm.sh`
* COCO_FOLDER in `./scripts/launchers/srun.sh`

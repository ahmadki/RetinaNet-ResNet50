r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import os
import time
import random
import argparse
import datetime

import numpy as np
import torch
import torch.utils.data
import torchvision

from coco_utils import get_coco

import utils
import presets
from engine import train_one_epoch, evaluate
from model.retinanet import (retinanet_resnet50_fpn,
                             retinanet_resnext50_32x4d_fpn)


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    # Model
    parser.add_argument('--backbone', default='resnet50_fpn', choices=['resnet50_fpn', 'resnext50_32x4d_fpn'],
                        help='The model backbone')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true", help="Use sync batch norm")
    parser.add_argument("--amp", action="store_true",
                        help="Whether to enable Automatic Mixed Precision (AMP). When false, uses TF32 on A100 and FP32 on V100 GPUS.")

    # Dataset
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--data-path', default='/datasets/coco2017', help='dataset')
    parser.add_argument('--image-size', default=[800, 800], nargs=2, type=int,
                        help='Image size for training')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')

    # Train parameters
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--output-dir', default=None, help='path where to save checkpoints.')
    parser.add_argument('--target-map', default=None, type=float, help='Stop training when target mAP is reached')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pre-trained models from the modelzoo")

    # Hyperparameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-e', '--eval-batch-size', default=None, type=int,
                        help='evaluation images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # Other
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed')
    parser.add_argument('--device', default='cuda', help='device')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    args.eval_batch_size = args.eval_batch_size or args.batch_size

    return args


def main(args):
    print(args)
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # set rank seeds according to MLPerf rules
    if args.distributed:
        args.seed = utils.broadcast(args.seed, src=1)
        args.seed = (args.seed + utils.get_rank()) % 2**32
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    # Print args
    print(args)

    # Data loading code
    print("Loading data")
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args.data_augmentation),
                                       args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args.data_augmentation), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.eval_batch_size or args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = None
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }
    if args.backbone=="resnet50_fpn":
        model = retinanet_resnet50_fpn(num_classes=num_classes, pretrained=args.pretrained,
                                       image_size=args.image_size,
                                       **kwargs)
    elif args.backbone=="resnext50_32x4d_fpn":
        model = retinanet_resnext50_32x4d_fpn(num_classes=num_classes, pretrained=args.pretrained,
                                              image_size=args.image_size,
                                              **kwargs)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device, args=args)
        return

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        # evaluate after every epoch
        coco_evaluator = evaluate(model, data_loader_test, device=device, args=args)
        accuracy = coco_evaluator.get_stats()['bbox'][0]
        if args.target_map and accuracy >= args.target_map:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = parse_args()
    main(args)

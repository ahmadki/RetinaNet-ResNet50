# TODO: clean
import copy
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision
import pandas

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from openimages import OpenImagesDetection

import transforms as T


class OpenImagesToCOCO(object):
    # TODO: docstring
    """Convert OpenImages dataset format to COCO format

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self):
        self.labels = pandas.read_csv("labels_text.csv") # TODO: paramtrize ?


    @staticmethod
    def hash_str(string, mod=10**12):
        return hash(string) % mod


    def __call__(self, image, target):
        w, h = image.size

        # image_id = torch.tensor([self.hash_str(t["ImageID"]) for t in target], dtype=torch.int64)
        # labels = torch.tensor([self.hash_str(t["LabelName"]) for t in target], dtype=torch.int64)
        image_id = torch.tensor([t["ImageIdx"] for t in target], dtype=torch.int64)
        labels = torch.tensor([t["LabelIdx"] for t in target], dtype=torch.int64)
        labels = torch.tensor([1 for t in target], dtype=torch.int64) # FIXME
        iscrowd = torch.tensor([t["IsGroupOf"] for t in target])

        boxes = [[t["XMin"]*w, t["YMin"]*h, t["XMax"]*w, t["YMax"]*h] for t in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        image_id = image_id[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        iscrowd = iscrowd[keep]

        target = dict(image_id=image_id, labels=labels, iscrowd=iscrowd, boxes=boxes)

        return image, target


def get_openimages(root, image_set, transforms):
    PATHS = {
        "train": ("train", os.path.join("annotations", "boxes", "oidv6-train-annotations-bbox.csv")),
        "test": ("test", os.path.join("annotations", "boxes", "test-annotations-bbox.csv")),
        "val": ("validation", os.path.join("annotations", "boxes", "validation-annotations-bbox.csv")),
    }

    t = [OpenImagesToCOCO()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = OpenImagesDetection(img_folder, ann_file, transforms=transforms)

    return dataset

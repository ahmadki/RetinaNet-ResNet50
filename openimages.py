from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import copy
import os.path
from typing import Any, Callable, Optional, Tuple, List
import pandas


class OpenImagesDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

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

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.df = pandas.read_csv(annFile, usecols=["ImageID", "LabelName", "IsGroupOf", "XMin","YMin", "XMax","YMax"])
        self.ids = sorted(set(self.df["ImageID"]))
        self.labels = sorted(set(self.df["LabelName"]))
        # self.df["ImageIdx"] = self.df.apply(lambda row: self.ids.index(row['ImageID']), axis=1)
        # self.df["LabelIdx"] = self.df.apply(lambda row: self.ids.index(row['LabelName']), axis=1)

    def _load_image(self, id: str) -> Image.Image:
        path = os.path.join(self.root, id + ".jpg")
        return Image.open(path).convert("RGB")

    def _load_annotations(self, id: str) -> List[Any]:
        annotations = self.df[self.df["ImageID"]==id].to_dict('records')
        for annotation in annotations:
            annotation['ImageIdx'] = self.ids.index(annotation['ImageID'])
            annotation['LabelIdx'] = self.labels.index(annotation['LabelName'])
        return annotations

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_annotations(id)
        target = copy.deepcopy(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    def get_ids(self) -> List[str]:
        return self.ids

    def get_labels(self) -> List[str]:
        return self.labels

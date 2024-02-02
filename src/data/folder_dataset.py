import os
import random
from typing import Dict, List, Tuple

import torch
import torchvision.transforms.v2 as v2
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils.data_utils import transform


class ImageFolderWithFilter(ImageFolder):
    def __init__(
        self,
        root: str,
        class_filter: str,
        num_classes=None,
        **kwargs,
    ):
        assert num_classes is None or num_classes > 0
        self.class_filter = class_filter
        self.num_classes = num_classes
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and str(self.class_filter) in str(entry)
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        if self.num_classes is not None:
            classes = classes[: self.num_classes]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class VideoFolder(Dataset):
    def __init__(
        self,
        hr_path,
        *,
        patch_size=None,
        augment=False,
        tempo_extent=None,
        hr_path_filter="",
        num_classes=None,
        jump_frames=1,
        **kwargs,
    ):
        self.hr_path = hr_path
        self.patch_size = patch_size
        self.augment_imgs = augment
        self.tempo_extent = tempo_extent
        self.hr_path_filter = hr_path_filter
        self.num_classes = num_classes
        self.jump_frames = jump_frames

        self.hr = ImageFolderWithFilter(
            hr_path,
            hr_path_filter,
            num_classes,
            transform=transform,
        )

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, item):
        if self.tempo_extent is None:
            hr_img = self.hr[item][0]

            if self.patch_size is not None:
                hr_img = self.crop(hr_img)
            if self.augment_imgs:
                hr_img = self.augment(hr_img)

            return hr_img
        else:
            target = self.hr.imgs[item][1]

            hr_frms = []

            # read frames
            if self.tempo_extent == 2:
                window = range(-1, 1)
            else:
                mid = self.tempo_extent // 2
                window = range(-mid, mid + 1)
            for i in window:
                cur_item = item + i * self.jump_frames
                cur_target = (
                    self.hr.imgs[cur_item][1]
                    if cur_item < len(self) and cur_item >= 0
                    else -1
                )
                if cur_target != target:
                    cur_item = -1  # black frame
                if cur_item != -1:
                    cur_frm, cur_target = self.hr[cur_item]
                    assert cur_target == target
                else:
                    cur_frm = torch.zeros_like(self.hr[item][0])
                hr_frms.append(cur_frm)

            hr_frms = torch.stack(hr_frms)  # t c h w

            if self.patch_size is not None:
                hr_frms = self.crop(hr_frms)
            if self.augment_imgs:
                hr_frms = self.augment(hr_frms)

            return hr_frms

    def crop(self, gt_frms):
        gt_csz = self.patch_size

        gt_h, gt_w = gt_frms.shape[-2:]
        assert (gt_csz <= gt_h) and (
            gt_csz <= gt_w
        ), "the crop size is larger than the image size"

        # crop lr
        gt_top = random.randint(0, gt_h - gt_csz)
        gt_left = random.randint(0, gt_w - gt_csz)

        # crop gt
        gt_top = gt_top
        gt_left = gt_left
        gt_pats = gt_frms[..., gt_top : gt_top + gt_csz, gt_left : gt_left + gt_csz]

        return gt_pats

    def augment(self, gt_pats):
        # flip
        axis = random.randint(1, 3)
        if axis == 2:
            gt_pats = v2.functional.vflip(gt_pats)
        if axis == 3:
            gt_pats = v2.functional.hflip(gt_pats)

        # rotate
        angle = (0.0, 90.0, 180.0, 270.0)[random.randint(0, 3)]
        gt_pats = v2.functional.rotate(gt_pats, angle)

        return gt_pats


class VideoFolderPaired(Dataset):
    def __init__(
        self,
        hr_path,
        lr_path="",
        *,
        patch_size=None,
        augment=False,
        tempo_extent=None,
        hr_path_filter="",
        lr_path_filter="",
        num_classes=None,
        jump_frames=1,
        dataset_upscale_factor=4,
        **kwargs,
    ):
        """
        Custom dataset for the training phase. The getitem method will return a couple (x, y), where x is the
        LowQuality input and y is the relative groundtruth. The relationship between the LQ and HQ samples depends on
        how the dataset is built.
        """

        self.hr_path = hr_path
        self.lr_path = lr_path
        self.patch_size = patch_size
        self.augment_imgs = augment
        self.tempo_extent = tempo_extent
        self.hr_path_filter = hr_path_filter
        self.lr_path_filter = lr_path_filter
        self.num_classes = num_classes
        self.jump_frames = jump_frames
        self.has_lowres = lr_path != ""
        self.upscale_factor = dataset_upscale_factor

        self.hr = ImageFolderWithFilter(
            hr_path,
            hr_path_filter,
            num_classes,
            transform=transform,
        )

        if self.has_lowres:
            self.lr = ImageFolderWithFilter(
                lr_path,
                lr_path_filter,
                num_classes,
                transform=transform,
            )
            assert len(self.hr) == len(
                self.lr
            ), "has_lowres is True but num of lr images does not correspond to hr images"

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, item):
        if self.tempo_extent is None:
            hr_img = self.hr[item][0]

            if self.has_lowres:
                lr_img = self.lr[item][0]
            else:
                lr_img = interpolate(
                    hr_img.unsqueeze(0),
                    scale_factor=1 / self.upscale_factor,
                    mode="bicubic",
                ).squeeze(0)

            if self.patch_size is not None:
                hr_img, lr_img = self.crop(hr_img, lr_img)
            if self.augment_imgs:
                hr_img, lr_img = self.augment(hr_img, lr_img)

            return hr_img, lr_img
        else:
            target = self.hr.imgs[item][1]

            hr_frms, lr_frms = [], []

            # read frames
            if self.tempo_extent == 2:
                window = range(-1, 1)
            else:
                mid = self.tempo_extent // 2
                window = range(-mid, mid + 1)
            for i in window:
                cur_item = item + i * self.jump_frames
                cur_target = (
                    self.hr.imgs[cur_item][1]
                    if cur_item < len(self) and cur_item >= 0
                    else -1
                )
                if cur_target != target:
                    cur_item = -1  # black frame
                if cur_item != -1:
                    hr_frm, cur_target = self.hr[cur_item]
                    assert cur_target == target
                else:
                    hr_frm = torch.zeros_like(self.hr[item][0])
                hr_frms.append(hr_frm)
                if self.has_lowres:
                    if cur_item != -1:
                        lr_frm, cur_target = self.lr[cur_item]
                        assert cur_target == target
                    else:
                        lr_frm = torch.zeros_like(self.lr[item][0])
                    lr_frms.append(lr_frm)

            hr_frms = torch.stack(hr_frms)  # t c h w
            if self.has_lowres:
                lr_frms = torch.stack(lr_frms)
            else:
                lr_frms = interpolate(
                    hr_frms,
                    scale_factor=1 / self.upscale_factor,
                    mode="bicubic",
                )

            if self.patch_size is not None:
                hr_frms, lr_frms = self.crop(hr_frms, lr_frms)
            if self.augment_imgs:
                hr_frms, lr_frms = self.augment(hr_frms, lr_frms)

            return hr_frms, lr_frms

    def crop(self, gt_frms, lr_frms):
        gt_csz = self.patch_size * self.upscale_factor
        lr_csz = self.patch_size

        lr_h, lr_w = lr_frms.shape[-2:]
        assert (lr_csz <= lr_h) and (
            lr_csz <= lr_w
        ), "the crop size is larger than the image size"

        # crop lr
        lr_top = random.randint(0, lr_h - lr_csz)
        lr_left = random.randint(0, lr_w - lr_csz)
        lr_pats = lr_frms[..., lr_top : lr_top + lr_csz, lr_left : lr_left + lr_csz]

        # crop gt
        gt_top = lr_top * self.upscale_factor
        gt_left = lr_left * self.upscale_factor
        gt_pats = gt_frms[..., gt_top : gt_top + gt_csz, gt_left : gt_left + gt_csz]

        return gt_pats, lr_pats

    def augment(self, gt_pats, lr_pats):
        # flip
        axis = random.randint(1, 3)
        if axis == 2:
            gt_pats = v2.functional.vflip(gt_pats)
            lr_pats = v2.functional.vflip(lr_pats)
        if axis == 3:
            gt_pats = v2.functional.hflip(gt_pats)
            lr_pats = v2.functional.hflip(lr_pats)

        # rotate
        angle = (0.0, 90.0, 180.0, 270.0)[random.randint(0, 3)]
        gt_pats = v2.functional.rotate(gt_pats, angle)
        lr_pats = v2.functional.rotate(lr_pats, angle)

        return gt_pats, lr_pats


if __name__ == "__main__":
    ds = VideoFolder(
        # hr_path="/home/DATASETS/BVI_DVC/frames_HQ",
        hr_path="../REDS/X4/test",
        num_classes=50,
        patch_size=128,
        augment=True,
        tempo_extent=2,
        jump_frames=1,
    )
    # ds = VideoFolderPaired(
    #     hr_path="/home/DATASETS/BVI_DVC/frames_HQ",
    #     # hr_path="../REDS/X4/test",
    #     lr_path="/home/DATASETS/BVI_DVC/frames/frames_CRF_22",
    #     hr_path_filter="1088",
    #     lr_path_filter="544",
    #     dataset_upscale_factor=4,
    #     num_classes=50,
    #     patch_size=64,
    #     augment=True,
    #     tempo_extent=7,
    #     jump_frames=5,
    # )
    el = ds[94]
    print()

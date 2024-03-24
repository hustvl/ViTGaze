import glob
from typing import Callable, Optional
from os import path as osp

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
from PIL import Image

from . import augmentation
from . import data_utils as utils
from .masking import MaskGenerator


class VideoAttentionTarget(Dataset):
    def __init__(
        self,
        image_root: str,
        anno_root: str,
        head_root: str,
        transform: Callable,
        input_size: int,
        output_size: int,
        quant_labelmap: bool = True,
        is_train: bool = True,
        *,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
    ):
        frames = []
        for show_dir in glob.glob(osp.join(anno_root, "*")):
            for sequence_path in glob.glob(osp.join(show_dir, "*", "*.txt")):
                df = pd.read_csv(
                    sequence_path,
                    header=None,
                    index_col=False,
                    names=[
                        "path",
                        "x_min",
                        "y_min",
                        "x_max",
                        "y_max",
                        "gaze_x",
                        "gaze_y",
                    ],
                )

                show_name = sequence_path.split("/")[-3]
                clip = sequence_path.split("/")[-2]
                df["path"] = df["path"].apply(
                    lambda path: osp.join(show_name, clip, path)
                )
                # Add two columns for the bbox center
                df["eye_x"] = (df["x_min"] + df["x_max"]) / 2
                df["eye_y"] = (df["y_min"] + df["y_max"]) / 2
                df = df.sample(frac=0.2, random_state=42)
                frames.extend(df.values.tolist())

        df = pd.DataFrame(
            frames,
            columns=[
                "path",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "gaze_x",
                "gaze_y",
                "eye_x",
                "eye_y",
            ],
        )
        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["x_min"].values,
                    df["y_min"].values,
                    df["x_max"].values,
                    df["y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)
        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        self.df = df
        self.length = len(df)

        self.data_dir = image_root
        self.head_dir = head_root
        self.transform = transform
        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )
        self.is_train = is_train

        self.input_size = input_size
        self.output_size = output_size

        if self.is_train:
            ## data augmentation
            self.augment = augmentation.AugmentationList(
                [
                    augmentation.ColorJitter(color_jitter),
                    augmentation.BoxJitter(bbox_jitter),
                    augmentation.RandomCrop(rand_crop),
                    augmentation.RandomFlip(rand_flip),
                    augmentation.RandomRotate(rand_rotate),
                    augmentation.RandomLSJ(rand_lsj),
                ]
            )

            self.mask_generator = mask_generator

    def __getitem__(self, index):
        (
            _,
            path,
            x_min,
            y_min,
            x_max,
            y_max,
            gaze_x,
            gaze_y,
            eye_x,
            eye_y,
        ) = self.df.iloc[index]
        gaze_inside = gaze_x != -1 or gaze_y != -1

        img = Image.open(osp.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        if osp.exists(osp.join(self.head_dir, path)):
            head_mask = Image.open(osp.join(self.head_dir, path)).resize(
                (width, height)
            )
        else:
            head_mask = Image.fromarray(np.zeros((height, width), dtype=np.float32))
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        gaze_x, gaze_y = gaze_x / width, gaze_y / height
        # expand face bbox a bit
        k = 0.1
        x_min = max(x_min - k * abs(x_max - x_min), 0)
        y_min = max(y_min - k * abs(y_max - y_min), 0)
        x_max = min(x_max + k * abs(x_max - x_min), width - 1)
        y_max = min(y_max + k * abs(y_max - y_min), height - 1)

        if self.is_train:
            img, bbox, gaze, head_mask, size = self.augment(
                img,
                (x_min, y_min, x_max, y_max),
                (gaze_x, gaze_y),
                head_mask,
                (width, height),
            )
            x_min, y_min, x_max, y_max = bbox
            gaze_x, gaze_y = gaze
            width, height = size

        head_channel = utils.get_head_box_channel(
            x_min,
            y_min,
            x_max,
            y_max,
            width,
            height,
            resolution=self.input_size,
            coordconv=False,
        ).unsqueeze(0)

        if self.is_train and self.mask_generator is not None:
            image_mask = self.mask_generator(
                x_min / width,
                y_min / height,
                x_max / width,
                y_max / height,
                head_channel,
            )

        if self.transform is not None:
            img = self.transform(img)
            head_mask = TF.to_tensor(
                TF.resize(head_mask, (self.input_size, self.input_size))
            )

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(
            self.output_size, self.output_size
        )  # set the size of the output

        gaze_heatmap = self.draw_labelmap(
            gaze_heatmap,
            [gaze_x * self.output_size, gaze_y * self.output_size],
            3,
            type="Gaussian",
        )

        imsize = torch.IntTensor([width, height])

        out_dict = {
            "images": img,
            "head_channels": head_channel,
            "heatmaps": gaze_heatmap,
            "gazes": torch.FloatTensor([gaze_x, gaze_y]),
            "gaze_inouts": torch.FloatTensor([gaze_inside]),
            "head_masks": head_mask,
            "imsize": imsize,
        }
        if self.is_train and self.mask_generator is not None:
            out_dict["image_masks"] = image_mask
        return out_dict

    def __len__(self):
        return self.length

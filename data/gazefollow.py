from os import path as osp
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd

from . import augmentation
from .masking import MaskGenerator
from . import data_utils as utils


class GazeFollow(Dataset):
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
        if is_train:
            column_names = [
                "path",
                "idx",
                "body_bbox_x",
                "body_bbox_y",
                "body_bbox_w",
                "body_bbox_h",
                "eye_x",
                "eye_y",
                "gaze_x",
                "gaze_y",
                "bbox_x_min",
                "bbox_y_min",
                "bbox_x_max",
                "bbox_y_max",
                "inout",
                "meta0",
                "meta1",
            ]
            df = pd.read_csv(
                anno_root,
                sep=",",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            df = df[
                df["inout"] != -1
            ]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            self.y_train = df[
                [
                    "bbox_x_min",
                    "bbox_y_min",
                    "bbox_x_max",
                    "bbox_y_max",
                    "eye_x",
                    "eye_y",
                    "gaze_x",
                    "gaze_y",
                    "inout",
                ]
            ]
            self.X_train = df["path"]
            self.length = len(df)
        else:
            column_names = [
                "path",
                "idx",
                "body_bbox_x",
                "body_bbox_y",
                "body_bbox_w",
                "body_bbox_h",
                "eye_x",
                "eye_y",
                "gaze_x",
                "gaze_y",
                "bbox_x_min",
                "bbox_y_min",
                "bbox_x_max",
                "bbox_y_max",
                "meta0",
                "meta1",
            ]
            df = pd.read_csv(
                anno_root,
                sep=",",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            df = df[
                [
                    "path",
                    "eye_x",
                    "eye_y",
                    "gaze_x",
                    "gaze_y",
                    "bbox_x_min",
                    "bbox_y_min",
                    "bbox_x_max",
                    "bbox_y_max",
                ]
            ].groupby(["path", "eye_x"])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)

        self.data_dir = image_root
        self.head_dir = head_root
        self.transform = transform
        self.is_train = is_train

        self.input_size = input_size
        self.output_size = output_size

        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )

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
        if not self.is_train:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for _, row in g.iterrows():
                path = row["path"]
                x_min = row["bbox_x_min"]
                y_min = row["bbox_y_min"]
                x_max = row["bbox_x_max"]
                y_max = row["bbox_y_max"]
                eye_x = row["eye_x"]
                eye_y = row["eye_y"]
                gaze_x = row["gaze_x"]
                gaze_y = row["gaze_y"]
                cont_gaze.append(
                    [gaze_x, gaze_y]
                )  # all ground truth gaze are stacked up
            for _ in range(len(cont_gaze), 20):
                cont_gaze.append(
                    [-1, -1]
                )  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True  # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            (
                x_min,
                y_min,
                x_max,
                y_max,
                eye_x,
                eye_y,
                gaze_x,
                gaze_y,
                inout,
            ) = self.y_train.iloc[index]
            gaze_inside = bool(inout)

        img = Image.open(osp.join(self.data_dir, path))
        img = img.convert("RGB")
        head_mask = Image.open(osp.join(self.head_dir, path))
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
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
        if not self.is_train:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap += self.draw_labelmap(
                        torch.zeros(self.output_size, self.output_size),
                        [gaze_x * self.output_size, gaze_y * self.output_size],
                        3,
                        type="Gaussian",
                    )
            gaze_heatmap /= num_valid
        else:
            # if gaze_inside:
            gaze_heatmap = self.draw_labelmap(
                gaze_heatmap,
                [gaze_x * self.output_size, gaze_y * self.output_size],
                3,
                type="Gaussian",
            )

        imsize = torch.IntTensor([width, height])

        if self.is_train:
            out_dict = {
                "images": img,
                "head_channels": head_channel,
                "heatmaps": gaze_heatmap,
                "gazes": torch.FloatTensor([gaze_x, gaze_y]),
                "gaze_inouts": torch.FloatTensor([gaze_inside]),
                "head_masks": head_mask,
                "imsize": imsize,
            }
            if self.mask_generator is not None:
                out_dict["image_masks"] = image_mask
            return out_dict
        else:
            return {
                "images": img,
                "head_channels": head_channel,
                "heatmaps": gaze_heatmap,
                "gazes": cont_gaze,
                "gaze_inouts": torch.FloatTensor([gaze_inside]),
                "head_masks": head_mask,
                "imsize": imsize,
            }

    def __len__(self):
        return self.length

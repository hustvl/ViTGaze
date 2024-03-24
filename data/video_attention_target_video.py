import math
from os import path as osp
from typing import Callable, Optional
import glob
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from .masking import MaskGenerator
from . import data_utils as utils


class VideoAttentionTargetVideo(Dataset):
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
        seq_len: int = 8,
        max_len: int = 32,
        *,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
    ):
        dfs = []
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
                cur_len = len(df.index)
                if is_train:
                    if cur_len <= max_len:
                        if cur_len >= seq_len:
                            dfs.append(df)
                        continue
                    remainder = cur_len % max_len
                    df_splits = [
                        df[i : i + max_len]
                        for i in range(0, cur_len - max_len, max_len)
                    ]
                    if remainder >= seq_len:
                        df_splits.append(df[-remainder:])
                    dfs.extend(df_splits)
                else:
                    if cur_len < seq_len:
                        continue
                    df_splits = [
                        df[i : i + seq_len]
                        for i in range(0, cur_len - seq_len, seq_len)
                    ]
                    dfs.extend(df_splits)

        for df in dfs:
            df.reset_index(inplace=True)
        self.dfs = dfs
        self.length = len(dfs)

        self.data_dir = image_root
        self.head_dir = head_root
        self.transform = transform
        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )
        self.is_train = is_train

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len

        if self.is_train:
            self.bbox_jitter = bbox_jitter
            self.rand_crop = rand_crop
            self.rand_flip = rand_flip
            self.color_jitter = color_jitter
            self.rand_rotate = rand_rotate
            self.rand_lsj = rand_lsj
            self.mask_generator = mask_generator

    def __getitem__(self, index):
        df = self.dfs[index]
        seq_len = len(df.index)
        for coord in ["x_min", "y_min", "x_max", "y_max"]:
            df[coord] = utils.smooth_by_conv(11, df, coord)

        if self.is_train:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            if cond_color < self.color_jitter:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)
            cond_crop = np.random.random_sample()
            cond_rotate = np.random.random_sample()
            if cond_rotate < self.rand_rotate:
                angle = (2 * np.random.random_sample() - 1) * 20
                angle = -math.radians(angle)
            cond_lsj = np.random.random_sample()
            if cond_lsj < self.rand_lsj:
                lsj_scale = 0.1 + np.random.random_sample() * 0.9

            # if longer than seq_len_limit, cut it down to the limit with the init index randomly sampled
            if seq_len > self.seq_len:
                sampled_ind = np.random.randint(0, seq_len - self.seq_len)
                seq_len = self.seq_len
            else:
                sampled_ind = 0

            if cond_crop < self.rand_crop:
                sliced_x_min = df["x_min"].iloc[sampled_ind : sampled_ind + seq_len]
                sliced_x_max = df["x_max"].iloc[sampled_ind : sampled_ind + seq_len]
                sliced_y_min = df["y_min"].iloc[sampled_ind : sampled_ind + seq_len]
                sliced_y_max = df["y_max"].iloc[sampled_ind : sampled_ind + seq_len]

                sliced_gaze_x = df["gaze_x"].iloc[sampled_ind : sampled_ind + seq_len]
                sliced_gaze_y = df["gaze_y"].iloc[sampled_ind : sampled_ind + seq_len]

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2 * seq_len

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min(
                        [sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()]
                    )
                    crop_y_min = np.min(
                        [sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()]
                    )
                    crop_x_max = np.max(
                        [sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()]
                    )
                    crop_y_max = np.max(
                        [sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()]
                    )

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = osp.join(self.data_dir, df["path"].iloc[0])
                img = Image.open(path)
                img = img.convert("RGB")
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Round to integers
                crop_y_min, crop_x_min, crop_height, crop_width = map(
                    int, map(round, (crop_y_min, crop_x_min, crop_height, crop_width))
                )
        else:
            sampled_ind = 0

        images = []
        head_channels = []
        heatmaps = []
        gazes = []
        gaze_inouts = []
        imsizes = []
        head_masks = []
        if self.is_train and self.mask_generator is not None:
            image_masks = []
        for i, row in df.iterrows():
            if self.is_train and (i < sampled_ind or i >= (sampled_ind + self.seq_len)):
                continue

            x_min = row["x_min"]  # note: Already in image coordinates
            y_min = row["y_min"]  # note: Already in image coordinates
            x_max = row["x_max"]  # note: Already in image coordinates
            y_max = row["y_max"]  # note: Already in image coordinates
            gaze_x = row["gaze_x"]  # note: Already in image coordinates
            gaze_y = row["gaze_y"]  # note: Already in image coordinates

            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            path = row["path"]
            img = Image.open(osp.join(self.data_dir, path)).convert("RGB")
            width, height = img.size
            imsize = torch.FloatTensor([width, height])
            imsizes.append(imsize)
            if osp.exists(osp.join(self.head_dir, path)):
                head_mask = Image.open(osp.join(self.head_dir, path)).resize(
                    (width, height)
                )
            else:
                head_mask = Image.fromarray(np.zeros((height, width), dtype=np.float32))

            x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if (
                    gaze_x < 0
                ):  # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True

            if self.is_train:
                ## data augmentation
                # Jitter (expansion-only) bounding box size.
                if cond_jitter < self.bbox_jitter:
                    k = cond_jitter * 0.1
                    x_min -= k * abs(x_max - x_min)
                    y_min -= k * abs(y_max - y_min)
                    x_max += k * abs(x_max - x_min)
                    y_max += k * abs(y_max - y_min)
                    x_min = np.clip(x_min, 0, width - 1)
                    x_max = np.clip(x_max, 0, width - 1)
                    y_min = np.clip(y_min, 0, height - 1)
                    y_max = np.clip(y_max, 0, height - 1)

                # Random color change
                if cond_color < self.color_jitter:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)

                # Random Crop
                if cond_crop < self.rand_crop:
                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    head_mask = TF.crop(
                        head_mask, crop_y_min, crop_x_min, crop_height, crop_width
                    )

                    # Record the crop's (x, y) offset
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    x_min, y_min, x_max, y_max = (
                        x_min - offset_x,
                        y_min - offset_y,
                        x_max - offset_x,
                        y_max - offset_y,
                    )
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x - offset_x), (gaze_y - offset_y)
                    else:
                        gaze_x = -1
                        gaze_y = -1

                    width, height = crop_width, crop_height

                # Flip?
                if cond_flip < self.rand_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    head_mask = head_mask.transpose(Image.FLIP_LEFT_RIGHT)
                    x_max_2 = width - x_min
                    x_min_2 = width - x_max
                    x_max = x_max_2
                    x_min = x_min_2
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x

                # Random Rotation
                if cond_rotate < self.rand_rotate:
                    rot_mat = [
                        round(math.cos(angle), 15),
                        round(math.sin(angle), 15),
                        0.0,
                        round(-math.sin(angle), 15),
                        round(math.cos(angle), 15),
                        0.0,
                    ]

                    def _transform(x, y, matrix):
                        return (
                            matrix[0] * x + matrix[1] * y + matrix[2],
                            matrix[3] * x + matrix[4] * y + matrix[5],
                        )

                    def _inv_transform(x, y, matrix):
                        x, y = x - matrix[2], y - matrix[5]
                        return (
                            matrix[0] * x + matrix[3] * y,
                            matrix[1] * x + matrix[4] * y,
                        )

                    # Calculate offsets
                    rot_center = (width / 2.0, height / 2.0)
                    rot_mat[2], rot_mat[5] = _transform(
                        -rot_center[0], -rot_center[1], rot_mat
                    )
                    rot_mat[2] += rot_center[0]
                    rot_mat[5] += rot_center[1]
                    xx = []
                    yy = []
                    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
                        x, y = _transform(x, y, rot_mat)
                        xx.append(x)
                        yy.append(y)
                    nw = math.ceil(max(xx)) - math.floor(min(xx))
                    nh = math.ceil(max(yy)) - math.floor(min(yy))
                    rot_mat[2], rot_mat[5] = _transform(
                        -(nw - width) / 2.0, -(nh - height) / 2.0, rot_mat
                    )

                    img = img.transform((nw, nh), Image.AFFINE, rot_mat, Image.BILINEAR)
                    head_mask = head_mask.transform(
                        (nw, nh), Image.AFFINE, rot_mat, Image.BILINEAR
                    )

                    xx = []
                    yy = []
                    for x, y in (
                        (x_min, y_min),
                        (x_min, y_max),
                        (x_max, y_min),
                        (x_max, y_max),
                    ):
                        x, y = _inv_transform(x, y, rot_mat)
                        xx.append(x)
                        yy.append(y)
                    x_max, x_min = min(max(xx), nw), max(min(xx), 0)
                    y_max, y_min = min(max(yy), nh), max(min(yy), 0)
                    gaze_x, gaze_y = _inv_transform(gaze_x, gaze_y, rot_mat)
                    width, height = nw, nh

                if cond_lsj < self.rand_lsj:
                    nh, nw = int(height * lsj_scale), int(width * lsj_scale)
                    img = TF.resize(img, (nh, nw))
                    img = ImageOps.expand(img, (0, 0, width - nw, height - nh))
                    head_mask = TF.resize(head_mask, (nh, nw))
                    head_mask = ImageOps.expand(
                        head_mask, (0, 0, width - nw, height - nh)
                    )
                    x_min, y_min, x_max, y_max = (
                        x_min * lsj_scale,
                        y_min * lsj_scale,
                        x_max * lsj_scale,
                        y_max * lsj_scale,
                    )
                    gaze_x, gaze_y = gaze_x * lsj_scale, gaze_y * lsj_scale

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
                image_masks.append(image_mask)

            if self.transform is not None:
                img = self.transform(img)
                head_mask = TF.to_tensor(
                    TF.resize(head_mask, (self.input_size, self.input_size))
                )

            if gaze_inside:
                gaze_x /= float(width)  # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(
                    self.output_size, self.output_size
                )  # set the size of the output
                gaze_map = self.draw_labelmap(
                    gaze_heatmap,
                    [gaze_x * self.output_size, gaze_y * self.output_size],
                    3,
                    type="Gaussian",
                )
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))
            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))
            images.append(img)
            head_channels.append(head_channel)
            head_masks.append(head_mask)
            heatmaps.append(gaze_map)
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))

        images = torch.stack(images)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        head_masks = torch.stack(head_masks)
        imsizes = torch.stack(imsizes)

        out_dict = {
            "images": images,
            "head_channels": head_channels,
            "heatmaps": heatmaps,
            "gazes": gazes,
            "gaze_inouts": gaze_inouts,
            "head_masks": head_masks,
            "imsize": imsizes,
        }
        if self.is_train and self.mask_generator is not None:
            out_dict["image_masks"] = torch.stack(image_masks)
        return out_dict

    def __len__(self):
        return self.length


def video_collate(batch):
    keys = batch[0].keys()
    return {key: torch.cat([item[key] for item in batch]) for key in keys}

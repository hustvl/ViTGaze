import random
import math
import numpy as np
import torch
from torch.nn import functional as F


class SceneMaskGenerator:
    def __init__(
        self,
        input_size,
        min_num_patches=16,
        max_num_patches_ratio=0.5,
        min_aspect=0.3,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.input_size = input_size
        self.num_patches = input_size[0] * input_size[1]

        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches_ratio * self.num_patches

        self.log_aspect_ratio = (math.log(min_aspect), -math.log(min_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(4):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            height, width = self.input_size
            if w < width and h < height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    mask[top : top + h, left : left + w] = 1
                    delta = h * w - num_masked
                    break
        return delta

    def __call__(self, head_mask):
        mask = np.zeros(shape=self.input_size, dtype=bool)
        mask_count = 0
        num_masking_patches = random.uniform(self.min_num_patches, self.max_num_patches)
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        mask = torch.from_numpy(mask).unsqueeze(0)
        head_mask = (
            F.interpolate(head_mask.unsqueeze(0), mask.shape[-2:]).squeeze(0) < 0.5
        )
        return torch.logical_and(mask, head_mask).squeeze(0)


class HeadMaskGenerator:
    def __init__(
        self,
        input_size,
        min_num_patches=4,
        max_num_patches_ratio=0.5,
        min_aspect=0.3,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.input_size = input_size
        self.num_patches = input_size[0] * input_size[1]

        self.min_num_patches = min_num_patches
        self.max_num_patches_ratio = max_num_patches_ratio

        self.log_aspect_ratio = (math.log(min_aspect), -math.log(min_aspect))

    def __call__(
        self,
        x_min,
        y_min,
        x_max,
        y_max,  # coords in [0,1]
    ):
        height = math.floor((y_max - y_min) * self.input_size[0])
        width = math.floor((x_max - x_min) * self.input_size[1])
        origin_area = width * height
        if origin_area < self.min_num_patches:
            return torch.zeros(size=self.input_size, dtype=bool)

        target_area = random.uniform(
            self.min_num_patches, self.max_num_patches_ratio * origin_area
        )
        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
        h = min(int(round(math.sqrt(target_area * aspect_ratio))), height)
        w = min(int(round(math.sqrt(target_area / aspect_ratio))), width)
        top = random.randint(0, height - h) + int(y_min * self.input_size[0])
        left = random.randint(0, width - w) + int(x_min * self.input_size[1])
        mask = torch.zeros(size=self.input_size, dtype=bool)
        mask[top : top + h, left : left + w] = True
        return mask


class MaskGenerator:
    def __init__(
        self,
        input_size,
        mask_scene: bool = False,
        mask_head: bool = False,
        min_scene_patches=16,
        max_scene_patches_ratio=0.5,
        min_head_patches=4,
        max_head_patches_ratio=0.5,
        min_aspect=0.3,
        mask_prob=0.2,
        head_prob=0.2,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.input_size = input_size
        if mask_scene:
            self.scene_mask_generator = SceneMaskGenerator(
                input_size, min_scene_patches, max_scene_patches_ratio, min_aspect
            )
        else:
            self.scene_mask_generator = None

        if mask_head:
            self.head_mask_generator = HeadMaskGenerator(
                input_size, min_head_patches, max_head_patches_ratio, min_aspect
            )
        else:
            self.head_mask_generator = None

        self.no_mask = not (mask_scene or mask_head)
        self.mask_head = mask_head and not mask_scene
        self.mask_scene = mask_scene and not mask_head
        self.scene_prob = mask_prob
        self.head_prob = head_prob

    def __call__(
        self,
        x_min,
        y_min,
        x_max,
        y_max,
        head_mask,
    ):
        mask_scene = random.random() < self.scene_prob
        mask_head = random.random() < self.head_prob
        no_mask = (
            self.no_mask
            or (self.mask_head and not mask_head)
            or (self.mask_scene and not mask_scene)
            or not (mask_scene or mask_head)
        )
        if no_mask:
            return torch.zeros(size=self.input_size, dtype=bool)
        if self.mask_scene:
            return self.scene_mask_generator(head_mask)
        if self.mask_head:
            return self.head_mask_generator(x_min, y_min, x_max, y_max)
        if mask_head and mask_scene:
            return torch.logical_or(
                self.scene_mask_generator(head_mask),
                self.head_mask_generator(x_min, y_min, x_max, y_max),
            )
        elif mask_head:
            return self.head_mask_generator(x_min, y_min, x_max, y_max)
        return self.scene_mask_generator(head_mask)

import math
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF


class Augmentation:
    def __init__(self, p: float) -> None:
        self.p = p

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        raise NotImplementedError

    def __call__(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        if np.random.random_sample() < self.p:
            return self.transform(image, bbox, gaze, head_mask, size)
        return image, bbox, gaze, head_mask, size


class AugmentationList:
    def __init__(self, augmentations: List[Augmentation]) -> None:
        self.augmentations = augmentations

    def __call__(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        for aug in self.augmentations:
            image, bbox, gaze, head_mask, size = aug(image, bbox, gaze, head_mask, size)
        return image, bbox, gaze, head_mask, size


class BoxJitter(Augmentation):
    # Jitter (expansion-only) bounding box size
    def __init__(self, p: float, expansion: float = 0.2) -> None:
        super().__init__(p)
        self.expansion = expansion

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        width, height = size
        k = np.random.random_sample() * self.expansion
        x_min = np.clip(x_min - k * abs(x_max - x_min), 0, width - 1)
        y_min = np.clip(y_min - k * abs(y_max - y_min), 0, height - 1)
        x_max = np.clip(x_max + k * abs(x_max - x_min), 0, width - 1)
        y_max = np.clip(y_max + k * abs(y_max - y_min), 0, height - 1)
        return image, (x_min, y_min, x_max, y_max), gaze, head_mask, size


class RandomCrop(Augmentation):
    def __init__(self, p: float) -> None:
        super().__init__(p)

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        gaze_x, gaze_y = gaze
        width, height = size
        # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
        crop_x_min = np.min([gaze_x * width, x_min, x_max])
        crop_y_min = np.min([gaze_y * height, y_min, y_max])
        crop_x_max = np.max([gaze_x * width, x_min, x_max])
        crop_y_max = np.max([gaze_y * height, y_min, y_max])

        # Randomly select a random top left corner
        crop_x_min = np.random.uniform(0, crop_x_min)
        crop_y_min = np.random.uniform(0, crop_y_min)

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

        # Crop it
        image = TF.crop(image, crop_y_min, crop_x_min, crop_height, crop_width)
        head_mask = TF.crop(head_mask, crop_y_min, crop_x_min, crop_height, crop_width)

        # convert coordinates into the cropped frame
        x_min, y_min, x_max, y_max = (
            x_min - crop_x_min,
            y_min - crop_y_min,
            x_max - crop_x_min,
            y_max - crop_y_min,
        )

        gaze_x = (gaze_x * width - crop_x_min) / float(crop_width)
        gaze_y = (gaze_y * height - crop_y_min) / float(crop_height)

        return (
            image,
            (x_min, y_min, x_max, y_max),
            (gaze_x, gaze_y),
            head_mask,
            (crop_width, crop_height),
        )


class RandomFlip(Augmentation):
    def __init__(self, p: float) -> None:
        super().__init__(p)

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        head_mask = head_mask.transpose(Image.FLIP_LEFT_RIGHT)
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = size[0] - x_max, size[0] - x_min
        gaze_x, gaze_y = 1 - gaze[0], gaze[1]
        return image, (x_min, y_min, x_max, y_max), (gaze_x, gaze_y), head_mask, size


class RandomRotate(Augmentation):
    def __init__(
        self, p: float, max_angle: int = 20, resample: int = Image.BILINEAR
    ) -> None:
        super().__init__(p)
        self.max_angle = max_angle
        self.resample = resample

    def _random_rotation_matrix(self):
        angle = (2 * np.random.random_sample() - 1) * self.max_angle
        angle = -math.radians(angle)
        return [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

    @staticmethod
    def _transform(x, y, matrix):
        return (
            matrix[0] * x + matrix[1] * y + matrix[2],
            matrix[3] * x + matrix[4] * y + matrix[5],
        )

    @staticmethod
    def _inv_transform(x, y, matrix):
        x, y = x - matrix[2], y - matrix[5]
        return matrix[0] * x + matrix[3] * y, matrix[1] * x + matrix[4] * y

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        gaze_x, gaze_y = gaze
        width, height = size
        rot_mat = self._random_rotation_matrix()

        # Calculate offsets
        rot_center = (width / 2.0, height / 2.0)
        rot_mat[2], rot_mat[5] = self._transform(
            -rot_center[0], -rot_center[1], rot_mat
        )
        rot_mat[2] += rot_center[0]
        rot_mat[5] += rot_center[1]
        xx = []
        yy = []
        for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
            x, y = self._transform(x, y, rot_mat)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))
        rot_mat[2], rot_mat[5] = self._transform(
            -(nw - width) / 2.0, -(nh - height) / 2.0, rot_mat
        )

        image = image.transform((nw, nh), Image.AFFINE, rot_mat, self.resample)
        head_mask = head_mask.transform((nw, nh), Image.AFFINE, rot_mat, self.resample)

        xx = []
        yy = []
        for x, y in (
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
        ):
            x, y = self._inv_transform(x, y, rot_mat)
            xx.append(x)
            yy.append(y)
        x_max, x_min = min(max(xx), nw), max(min(xx), 0)
        y_max, y_min = min(max(yy), nh), max(min(yy), 0)

        gaze_x, gaze_y = self._inv_transform(gaze_x * width, gaze_y * height, rot_mat)
        gaze_x = max(min(gaze_x / nw, 1), 0)
        gaze_y = max(min(gaze_y / nh, 1), 0)

        return (
            image,
            (x_min, y_min, x_max, y_max),
            (gaze_x, gaze_y),
            head_mask,
            (nw, nh),
        )


class ColorJitter(Augmentation):
    def __init__(
        self,
        p: float,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        super().__init__(p)
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        return self.color_jitter(image), bbox, gaze, head_mask, size


class RandomLSJ(Augmentation):
    def __init__(self, p: float, min_scale: float = 0.1) -> None:
        super().__init__(p)
        self.min_scale = min_scale

    def transform(
        self,
        image: Image,
        bbox: Tuple[float],
        gaze: Tuple[float],
        head_mask: Image,
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        gaze_x, gaze_y = gaze
        width, height = size

        scale = self.min_scale + np.random.random_sample() * (1 - self.min_scale)
        nh, nw = int(height * scale), int(width * scale)

        image = TF.resize(image, (nh, nw))
        image = ImageOps.expand(image, (0, 0, width - nw, height - nh))
        head_mask = TF.resize(head_mask, (nh, nw))
        head_mask = ImageOps.expand(head_mask, (0, 0, width - nw, height - nh))

        x_min, y_min, x_max, y_max = (
            x_min * scale,
            y_min * scale,
            x_max * scale,
            y_max * scale,
        )
        gaze_x, gaze_y = gaze_x * scale, gaze_y * scale
        return image, (x_min, y_min, x_max, y_max), (gaze_x, gaze_y), head_mask, size

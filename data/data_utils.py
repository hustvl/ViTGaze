from typing import Tuple
import torch
from torchvision import transforms
import numpy as np
import pandas as pd


def to_numpy(tensor: torch.Tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray: np.ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def get_head_box_channel(
    x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False
):
    head_box = (
        np.array([x_min / width, y_min / height, x_max / width, y_max / height])
        * resolution
    )
    int_head_box = head_box.astype(int)
    int_head_box = np.clip(int_head_box, 0, resolution - 1)
    if int_head_box[0] == int_head_box[2]:
        if int_head_box[0] == 0:
            int_head_box[2] = 1
        elif int_head_box[2] == resolution - 1:
            int_head_box[0] = resolution - 2
        elif abs(head_box[2] - int_head_box[2]) > abs(head_box[0] - int_head_box[0]):
            int_head_box[2] += 1
        else:
            int_head_box[0] -= 1
    if int_head_box[1] == int_head_box[3]:
        if int_head_box[1] == 0:
            int_head_box[3] = 1
        elif int_head_box[3] == resolution - 1:
            int_head_box[1] = resolution - 2
        elif abs(head_box[3] - int_head_box[3]) > abs(head_box[1] - int_head_box[1]):
            int_head_box[3] += 1
        else:
            int_head_box[1] -= 1
    head_box = int_head_box
    if coordconv:
        unit = np.array(range(0, resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit + i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution, resolution), dtype=np.float32)
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def draw_labelmap(img, pt, sigma, type="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    size = int(6 * sigma + 1)
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [ul[0] + size, ul[1] + size]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif type == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    # img = img / np.max(img)
    return to_torch(img)


def draw_labelmap_no_quant(img, pt, sigma, type="Gaussian"):
    img = to_numpy(img)
    shape = img.shape
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dist_matrix = (yy - float(pt[0])) ** 2 + (xx - float(pt[1])) ** 2
    if type == "Gaussian":
        g = np.exp(-dist_matrix / (2 * sigma**2))
    elif type == "Cauchy":
        g = sigma / ((dist_matrix + sigma**2) ** 1.5)
    g[dist_matrix > 10 * sigma**2] = 0
    img += g
    # img = img / np.max(img)
    return to_torch(img)


def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * float(w), p[1] * float(h)])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1
    return target_map


def get_cone(tgt, src, wh, theta=150):
    eye = src * wh
    gaze = tgt * wh

    pixel_mat = np.stack(
        np.meshgrid(np.arange(wh[0]), np.arange(wh[1])),
        -1,
    )

    dot_prod = np.sum((pixel_mat - eye) * (gaze - eye), axis=-1)
    gaze_vector_norm = np.sqrt(np.sum((gaze - eye) ** 2))
    pixel_mat_norm = np.sqrt(np.sum((pixel_mat - eye) ** 2, axis=-1))

    gaze_cones = dot_prod / (gaze_vector_norm * pixel_mat_norm)
    gaze_cones = np.nan_to_num(gaze_cones, nan=1)

    theta = theta * (np.pi / 180)
    beta = np.arccos(gaze_cones)
    # Create mask where true if beta is less than theta/2
    pixel_mat_presence = beta < (theta / 2)

    # Zero out values outside the gaze cone
    gaze_cones[~pixel_mat_presence] = 0
    gaze_cones = np.clip(gaze_cones, 0, None)

    return torch.from_numpy(gaze_cones).unsqueeze(0).float()


def get_transform(
    input_resolution: int, mean: Tuple[int, int, int], std: Tuple[int, int, int]
):
    return transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def smooth_by_conv(window_size, df, col):
    padded_track = pd.concat(
        [
            pd.DataFrame([[df.iloc[0][col]]] * (window_size // 2), columns=[0]),
            df[col],
            pd.DataFrame([[df.iloc[-1][col]]] * (window_size // 2), columns=[0]),
        ]
    )
    smoothed_signals = np.convolve(
        padded_track.squeeze(), np.ones(window_size) / window_size, mode="valid"
    )
    return smoothed_signals

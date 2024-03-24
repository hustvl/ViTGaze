from typing import Union, Iterable, Tuple
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(
            np.reshape(onehot_im, onehot_im.size), np.reshape(heatmap, heatmap.size)
        )
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float, idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * w.float(), p[1] * h.float()])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1
    return target_map


def inverse_transform(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().permute(0, 2, 3, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std + mean
    return cv2.cvtColor((tensor.numpy() * 255).astype(np.uint8)[0], cv2.COLOR_RGB2BGR)


def draw(data, heatmap, out_path, on_img=True):
    img = inverse_transform(data["images"])
    head_channel = cv2.applyColorMap(
        (data["head_channels"].squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_BONE,
    )
    hm = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = hm
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    if on_img:
        img = cv2.addWeighted(img, 1, heatmap, 0.5, 1)
    else:
        img = heatmap
    # img = cv2.addWeighted(img, 1, head_channel, 0.1, 1)
    cv2.imwrite(out_path, img)


def draw_origin_img(data, out_path):
    img = inverse_transform(data["images"])
    hm = cv2.applyColorMap(
        (data["heatmaps"].squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    hm[data["heatmaps"].squeeze().detach().cpu().numpy() == 0] = 0
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    head_channel = cv2.applyColorMap(
        (data["head_channels"].squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_BONE,
    )
    head_channel[data["head_channels"].squeeze().detach().cpu().numpy() < 0.1] = 0
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    ori = cv2.addWeighted(img, 1, hm, 0.5, 1)
    ori = cv2.addWeighted(ori, 1, head_channel, 0.1, 1)
    cv2.imwrite(out_path, ori)


class __Image2MP4:
    def __init__(self):
        self.Fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def __call__(
        self,
        frames: Union[Iterable[np.ndarray], str],
        path: str,
        fps: float = 30.0,
        isize: Tuple[int, int] = None,
    ):
        if isinstance(frames, str):  # directory of img files
            from os import listdir, path as osp

            imgs = sorted(listdir(frames))
            frames = [
                cv2.imread(osp.join(frames, img), cv2.IMREAD_COLOR) for img in imgs
            ]

        if isize is None:
            isize = (frames[0].shape[1], frames[0].shape[0])

        output_video = cv2.VideoWriter(path, self.Fourcc, fps, isize)
        for frame in frames:
            frame = cv2.resize(frame, isize)
            output_video.write(frame)
        output_video.release()


img2mp4 = __Image2MP4()


def dark_inference(heatmap: np.ndarray, gaussian_kernel: int = 39):
    pred_x, pred_y = argmax_pts(heatmap)
    pred_x, pred_y = int(pred_x), int(pred_y)
    height, width = heatmap.shape[-2:]
    # Gaussian blur
    orig_max = heatmap.max()
    border = (gaussian_kernel - 1) // 2
    dr = np.zeros((height + 2 * border, width + 2 * border))
    dr[border:-border, border:-border] = heatmap.copy()
    dr = cv2.GaussianBlur(dr, (gaussian_kernel, gaussian_kernel), 0)
    heatmap = dr[border:-border, border:-border].copy()
    heatmap *= orig_max / np.max(heatmap)
    # Log-likelihood
    heatmap = np.maximum(heatmap, 1e-10)
    heatmap = np.log(heatmap)
    # DARK
    if 1 < pred_x < width - 2 and 1 < pred_y < height - 2:
        dx = 0.5 * (heatmap[pred_y][pred_x + 1] - heatmap[pred_y][pred_x - 1])
        dy = 0.5 * (heatmap[pred_y + 1][pred_x] - heatmap[pred_y - 1][pred_x])
        dxx = 0.25 * (
            heatmap[pred_y][pred_x + 2]
            - 2 * heatmap[pred_y][pred_x]
            + heatmap[pred_y][pred_x - 2]
        )
        dxy = 0.25 * (
            heatmap[pred_y + 1][pred_x + 1]
            - heatmap[pred_y - 1][pred_x + 1]
            - heatmap[pred_y + 1][pred_x - 1]
            + heatmap[pred_y - 1][pred_x - 1]
        )
        dyy = 0.25 * (
            heatmap[pred_y + 2][pred_x]
            - 2 * heatmap[pred_y][pred_x]
            + heatmap[pred_y - 2][pred_x]
        )
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset_x, offset_y = np.squeeze(np.array(offset.T), axis=0)
            pred_x += offset_x
            pred_y += offset_y
    return pred_x, pred_y

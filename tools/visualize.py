import sys
import os
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *


warnings.simplefilter(action="ignore", category=FutureWarning)


@torch.no_grad()
def test_plot(model, dataloader):
    for i, data in enumerate(dataloader, start=1):
        val_gaze_heatmap_pred, _ = model(data)
        val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()

        # remove padding and recover valid ground truth points
        valid_gaze = data["gazes"][0]
        valid_gaze = valid_gaze[valid_gaze != -1].view(-1, 2)
        # AUC: area under curve of ROC
        multi_hot = multi_hot_targets(data["gazes"][0], data["imsize"][0])
        pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[0])
        norm_p = [
            pred_x / val_gaze_heatmap_pred[0].shape[-2],
            pred_y / val_gaze_heatmap_pred[0].shape[-1],
        ]
        scaled_heatmap = np.array(
            Image.fromarray(val_gaze_heatmap_pred[0]).resize(
                data["imsize"][0],
                resample=Image.BILINEAR,
            )
        )
        auc_score = auc(scaled_heatmap, multi_hot)
        # min distance: minimum among all possible pairs of <ground truth point, predicted point>
        all_distances = []
        for gt_gaze in valid_gaze:
            all_distances.append(L2_dist(gt_gaze, norm_p))
        min_dist = min(all_distances)
        # average distance: distance between the predicted point and human average point
        mean_gt_gaze = torch.mean(valid_gaze, 0)
        avg_dist = L2_dist(mean_gt_gaze, norm_p)
        good_case = auc_score > 0.995 and min_dist < 0.005
        bad_case = auc_score < 0.5 or min_dist > 0.4
        if good_case or bad_case:
            root = "vis_output/good_cases/" if good_case else "vis_output/bad_cases/"
            print(f"{i}: {auc_score}\t{min_dist}\t{avg_dist}")
            os.makedirs(f"{root}{i}", exist_ok=True)
            draw_origin_img(data, f"{root}{i}/origin.png")
            draw(data, scaled_heatmap, f"{root}{i}/result.png")
            # normalize heatmap to highlight the peak
            # useful for checking whether the model has learned to focus on the right region
            # not for quantitative evaluation
            scaled_heatmap = (
                scaled_heatmap - scaled_heatmap.min()
            ) / scaled_heatmap.ptp()
            draw(data, scaled_heatmap, f"{root}{i}/normed_result.png")
            out_dict = model.forward_backbone(data)
            attention_maps = [
                attn_map
                for attn_map in out_dict["attention_maps"][0].cpu().detach().numpy()
            ]
            for a_i, attn_map in enumerate(attention_maps):
                attn_map = np.array(
                    Image.fromarray(attn_map).resize(
                        data["imsize"][0], resample=Image.BILINEAR
                    )
                )
                attn_map = (attn_map - attn_map.min()) / attn_map.ptp()
                draw(data, attn_map, f"{root}{i}/amap{a_i}.png")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.nn.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights)["model"])
    model.to(cfg.train.device).train(False)
    cfg.dataloader.val.batch_size = 1
    dataloader = instantiate(cfg.dataloader.val)
    test_plot(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="config file",
    )
    parser.add_argument(
        "--model_weights",
        "-w",
        type=str,
        help="model weights",
    )
    args = parser.parse_args()
    main(args)

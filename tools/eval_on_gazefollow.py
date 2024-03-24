import sys
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


def do_test(cfg, model, use_dark_inference=False):
    val_loader = instantiate(cfg.dataloader.val)

    model.train(False)
    AUC = []
    min_dist = []
    avg_dist = []
    with torch.no_grad():
        for data in val_loader:
            val_gaze_heatmap_pred, _ = model(data)
            val_gaze_heatmap_pred = (
                val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )

            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(val_gaze_heatmap_pred)):
                # remove padding and recover valid ground truth points
                valid_gaze = data["gazes"][b_i]
                valid_gaze = valid_gaze[valid_gaze != -1].view(-1, 2)
                # AUC: area under curve of ROC
                multi_hot = multi_hot_targets(data["gazes"][b_i], data["imsize"][b_i])
                if use_dark_inference:
                    pred_x, pred_y = dark_inference(val_gaze_heatmap_pred[b_i])
                else:
                    pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[b_i])
                norm_p = [
                    pred_x / val_gaze_heatmap_pred[b_i].shape[-2],
                    pred_y / val_gaze_heatmap_pred[b_i].shape[-1],
                ]
                scaled_heatmap = np.array(
                    Image.fromarray(val_gaze_heatmap_pred[b_i]).resize(
                        data["imsize"][b_i],
                        resample=Image.BILINEAR,
                    )
                )
                auc_score = auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)
                # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                all_distances = []
                for gt_gaze in valid_gaze:
                    all_distances.append(L2_dist(gt_gaze, norm_p))
                min_dist.append(min(all_distances))
                # average distance: distance between the predicted point and human average point
                mean_gt_gaze = torch.mean(valid_gaze, 0)
                avg_distance = L2_dist(mean_gt_gaze, norm_p)
                avg_dist.append(avg_distance)

    print("|AUC   |min dist|avg dist|")
    print(
        "|{:.4f}|{:.4f}  |{:.4f}  |".format(
            torch.mean(torch.tensor(AUC)),
            torch.mean(torch.tensor(min_dist)),
            torch.mean(torch.tensor(avg_dist)),
        )
    )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights)["model"])
    model.to(cfg.train.device)
    do_test(cfg, model, use_dark_inference=args.use_dark_inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="model weights",
    )
    parser.add_argument("--use_dark_inference", action="store_true")
    args = parser.parse_args()
    main(args)

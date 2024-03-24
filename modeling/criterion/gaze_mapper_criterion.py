from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit


class GazeMapperCriterion(nn.Module):
    def __init__(
        self,
        heatmap_weight: float = 10000,
        inout_weight: float = 100,
        aux_weight: float = 100,
        use_aux_loss: bool = False,
        aux_head_thres: float = 0,
        use_focal_loss: bool = False,
        alpha: float = -1,
        gamma: float = 2,
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.inout_weight = inout_weight
        self.aux_weight = aux_weight
        self.aux_head_thres = aux_head_thres

        self.heatmap_loss = nn.MSELoss(reduce=False)

        if use_focal_loss:
            self.inout_loss = partial(
                sigmoid_focal_loss_jit, alpha=alpha, gamma=gamma, reduction="mean"
            )
        else:
            self.inout_loss = nn.BCEWithLogitsLoss()

        if use_aux_loss:
            self.aux_loss = nn.BCEWithLogitsLoss()
        else:
            self.aux_loss = None

    def forward(
        self,
        pred_heatmap,
        pred_inout,
        gt_heatmap,
        gt_inout,
        pred_head_masks=None,
        gt_head_masks=None,
    ):
        loss_dict = {}

        pred_heatmap = F.interpolate(
            pred_heatmap,
            size=tuple(gt_heatmap.shape[-2:]),
            mode="bilinear",
            align_corners=True,
        )
        heatmap_loss = (
            self.heatmap_loss(pred_heatmap.squeeze(1), gt_heatmap) * self.heatmap_weight
        )
        heatmap_loss = torch.mean(heatmap_loss, dim=(-2, -1))
        heatmap_loss = torch.sum(heatmap_loss.reshape(-1) * gt_inout.reshape(-1))
        # Check whether all outside, avoid 0/0 to be nan
        if heatmap_loss > 1e-7:
            heatmap_loss = heatmap_loss / torch.sum(gt_inout)
            loss_dict["regression loss"] = heatmap_loss
        else:
            loss_dict["regression loss"] = heatmap_loss * 0

        inout_loss = (
            self.inout_loss(pred_inout.reshape(-1), gt_inout.reshape(-1))
            * self.inout_weight
        )
        loss_dict["classification loss"] = inout_loss

        if self.aux_loss is not None:
            pred_head_masks = F.interpolate(
                pred_head_masks,
                size=tuple(gt_head_masks.shape[-2:]),
                mode="bilinear",
                align_corners=True,
            )
            aux_loss = (
                torch.clamp(
                    self.aux_loss(
                        pred_head_masks.reshape(-1), gt_head_masks.reshape(-1)
                    )
                    - self.aux_head_thres,
                    0,
                )
                * self.aux_weight
            )
            loss_dict["aux head loss"] = aux_loss

        return loss_dict

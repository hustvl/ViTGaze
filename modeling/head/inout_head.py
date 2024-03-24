from typing import OrderedDict
import torch
from torch import nn
from detectron2.utils.registry import Registry


INOUT_HEAD_REGISTRY = Registry("INOUT_HEAD_REGISTRY")
INOUT_HEAD_REGISTRY.__doc__ = "Registry for inout head"


@INOUT_HEAD_REGISTRY.register()
class SimpleLinear(nn.Module):
    def __init__(self, in_channel: int, dropout: float = 0) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.classifier = nn.Sequential(
            OrderedDict(
                [("dropout", nn.Dropout(dropout)), ("linear", nn.Linear(in_channel, 1))]
            )
        )

    def get_feat(self, x, masks):
        feats = x["head_feat"]
        if masks is not None:
            B, C = x["last_feat"].shape[:2]
            scene_feats = x["last_feat"].view(B, C, -1).permute(0, 2, 1)
            masks = masks / (masks.sum(dim=-1, keepdim=True) + 1e-6)
            scene_feats = (scene_feats * masks.unsqueeze(-1)).sum(dim=1)
            feats = torch.cat((feats, scene_feats), dim=1)
        return feats

    def forward(self, x, masks=None):
        feat = self.get_feat(x, masks)
        return self.classifier(feat)


@INOUT_HEAD_REGISTRY.register()
class SimpleMlp(SimpleLinear):
    def __init__(self, in_channel: int, dropout: float = 0) -> None:
        super().__init__(in_channel, dropout)
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("dropout0", nn.Dropout(dropout)),
                    ("linear0", nn.Linear(in_channel, in_channel)),
                    ("relu", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("linear1", nn.Linear(in_channel, 1)),
                ]
            )
        )


def build_inout_head(name, *args, **kwargs):
    return INOUT_HEAD_REGISTRY.get(name)(*args, **kwargs)

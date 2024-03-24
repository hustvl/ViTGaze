import torch
from torch import nn
from detectron2.utils.registry import Registry
from typing import Literal, List, Dict, Optional, OrderedDict


HEATMAP_HEAD_REGISTRY = Registry("HEATMAP_HEAD_REGISTRY")
HEATMAP_HEAD_REGISTRY.__doc__ = "Registry for heatmap head"


class BaseHeatmapHead(nn.Module):
    def __init__(
        self,
        in_channel: int,
        deconv_cfgs: List[Dict],
        dim: int = 96,
        use_conv: bool = False,
        use_residual: bool = False,
        feat_type: Literal["attn", "both"] = "both",
        attn_layer: Optional[str] = None,
        pre_norm: bool = False,
        use_head: bool = False,
    ) -> None:
        super().__init__()
        self.feat_type = feat_type
        self.use_head = use_head

        if pre_norm:
            self.pre_norm = nn.Sequential(
                OrderedDict(
                    [
                        ("bn", nn.BatchNorm2d(in_channel)),
                        ("relu", nn.ReLU(inplace=True)),
                    ]
                )
            )
        else:
            self.pre_norm = nn.Identity()

        if use_conv:
            if use_residual:
                from timm.models.resnet import Bottleneck, downsample_conv

                self.conv = Bottleneck(
                    in_channel,
                    dim // 4,
                    downsample=downsample_conv(in_channel, dim, 1)
                    if in_channel != dim
                    else None,
                    attn_layer=attn_layer,
                )
            else:
                self.conv = nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channel, dim, 3, 1, 1)),
                            ("bn", nn.BatchNorm2d(dim)),
                            ("relu", nn.ReLU(inplace=True)),
                        ]
                    )
                )
        else:
            self.conv = nn.Identity()

        self.decoder: nn.Module = None

    def get_feat(self, x):
        if self.feat_type == "attn":
            feat = x["attention_maps"]
        elif self.feat_type == "feat":
            feat = x["last_feat"]
        return feat

    def forward(self, x):
        feat = self.get_feat(x)
        feat = self.pre_norm(feat)
        feat = self.conv(feat)
        return self.decoder(feat)


@HEATMAP_HEAD_REGISTRY.register()
class SimpleDeconv(BaseHeatmapHead):
    def __init__(
        self,
        in_channel: int,
        deconv_cfgs: List[Dict],
        dim: int = 96,
        use_conv: bool = False,
        use_residual: bool = False,
        feat_type: Literal["attn", "both"] = "both",
        attn_layer: Optional[str] = None,
        pre_norm: bool = False,
        use_head: bool = False,
    ) -> None:
        super().__init__(
            in_channel,
            deconv_cfgs,
            dim,
            use_conv,
            use_residual,
            feat_type,
            attn_layer,
            pre_norm,
            use_head,
        )
        decoder_layers = []
        for i, deconv_cfg in enumerate(deconv_cfgs, start=1):
            decoder_layers.extend(
                [
                    (
                        "".join(["deconv", str(i)]),
                        nn.ConvTranspose2d(**deconv_cfg),
                    ),
                    (
                        "".join(["bn", str(i)]),
                        nn.BatchNorm2d(deconv_cfg["out_channels"]),
                    ),
                    ("".join(["relu", str(i)]), nn.ReLU(inplace=True)),
                ]
            )
        decoder_layers.append(("conv", nn.Conv2d(1, 1, 1)))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))


@HEATMAP_HEAD_REGISTRY.register()
class UpSampleConv(BaseHeatmapHead):
    def __init__(
        self,
        in_channel: int,
        deconv_cfgs: List[Dict],
        dim: int = 96,
        use_conv: bool = False,
        use_residual: bool = False,
        feat_type: Literal["attn", "both"] = "both",
        attn_layer: Optional[str] = None,
        pre_norm: bool = False,
        use_head: bool = False,
    ) -> None:
        super().__init__(
            in_channel,
            deconv_cfgs,
            dim,
            use_conv,
            use_residual,
            feat_type,
            attn_layer,
            pre_norm,
            use_head,
        )
        decoder_layers = []
        for i, deconv_cfg in enumerate(deconv_cfgs, start=1):
            decoder_layers.extend(
                [
                    (
                        "".join(["upsample", str(i)]),
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "".join(["conv", str(i)]),
                        nn.Conv2d(**deconv_cfg),
                    ),
                    (
                        "".join(["bn", str(i)]),
                        nn.BatchNorm2d(deconv_cfg["out_channels"]),
                    ),
                    ("".join(["relu", str(i)]), nn.ReLU(inplace=True)),
                ]
            )
        decoder_layers.append(("conv", nn.Conv2d(1, 1, 1)))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))


@HEATMAP_HEAD_REGISTRY.register()
class PixelShuffle(BaseHeatmapHead):
    def __init__(
        self,
        in_channel: int,
        deconv_cfgs: List[Dict],
        dim: int = 96,
        use_conv: bool = False,
        use_residual: bool = False,
        feat_type: Literal["attn", "both"] = "both",
        attn_layer: Optional[str] = None,
        pre_norm: bool = False,
        use_head: bool = False,
    ) -> None:
        super().__init__(
            in_channel,
            deconv_cfgs,
            dim,
            use_conv,
            use_residual,
            feat_type,
            attn_layer,
            pre_norm,
            use_head,
        )
        decoder_layers = []
        for i, deconv_cfg in enumerate(deconv_cfgs, start=1):
            deconv_cfg["out_channels"] = deconv_cfg["out_channels"] * 4
            decoder_layers.extend(
                [
                    (
                        "".join(["conv", str(i)]),
                        nn.Conv2d(**deconv_cfg),
                    ),
                    (
                        "".join(["pixel_shuffle", str(i)]),
                        nn.PixelShuffle(upscale_factor=2),
                    ),
                    (
                        "".join(["bn", str(i)]),
                        nn.BatchNorm2d(deconv_cfg["out_channels"] // 4),
                    ),
                    ("".join(["relu", str(i)]), nn.ReLU(inplace=True)),
                ]
            )
        decoder_layers.append(("conv", nn.Conv2d(1, 1, 1)))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))


def build_heatmap_head(name, *args, **kwargs):
    return HEATMAP_HEAD_REGISTRY.get(name)(*args, **kwargs)

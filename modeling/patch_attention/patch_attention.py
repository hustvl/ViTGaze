from typing import OrderedDict
from torch import nn
from torch.nn import functional as F
from detectron2.utils.registry import Registry
from fvcore.nn import c2_msra_fill


SPATIAL_GUIDANCE_REGISTRY = Registry("SPATIAL_GUIDANCE_REGISTRY")
SPATIAL_GUIDANCE_REGISTRY.__doc__ = "Registry for 2d spatial guidance"


class _PoolFusion(nn.Module):
    def __init__(self, patch_size: int, use_avgpool: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.attn_reducer = F.avg_pool2d if use_avgpool else F.max_pool2d

    def forward(self, scenes, heads):
        attn_masks = self.attn_reducer(
            heads,
            (self.patch_size, self.patch_size),
            (self.patch_size, self.patch_size),
            (0, 0),
        )
        patch_attn = attn_masks.masked_fill(attn_masks <= 0, -1e9)
        return F.softmax(patch_attn.view(len(patch_attn), -1), dim=1).view(
            *patch_attn.shape
        )


@SPATIAL_GUIDANCE_REGISTRY.register()
class AvgFusion(_PoolFusion):
    def __init__(self, patch_size: int) -> None:
        super().__init__(patch_size, False)


@SPATIAL_GUIDANCE_REGISTRY.register()
class MaxFusion(_PoolFusion):
    def __init__(self, patch_size: int) -> None:
        super().__init__(patch_size, True)


@SPATIAL_GUIDANCE_REGISTRY.register()
class PatchPAM(nn.Module):
    def __init__(
        self,
        patch_size: int,
        act_layer=nn.ReLU,
        embed_dim: int = 768,
        use_aux_loss: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        patch_embed = nn.Conv2d(
            3, embed_dim, (patch_size, patch_size), (patch_size, patch_size), (0, 0)
        )
        c2_msra_fill(patch_embed)
        conv = nn.Conv2d(embed_dim, 1, (1, 1), (1, 1), (0, 0))
        c2_msra_fill(conv)
        self.use_aux_loss = use_aux_loss
        if use_aux_loss:
            self.patch_embed = nn.Sequential(
                OrderedDict(
                    [
                        ("patch_embed", patch_embed),
                        ("act_layer", act_layer(inplace=True)),
                    ]
                )
            )
            self.embed = conv
            conv = nn.Conv2d(embed_dim, 1, (1, 1), (1, 1), (0, 0))
            c2_msra_fill(conv)
            self.aux_embed = conv
        else:
            self.embed = nn.Sequential(
                OrderedDict(
                    [
                        ("patch_embed", patch_embed),
                        ("act_layer", act_layer(inplace=True)),
                        ("embed", conv),
                    ]
                )
            )

    def forward(self, scenes, heads):
        attn_masks = F.max_pool2d(
            heads,
            (self.patch_size, self.patch_size),
            (self.patch_size, self.patch_size),
            (0, 0),
        )
        if self.use_aux_loss:
            embed = self.patch_embed(scenes)
            aux_masks = self.aux_embed(embed)
            patch_attn = self.embed(embed) * attn_masks
        else:
            patch_attn = self.embed(scenes) * attn_masks
        patch_attn = patch_attn.masked_fill(attn_masks <= 0, -1e9)
        patch_attn = F.softmax(patch_attn.view(len(patch_attn), -1), dim=1).view(
            *patch_attn.shape
        )
        return (patch_attn, aux_masks) if self.use_aux_loss else patch_attn


def build_pam(name, *args, **kwargs):
    return SPATIAL_GUIDANCE_REGISTRY.get(name)(*args, **kwargs)

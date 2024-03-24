import logging
from typing import Literal, Union
from functools import partial
import torch
import torch.nn as nn
from detectron2.modeling import Backbone

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_ON = True
except ImportError:
    XFORMERS_ON = False
from .utils import (
    PatchEmbed,
    get_abs_pos,
    DropPath,
    Mlp,
)


logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        return_softmax_attn=True,
        use_proj=True,
        patch_token_offset=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim) if use_proj else nn.Identity()

        self.return_softmax_attn = return_softmax_attn

        self.patch_token_offset = patch_token_offset

    def forward(self, x, return_attention=False, extra_token_offset=None):
        B, L, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, L, -1).unbind(0)

        if return_attention or not XFORMERS_ON:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if return_attention and not self.return_softmax_attn:
                out_attn = attn
            attn = attn.softmax(dim=-1)
            if return_attention and self.return_softmax_attn:
                out_attn = attn
            x = attn @ v
        else:
            x = memory_efficient_attention(q, k, v, scale=self.scale)

        x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)
        x = self.proj(x)

        if return_attention:
            out_attn = out_attn.reshape(B, self.num_heads, L, -1)
            out_attn = out_attn[
                :,
                :,
                self.patch_token_offset : extra_token_offset,
                self.patch_token_offset : extra_token_offset,
            ]
            return x, out_attn
        else:
            return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        init_values=None,
        return_softmax_attn=True,
        attention_map_only=False,
        patch_token_offset=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.attention_map_only = attention_map_only
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            return_softmax_attn=return_softmax_attn,
            use_proj=return_softmax_attn or not attention_map_only,
            patch_token_offset=patch_token_offset,
        )

        if attention_map_only:
            return

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x, return_attention=False, extra_token_offset=None):
        shortcut = x
        x = self.norm1(x)

        if return_attention:
            x, attn = self.attn(x, True, extra_token_offset)
        else:
            x = self.attn(x)

        if self.attention_map_only:
            return x, attn

        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn
        else:
            return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ViT(Backbone):
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        init_values=None,
        use_cls_token=False,
        use_mask_token=False,
        norm_features=False,
        return_softmax_attn=True,
        num_register_tokens=0,
        num_msg_tokens=0,
        register_as_msg=False,
        shift_strides=None,  # [1, -1, 2, -2],
        cls_shift=False,
        num_extra_tokens=4,
        use_extra_embed=False,
        num_frames=None,
        out_feature=True,
        out_attn=(),
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

        self.use_cls_token = use_cls_token
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        )

        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )

        # We tried to leverage temporal information with TeViT while it doesn't work
        assert num_msg_tokens >= 0
        self.num_msg_tokens = num_msg_tokens
        if register_as_msg:
            self.num_msg_tokens += num_register_tokens
        self.msg_tokens = (
            nn.Parameter(torch.zeros(1, num_msg_tokens, embed_dim))
            if num_msg_tokens > 0
            else None
        )

        patch_token_offset = (
            num_msg_tokens + num_register_tokens + int(self.use_cls_token)
        )
        self.patch_token_offset = patch_token_offset

        self.msg_shift = None
        if shift_strides is not None:
            self.msg_shift = []
            for i in range(depth):
                if i % 2 == 0:
                    self.msg_shift.append([_ for _ in shift_strides])
                else:
                    self.msg_shift.append([-_ for _ in shift_strides])

        self.cls_shift = None
        if cls_shift:
            self.cls_shift = [(-1) ** idx for idx in range(depth)]

        assert num_extra_tokens >= 0
        self.num_extra_tokens = num_extra_tokens
        self.extra_pos_embed = (
            nn.Linear(embed_dim, embed_dim)
            if num_extra_tokens > 0 and use_extra_embed
            else nn.Identity()
        )

        self.num_frames = num_frames

        # Mask token for masking augmentation
        self.use_mask_token = use_mask_token
        self.mask_token = (
            nn.Parameter(torch.zeros(1, embed_dim)) if use_mask_token else None
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                init_values=init_values,
                return_softmax_attn=return_softmax_attn,
                attention_map_only=(i == depth - 1) and not out_feature,
                patch_token_offset=patch_token_offset,
            )
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim) if norm_features else nn.Identity()

        self._out_features = out_feature
        self._out_attn = out_attn

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks=None, guidance=None):
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        B, H, W, _ = x.shape
        x = x.reshape(B, H * W, -1)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(len(x), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(len(x), -1, -1)
            x = torch.cat((register_tokens, x), dim=1)

        if self.msg_tokens is not None:
            msg_tokens = self.msg_tokens.expand(len(x), -1, -1)
            x = torch.cat((msg_tokens, x), dim=1)
        # [MSG, REG, CLS, PAT]

        extra_tokens_offset = None
        if guidance is not None:
            guidance = guidance.reshape(len(guidance), -1, 1)
            extra_tokens = (
                (x[:, self.patch_token_offset :] * guidance)
                .sum(dim=1, keepdim=True)
                .expand(-1, self.num_extra_tokens, -1)
            )
            extra_tokens = self.extra_pos_embed(extra_tokens)
            x = torch.cat((x, extra_tokens), dim=1)
            extra_tokens_offset = -self.num_extra_tokens
        # [MSG, REG, CLS, PAT, EXT]

        attn_maps = []
        for idx, blk in enumerate(self.blocks):
            if idx in self._out_attn:
                x, attn = blk(x, True, extra_tokens_offset)
                attn_maps.append(attn)
            else:
                x = blk(x)

            if self.msg_shift is not None:
                msg_shift = self.msg_shift[idx]
                msg_tokens = (
                    x[:, : self.num_msg_tokens]
                    if guidance is None
                    else x[:, extra_tokens_offset:]
                )
                msg_tokens = msg_tokens.reshape(
                    -1, self.num_frames, *msg_tokens.shape[1:]
                )
                msg_tokens = msg_tokens.chunk(len(msg_shift), dim=2)
                msg_tokens = [
                    torch.roll(tokens, roll, dims=1)
                    for tokens, roll in zip(msg_tokens, msg_shift)
                ]
                msg_tokens = torch.cat(msg_tokens, dim=2).flatten(0, 1)
                if guidance is None:
                    x = torch.cat([msg_tokens, x[:, self.num_msg_tokens :]], dim=1)
                else:
                    x = torch.cat([x[:, :extra_tokens_offset], msg_tokens], dim=1)

            if self.cls_shift is not None:
                cls_tokens = x[:, self.patch_token_offset - 1]
                cls_tokens = cls_tokens.reshape(
                    -1, self.num_frames, 1, *cls_tokens.shape[1:]
                )
                cls_tokens = torch.roll(cls_tokens, self.cls_shift[idx], dims=1)
                x = torch.cat(
                    [
                        x[:, : self.patch_token_offset - 1],
                        cls_tokens.flatten(0, 1),
                        x[:, self.patch_token_offset :],
                    ],
                    dim=1,
                )

        x = self.norm(x)

        outputs = {}
        outputs["attention_maps"] = torch.cat(attn_maps, dim=1).reshape(
            B, -1, H * W, H, W
        )
        if self._out_features:
            outputs["last_feat"] = (
                x[:, self.patch_token_offset : extra_tokens_offset]
                .reshape(B, H, W, -1)
                .permute(0, 3, 1, 2)
            )

        return outputs


def vit_tiny(**kwargs):
    model = ViT(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(**kwargs):
    model = ViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(**kwargs):
    model = ViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def dinov2_base(**kwargs):
    model = ViT(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrain_img_size=518,
        init_values=1,
        **kwargs
    )
    return model


def dinov2_small(**kwargs):
    model = ViT(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrain_img_size=518,
        init_values=1,
        **kwargs
    )
    return model


def build_backbone(
    name: Literal["tiny", "small", "base", "dinov2_base", "dinov2_small"], **kwargs
):
    vit_dict = {
        "tiny": vit_tiny,
        "small": vit_small,
        "base": vit_base,
        "dinov2_base": dinov2_base,
        "dinov2_small": dinov2_small,
    }
    return vit_dict[name](**kwargs)

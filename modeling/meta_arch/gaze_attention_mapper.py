import torch
from torch import nn
from typing import Dict, Union


class GazeAttentionMapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        regressor: nn.Module,
        classifier: nn.Module,
        criterion: nn.Module,
        pam: nn.Module,
        use_aux_loss: bool = False,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pam = pam
        self.regressor = regressor
        self.classifier = classifier
        self.criterion = criterion
        self.use_aux_loss = use_aux_loss
        self.device = torch.device(device)

    def forward(self, x):
        (
            scenes,
            heads,
            gt_heatmaps,
            gt_inouts,
            head_masks,
            image_masks,
        ) = self.preprocess_inputs(x)
        # Calculate patch weights based on head position
        embedded_heads = self.pam(scenes, heads)
        aux_masks = None
        if self.use_aux_loss:
            embedded_heads, aux_masks = embedded_heads

        # Get out-dict
        x = self.backbone(
            scenes,
            image_masks,
            None,
        )

        # Apply patch weights to get the final feats and attention maps
        feats = x.get("last_feat", None)
        if feats is not None:
            x["head_feat"] = (
                (embedded_heads.repeat(1, feats.shape[1], 1, 1) * feats)
                .sum(dim=(2, 3))
                .reshape(len(feats), -1)
            )  # BC

        attn_maps = x["attention_maps"]
        B, C, *_ = attn_maps.shape
        x["attention_maps"] = (
            attn_maps * embedded_heads.reshape(B, 1, -1, 1, 1).repeat(1, C, 1, 1, 1)
        ).sum(
            dim=2
        )  # BCHW

        # Apply heads
        heatmaps = self.regressor(x)
        inouts = self.classifier(x, None)

        if self.training:
            return self.criterion(
                heatmaps,
                inouts,
                gt_heatmaps,
                gt_inouts,
                aux_masks,
                head_masks,
            )
        # Inference
        return heatmaps, inouts.sigmoid()

    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        return (
            batched_inputs["images"].to(self.device),
            batched_inputs["head_channels"].to(self.device),
            batched_inputs["heatmaps"].to(self.device)
            if "heatmaps" in batched_inputs.keys()
            else None,
            batched_inputs["gaze_inouts"].to(self.device)
            if "gaze_inouts" in batched_inputs.keys()
            else None,
            batched_inputs["head_masks"].to(self.device)
            if "head_masks" in batched_inputs.keys()
            else None,
            batched_inputs["image_masks"].to(self.device)
            if "image_masks" in batched_inputs.keys()
            else None,
        )

from detectron2 import model_zoo


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


class LRDecayRater:
    def __init__(self, lr_decay_rate=1.0, num_layers=12, backbone_multiplier=1.0, freeze_pe=False, pam_lr_decay=1):
        self.lr_decay_rate = lr_decay_rate
        self.num_layers = num_layers
        self.backbone_multiplier = backbone_multiplier
        self.freeze_pe = freeze_pe
        self.pam_lr_decay = pam_lr_decay

    def __call__(self, name):
        if name.startswith("backbone"):
            if self.freeze_pe and ".pos_embed" in name or ".patch_embed" in name:
                return 0
            return self.backbone_multiplier * get_vit_lr_decay_rate(
                name, self.lr_decay_rate, self.num_layers
            )
        if name.startswith("pam"):
            return self.pam_lr_decay
        return 1


# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = LRDecayRater(num_layers=12, lr_decay_rate=0.65)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

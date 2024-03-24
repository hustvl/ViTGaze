from detectron2.config import LazyCall as L

from modeling import backbone, patch_attention, meta_arch, head, criterion


model = L(meta_arch.GazeAttentionMapper)()
model.backbone = L(backbone.build_backbone)(
    name="small", out_attn=[2, 5, 8, 11]
)
model.pam = L(patch_attention.build_pam)(name="PatchPAM", patch_size=16)
model.regressor = L(head.build_heatmap_head)(
    name="SimpleDeconv",
    in_channel=24,
    deconv_cfgs=[
        {
            "in_channels": 24,
            "out_channels": 12,
            "kernel_size": 3,
            "stride": 2,
        },
        {
            "in_channels": 12,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 2,
        },
        {
            "in_channels": 6,
            "out_channels": 3,
            "kernel_size": 3,
            "stride": 2,
        },
        {
            "in_channels": 3,
            "out_channels": 1,
            "kernel_size": 3,
            "stride": 2,
        },
    ],
    feat_type="attn",
)
model.classifier = L(head.build_inout_head)(name="SimpleLinear", in_channel=384)
model.criterion = L(criterion.GazeMapperCriterion)()
model.device = "cuda"

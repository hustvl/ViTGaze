from .common.dataloader import dataloader
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.train import train
from os.path import join, basename
from torch.cuda import device_count


num_gpu = device_count()
ins_per_iter = 48
len_dataset = 126000
num_epoch = 14
# dataloader
dataloader = dataloader.gazefollow
dataloader.train.batch_size = ins_per_iter // num_gpu
dataloader.train.num_workers = dataloader.val.num_workers = 14
dataloader.train.distributed = num_gpu > 1
dataloader.train.rand_rotate = 0.5
dataloader.train.rand_lsj = 0.5
dataloader.train.input_size = dataloader.val.input_size = 434
dataloader.train.mask_scene = True
dataloader.train.mask_prob = 0.5
dataloader.train.mask_size = dataloader.train.input_size // 14
dataloader.train.max_scene_patches_ratio = 0.5
dataloader.val.batch_size = 64
dataloader.val.distributed = False
# train
train.init_checkpoint = "pretrained/dinov2_small.pth"
train.output_dir = join("./output", basename(__file__).split(".")[0])
train.max_iter = len_dataset * num_epoch // ins_per_iter
train.log_period = len_dataset // (ins_per_iter * 100)
train.checkpointer.max_to_keep = 10
train.checkpointer.period = len_dataset // ins_per_iter
train.seed = 0
# optimizer
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.99)
lr_multiplier.scheduler.typ = "cosine"
lr_multiplier.scheduler.start_value = 1
lr_multiplier.scheduler.end_value = 0.1
lr_multiplier.warmup_length = 1e-2
# model
model.use_aux_loss = model.pam.use_aux_loss = model.criterion.use_aux_loss = True
model.pam.name = "PatchPAM"
model.pam.embed_dim = 8
model.pam.patch_size = 14
model.backbone.name = "dinov2_small"
model.backbone.return_softmax_attn = True
model.backbone.out_attn = [2, 5, 8, 11]
model.backbone.use_cls_token = True
model.backbone.use_mask_token = True
model.regressor.name = "UpSampleConv"
model.regressor.in_channel = 24
model.regressor.use_conv = False
model.regressor.dim = 24
model.regressor.deconv_cfgs = [
    dict(
        in_channels=24,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
    ),
    dict(
        in_channels=16,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
    ),
    dict(
        in_channels=8,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
    ),
]
model.regressor.feat_type = "attn"
model.classifier.name = "SimpleMlp"
model.classifier.in_channel = 384
model.criterion.aux_weight = 100
model.criterion.aux_head_thres = 0.05
model.criterion.use_focal_loss = True
model.device = "cuda"

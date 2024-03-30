## Train

### Training Dataset

You should prepare GazeFollow and VideoAttentionTarget for training.

* Get [GazeFollow](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0).
* If train with auxiliary regression, use `scripts\gen_gazefollow_head_masks.py` to generate head masks.
* Get [VideoAttentionTarget](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0).

Check `ViTGaze/configs/common/dataloader` to modify DATA_ROOT.

### Pretrained Model

* Get [DINOv2](https://github.com/facebookresearch/dinov2) pretrained ViT-S.
* Or you could download and preprocess pretrained weights by

  ```
  cd ViTGaze
  mkdir pretrained && cd pretrained
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
  ```
* Preprocess the model weights with `scripts\convert_pth.py` to fit Detectron2 format.
### Train ViTGaze

You can modify configs in `configs/gazefollow.py`, `configs/gazefollow_518.py` and `configs/videoattentiontarget.py`.

Run:

```
    bash train.sh
```

to train ViTGaze on the two datasets.

Training output will be saved in `ViTGaze/output/`.

## Eval
### Testing Dataset

You should prepare GazeFollow and VideoAttentionTarget for training.

* Get [GazeFollow](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0).
* If train with auxiliary regression, use `scripts\gen_gazefollow_head_masks.py` to generate head masks.
* Get [VideoAttentionTarget](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0).

Check `ViTGaze/configs/common/dataloader` to modify DATA_ROOT.

### Evaluation

Run
```
bash val.sh configs/gazefollow_518.py ${Path2checkpoint} gf
```
to evaluate on GazeFollow.

Run
```
bash val.sh configs/videoattentiontarget.py ${Path2checkpoint} vat
```
to evaluate on VideoAttentionTarget.

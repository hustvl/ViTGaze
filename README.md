<div align="center">
<h1>ViTGaze 👀</h1>
<h3>Gaze Following with Interaction Features in Vision Transformers</h3>

[Yuehao Song](https://scholar.google.com/citations?user=7sqkA-MAAAAJ)<sup>1</sup> , [Xinggang Wang](https://xwcv.github.io)<sup>1,✉️</sup> , [Jingfeng Yao](https://scholar.google.com/citations?user=4qc1qJ0AAAAJ)<sup>1</sup> , [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup> , Jinglin Zhang<sup>2</sup> , Xiangmin Xu<sup>3</sup>

<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> Shandong University, <sup>3</sup> South China University of Technology, <sup>✉️</sup> corresponding author

Accepted by Visual Intelligence ([Paper](https://link.springer.com/article/10.1007/s44267-024-00064-9))

[![arxiv paper](https://img.shields.io/badge/arXiv-Preprint-red)](https://arxiv.org/abs/2403.12778) [![🤗HF models](https://img.shields.io/badge/HF%20🤗-Models-orange)](https://huggingface.co/yhsong/ViTGaze) [![PaperwithCode](https://img.shields.io/badge/Paperswithcode-blue)](https://huggingface.co/yhsong/ViTGaze)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitgaze-gaze-following-with-interaction/gaze-target-estimation-on-gazefollow)](https://paperswithcode.com/sota/gaze-target-estimation-on-gazefollow?p=vitgaze-gaze-following-with-interaction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitgaze-gaze-following-with-interaction/gaze-target-estimation-on)](https://paperswithcode.com/sota/gaze-target-estimation-on?p=vitgaze-gaze-following-with-interaction)

</div>

![Demo0](assets/demo0.gif)
![Demo1](assets/demo1.gif)
### News
* **`Nov. 21th, 2024`:** ViTGaze is accepted by Visual Intelligence! 🎉
* **`Mar. 25th, 2024`:** We release an initial version of ViTGaze.
* **`Mar. 19th, 2024`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Introduction
<div align="center"><h5>Plain Vision Transformer could also do gaze following with the simple ViTGaze framework!</h5></div>

![framework](assets/pipeline.png "framework")

Inspired by the remarkable success of pre-trained plain Vision Transformers (ViTs), we introduce a novel single-modality gaze following framework, **ViTGaze**. In contrast to previous methods, it creates a brand new gaze following framework based mainly on powerful encoders (relative decoder parameter less than 1%). Our principal insight lies in that the inter-token interactions within self-attention can be transferred to interactions between humans and scenes. Our method achieves state-of-the-art (SOTA) performance among all single-modality methods (3.4% improvement on AUC, 5.1% improvement on AP) and very comparable performance against multi-modality methods with 59% number of parameters less.

## Results
> Results from the [ViTGaze paper](https://link.springer.com/article/10.1007/s44267-024-00064-9)

![comparison](assets/comparion.png "comparison")

<table align="center">
  <tr>
    <th colspan="3">Results on <a herf=http://gazefollow.csail.mit.edu/index.html>GazeFollow</a></th>
    <th colspan="3">Results on <a herf=https://github.com/ejcgt/attention-target-detection>VideoAttentionTarget</a></th>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td><b>Avg. Dist.</b></td>
    <td><b>Min. Dist.</b></td>
    <td><b>AUC</b></td>
    <td><b>Dist.</b></td>
    <td><b>AP</b></td>
  </tr>
  <tr>
    <td>0.949</td>
    <td>0.105</td>
    <td>0.047</td>
    <td>0.938</td>
    <td>0.102</td>
    <td>0.905</td>
  </tr>
</table>

Corresponding checkpoints are released:
- GazeFollow: [GoogleDrive](https://drive.google.com/file/d/164c4woGCmUI8UrM7GEKQrV1FbA3vGwP4/view?usp=drive_link)
- VideoAttentionTarget: [GoogleDrive](https://drive.google.com/file/d/11_O4Jm5wsvQ8qfLLgTlrudqSNvvepsV0/view?usp=drive_link)
## Getting Started
- [Installation](docs/install.md)
- [Train](docs/train.md)
- [Eval](docs/eval.md)

## Acknowledgements
ViTGaze is based on [detectron2](https://github.com/facebookresearch/detectron2). We use the efficient multi-head attention implemented in the [xFormers](https://github.com/facebookresearch/xformers) library.

## Citation
If you find ViTGaze is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@article{song2024vitgaze,
  title   = {ViTGaze: Gaze Following with Interaction Features in Vision Transformers},
  author  = {Song, Yuehao and Wang, Xinggang and Yao, Jingfeng and Liu, Wenyu and Zhang, Jinglin and Xu, Xiangmin},
  journal = {Visual Intelligence},
  volume  = {2},
  number  = {31},
  year    = {2024},
  url     = {https://doi.org/10.1007/s44267-024-00064-9}
}
```

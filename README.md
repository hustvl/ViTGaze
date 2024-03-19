<div align="center">
<h1>ViTGaze</h1>
<h3>Gaze Following with Interaction Features in Vision Transformers</h3>

Yuehao Song<sup>1</sup> , Xinggang Wang<sup>1 :email:</sup> , Jingfeng Yao<sup>1</sup> , Wenyu Liu<sup>1</sup> , Jinglin Zhang<sup>2</sup> , Xiangmin Xu<sup>3</sup>

<sup>1</sup> School of EIC, HUST, <sup>2</sup> Shandong University, <sup>3</sup> South China University of Technology

(<sup>:email:</sup>) corresponding author.

<!-- ArXiv Preprint ([arXiv ](https://arxiv.org/abs/)) -->

<!-- [openreview ICLR'23](https://openreview.net/forum?id=k7p_YAO7yE), accepted as **ICLR Spotlight** -->

</div>

#
### News
* **`Mar. 20th, 2024`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Introduction
<div align="center"><h5>Plain Vision Transformer could also do gaze following with simple ViTGaze framework!</h5></div>

![framework](assets/pipeline.png "framework")

Inspired by the remarkable success of pre-trained plain Vision Transformers (ViTs), we introduce a novel single-modality gaze following framework, **ViTGaze**. In contrast to previous methods, it creates a brand new gaze following framework based mainly on powerful encoders (relative decoder parameter less than 1%). Our principal insight lies in that the inter-token interactions within self-attention can be transferred to interactions between humans and scenes. Our method achieves state-of-the-art (SOTA) performance among all single-modality methods (3.4% improvement on AUC, 5.1% improvement on AP) and very comparable performance against multi-modality methods with 59% number of parameters less.

## Demo
![Demo0](assets/demo0.gif)
![Demo1](assets/demo1.gif)

<h3>Code Coming Soon!</h3>

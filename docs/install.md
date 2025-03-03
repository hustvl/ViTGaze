## Installation

* Create a conda virtual env and activate it.

  ```
  conda create -n ViTGaze python==3.9.18
  conda activate ViTGaze
  ```
* Install packages.

  ```
  cd path/to/ViTGaze
  pip install -r requirements.txt
  ```
* Install [detectron2](https://github.com/facebookresearch/detectron2) , follow its [documentation](https://detectron2.readthedocs.io/en/latest/).
  ~~For ViTGaze, we recommend to build it from latest source code.~~

  To ensure compatibility, we recommend to build it from the specific version of Detectron2 (refer to [Issue #9](https://github.com/hustvl/ViTGaze/issues/9) for detailed discussion).

  ```
  pip install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f#egg=detectron2"
  ```

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
  For ViTGaze, we recommend to build it from latest source code.
  ```
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```

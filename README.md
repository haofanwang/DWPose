# DWPose
This repository refactors the inference from the official implementation of [DWPose](https://github.com/IDEA-Research/DWPose/tree/main).

<center><img src="https://github.com/haofanwang/DWPose/raw/main/assets/asset-example.png" width="80%" height="80%"></center> 


## Installation

```bash
# git clone this repository
git clone https://github.com/haofanwang/DWPose.git
cd DWPose

# install required packages
pip install -r requirements.txt

# Set environment
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```
If you meet any download issues, please refer to [installation instructions](https://github.com/IDEA-Research/DWPose/blob/main/INSTALL.md).

## Download Checkpoints

Download the pretrained [detection model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) and [pose model](https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth), and save them under `./ckpts`. It is also possible to use [other detection models](https://github.com/open-mmlab/mmdetection/tree/main/configs) from MMDetection and [pose model](https://github.com/IDEA-Research/DWPose/tree/main).


## Quick Inference

```bash
import cv2
import numpy as np
from PIL import Image

from utils import *
from dwpose import DWposeDetector

# set configs
det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'

# set device
device = "cuda:0"

# init
dwprocessor = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt, device)

# infer
image_dir = "./assets/test.jpeg"
input_image = cv2.imread(image_dir)
input_image = HWC3(input_image)
input_image = resize_image(input_image, resolution=512)

#height and weight of original image
H, W = input_image.shape[:2]

detected_map = dwprocessor(input_image)
detected_map = HWC3(detected_map)

detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(image_dir.split('/')[-1], detected_map)
```


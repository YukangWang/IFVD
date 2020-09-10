# Intra-class Feature Variation Distillation for Semantic Segmentation

## Introduction

This repository contains the PyTorch implementation of: 

Intra-class Feature Variation Distillation for Semantic Segmentation, ECCV 2020 [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520341.pdf)

## Requirements

All the codes are tested in the following environment:

* Linux (tested on Ubuntu 16.04 / CentOS 7.6)
* Python 3.6.2
* PyTorch 0.4.1
* Single TITAN Xp GPU

## Installation

* Install PyTorch: ` conda install pytorch=0.4.1 cuda90 torchvision -c pytorch `
* Install other dependences: ` pip install opencv-python scipy `
* Install InPlace-ABN:
```bash
cd libs
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

## Dataset & Models

* Dataset: [[Cityscapes]](https://www.cityscapes-dataset.com/)

* Teacher: PSPNet (ResNet-101) trained on Cityscapes [[Google Drive]](https://drive.google.com/file/d/1epiJnLiPYSAgT2IHP0UhYkTSRMb8twpJ/view?usp=sharing)

* Student: ResNet-18 pretrained on ImageNet [[Google Drive]](https://drive.google.com/file/d/17ewTEr-FZ8x0Lc9XMMR5VbgOupq48s9q/view?usp=sharing)

* After distillation: PSPNet (ResNet-18) [[Google Drive]](https://drive.google.com/file/d/1dNjKbj7Cm2_JSr9HzrmqeS9S081nNyvw/view?usp=sharing)

Please create a new folder `ckpt` and move all downloaded models to it.

## Usage

#### 1. Trainning with evaluation

```bash

python train.py --data-dir /path/to/cityscapes --save-name /path/to/save --gpu /device/id

```

#### 2. Inference with evaluation

```bash

python val.py --data-dir /path/to/cityscapes --restore-from /path/to/pth --gpu /device/id

```  

#### 3. Inference only
  

```bash

python test.py --data-dir /path/to/cityscapes --restore-from /path/to/pth --gpu /device/id

```

## Citation

Please consider citing this work if it helps your research:

```

@inproceedings{wang2020ifvd,
  title={Intra-class Feature Variation Distillation for Semantic Segmentation},
  author={Wang, Yukang and Zhou, Wei and Jiang, Tao and Bai, Xiang and Xu, Yongchao},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}

```

## Acknowledgment

This codebase is heavily borrowed from [pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox) and [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation). Thanks for their excellent works.

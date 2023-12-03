# Knockoffs-SPR: Clean Sample Selection in Learning with Noisy Labels
\[[paper](https://arxiv.org/abs/2301.00545)\]

## Overview
This is the official repo for our paper "Knockoffs-SPR: Clean Sample Selection in Learning with Noisy Labels".

> Knockoffs-SPR is a theoretically guaranteed sample selection algorithm for learning with noisy labels, which is provable to control the False-Selection-Rate (FSR) in the selected clean data.

## Requirements
```
python
numpy
scipy
scikit-learn
torch
torchvision
```

## Data Preparing

CIFAR-10 and CIFAR-100 can be downloaded using *torchvision*. The other datasets can be downloaded from the official link: [WebVision](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html), [Clothing1M](https://github.com/Cysu/noisy_label).

The datasets are expected to be stored in the folder **../data** or specified by the *root* parameter, and arranged as follows:
```
│data/
├── CIFAR10/
│   ├── ......
├── CIFAR100/
│   ├── ......
├── webvision/
│   ├── info/
│   │   ├── ......
│   ├── google/
│   │   ├── ......
│   ├── val_images_256/
│   │   ├── ......
├── clothing1m/
│   ├── ......
(Optional)
├── imagenet/
│   ├── meta.mat
│   ├── ILSVRC2012_validation_ground_truth.txt
│   ├── val/
│   │   ├── ......
```


## Pretrained Model
The pretained models can be downloaded from [here](https://drive.google.com/drive/folders/1oONTRmng9JrDNmrf_PFM561_pfP18vqC?usp=sharing) and should be put in the folder **ckpt**.

## Training

### Pretrain
We first pretrain the backbone network use SimSiam with the following command.
```
python pretrain_simsiam.py -a $ARCH --dataset $DATASET --num_classes $NUM_CLASS
```
where $ARCH is the architecture of the backbone network, $DATASET is the dataset name, and $NUM_CLASS is the number of classes in the dataset. 
Specific running commands are listed in the folder **scripts**.

### Train with Knockoffs-SPR
Example training commands are listed in the folder **scripts**.
You could try the following commands as a start.

Train on CIFAR10 with different noise setting:
```
python scripts/train_cifar.py
```

## Evaluation
Example evaluation commands are listed in the folder **scripts**.
You could try the following commands as a start.

Test on CIFAR10 with different noise setting:
```
python scripts/eval_cifar.py
```

## Acknowledgements
Thanks to everyone who makes their code and models available. In particular,

- [SimSiam](https://github.com/facebookresearch/simsiam)
- [CutMix](https://github.com/clovaai/CutMix-PyTorch)
- [SR](https://github.com/hitcszx/lnl_sr)
- [TopoFilter](https://github.com/pxiangwu/TopoFilter)

## Contact Information
For issues using Knockoffs-SPR, please submit a GitHub issue.

## Citation

If you found the provided code useful, please cite our work.

```
@article{wang2023knockoffs,
title={Knockoffs-SPR: Clean Sample Selection in Learning with Noisy Labels},
author={Wang, Yikai and Fu, Yanwei and Sun, Xinwei},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
year={2023},
doi={10.1109/TPAMI.2023.3338268},
}
```
# IKMR
Implement of Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning

Our Performance shown on project page: https://cybercal.github.io/webpage.ikmr/

Paper Link: https://arxiv.org/pdf/2509.15443


## Overview 
a light skeleton-topolog based nerual retergeting from human to humanoid via style transfer

![Performance](pic.png)

## Structure
```
IKMR/
├── posebox/                    ← motion data converter
├── poselib/                    ← motion pose lib
├── retargeting/
│   ├── train_cmp.py            ← pre-train script
│   └── finetune_cmp.py         ← fine-tuning script
└── readme.md
```


## Quick Start
```bash
cd retargeting

## train both smpl and g1's encoder and decoder
python train_cmp.py --save_dir=./exp/pretrain

## fine tuning decoder
python finetune_cmp.py --save_dir=./exp/finetune

## check whether converge
tensorboard --logdir=./retargeting/exp/your_exp/logs

## inference retargeting stream
python test_cmp.py
```


## Acknowledgments
We would like to acknowledge the following projects from which parts of the code in this repo are derived from:
- DeepMotionEditing: https://github.com/DeepMotionEditing
- Poselib: https://github.com/ZhengyiLuo/PHC/blob/master/poselib
# IKMR: Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning
<div align="center">
  <p align="center">
    <a href='https://cybercal.github.io/'>Xingyu Chen</a>,
    Hanyu Wu, Sikai Wu, Mingliang Zhou, Diyun Xiang, Haodong Zhang
    <br>
    Xiaomi Robotics Lab, ETH Zurich, Zhejiang University
  </p>
</p>

[![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://cybercal.github.io/webpage.ikmr/)
[![Arxiv](https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow)](https://arxiv.org/pdf/2509.15443)
[![code](https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white)](https://github.com/Cybercal/IKMR)

</div>


## Overview 
Implement of Implicit Kinodynamic Motion Retargeting (IKMR) for Human-to-humanoid Imitation Learning. A lightweight neural retargeting method from human to humanoid motion through style transfer,
It is design for end-to-end motion retargeting based on skeletal-topology, 
and this framework supports large-scale data and online stream motion.
![Performance](pic.png)


Our Performance shown on project page: https://cybercal.github.io/webpage.ikmr/

Paper Link: https://arxiv.org/pdf/2509.15443

## Code Structure
```
IKMR/
├── posebox/                    ← motion data proc
├── poselib/                    ← motion pose lib
├── retargeting/
│   ├── train_cmp.py            ← pre-train script
│   └── finetune_cmp.py         ← fine-tuning script
└── readme.md
```
Here, `/retargeting/datasets/CMP` only releases 10 processed motion sequence for SMPL-G1 pairs.
We suggest to use 2000+ motion for a stable and good retargeting quality.
More motion pairs will achieve better performance.

## Quick Start
```bash
# simple env setup
conda create -y -n ikmr python=3.10
conda activate ikmr
pip install numpy==1.21
pip install scipy==1.7.3
pip install matplotlib==3.5
pip install setuptools==59.5.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard

## install poselib
cd poselib
pip install -e .

## train both smpl and g1's encoder and decoder
cd retargeting
python train_cmp.py --save_dir=./exp/pretrain

## fine tuning decoder
python finetune_cmp.py --save_dir=./exp/finetune

## check whether converge
tensorboard --logdir=./retargeting/exp/your_exp/logs

## inference retargeting stream
python test_cmp.py
```
Training trick: The forward kinematics for end-effector calculation is time-cost, you can comment out this part at beginning, then apply them later.
Feel free to open an issue or discussion if you encounter any problems or have questions about this training process.

## Acknowledgments
We would like to acknowledge the following projects from which parts of the code in this repo are derived from:
- DeepMotionEditing: https://github.com/DeepMotionEditing
- Poselib: https://github.com/ZhengyiLuo/PHC/blob/master/poselib

## Citation
If you find our work helpful, please cite:
```bibtex
@article{Chen2025ImplicitKM,
  title={Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning},
  author={Xingyu Chen and Hanyu Wu and Sikai Wu and Mingliang Zhou and Diyun Xiang and Haodong Zhang},
  journal={arXiv preprint arXiv:2509.15443},
  year={2025},
  url={https://arxiv.org/abs/2509.15443}
}
```
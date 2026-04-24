import os
import sys
# os.chdir('/home/nuc/hemera_workspace/hemera_poselib_3.8/')
# sys.path.append('/home/nuc/hemera_workspace/hemera_poselib_3.8/')

import torch
import numpy as np
# from poselib.core.rotation3d import *
# from poselib.core.write_data_to_file import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

motion_file = '/home/nuc/Datasets/CMP/AMASS_smpl/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose.npy'
# motion_file = '/home/nuc/Datasets/CMP/AMASS_g1/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose_g1_g1_29.npy'

# motion_file = "/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/smpl.npy"

skm = SkeletonMotion.from_file(motion_file)
skt = skm.skeleton_tree 

local_rotation = skm.local_rotation.numpy()  # size(frames, joints, 4), q = [x, y, z, w]
root_translation = skm.root_translation.numpy()

list_of_dicts = [{'rot': local_rotation, 'trans': root_translation}]

# 将列表保存为 .npy 文件
# np.save("/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/smpl.npy", list_of_dicts)

# plot_skeleton_state(SkeletonState.zero_pose(skt))
# plot_skeleton_motion_interactive(skm)
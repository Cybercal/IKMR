import os
import sys
import glob
# os.chdir('/home/nuc/hemera_workspace/hemera_poselib_3.7/')
# sys.path.append('/home/nuc/hemera_workspace/hemera_poselib_3.7/')

import torch
import numpy as np
from utils.rotation3d import *
# from poselib.core.rotation3d import *
# from poselib.core.write_data_to_file import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive


def single_file():
    # motion_file = '/home/nuc/Datasets/CMP/AMASS_smpl/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose.npy'
    motion_file = '/home/nuc/Datasets/CMP/AMASS_g1/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose_g1_g1_29.npy'

    skm = SkeletonMotion.from_file(motion_file)
    skt = skm.skeleton_tree 

    plot_skeleton_state(SkeletonState.zero_pose(skt))
    plot_skeleton_motion_interactive(skm)

    # process motion from zero 
    if False:
        trans_modified = skm.root_translation.clone().float()   # [639, 3]
        rot_modified = skm.local_rotation.clone().float() # [639, 24, 4]
        # adjust_info["xy_start_from_0"]:
        trans_xy0 = trans_modified[0,:2].clone()    # [x0, y0]
        trans_modified[:,:2] -= trans_xy0
        # adjust_info["yaw_start_from_0"]:
        yaw0 = rot_modified[0, 0, :].clone().unsqueeze(0)   #  q0 = (1,4) = [x, y, z, w]
        heading_rot_inv = calc_heading_quat_inv(yaw0)
        rot_modified[:, 0, :] = quat_mul(heading_rot_inv, rot_modified[:, 0, :])
        heading_rot_inv_expand = heading_rot_inv.repeat((1, trans_modified.shape[0], 1))
        flat_heading_rot_inv = heading_rot_inv_expand.view(trans_modified.shape[0], heading_rot_inv.shape[1])
        trans_modified = my_quat_rotate(flat_heading_rot_inv, trans_modified)
        
        state_new = SkeletonState.from_rotation_and_root_translation(skeleton_tree=skt, r=rot_modified, t=trans_modified, is_local=True)
        motion_new = SkeletonMotion.from_skeleton_state(state_new, fps=30)
        skm = motion_new

        plot_skeleton_state(SkeletonState.zero_pose(skm.skeleton_tree))
        plot_skeleton_motion_interactive(skm)

    
    local_rotation = skm.local_rotation.numpy()  # size(frames, joints, 4), q = [x, y, z, w]
    root_translation = skm.root_translation.numpy()

    list_of_dicts = [{'rot': local_rotation, 'trans': root_translation}]

    # 将列表保存为 .npy 文件
    np.save("/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/smpl.npy", list_of_dicts)



def main_mult_files():
    # modify = True
    # input_dir = '/home/nuc/Datasets/CMP/AMASS_smpl/'
    # output_file = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/smpl.npy'

    modify = False
    input_dir = '/home/nuc/Datasets/CMP/AMASS_g1/'
    output_file = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/g1.npy'

    npy_files = glob.glob(os.path.join(input_dir, '**', '*.npy'), recursive=True)
    npy_files = sorted(npy_files)
    # print(npy_files)

    # np.save('/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/all_list.npy', npy_files)
    with open('/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/all_list.txt', 'w', encoding='utf-8') as f:
        for item in npy_files:
            f.write(item + '\n')

    # with open('/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/all_list.txt', 'r', encoding='utf-8') as f:
    #     my_list = [line.strip() for line in f]


    # 确保输出目录存在
    # os.makedirs(output_dir, exist_ok=True)
    list_of_dicts = []

    # 遍历所有 .npy 文件并进行转换
    for npy_file in npy_files:
        # 读取 SkeletonMotion
        skm = SkeletonMotion.from_file(npy_file)
        skt = skm.skeleton_tree

        # process motion from zero 
        if modify == True:
            trans_modified = skm.root_translation.clone().float()   # [639, 3]
            rot_modified = skm.local_rotation.clone().float() # [639, 24, 4]
            # adjust_info["xy_start_from_0"]:
            trans_xy0 = trans_modified[0,:2].clone()    # [x0, y0]
            trans_modified[:,:2] -= trans_xy0
            # adjust_info["yaw_start_from_0"]:
            yaw0 = rot_modified[0, 0, :].clone().unsqueeze(0)   #  q0 = (1,4) = [x, y, z, w]
            heading_rot_inv = calc_heading_quat_inv(yaw0)
            rot_modified[:, 0, :] = quat_mul(heading_rot_inv, rot_modified[:, 0, :])
            heading_rot_inv_expand = heading_rot_inv.repeat((1, trans_modified.shape[0], 1))
            flat_heading_rot_inv = heading_rot_inv_expand.view(trans_modified.shape[0], heading_rot_inv.shape[1])
            trans_modified = my_quat_rotate(flat_heading_rot_inv, trans_modified)
            
            state_new = SkeletonState.from_rotation_and_root_translation(skeleton_tree=skt, r=rot_modified, t=trans_modified, is_local=True)
            motion_new = SkeletonMotion.from_skeleton_state(state_new, fps=30)
            skm = motion_new
    
        local_rotation = skm.local_rotation.numpy()  # size(frames, joints, 4), q = [x, y, z, w]
        root_translation = skm.root_translation.numpy()
        list_of_dicts.append({'rot': local_rotation, 'trans': root_translation})

    # 写入 BVH 文件
    np.save(output_file, list_of_dicts)
    print(f"whether rotation {modify}")
    print(f"Final converted {npy_file} to {output_file}")


if __name__ == '__main__':
    single_file()
    # main_mult_files()
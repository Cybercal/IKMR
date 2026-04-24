import os
import sys
import time

# if debug
current_working_dir = os.getcwd()
sys.path.append(current_working_dir)
print(f"当前工作目录: {current_working_dir}")
new_working_dir = os.path.join(current_working_dir, 'retargeting/')
os.chdir(new_working_dir)
print(f"当前工作目录: {new_working_dir}")

from os.path import join as pjoin
# from get_error import full_batch
import numpy as np
import option_parser
# from option_parser import try_mkdir
import argparse
import torch
from tqdm import tqdm
from models import create_model
from datasets.motionloader import HRdataset
# from datasets import create_dataset, get_character_names
# from poselib.core.rotation3d import *
# from poselib.core.write_data_to_file import *
from tools.rotation3d import *
from tools.write_data_to_file import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive


def dict_to_string(a):
    res = "{\n"
    # 缩进
    indent = 0
    # 当前的遍历的位置
    index = 0
    # 当前字典(json数据) 的可迭代长度
    keyLen = len(a.keys())

    # print("a.keys():", a.keys())

    for i in a:
        # 得到当前值
        info = a[i]
        # 获取到当前值的类型
        dataTp = type(info)
        # 给末尾添加的额外元素
        endStr = ",\n"
        # 如果当前值是数组, 需要特殊处理
        if dataTp == list:
            # 判断数组的第一个元素是否是 二维数组,这里只实现 二维数组的方法, 一维数组直接写入, 其他多维的数组方法同理
            # 实现的功能是保存 用二维数组渲染的游戏地图, 所以子元素的类型都一致,只用判断一个就行
            if type(info[0]) == list:
                info = "[\n"
                lIndex = 0
                # 属于更下一级的子元素, 缩进需要再进一步
                indent = 2
                for ci in a[i]:
                    # print("ci:", ci)
                    # print("a[i]:", a[i])
                    # 判断是否到了迭代的末尾, 末尾的数据就不用加 逗号和换行了
                    if lIndex == len(a[i]) -1:
                        endStr = ""
                    # else:
                    info += " "*indent+f"{ci}{endStr}"

                    # print("info:", info)
                    # 遍历位置+1
                    lIndex += 1
                    
                info += "\n]"
                endStr = ",\n"
                #  结束二维数组的处理之后, 将缩进改回来
                indent = 0
        # 根据当前行的数据类型判断是否需要 加上双引号包裹起来
        if dataTp != dict and dataTp != list:
            info = f'"{info}"'

        if index == keyLen -1:
            endStr = ""
        # 将当前行的最终数据追加到res
        res += " "*indent+f'"{i}": {info}{endStr}'
        index += 1

    res += "\n}"
    return res


def local_rotation_to_dof(local_rot, node_id, n_frames, num_dof):
    dof_pos = torch.zeros((n_frames, num_dof))
    x_axis_joint = [1, 5, 7, 11, 13, 16, 19, 23, 26]
    z_axis_joint = [2, 8, 12, 17, 21, 24, 28]

    for i in range(len(node_id)):
        # print("i:", i)
        joint_theta, joint_axis = quat_to_angle_axis(local_rot[:, node_id[i], :])
        # print("local_rot[node_id[i]]:", local_rot[:, node_id[i], :].shape)
        if i in x_axis_joint:
            joint_theta = joint_theta * joint_axis[..., 0] # assume joint is along x axis
            # print("x axis joint")
        elif i in z_axis_joint:
            joint_theta = joint_theta * joint_axis[..., 2] # assume joint is along z axis
            # print("z axis joint")
        else:
            joint_theta = joint_theta * joint_axis[..., 1] # assume joint is along y axis
            # print("y axis joint")
        joint_theta = normalize_angle(joint_theta)
        # print("joint_theta:", joint_theta.shape)
        dof_pos[:, i] = joint_theta
    return dof_pos


def local_rotation_to_dof_vel(local_rot0, local_rot1, node_id, n_frames, num_dof, dt):
    dof_vel = torch.zeros(num_dof)
    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
    root_rot_vel = local_vel[0, :]
    # print("local_vel:", local_vel.shape)

    x_axis_joint = [1, 5, 7, 11, 13, 16, 19, 23, 26]
    z_axis_joint = [2, 8, 12, 17, 21, 24, 28]

    for i in range(len(node_id)):
        joint_vel = local_vel[node_id[i], :]
        if i in x_axis_joint:
            joint_vel = joint_vel[0] # assume joint is along x axis
            # print("x axis joint")
        elif i in z_axis_joint:
            joint_vel = joint_vel[2] # assume joint is along z axis
            # print("z axis joint")
        else:
            joint_vel = joint_vel[1] # assume joint is along y axis
            # print("y axis joint")
        # print("joint_vel:", joint_vel.shape)
        dof_vel[i] = joint_vel
    return root_rot_vel, dof_vel


def save_json(rtg_data, id):
    motion_file = '/home/nuc/Datasets/CMP/AMASS_g1/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose_g1_g1_29.npy'
    source_motion = SkeletonMotion.from_file(motion_file)
    robot_motion_fps = 50
    rtg_motion = rtg_data[0]
    global_motion = rtg_data[1]

    weight = 1.0
    new_motion_dict = {}
    in_local_yaw_frame = True
    num_dof = 29
    
    # necessary information
    new_motion_dict["LoopMode"] = "Wrap"
    new_motion_dict["FrameDuration"] = 1.0 / robot_motion_fps #source_motion.fps
    new_motion_dict["EnableCycleOffsetPosition"] = True
    new_motion_dict["EnableCycleOffsetRotation"] = True
    new_motion_dict["MotionWeight"] = weight
    print("source_motion.fps", new_motion_dict["FrameDuration"])

    # get rotation and translation from motion
    n_frames = rtg_motion.shape[0]
    local_rot = rtg_motion[:, :-3].reshape(n_frames, -1, 4) # source_motion.local_rotation
    root_rot = local_rot[:, 0, :]
    root_pos = rtg_motion[:, -3:] # source_motion.root_translation
    node_names = source_motion.skeleton_tree.node_names

    global_trans = global_motion[..., 4:]
    global_rotat = global_motion[..., :4]

    '''
    node_names: 
        ['pelvis', 
         'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
         'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
         'waist_yaw_link', 
         'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
         'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
    '''
    
    # get node id according to the node_names
    left_hip_pitch_id = source_motion.skeleton_tree._node_indices["left_hip_pitch_link"]
    left_hip_roll_id = source_motion.skeleton_tree._node_indices["left_hip_roll_link"]
    left_hip_yaw_id = source_motion.skeleton_tree._node_indices["left_hip_yaw_link"]
    left_knee_id = source_motion.skeleton_tree._node_indices["left_knee_link"]
    left_ankle_pitch_id = source_motion.skeleton_tree._node_indices["left_ankle_pitch_link"]
    left_ankle_roll_id = source_motion.skeleton_tree._node_indices["left_ankle_roll_link"]

    right_hip_pitch_id = source_motion.skeleton_tree._node_indices["right_hip_pitch_link"]
    right_hip_roll_id = source_motion.skeleton_tree._node_indices["right_hip_roll_link"]
    right_hip_yaw_id = source_motion.skeleton_tree._node_indices["right_hip_yaw_link"]
    right_knee_id = source_motion.skeleton_tree._node_indices["right_knee_link"]
    right_ankle_pitch_id = source_motion.skeleton_tree._node_indices["right_ankle_pitch_link"]
    right_ankle_roll_id = source_motion.skeleton_tree._node_indices["right_ankle_roll_link"]
    
    waist_yaw_id = source_motion.skeleton_tree._node_indices["waist_yaw_link"]
    waist_roll_id = source_motion.skeleton_tree._node_indices["waist_roll_link"]
    waist_pitch_id = source_motion.skeleton_tree._node_indices["torso_link"]

    left_shoulder_pitch_id = source_motion.skeleton_tree._node_indices["left_shoulder_pitch_link"]
    left_shoulder_roll_id = source_motion.skeleton_tree._node_indices["left_shoulder_roll_link"]
    left_shoulder_yaw_id = source_motion.skeleton_tree._node_indices["left_shoulder_yaw_link"]
    left_elbow_id = source_motion.skeleton_tree._node_indices["left_elbow_link"]
    left_wrist_roll_id = source_motion.skeleton_tree._node_indices["left_wrist_roll_link"]
    left_wrist_pitch_id = source_motion.skeleton_tree._node_indices["left_wrist_pitch_link"]
    left_wrist_yaw_id = source_motion.skeleton_tree._node_indices["left_wrist_yaw_link"]

    right_shoulder_pitch_id = source_motion.skeleton_tree._node_indices["right_shoulder_pitch_link"]
    right_shoulder_roll_id = source_motion.skeleton_tree._node_indices["right_shoulder_roll_link"]
    right_shoulder_yaw_id = source_motion.skeleton_tree._node_indices["right_shoulder_yaw_link"]
    right_elbow_id = source_motion.skeleton_tree._node_indices["right_elbow_link"]
    right_wrist_roll_id = source_motion.skeleton_tree._node_indices["right_wrist_roll_link"]
    right_wrist_pitch_id = source_motion.skeleton_tree._node_indices["right_wrist_pitch_link"]
    right_wrist_yaw_id = source_motion.skeleton_tree._node_indices["right_wrist_yaw_link"]

    # joint index
    node_id = [left_hip_pitch_id, left_hip_roll_id, left_hip_yaw_id, left_knee_id, left_ankle_pitch_id, left_ankle_roll_id,\
            right_hip_pitch_id, right_hip_roll_id, right_hip_yaw_id, right_knee_id, right_ankle_pitch_id, right_ankle_roll_id,\
            waist_yaw_id, waist_roll_id, waist_pitch_id,\
            left_shoulder_pitch_id, left_shoulder_roll_id, left_shoulder_yaw_id, left_elbow_id, left_wrist_roll_id, left_wrist_pitch_id, left_wrist_yaw_id,\
            right_shoulder_pitch_id, right_shoulder_roll_id, right_shoulder_yaw_id, right_elbow_id, right_wrist_roll_id, right_wrist_pitch_id, right_wrist_yaw_id]
    # print("selected node_id:", node_id)

    dof_pos = local_rotation_to_dof(local_rot, node_id, n_frames, num_dof)
    # print("dof_pos:", dof_pos[:, 0])
    # print("local_rot:", local_rot[:, 4, :])
 
    root_rot_vel = torch.zeros((n_frames, 3))
    root_vel = torch.zeros((n_frames, 3))
    dof_vel = torch.zeros((n_frames, num_dof))
    for f in range(n_frames - 1):
        # print("f:", f)
        local_rot0 = local_rot[f]
        # print("local_rot0:", local_rot0.shape)
        local_rot1 = local_rot[f + 1]
        frame_root_rot_vel, frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, node_id, n_frames, num_dof, 1.0 / robot_motion_fps)
        # root rotation velocity
        root_rot_vel[f, :] = frame_root_rot_vel
        # root linear velocity
        root_vel[f, :] = (root_pos[f + 1] - root_pos[f]) * robot_motion_fps
        # dofs velocity
        dof_vel[f, :] = frame_dof_vel
    
    # the last frame use the same value as the second last frame
    dof_vel[-1, :] = dof_vel[-2, :]
    root_rot_vel[-1, :] = root_rot_vel[-2, :]
    root_vel[-1, :] = root_vel[-2, :]

    heading_rot_inv = calc_heading_quat_inv(root_rot)
    local_root_rot_vel = quat_rotate(quat_inverse(root_rot), root_rot_vel)
    local_root_vel = quat_rotate(quat_inverse(root_rot), root_vel)
    local_yaw_root_rot_vel = quat_rotate(heading_rot_inv, root_rot_vel)
    local_yaw_root_vel = quat_rotate(heading_rot_inv, root_vel)
    # print("root_vel:", root_vel)

    # generate frames acoordint to the root and dofs state
    frame_cols = 3 + 4 + 29 + 12 + 3 + 3 + 29 + 12 + 16  # 107 = root_pos(3) + root_rot(4) + dof_pos(27) + key_pos(12) + root_vel(3) + root_ang_vel(3) + dof_vel(27) + key_vel(12) + key_rot(16)
    frames = torch.zeros((n_frames, frame_cols))
    print("n_frames:", n_frames)
    # print('frames shape:', frames.shape)

    # get key positions in global coordinate
    key_pos_global = torch.zeros((n_frames, 12))
    key_pos_local = torch.zeros((n_frames, 12))
    key_pos_global[:, 0:3] = global_trans[..., left_ankle_roll_id, :]
    key_pos_global[:, 3:6] = global_trans[..., right_ankle_roll_id, :]
    key_pos_global[:, 6:9] = global_trans[..., left_wrist_yaw_id, :]
    key_pos_global[:, 9:12] = global_trans[..., right_wrist_yaw_id, :]

    # key_pos_global1 = torch.zeros((n_frames, 12))
    # key_pos_global1[:, 0:3] = source_motion.global_translation2[..., left_ankle_roll_id, :]

    # print("key_pos_global:", key_pos_global[:, 0:3])
    # print("key_pos_global1:", key_pos_global1[:, 0:3])
    # exit()

    # transform key positions to local coordinate
    key_pos_to_root_global = key_pos_global[:, 0:3] - root_pos
    key_pos_local[:, 0:3] = quat_rotate(quat_inverse(root_rot), key_pos_to_root_global)
    key_pos_to_root_global = key_pos_global[:, 3:6] - root_pos
    key_pos_local[:, 3:6] = quat_rotate(quat_inverse(root_rot), key_pos_to_root_global)
    key_pos_to_root_global = key_pos_global[:, 6:9] - root_pos
    key_pos_local[:, 6:9] = quat_rotate(quat_inverse(root_rot), key_pos_to_root_global)
    key_pos_to_root_global = key_pos_global[:, 9:12] - root_pos
    key_pos_local[:, 9:12] = quat_rotate(quat_inverse(root_rot), key_pos_to_root_global)
    # transform key positions to local yaw coordinate
    key_pos_local_yaw = torch.zeros((n_frames, 12))
    key_pos_to_root_global_yaw = key_pos_global[:, 0:3] - root_pos
    key_pos_local_yaw[:, 0:3] = quat_rotate(heading_rot_inv, key_pos_to_root_global_yaw)
    key_pos_to_root_global_yaw = key_pos_global[:, 3:6] - root_pos
    key_pos_local_yaw[:, 3:6] = quat_rotate(heading_rot_inv, key_pos_to_root_global_yaw)
    key_pos_to_root_global_yaw = key_pos_global[:, 6:9] - root_pos
    key_pos_local_yaw[:, 6:9] = quat_rotate(heading_rot_inv, key_pos_to_root_global_yaw)
    key_pos_to_root_global_yaw = key_pos_global[:, 9:12] - root_pos
    key_pos_local_yaw[:, 9:12] = quat_rotate(heading_rot_inv, key_pos_to_root_global_yaw)

    # get key velocities in local coordinate
    key_vel_local = torch.zeros((n_frames, 12))

    # get key orientations in global coordinate
    key_rot_global = torch.zeros((n_frames, 16))
    key_rot_global[:, 0:4] = global_rotat[..., left_ankle_roll_id, :]
    key_rot_global[:, 4:8] = global_rotat[..., right_ankle_roll_id, :]
    key_rot_global[:, 8:12] = global_rotat[..., left_wrist_yaw_id, :]
    key_rot_global[:, 12:16] = global_rotat[..., right_wrist_yaw_id, :]
    
    # transform key orientations to local coordinate
    key_rot_local = torch.zeros((n_frames, 16))
    key_rot_local[:, 0:4] = quat_mul_norm(quat_inverse(root_rot), key_rot_global[:, 0:4])
    key_rot_local[:, 4:8] = quat_mul_norm(quat_inverse(root_rot), key_rot_global[:, 4:8])
    key_rot_local[:, 8:12] = quat_mul_norm(quat_inverse(root_rot), key_rot_global[:, 8:12])
    key_rot_local[:, 12:16] = quat_mul_norm(quat_inverse(root_rot), key_rot_global[:, 12:16])
    
    # transform key orientations to local yaw coordinate
    key_rot_local_yaw = torch.zeros((n_frames, 16))
    key_rot_local_yaw[:, 0:4] = quat_mul_norm(heading_rot_inv, key_rot_global[:, 0:4])
    key_rot_local_yaw[:, 4:8] = quat_mul_norm(heading_rot_inv, key_rot_global[:, 4:8])
    key_rot_local_yaw[:, 8:12] = quat_mul_norm(heading_rot_inv, key_rot_global[:, 8:12])
    key_rot_local_yaw[:, 12:16] = quat_mul_norm(heading_rot_inv, key_rot_global[:, 12:16])
    
    # get key angular velocities in local coordinate
    key_ang_vel_local = torch.zeros((n_frames, 12))

    # root_pos
    ind = 0
    frames[:, ind:ind+3] = root_pos
    ind += 3
    # root_rot
    frames[:, ind:ind+4] = root_rot
    ind += 4
    # dof_pos
    frames[:, ind:ind+29] = dof_pos
    ind += 29
    if in_local_yaw_frame:
        # key_pos
        frames[:, ind:ind+12] = key_pos_local_yaw
        ind += 12
        # root vel
        frames[:, ind:ind+3] = local_yaw_root_vel
        ind += 3
        # root ang vel
        frames[:, ind:ind+3] = local_yaw_root_rot_vel
        ind += 3
    else:
        # key_pos
        frames[:, ind:ind+12] = key_pos_local
        ind += 12
        # root vel
        frames[:, ind:ind+3] = local_root_vel
        ind += 3
        # root ang vel
        frames[:, ind:ind+3] = local_root_rot_vel
        ind += 3
    # dof vel
    frames[:, ind:ind+29] = dof_vel
    ind += 29
    # key vel, not used
    frames[:, ind:ind+12] = key_vel_local
    ind += 12
    if in_local_yaw_frame:
        # key rot
        frames[:, ind:ind+16] = key_rot_local_yaw
        ind += 16
    else:
        # key rot
        frames[:, ind:ind+16] = key_rot_local
        ind += 16
    print("ind", ind)
    # key ang vel, not used
    # frames[:, 107:119] = key_ang_vel_local

    nan_mask = torch.isnan(frames)
    nan_mask_num = torch.sum(nan_mask)

    frames_list = frames.numpy().tolist()
    frames_list = np.round(frames_list, 5)
    frames_list = frames_list.tolist()

    new_motion_dict["Frames"] = frames_list

    new_motion_str = dict_to_string(new_motion_dict)

    save_path = '/home/nuc/MoBox/deep-motion-editing/retargeting/exp/val/results/bvh/'
    file_path = f'{save_path}g1_motion_{id}.json'
    with open(file_path, "w") as f:
        f.write(new_motion_str)


def eval(save_dir, test_device='cpu'):
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    # args.eval_seq = eval_seq
    args.save_dir = save_dir
    character_names = [['smpl'], ['g1']]

    dataset = HRdataset(args, character_names)
    # torch.save(dataset, '/home/nuc/MoBox/pose2bvh/results/dataset.pth')
    # get modified datasets topology part  
    # None

    model = create_model(args, character_names, dataset)
    model.load(epoch=11000)

    # with open('/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/all_list.txt', 'r', encoding='utf-8') as f:
    #     my_list = [line.strip() for line in f]

    id = 0
    for i, motions in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(motions)
        # model.test(); print(i)
        if motions[0][0].shape[2] < 64*5/3:
            print(f'Skipping motion {i} due to insufficient {motions[0][0].shape[2]} frames')

            # my_list.pop(i)

            continue
        else:
            
            # f1 = time.time()
            rtg_data = model.usage()
            # rtg_data = model.recon()
            # f2 = time.time()
            save_json(rtg_data, id)
            # print(f'Time taken for inference: {f2 - f1} seconds')
            id += 1

    # with open('/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/test_list.txt', 'w', encoding='utf-8') as f:
    #     for item in my_list:
    #         f.write(item + '\n')


# def batch_copy(source_path, suffix, dest_path, dest_suffix=None):
#     option_parser.try_mkdir(dest_path)
#     files = [f for f in os.listdir(source_path) if f.endswith('_{}.bvh'.format(suffix))]

#     length = len('_{}.bvh'.format(suffix))
#     for f in files:
#         if dest_suffix is not None:
#             cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '_{}.bvh'.format(dest_suffix)))
#         else:
#             cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '.bvh'))
#         os.system(cmd)


if __name__ == '__main__':
    # test_characters = ['g1'] # ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']

    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_dir', type=str, default='./pretrained/')
    parser.add_argument('--save_dir', type=str, default='./exp/val/')
    args = parser.parse_args()
    prefix = args.save_dir
    args.test_data = 0  
    # 0 for train, 1 for eval data

    cross_dest_path = pjoin(prefix, 'results/cross_structure/')
    intra_dest_path = pjoin(prefix, 'results/intra_structure/')
    source_path = pjoin(prefix, 'results/bvh/')

    cross_error = []
    intra_error = []
    # for i in range(4):
    #     print('Batch [{}/4]'.format(i + 1))
    #     eval(i, prefix)
    eval(prefix)
    #     print('Collecting test error...')
    #     if i == 0:
    #         cross_error += full_batch(0, prefix)
    #         for char in test_characters:
    #             batch_copy(os.path.join(source_path, char), 0, os.path.join(cross_dest_path, char))
    #             batch_copy(os.path.join(source_path, char), 'gt', os.path.join(cross_dest_path, char), 'gt')

    #     intra_dest = os.path.join(intra_dest_path, 'from_{}'.format(test_characters[i]))
    #     for char in test_characters:
    #         for char in test_characters:
    #             batch_copy(os.path.join(source_path, char), 1, os.path.join(intra_dest, char))
    #             batch_copy(os.path.join(source_path, char), 'gt', os.path.join(intra_dest, char), 'gt')

    #     intra_error += full_batch(1, prefix)

    # cross_error = np.array(cross_error)
    # intra_error = np.array(intra_error)

    # cross_error_mean = cross_error.mean()
    # intra_error_mean = intra_error.mean()

    # os.system('rm -r %s' % pjoin(prefix, 'results/bvh'))

    # print('Intra-retargeting error:', intra_error_mean)
    # print('Cross-retargeting error:', cross_error_mean)
    print('Evaluation finished!')

import numpy as np
import joblib
import torch
import glob
import os
from poselib.core.rotation3d import *
# from poselib.core.write_data_to_file import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

def convert_pose_to_quat(root_traj, root_quat, joint_pos):

    frames_array = np.concatenate([np.zeros((root_traj.shape[0], 7)),
                                   joint_pos], axis=1) 
    
    left_hip_pitch = torch.tensor(frames_array[:, 7:8])
    left_hip_roll = torch.tensor(frames_array[:, 8:9])
    left_hip_yaw = torch.tensor(frames_array[:, 9:10])
    left_knee_pitch = torch.tensor(frames_array[:, 10:11])
    left_ankle_pitch = torch.tensor(frames_array[:, 11:12])
    left_ankle_roll = torch.tensor(frames_array[:, 12:13])

    right_hip_pitch = torch.tensor(frames_array[:, 13:14])
    right_hip_roll = torch.tensor(frames_array[:, 14:15])
    right_hip_yaw = torch.tensor(frames_array[:, 15:16])
    right_knee_pitch = torch.tensor(frames_array[:, 16:17])
    right_ankle_pitch = torch.tensor(frames_array[:, 17:18])
    right_ankle_roll = torch.tensor(frames_array[:, 18:19])

    waist_yaw = torch.tensor(frames_array[:, 19:20])
    waist_roll = torch.tensor(frames_array[:, 20:21])
    waist_pitch = torch.tensor(frames_array[:, 21:22])

    left_shoulder_pitch = torch.tensor(frames_array[:, 22:23])
    left_shoulder_roll = torch.tensor(frames_array[:, 23:24])
    left_shoulder_yaw = torch.tensor(frames_array[:, 24:25])
    left_elbow_pitch = torch.tensor(frames_array[:, 25:26])
    left_wrist_roll = torch.tensor(frames_array[:, 26:27])
    left_wrist_pitch = torch.tensor(frames_array[:, 27:28])
    left_wrist_yaw = torch.tensor(frames_array[:, 28:29])

    right_shoulder_pitch = torch.tensor(frames_array[:, 29:30])
    right_shoulder_roll = torch.tensor(frames_array[:, 30:31])
    right_shoulder_yaw = torch.tensor(frames_array[:, 31:32])
    right_elbow_pitch = torch.tensor(frames_array[:, 32:33])
    right_wrist_roll = torch.tensor(frames_array[:, 33:34])
    right_wrist_pitch = torch.tensor(frames_array[:, 34:35])
    right_wrist_yaw = torch.tensor(frames_array[:, 35:36])

    # convert joint angle to quaternion
    left_hip_pitch_quat = quat_from_angle_axis(left_hip_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    left_hip_roll_quat = quat_from_angle_axis(left_hip_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    left_hip_yaw_quat = quat_from_angle_axis(left_hip_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)
    left_knee_pitch_quat = quat_from_angle_axis(left_knee_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    left_ankle_pitch_quat = quat_from_angle_axis(left_ankle_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    left_ankle_roll_quat = quat_from_angle_axis(left_ankle_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)

    right_hip_pitch_quat = quat_from_angle_axis(right_hip_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    right_hip_roll_quat = quat_from_angle_axis(right_hip_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    right_hip_yaw_quat = quat_from_angle_axis(right_hip_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)
    right_knee_pitch_quat = quat_from_angle_axis(right_knee_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    right_ankle_pitch_quat = quat_from_angle_axis(right_ankle_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    right_ankle_roll_quat = quat_from_angle_axis(right_ankle_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    
    waist_yaw_quat = quat_from_angle_axis(waist_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)
    waist_roll_quat = quat_from_angle_axis(waist_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    waist_pitch_quat = quat_from_angle_axis(waist_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)

    left_shoulder_pitch_quat = quat_from_angle_axis(left_shoulder_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    left_shoulder_roll_quat = quat_from_angle_axis(left_shoulder_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    left_shoulder_yaw_quat = quat_from_angle_axis(left_shoulder_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)
    left_elbow_pitch_quat = quat_from_angle_axis(left_elbow_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    left_wrist_roll_quat = quat_from_angle_axis(left_wrist_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    left_wrist_pitch_quat = quat_from_angle_axis(left_wrist_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    left_wrist_yaw_quat = quat_from_angle_axis(left_wrist_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)

    right_shoulder_pitch_quat = quat_from_angle_axis(right_shoulder_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    right_shoulder_roll_quat = quat_from_angle_axis(right_shoulder_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    right_shoulder_yaw_quat = quat_from_angle_axis(right_shoulder_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)
    right_elbow_pitch_quat = quat_from_angle_axis(right_elbow_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    right_wrist_roll_quat = quat_from_angle_axis(right_wrist_roll, torch.tensor([1.0, 0.0, 0.0]), degree=False)
    right_wrist_pitch_quat = quat_from_angle_axis(right_wrist_pitch, torch.tensor([0.0, 1.0, 0.0]), degree=False)
    right_wrist_yaw_quat = quat_from_angle_axis(right_wrist_yaw, torch.tensor([0.0, 0.0, 1.0]), degree=False)

    
    # Construct poselib
    dt = 1/50
    trans = torch.tensor(root_traj)
    poselib_source_file = '/home/nuc/Datasets/CMP/AMASS_g1/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose_g1_g1_29.npy'
    skm = SkeletonMotion.from_file(poselib_source_file)
    skt = skm.skeleton_tree
    rot = quat_identity([len(root_quat), len(skt.node_names)])

    rot[:, 0] = torch.tensor(root_quat)
    rot[:, 1] = left_hip_pitch_quat.squeeze()
    rot[:, 2] = left_hip_roll_quat.squeeze()
    rot[:, 3] = left_hip_yaw_quat.squeeze()
    rot[:, 4] = left_knee_pitch_quat.squeeze()
    rot[:, 5] = left_ankle_pitch_quat.squeeze()
    rot[:, 6] = left_ankle_roll_quat.squeeze()

    rot[:, 7] = right_hip_pitch_quat.squeeze()
    rot[:, 8] = right_hip_roll_quat.squeeze()
    rot[:, 9] = right_hip_yaw_quat.squeeze()
    rot[:, 10] = right_knee_pitch_quat.squeeze()
    rot[:, 11] = right_ankle_pitch_quat.squeeze()
    rot[:, 12] = right_ankle_roll_quat.squeeze()

    rot[:, 13] = waist_yaw_quat.squeeze()
    rot[:, 14] = waist_roll_quat.squeeze()
    rot[:, 15] = waist_pitch_quat.squeeze()

    rot[:, 16] = left_shoulder_pitch_quat.squeeze()
    rot[:, 17] = left_shoulder_roll_quat.squeeze()
    rot[:, 18] = left_shoulder_yaw_quat.squeeze()
    rot[:, 19] = left_elbow_pitch_quat.squeeze()
    rot[:, 20] = left_wrist_roll_quat.squeeze()
    rot[:, 21] = left_wrist_pitch_quat.squeeze()
    rot[:, 22] = left_wrist_yaw_quat.squeeze()

    rot[:, 23] = right_shoulder_pitch_quat.squeeze()
    rot[:, 24] = right_shoulder_roll_quat.squeeze()
    rot[:, 25] = right_shoulder_yaw_quat.squeeze()
    rot[:, 26] = right_elbow_pitch_quat.squeeze()
    rot[:, 27] = right_wrist_roll_quat.squeeze()
    rot[:, 28] = right_wrist_pitch_quat.squeeze()
    rot[:, 29] = right_wrist_yaw_quat.squeeze()

    if False:
        new_state = SkeletonState.from_rotation_and_root_translation(skt, rot, trans, is_local=True)
        new_motion = SkeletonMotion.from_skeleton_state(new_state, fps=np.round(1.0/dt))
        print("===========This is the new motion===========")
        plot_skeleton_motion_interactive(new_motion)

    return trans.numpy(), rot.numpy()



def single_file():
    ## LOAD PKL
    # file1 = './example/ref_data/g1_run9b_10k_779_GS3.pkl'
    # file2 = './logs/icra_policy/779/metrics/ckpt_48000/tmp.pkl'

    file1 = '/home/nuc/MoGym/PBHC/example/ref_data/g1_run9b_5k_558_GS3.pkl'
    file2 = '/home/nuc/MoGym/PBHC/logs/icra_policy/558/metrics/ckpt_59000/tmp.pkl'

    data1 = joblib.load(file1)
    data2 = joblib.load(file2)

    ## GET DATA
    root_traj_1 = data1['cmp_retarget_motion']['root_trans_offset']
    root_traj_2 = data2['motion0']['root_trans_offset']

    root_quat_2 = data2['motion0']['root_rot']
    dof_pose_2 = data2['motion0']['dof']   # (frames, 23)

    ## Recover DATA
    num_frames = dof_pose_2.shape[0]
    joint_pos_2 = np.concatenate([dof_pose_2[:, 0:19], 
                                  np.zeros((num_frames, 3)),
                                  dof_pose_2[:, 19:], 
                                  np.zeros((num_frames, 3))], axis=1) 
    # 23 to 29

    ## Convert to Quat
    root_translation, local_rotation = convert_pose_to_quat(root_traj_2, root_quat_2, joint_pos_2)

    list_of_dicts = []
    list_of_dicts.append({'rot': local_rotation, 'trans': root_translation})
    output_file = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP_finetune/g1.npy'
    # np.save(output_file, list_of_dicts)
    print(f"Final converted {file2} to {output_file}")
    ## plot 3D root trajectory
    if False:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 创建3D坐标系
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')  # 创建3D坐标轴

        ax.set_xlim(-0.2, 2.5)  # 设置X轴范围
        ax.set_ylim(-0.5, 0.5)  # 设置Y轴范围
        ax.set_zlim(-0.0, 1.5) 


        # 绘制第一条轨迹
        ax.scatter(root_traj_1[:, 0], root_traj_1[:, 1], root_traj_1[:, 2], c='blue', label='retarget_traj', alpha=0.7)

        # 绘制第二条轨迹
        ax.scatter(root_traj_2[:, 0], root_traj_2[:, 1], root_traj_2[:, 2], c='red', label='simulated_traj', alpha=0.7)

        # 添加图例
        plt.legend()

        # 添加坐标轴标签（使用ax.set_xlabel等方法）
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')  # 这里使用ax对象的zlabel方法

        # 添加标题
        ax.set_title('3D Trajectory Visualization')

        plt.show()
    return


def multi_files():
    ## LOAD PKL
    input_dir = '/home/nuc/MoGym/PBHC/example/collect_gt'
    output_file = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP_finetune/g1.npy'
    output_file = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP_finetune/smpl.npy'


    pkl_files = glob.glob(os.path.join(input_dir, '**', '*.pkl'), recursive=True)
    pkl_files = sorted(pkl_files)
    list_of_dicts = []

    with open('/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/all_list.txt', 'w', encoding='utf-8') as f:
        for item in pkl_files:
            f.write(item + '\n')

    # 遍历所有 .npy 文件并进行转换
    for pkl_file in pkl_files:
        data1 = joblib.load(pkl_file)

        ## GET DATA
        root_traj_1 = data1['motion0']['root_trans_offset']
        root_quat_1 = data1['motion0']['root_rot']
        dof_pose_1 = data1['motion0']['dof']   # (frames, 23)

        ## Recover DATA
        num_frames = dof_pose_1.shape[0]
        joint_pos_1 = np.concatenate([dof_pose_1[:, 0:19], 
                                    np.zeros((num_frames, 3)),
                                    dof_pose_1[:, 19:], 
                                    np.zeros((num_frames, 3))], axis=1) 

        ## Convert to Quat
        root_translation, local_rotation = convert_pose_to_quat(root_traj_1, root_quat_1, joint_pos_1)

        if True: # make fake smpl
            local_rotation = local_rotation[:, 0:24]
        list_of_dicts.append({'rot': local_rotation, 'trans': root_translation})
    
    np.save(output_file, list_of_dicts)
    print(f"Final converted {len(list_of_dicts)} files to {output_file}")


if __name__ == "__main__":
    single_file()
    # multi_files()



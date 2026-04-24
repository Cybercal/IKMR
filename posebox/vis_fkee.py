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


def transform_from_quaternion(quater: torch.Tensor):
    qw = quater[..., 0]
    qx = quater[..., 1]
    qy = quater[..., 2]
    qz = quater[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def vis_points(ee_id, data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 选择要绘制的帧数
    frames_to_plot = [0, 200, 400, 600]  # 选择几个特定的帧

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(15, 10))

    # 遍历选择的帧
    for i, frame_idx in enumerate(frames_to_plot):
        ax = fig.add_subplot(1, len(frames_to_plot), i + 1, projection='3d')
        points = data[frame_idx].numpy()  # 获取当前帧的点

        # 绘制点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
        ax.scatter(points[ee_id, 0], points[ee_id, 1], points[ee_id, 2], c='r', marker=',')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置标题
        ax.set_title(f'Frame {frame_idx}')

    # 调整子图布局
    plt.tight_layout()
    plt.show()


def vis_joint_link(data, ref_data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    Box_length = 1.5

    # 选择要绘制的帧数
    frames_to_plot = [0]  # 选择几个特定的帧

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(15, 10))

    # 遍历选择的帧
    for i, frame_idx in enumerate(frames_to_plot):
        ax = fig.add_subplot(1, len(frames_to_plot), i + 1, projection='3d')
        points = data[frame_idx].numpy()  # 获取当前帧的点
        points_ref = ref_data[frame_idx].numpy()  # 获取当前帧的点

        # 绘制点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
        ax.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], c='r', marker=',')

        # 连接点
        for j in range(points.shape[0] - 1):
            ax.plot([points[j, 0], points[j + 1, 0]],
                    [points[j, 1], points[j + 1, 1]],
                    [points[j, 2], points[j + 1, 2]], c='b')

        for j in range(points_ref.shape[0] - 1):
            ax.plot([points_ref[j, 0], points_ref[j + 1, 0]],
                    [points_ref[j, 1], points_ref[j + 1, 1]],
                    [points_ref[j, 2], points_ref[j + 1, 2]], c='r')
            
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置标题
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlim3d([- Box_length, Box_length])
        ax.set_ylim3d([- Box_length, Box_length])
        ax.set_zlim3d([- Box_length, Box_length])

    # 调整子图布局
    plt.tight_layout()
    plt.show()


def vis_skeleton(topology, data, ref_data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    Box_length = 1

    # 选择要绘制的帧数
    # frames_to_plot = [0, 20, 40, 60]  # 选择几个特定的帧
    frames_to_plot = [0]
    # 创建一个 3D 图形
    fig = plt.figure(figsize=(40, 10))

    # 遍历选择的帧
    for i, frame_idx in enumerate(frames_to_plot):
        ax = fig.add_subplot(1, len(frames_to_plot), i + 1, projection='3d')
        points = data[frame_idx].numpy()  # 获取当前帧的点
        points_ref = ref_data[frame_idx].numpy()  # 获取当前帧的点

        # 绘制点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
        ax.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], c='r', marker=',')

        # 连接点
        for j, pi in enumerate(topology):
            if pi == -1:    # if root skip
                assert j == 0
                continue

            ax.plot([points[j, 0], points[pi, 0]],
                    [points[j, 1], points[pi, 1]],
                    [points[j, 2], points[pi, 2]], c='b')

            ax.plot([points_ref[j, 0], points_ref[pi, 0]],
                    [points_ref[j, 1], points_ref[pi, 1]],
                    [points_ref[j, 2], points_ref[pi, 2]], c='r')
            
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置标题
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlim3d([- Box_length, Box_length])
        ax.set_ylim3d([- Box_length, Box_length])
        ax.set_zlim3d([- Box_length, Box_length])

    # 调整子图布局
    plt.tight_layout()
    plt.show()


def vis_all_motion(topology1, topology2, ref_datas, res_datas, rtg_datas):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    Box_length = 1
    topology = [topology1, topology2]
    # 选择要绘制的帧数
    frames_to_plot = [0,10, 20, 30, 40,50, 60]  # 选择几个特定的帧
    # frames_to_plot = [0]
    # 创建一个 3D 图形
    fig = plt.figure(figsize=(15, 10))
    
    # ax = fig.add_subplot(111, projection='3d')

    # 遍历选择的帧
    for n in range(len(ref_datas)):
        
        ax = fig.add_subplot(1, len(ref_datas), n + 1, projection='3d')
        # torch.Size([n][1, 64, 22, 3])
        ref_data = ref_datas[n][0, ...].detach().clone().cpu()
        res_data = res_datas[n][0, ...].detach().clone().cpu()
        rtg_data = rtg_datas[n+1][0, ...].detach().clone().cpu()

        for i, frame_idx in enumerate(frames_to_plot):
            
            
            points_ref = ref_data[frame_idx].numpy()  # 获取当前帧的点
            points_res = res_data[frame_idx].numpy()  # 获取当前帧的点
            points_rtg = rtg_data[frame_idx].numpy()  # 获取当前帧的点

            points_res[:, 1] += 0.5
            points_rtg[:, 1] -= 0.5
            # 绘制点
            ax.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], c='r', marker=',')
            ax.scatter(points_res[:, 0], points_res[:, 1], points_res[:, 2], c='b', marker='.')
            ax.scatter(points_rtg[:, 0], points_rtg[:, 1], points_rtg[:, 2], c='g', marker='.')

            # 连接点
            for j, pi in enumerate(topology[n]):
                if pi == -1:    # if root skip
                    assert j == 0
                    continue

                ax.plot([points_ref[j, 0], points_ref[pi, 0]],
                        [points_ref[j, 1], points_ref[pi, 1]],
                        [points_ref[j, 2], points_ref[pi, 2]], c='r')
                
                ax.plot([points_res[j, 0], points_res[pi, 0]],
                        [points_res[j, 1], points_res[pi, 1]],
                        [points_res[j, 2], points_res[pi, 2]], c='b')
                
            for j, pi in enumerate(topology[1-n]):
                if pi == -1:    # if root skip
                    assert j == 0
                    continue

                ax.plot([points_rtg[j, 0], points_rtg[pi, 0]],
                        [points_rtg[j, 1], points_rtg[pi, 1]],
                        [points_rtg[j, 2], points_rtg[pi, 2]], c='g')

            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # 设置标题
            ax.set_title(f'Skeleton {n}')
            ax.set_xlim3d([- Box_length, Box_length])
            ax.set_ylim3d([- Box_length, Box_length])
            ax.set_zlim3d([0, 2 * Box_length])

    # 调整子图布局
    plt.tight_layout()
    plt.show()


# def vis_animation(topology1, topology2, ref_datas, res_datas, rtg_datas):
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     Box_length = 1
#     topology = [topology1, topology2]
#     # 选择要绘制的帧数
#     frames_to_plot = list(range(64))  # 选择所有帧

#     # 创建一个 3D 图形
#     fig = plt.figure(figsize=(15, 10))
    
#     # ax = fig.add_subplot(111, projection='3d')

#     # 遍历两个subplot子图
#     for n in range(len(ref_datas)):
        
#         ax = fig.add_subplot(1, len(ref_datas), n + 1, projection='3d')
#         # torch.Size([n][1, 64, 22, 3])
#         ref_data = ref_datas[n][0, ...].detach().clone().cpu()
#         res_data = res_datas[n][0, ...].detach().clone().cpu()
#         rtg_data = rtg_datas[n+1][0, ...].detach().clone().cpu()

#         # 遍历选择的帧
#         for i, frame_idx in enumerate(frames_to_plot):
            
            
#             points_ref = ref_data[frame_idx].numpy()  # 获取当前帧的点
#             points_res = res_data[frame_idx].numpy()  # 获取当前帧的点
#             points_rtg = rtg_data[frame_idx].numpy()  # 获取当前帧的点

#             points_res[:, 1] += 0.5
#             points_rtg[:, 1] -= 0.5
#             # 绘制点
#             ax.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], c='r', marker=',')
#             ax.scatter(points_res[:, 0], points_res[:, 1], points_res[:, 2], c='b', marker='.')
#             ax.scatter(points_rtg[:, 0], points_rtg[:, 1], points_rtg[:, 2], c='g', marker='.')

#             # 连接点
#             for j, pi in enumerate(topology[n]):
#                 if pi == -1:    # if root skip
#                     assert j == 0
#                     continue

#                 ax.plot([points_ref[j, 0], points_ref[pi, 0]],
#                         [points_ref[j, 1], points_ref[pi, 1]],
#                         [points_ref[j, 2], points_ref[pi, 2]], c='r')
                
#                 ax.plot([points_res[j, 0], points_res[pi, 0]],
#                         [points_res[j, 1], points_res[pi, 1]],
#                         [points_res[j, 2], points_res[pi, 2]], c='b')
                
#             for j, pi in enumerate(topology[1-n]):
#                 if pi == -1:    # if root skip
#                     assert j == 0
#                     continue

#                 ax.plot([points_rtg[j, 0], points_rtg[pi, 0]],
#                         [points_rtg[j, 1], points_rtg[pi, 1]],
#                         [points_rtg[j, 2], points_rtg[pi, 2]], c='g')

#             # 设置坐标轴标签
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')

#             # 设置标题
#             ax.set_title(f'Skeleton {n}')
#             ax.set_xlim3d([- Box_length, Box_length])
#             ax.set_ylim3d([- Box_length, Box_length])
#             ax.set_zlim3d([0, 2 * Box_length])

#     # 调整子图布局
#     plt.tight_layout()
#     plt.show()
Box_length = 1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def update(frame_idx, ref_datas, res_datas, rtg_datas, topology, ax_list):
    # 清除之前的绘图
    for ax in ax_list:
        ax.cla()

    # 绘制每个子图
    for n in range(len(ref_datas)):
        ax = ax_list[n]
        ref_data = ref_datas[n][0, frame_idx].detach().clone().cpu()
        res_data = res_datas[n][0, frame_idx].detach().clone().cpu()
        rtg_data = rtg_datas[n+1][0, frame_idx].detach().clone().cpu()

        points_ref = ref_data.numpy()
        points_res = res_data.numpy()
        points_rtg = rtg_data.numpy()

        points_res[:, 1] += 0.5
        points_rtg[:, 1] -= 0.5

        # 绘制点
        ax.scatter(points_ref[:, 0], points_ref[:, 1], points_ref[:, 2], c='r', marker=',')
        ax.scatter(points_res[:, 0], points_res[:, 1], points_res[:, 2], c='b', marker='.')
        ax.scatter(points_rtg[:, 0], points_rtg[:, 1], points_rtg[:, 2], c='g', marker='.')

        # 连接点
        for j, pi in enumerate(topology[n]):
            if pi == -1:  # if root skip
                assert j == 0
                continue

            ax.plot([points_ref[j, 0], points_ref[pi, 0]],
                    [points_ref[j, 1], points_ref[pi, 1]],
                    [points_ref[j, 2], points_ref[pi, 2]], c='r')

            ax.plot([points_res[j, 0], points_res[pi, 0]],
                    [points_res[j, 1], points_res[pi, 1]],
                    [points_res[j, 2], points_res[pi, 2]], c='b')

        for j, pi in enumerate(topology[1-n]):
            if pi == -1:  # if root skip
                assert j == 0
                continue

            ax.plot([points_rtg[j, 0], points_rtg[pi, 0]],
                    [points_rtg[j, 1], points_rtg[pi, 1]],
                    [points_rtg[j, 2], points_rtg[pi, 2]], c='g')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置标题
        ax.set_title(f'Skeleton {n}')
        ax.set_xlim3d([- Box_length, Box_length])
        ax.set_ylim3d([- Box_length, Box_length])
        ax.set_zlim3d([0, 2 * Box_length])


def vis_animation(topology1, topology2, ref_datas, res_datas, rtg_datas):
    Box_length = 1
    topology = [topology1, topology2]
    frames_to_plot = list(range(64))  # 选择所有帧

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(15, 10))
    ax_list = [fig.add_subplot(1, len(ref_datas), n + 1, projection='3d') for n in range(len(ref_datas))]

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=frames_to_plot, fargs=(ref_datas, res_datas, rtg_datas, topology, ax_list), interval=100)

    # 保存为 mp4 文件
    ani.save('skeleton_animation.mp4', writer='ffmpeg')

    plt.show()



def single_file():
    motion_file = '/home/nuc/Datasets/CMP/AMASS_smpl/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose.npy'
    motion_file = '/home/nuc/Datasets/CMP/AMASS_smpl/2_BioMotionLab_NTroje_rub093_0027_circle_walk_poses_nlpose.npy'
    
    # motion_file = '/home/nuc/Datasets/CMP/AMASS_g1/2_ACCAD_Female1General_c3d_A4___look_stageii_nlpose_g1_g1_29.npy'

    skm = SkeletonMotion.from_file(motion_file)
    skt = skm.skeleton_tree 

    # correct to zero
    if True:
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

    # plot_skeleton_state(SkeletonState.zero_pose(skt))
    # plot_skeleton_motion_interactive(skm)

    # process motion from fk_ee     
    topology = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 11, 17, 18, 19, 20]
    ee_id = [3, 7, 11, 16, 21]
    local_rotation = skm.local_rotation.clone()  # size(frames, joints, 4), q = [x, y, z, w]
    root_translation = skm.root_translation.clone()
    # offset = skt.local_translation.clone().reshape((1, -1, 3, 1))
    offset = skt.local_translation.clone().reshape((1, -1, 3))

    # if smpl
    W = local_rotation.shape[0]
    local_rotation = torch.cat([local_rotation[:, :12, :], local_rotation[:, 14:, :]], dim=1)
    local_rotation[:, :1, :] = torch.tensor((0, 0, 0, 1)).repeat(W, 1, 1)   
    # local_rotation = torch.cat([local_rotation[:, :, 3:], local_rotation[:, :, :3]], dim=2) # q = [w, x, y, z] 

    offset = torch.cat([offset[:, :12, :], offset[:, 14:, :]], dim=1)
    # offset[0,0,:] = torch.tensor([0.00, 0.00,  0.9037])
    offset = offset.repeat(W, 1, 1)
    offset[:,0,:] = root_translation
    
    res = torch.empty([W, len(topology), 3]) # [frames, joints, 3]
    
    
    # transform = transform_from_quaternion(local_rotation)
    # torch.cat[r, t]
    local_transformation = transform_from_rotation_translation(r=local_rotation, 
                                                               t=offset)
    global_transformation = local_transformation.clone()

    for i, pi in enumerate(topology):
        if pi == -1:    # if root skip
            assert i == 0
            continue
        # FK for calculating global transform

        # the original version in SKN
        # transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
        # # transform[..., i, :, :] = torch.matmul(transform[..., i, :, :].clone(), transform[..., pi, :, :].clone())
        # RT = transform[..., i, :3, :3]
        # # RT = torch.transpose(transform[..., i, :3, :3], -2, -1)
        # result[..., i, :] = torch.matmul(RT, offset[..., i, :, :]).squeeze()
        
        # the poselib version
        global_transformation[..., i, :] = transform_mul(global_transformation[..., pi, :], 
                                                            global_transformation[..., i, :],
                                                            )
    res = transform_translation(global_transformation)
    # res[..., 0, :] = root_translation    # root xyz = traj



    # res = res.clone()
    # for i, pi in enumerate(topology):
    #     if pi == 0 or pi == -1:
    #         continue
    #     res[..., i, :] += res[..., pi, :]

    
    # list_of_dicts = [{'rot': local_rotation, 'trans': root_translation}]

    gt1 = skm.global_translation.clone()
    # gt2 = skm.local_translation_to_root.clone()
    gt1 = torch.cat([gt1[:, :12, :], gt1[:, 14:, :]], dim=1).clone()
    # gt2 = torch.cat([gt2[:, :12, :], gt2[:, 14:, :]], dim=1)

    # vis_points(ee_id, res)
    # vis_points(ee_id, gt1)
    # vis_points(ee_id, gt2)

    # vis_joint_link(res[:, :5, :], gt1[:, :5, :])
    vis_skeleton(topology, res, gt1)
    
    # 将列表保存为 .npy 文件
    # np.save("/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/smpl.npy", list_of_dicts)


def check_dataset():
    smpl_path = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/smpl.npy'
    g1_path = '/home/nuc/MoBox/deep-motion-editing/retargeting/datasets/CMP/g1.npy'
    smpl_motions = np.load(smpl_path, allow_pickle=True)
    g1_motions = np.load(g1_path, allow_pickle=True)

    select_idx = [0, 10, 100, 1000, 2000, 2222]
    for idx in select_idx:
        len1 = smpl_motions[idx]['trans'].shape[0]
        len2 = g1_motions[idx]['trans'].shape[0]
        print(f"smpl len: {len1}, g1 len: {len2}, whether same:{len1 == len2}")

    # smpl len: 432, g1 len: 1759, whether same:False
    # smpl len: 249, g1 len: 893, whether same:False
    # smpl len: 242, g1 len: 253, whether same:False
    # smpl len: 783, g1 len: 851, whether same:False
    # smpl len: 990, g1 len: 845, whether same:False
    # smpl len: 1093, g1 len: 236, whether same:False
    return None



if __name__ == '__main__':
    # single_file()
    # check_dataset()

    # if debug SKN motion
    # a = torch.load('/home/nuc/MoBox/pose2bvh/debug/tensor.pt') #.detach().cpu()

    # forward 66, 54, 58 side 88 upper 20
    # ERROE 69
    a = torch.load('/home/nuc/MoBox/deep-motion-editing/retargeting/exp/val/results/motion_dict_54.pth')
    # tensor_dict = {
    #     'motion_denorm': self.motion_denorm,
    #     'pos_ref': self.pos_ref,
    #     'res_denorm': self.res_denorm,
    #     'res_pos': self.res_pos,
    #     'fake_res_denorm': self.fake_res_denorm,
    #     'fake_pos': self.fake_pos
    # }

    # torch.Size([4096, 64, 30, 3])
    topology1 = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 11, 17, 18, 19, 20]
    topology2 = [-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 15, 23, 24, 25, 26, 27, 28]
    
    # vis_skeleton(topology1, a[0][0, ...].detach().cpu(), a[0][1, ...].detach().cpu())
    # vis_skeleton(topology2, a[1][0, ...].detach().cpu(), a[1][2, ...].detach().cpu())
    # vis_all_motion(topology1, topology2,
    #            a['pos_ref'][0][0, ...].detach().cpu(), 
    #            a['res_pos'][0][0, ...].detach().cpu(),
    #            a['fake_pos'][0][0, ...].detach().cpu())
    vis_animation(topology1, topology2, a['pos_ref'], a['res_pos'], a['fake_pos'])
    # vis_all_motion(topology1, topology2, a['pos_ref'], a['res_pos'], a['fake_pos'])


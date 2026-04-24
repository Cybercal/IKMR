import os
import glob
import torch
import warnings
import numpy as np
from pathlib import Path
from bvh_skeleton import math3d
from bvh_skeleton import bvh_helper
from scipy.spatial.transform import Rotation as R
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive


# adjust_q = convert_quaternion(q) 会有万向锁问题
# def convert_quaternion(q):
#     # 定义坐标系变换的旋转矩阵
#     # R_matrix = np.array([
#     #     [0, 1, 0],
#     #     [0, 0, 1],
#     #     [1, 0, 0]
#     # ])
#     # R_matrix = np.array([
#     #     [1, 0, 0],
#     #     [0, 1, 0],
#     #     [0, 0, 1]
#     # ])
#     R_matrix = np.array([
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, -1, 0]
#     ])
#     # 将旋转矩阵转换为四元数
#     rotation = R.from_matrix(R_matrix)
#     q_R = rotation.as_quat()  # q_R 是 (x, y, z, w) 形式的四元数

#     # 将原来的四元数 q 转换为 Rotation 对象
#     original_rotation = R.from_quat(q)

#     # 应用四元数变换 q' = q_R * q * q_R^*
#     q_prime = (R.from_quat(q_R) * original_rotation * R.from_quat(q_R).inv()).as_quat()

#     return q_prime


# 将四元数转换为欧拉角 (XYZ 顺序)
def convert_q2xyz(q):
    rotation = R.from_quat(q)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles

class Poselib_g1:

    def get_bvh_header(self, skt):
        # 提取所有节点信息
        node_names = skt.node_names
        parent_indices = skt.parent_indices.numpy()
        # local_translation = [x.numpy() for x in skt.local_translation]
        # if vis for blender
        local_translation = [100*x.numpy() for x in skt.local_translation]

        # 调整零位
        if True:
            sk = np.array(local_translation)
            z_offset = -sk[:,2].min()*2.2
            y_offset = (sk[:,1].max() - sk[:,1].min())/1.4
        # trans axis
        # local_translation = [np.array([t[1], t[2], t[0]]) for t in local_translation]
        local_translation = [np.array([t[0], t[2], -t[1]]) for t in local_translation]
        local_translation[0][1] += z_offset
        local_translation[0][2] -= y_offset


        # 创建临时节点结构
        nodes = []
        for i in range(len(node_names)):
            nodes.append({
                'name': node_names[i],
                'offset': local_translation[i],
                'parent_idx': parent_indices[i],
                'children': [],
                'is_root': parent_indices[i] == -1,
                'rotation_order': 'xyz'  
            })

        # 填充子节点列表
        for i, node in enumerate(nodes):
            parent_idx = node['parent_idx']
            if parent_idx != -1:
                nodes[parent_idx]['children'].append(i)

        # 标记末端站点（无子节点的节点）
        for node in nodes:
            node['is_end_site'] = len(node['children']) == 0

        # 从根开始递归构建BvhNode树
        def build_bvh_node(node_idx, parent_bvh_node=None):
            node_info = nodes[node_idx]
            is_root = node_info['is_root']
            is_end = node_info['is_end_site']

            # 末端站点不需要rotation_order
            rotation_order = node_info['rotation_order'] # if not is_end else None

            # 当前节点
            current_node = bvh_helper.BvhNode(
                name=node_info['name'],
                offset=node_info['offset'],
                rotation_order=rotation_order,
                children=[],
                parent=parent_bvh_node,
                is_root=is_root,
                is_end_site=is_end
            )

            # 递归构建子节点
            for child_idx in node_info['children']:
                child_node = build_bvh_node(child_idx, current_node)
                current_node.children.append(child_node)

            return current_node

        # 查找根节点
        root_idx = [i for i, n in enumerate(nodes) if n['is_root']][0]
        root_node = build_bvh_node(root_idx)

        # 收集所有节点（按深度优先遍历顺序）
        all_nodes = []
        def collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        collect_nodes(root_node)

        self.header = bvh_helper.BvhHeader(root=root_node, nodes=all_nodes)
        return self.header




    def get_bvh_channels(self, skm):
        # 提取所有帧的旋转和位置数据
        local_rotation = skm.local_rotation.numpy()  # size(frames, joints, 4), q = [x, y, z, w]
        root_translation = skm.root_translation.numpy()

        # 调整零位
        root_translation[:,0] -= root_translation[0,0]
        root_translation[:,1] -= root_translation[0,1]
        root_rotation = skm.global_root_rotation.numpy()
        print(root_translation[100])
        # 调整根节点的平移数据
        # root_translation = np.stack([root_translation[:, 1], root_translation[:, 2], root_translation[:, 0]], axis=-1)
        # root_translation = np.stack([root_translation[:, 0], root_translation[:, 1], root_translation[:, 2]], axis=-1)

        # if vis in blender
        # root_translation = 100 * np.stack([root_translation[:, 0], root_translation[:, 2], -root_translation[:, 1]], axis=-1)
        root_translation = 100 * np.stack([-root_translation[:, 0], root_translation[:, 2], root_translation[:, 1]], axis=-1)
        


        # 调整局部旋转数据
        adjusted_local_rotation = []
        for frame in range(local_rotation.shape[0]):
            adjusted_frame = []
            for i in range(local_rotation.shape[1]):
                if i == 0:
                    q = root_rotation[frame]
                    rot_xyz = convert_q2xyz(q)
                    # rot_xyz = [rot_xyz[0], rot_xyz[2]+180, -rot_xyz[1]]
                    rot_xyz = [rot_xyz[0], rot_xyz[2]+180, -rot_xyz[1]]
                else:
                    q = local_rotation[frame, i]
                    rot_xyz = convert_q2xyz(q)
                    rot_xyz = [rot_xyz[0], rot_xyz[2], -rot_xyz[1]]
                
                
                
                adjusted_frame.append(rot_xyz)
            adjusted_local_rotation.append(adjusted_frame)
        adjusted_local_rotation = np.array(adjusted_local_rotation)


        channels = []
        for frame in range(adjusted_local_rotation.shape[0]):
            frame_values = []
            for i, node in enumerate(self.header.nodes):
                if node.is_root:
                    # 根节点的 XYZ 位置和旋转
                    frame_values.extend(root_translation[frame])
                    frame_values.extend(adjusted_local_rotation[frame, i])
                else:
                    # 其他关节的旋转
                    frame_values.extend(adjusted_local_rotation[frame, i])
            channels.append(frame_values)

        return channels


def main_debug():
    # motion_file = '/home/nuc/Datasets/g1/mixamo/tnpy_fps30/2_2_byHand_Talking_At_Watercooler_nlpose_g1_g1_29_tpose.npy' # '/home/nuc/Datasets/ultron/test6_endstand_double.npy' 
    # motion_file = '/home/nuc/Datasets/CMP/AMASS_smpl/2_KIT_674_wash_back02_poses_nlpose.npy'
    # motion_file = '/home/nuc/Datasets/CMP/AMASS_g1/2_KIT_674_wash_back02_poses_nlpose_g1_g1_29.npy'
    motion_file = '/home/nuc/MoGen/MDM/save/my_20250205_transenc_512/samples_my_20250205_transenc_512_000450000_seed10_amp_prompts/rev_motion/ahead_m92b.npy'
    # motion_file = '/home/nuc/Pictures/bvh/smpl.bvh'
    
    # output_dir = '/home/nuc/Datasets/CMP/bvh' #'/home/nuc/MoBox/pose2bvh/debug'
    output_dir = '/home/nuc/Pictures/bvh'
    
    bvh_file =  'smpl_Tpose.bvh'
    output_file = os.path.join(output_dir, bvh_file)

    skm = SkeletonMotion.from_file(motion_file)
    skt = skm.skeleton_tree 

    g1_skeleton = Poselib_g1()
    header = g1_skeleton.get_bvh_header(skt)
    channels = g1_skeleton.get_bvh_channels(skm)
    bvh_helper.write_bvh(output_file, header, channels, frame_rate=skm.fps)

    # plot_skeleton_motion_interactive(skm)


def main_mult_files():
    input_dir = '/home/nuc/Datasets/g1/mixamo/tnpy_fps30'
    output_dir = '/home/nuc/MoBox/pose2bvh/results'
    npy_files = glob.glob(os.path.join(input_dir, '**', '*.npy'), recursive=True)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有 .npy 文件并进行转换
    for npy_file in npy_files:
        # 获取文件名和输出文件路径
        bvh_file = os.path.basename(npy_file).replace('.npy', '.bvh')
        output_file = os.path.join(output_dir, bvh_file)

        # 读取 SkeletonMotion
        skm = SkeletonMotion.from_file(npy_file)
        skt = skm.skeleton_tree

        # 创建 Poselib_g1 实例
        g1_skeleton = Poselib_g1()
        header = g1_skeleton.get_bvh_header(skt)
        channels = g1_skeleton.get_bvh_channels(skm)

        # 写入 BVH 文件
        bvh_helper.write_bvh(output_file, header, channels, frame_rate=skm.fps)

        print(f"Converted {npy_file} to {output_file}")


def process_file(npy_file, output_dir):
    try:
        with warnings.catch_warnings(record=True) as w:
            # 重新设置警告过滤器
            warnings.simplefilter("always")
            
            # 读取 SkeletonMotion
            skm = SkeletonMotion.from_file(npy_file)
            skt = skm.skeleton_tree

            # 创建 Poselib_g1 实例
            g1_skeleton = Poselib_g1()
            header = g1_skeleton.get_bvh_header(skt)
            channels = g1_skeleton.get_bvh_channels(skm)
            
            # 检查是否存在万向锁警告
            if any(issubclass(ww.category, UserWarning) and "Gimbal lock detected" in str(ww.message) for ww in w):
                print(f"Warning: Gimbal lock detected in file {npy_file}. Skipping this file.")
                return False

            # 获取文件名和输出文件路径
            bvh_file = os.path.basename(npy_file).replace('.npy', '.bvh')
            output_file = os.path.join(output_dir, bvh_file)

            # 写入 BVH 文件
            bvh_helper.write_bvh(output_file, header, channels, frame_rate=skm.fps)
            # print(f"Converted {npy_file} to {output_file}")
            return True

    except Exception as e:
        print(f"Error processing file {npy_file}: {e}")
        return False            


def main_filt_files():
    input_dir = '/home/nuc/Datasets/g1/mixamo/tnpy_fps30'
    output_dir = '/home/nuc/MoBox/pose2bvh/filtered'
    npy_files = glob.glob(os.path.join(input_dir, '**', '*.npy'), recursive=True)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有 .npy 文件并进行转换
    for npy_file in npy_files:
        if process_file(npy_file, output_dir):
            print(f"Processed file {npy_file} successfully.")
        else:
            print(f"Skipped file {npy_file} due to Gimbal lock.")


if  __name__ == '__main__':
    main_debug()
    # main_mult_files()
    # main_filt_files()

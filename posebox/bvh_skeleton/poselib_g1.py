import numpy as np

from . import bvh_helper
from . import math3d


class Poselib_g1:

    def __init__(self):
        self.nodes = 30
        self.rotation_order = 'xyz'

    def from_poselibtree(self, skt):
        # 提取所有节点信息
        node_names = skt.node_names
        parent_indices = skt.parent_indices.numpy()
        local_translation = [x.numpy() for x in skt.local_translation]

        # 创建临时节点结构
        nodes = []
        for i in range(len(node_names)):
            nodes.append({
                'name': node_names[i],
                'offset': local_translation[i],
                'parent_idx': parent_indices[i],
                'children': [],
                'is_root': parent_indices[i] == -1,
                # 默认旋转顺序为'zyx'（根据需要调整逻辑）
                'rotation_order': 'zyx'  
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
            rotation_order = node_info['rotation_order'] if not is_end else None

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

        return bvh_helper.BvhHeader(root=root_node, nodes=all_nodes)


    def convert_to_bvh(self, skm, output_path):
        header = self.get_bvh_header()
        channels = self.get_bvh_channels()
        bvh_helper.write_bvh(output_path, header, channels, frame_rate=30)



    def get_bvh_header(self):
        def _convert_node(poselib_node, parent=None, is_root=False):
            is_end = poselib_node.is_end_site
            children = []
            if not is_end:
                children = [self._convert_node(child, parent=poselib_node)
                            for child in poselib_node.children]
            return bvh_helper.BvhNode(
                name=poselib_node.name,
                offset=poselib_node.offset,
                rotation_order=poselib_node.rotation_order if not is_end else None,
                children=children,
                parent=parent,
                is_root=is_root,
                is_end_site=is_end
            )

        bvh_root = _convert_node(self.hierarchy_root, is_root=True)
        nodes = []
        def _collect_nodes(node):
            nodes.append(node)
            for child in node.children:
                _collect_nodes(child)
        _collect_nodes(bvh_root)
        return bvh_helper.BvhHeader(bvh_root, nodes)

    def get_bvh_channels(self):
        header = self.get_bvh_header()
        channels = []
        for frame in self.motion_frames:
            frame_values = []
            for node in header.nodes:
                if node.is_end_site:
                    continue
                if node.is_root:
                    # 添加根节点的 XYZ 位置和旋转
                    frame_values.extend(frame[node.name]['position'])
                    reordered_rot = []
                    for axis in node.rotation_order:
                        reordered_rot.append(frame[node.name]['rotation'][axis])
                    frame_values.extend(reordered_rot)
                else:
                    # 添加其他关节的旋转（按 rotation_order 顺序）
                    reordered_rot = []
                    for axis in node.rotation_order:
                        reordered_rot.append(frame[node.name]['rotation'][axis])
                    frame_values.extend(reordered_rot)
            channels.append(frame_values)
        return channels
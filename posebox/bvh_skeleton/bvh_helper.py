import os
from pathlib import Path


class BvhNode(object):
    def __init__(
            self, name, offset, rotation_order,
            children=None, parent=None, is_root=False, is_end_site=False
    ):
        # if not is_end_site and \
        #         rotation_order not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
        #     raise ValueError(f'Rotation order invalid.')
        if rotation_order not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
            raise ValueError(f'Rotation order invalid.')

        self.name = name
        self.offset = offset
        self.rotation_order = rotation_order
        self.children = children
        self.parent = parent
        self.is_root = is_root
        self.is_end_site = is_end_site


class BvhHeader(object):
    def __init__(self, root, nodes):
        self.root = root
        self.nodes = nodes


# def write_header(writer, node, level):
#     indent = ' ' * 4 * level
#     if node.is_root:
#         writer.write(f'{indent}ROOT {node.name}\n')
#         channel_num = 6
#     elif node.is_end_site:
#         writer.write(f'{indent}End Site\n')
#         channel_num = 0
#     else:
#         writer.write(f'{indent}JOINT {node.name}\n')
#         channel_num = 3
#     writer.write(f'{indent}{"{"}\n')

#     indent = ' ' * 4 * (level + 1)
#     writer.write(
#         f'{indent}OFFSET '
#         f'{node.offset[0]} {node.offset[1]} {node.offset[2]}\n'
#     )
#     if channel_num:
#         channel_line = f'{indent}CHANNELS {channel_num} '
#         if node.is_root:
#             channel_line += f'Xposition Yposition Zposition '
#         channel_line += ' '.join([
#             f'{axis.upper()}rotation'
#             for axis in node.rotation_order
#         ])
#         writer.write(channel_line + '\n')

#     for child in node.children:
#         write_header(writer, child, level + 1)

#     indent = ' ' * 4 * level
#     writer.write(f'{indent}{"}"}\n')


def write_header(writer, node, level):
    indent = ' ' * 8 * level

    if node.is_root:
        writer.write(f'{indent}ROOT {node.name}\n')
        channel_num = 6
    else:
        writer.write(f'{indent}JOINT {node.name}\n')
        channel_num = 3

    writer.write(f'{indent}{"{"}\n')

    indent = ' ' * 8 * (level + 1)
    writer.write(
        f'{indent}OFFSET '
        f'{node.offset[0]} {node.offset[1]} {node.offset[2]}\n'
    )
    if channel_num:
        channel_line = f'{indent}CHANNELS {channel_num} '
        if node.is_root:
            channel_line += f'Xposition Yposition Zposition '
        if node.rotation_order:
            channel_line += ' '.join([
                f'{axis.upper()}rotation'
                for axis in node.rotation_order
            ])
        writer.write(channel_line + '\n')

    if node.is_end_site and not node.children:
        # 写入 End Site 标记
        indent = ' ' * 8 * (level + 1)
        writer.write(f'{indent}End Site\n')
        writer.write(f'{indent}{{\n')
        indent = ' ' * 8 * (level + 2)
        writer.write(f'{indent}OFFSET 0.000000 0.000000 0.000000\n')
        indent = ' ' * 8 * (level + 1)
        writer.write(f'{indent}}}\n')
    else:
        for child in node.children:
            write_header(writer, child, level + 1)

    indent = ' ' * 8 * level
    writer.write(f'{indent}{"}"}\n')


def write_bvh(output_file, header, channels, frame_rate=30):
    output_file = Path(output_file)
    if not output_file.parent.exists():
        os.makedirs(output_file.parent)

    with output_file.open('w') as f:
        f.write('HIERARCHY\n')
        write_header(writer=f, node=header.root, level=0)

        f.write('MOTION\n')
        f.write(f'Frames: {len(channels)}\n')
        f.write(f'Frame Time: {1 / frame_rate}\n')

        for channel in channels:
            f.write(' '.join([f'{element}' for element in channel]) + '\n')


# a demo how to use the above code
class Demo:

    def get_bvh_header(self):
        def _convert_node(poselib_node, parent=None, is_root=False):
            is_end = poselib_node.is_end_site
            children = []
            if not is_end:
                children = [self._convert_node(child, parent=poselib_node)
                            for child in poselib_node.children]
            return BvhNode(
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
        return BvhHeader(bvh_root, nodes)

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

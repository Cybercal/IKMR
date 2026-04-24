import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# from poselib.core.rotation3d import *
from tools.rotation3d import *

@torch.jit.script
def quatToZyx(q):
  zyx = torch.zeros((q.shape[0], 3), dtype=torch.float)

  for i in range(q.shape[0]):
    qx = q[i, 0]
    qy = q[i, 1]
    qz = q[i, 2]
    qw = q[i, 3]
    temp = min(-2. * (qx * qz - qw * qy), .99999)
    zyx[i, 0] = torch.atan2(2 * (qx * qy + qw * qz), torch.square(qw) + torch.square(qx) - torch.square(qy) - torch.square(qz))
    zyx[i, 1] = torch.asin(temp)
    zyx[i, 2] = torch.atan2(2 * (qy * qz + qw * qx), torch.square(qw) - torch.square(qx) - torch.square(qy) + torch.square(qz))

  return zyx

@torch.jit.script
def normalize_angle_positive(angle):
    """ Normalizes the angle to be 0 to 2*pi
        It takes and returns radians. """
    return torch.fmod(torch.fmod(angle, 2.0*torch.pi) + 2.0*torch.pi, 2.0*torch.pi)

# @torch.jit.script
def normalize_angle(angle):
    """ Normalizes the angle to be -pi to +pi
        It takes and returns radians."""
    a = normalize_angle_positive(angle)
    index = a > torch.pi
    a[index] -= 2.0 * torch.pi
    # if a > torch.pi:
    #     a -= 2.0 * torch.pi
    return a

# @torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

def local_rotation_to_dof(dof_quat, dof_body_ids,
                     num_dof = 20,
                     x_axis_joint = [1, 7, 5, 11, 13, 17],
                     z_axis_joint = [0, 6, 14, 18]):

    numFrame = max(len(sublist) for sublist in dof_quat)
    dof_pos = torch.zeros((numFrame, num_dof), dtype=torch.float)

    for joint_id in range(num_dof):
        id = dof_body_ids[joint_id]
        joint_q = dof_quat[id]

        joint_theta, joint_axis = quat_to_angle_axis(joint_q)

        if (joint_id in z_axis_joint):
            joint_theta = joint_theta * joint_axis[..., 2] # assume joint is along z axis
        elif (joint_id in x_axis_joint):
            joint_theta = joint_theta * joint_axis[..., 0] # assume joint is along x axis
        else:    
            joint_theta = joint_theta * joint_axis[..., 1] # assume joint is along y axis

        joint_theta = normalize_angle(joint_theta)
        dof_pos[:, joint_id] = joint_theta
    
    return dof_pos

def dof_to_local_rotation(dof_pos,
                          num_dof = 20,
                          x_axis_joint = [1, 7, 5, 11, 13, 17],
                          z_axis_joint = [0, 6, 14, 18]):
    dof_quat = torch.zeros((dof_pos.shape[0], dof_pos.shape[1], 4), dtype=torch.float)
    for joint_id in range(num_dof):
        joint_theta = dof_pos[:, joint_id]
        if (joint_id in x_axis_joint):
            joint_axis = torch.tensor(np.array([[1.0, 0.0, 0.0]]))
        elif (joint_id in z_axis_joint):
            joint_axis = torch.tensor(np.array([[0.0, 0.0, 1.0]]))
        else:
            joint_axis = torch.tensor(np.array([[0.0, 1.0, 0.0]]))

        joint_theta = quat_from_angle_axis(joint_theta, joint_axis, degree=False)
        dof_quat[:, joint_id, :] = joint_theta

    return dof_quat

def limit_joint_pos(dof_pos, joint_limit):
    assert len(dof_pos[0]) == len(joint_limit["joint_pos_upper_bound"])
    assert len(dof_pos[0]) == len(joint_limit["joint_pos_lower_bound"])
    limited_joint_pos = dof_pos.clone()

    for joint_id in range(len(dof_pos[0])):
        joint_theta = dof_pos[:, joint_id]

        joint_theta = torch.clamp(joint_theta, joint_limit["joint_pos_lower_bound"][joint_id] + joint_limit["joint_limit_edge"], joint_limit["joint_pos_upper_bound"][joint_id] - joint_limit["joint_limit_edge"])
        limited_joint_pos[:, joint_id] = joint_theta

    return limited_joint_pos

def limit_joint_quat(local_rotation_input, joint_limit, dof_body_ids,
                     num_dof = 20,
                     x_axis_joint = [1, 7, 5, 11, 13, 17],
                     z_axis_joint = [0, 6, 14, 18]):
    local_rotation = local_rotation_input.clone()
    dof_quat = []
    for i in range(local_rotation.shape[1]):
        temp = local_rotation[..., i, :]
        dof_quat.append(temp)

    dof_pos = local_rotation_to_dof(dof_quat, dof_body_ids, num_dof, x_axis_joint, z_axis_joint)
    limited_joint_pos = limit_joint_pos(dof_pos, joint_limit)
    limited_joint_quat_part = dof_to_local_rotation(limited_joint_pos, num_dof, x_axis_joint, z_axis_joint)

    limited_joint_quat = local_rotation.clone()
    for joint_id in range(len(dof_body_ids)):
        id = dof_body_ids[joint_id]
        limited_joint_quat[:, id, :] = limited_joint_quat_part[:, joint_id, :]

    return limited_joint_quat

def limit_joint_pos2(dof_pos, joint_limit):
    assert len(dof_pos[0]) == len(joint_limit["joint_pos_upper_bound"])
    assert len(dof_pos[0]) == len(joint_limit["joint_pos_lower_bound"])
    limited_joint_pos = dof_pos.clone()

    joint_changed = False
    for joint_id in range(len(dof_pos[0])):
        joint_theta = dof_pos[:, joint_id]

        joint_theta_temp = torch.clamp(joint_theta, joint_limit["joint_pos_lower_bound"][joint_id] + joint_limit["joint_limit_edge"], joint_limit["joint_pos_upper_bound"][joint_id] - joint_limit["joint_limit_edge"])
        limited_joint_pos[:, joint_id] = joint_theta_temp

        for i in range(len(joint_theta)):
            if torch.abs(joint_theta[i] - joint_theta_temp[i]) > 1e-4:
                joint_changed = True

    return limited_joint_pos, joint_changed

def limit_joint_quat2(local_rotation_input, joint_limit, dof_body_ids,
                     num_dof = 20,
                     x_axis_joint = [1, 7, 5, 11, 13, 17],
                     z_axis_joint = [0, 6, 14, 18]):
    local_rotation = local_rotation_input.clone()
    dof_quat = []
    for i in range(local_rotation.shape[1]):
        temp = local_rotation[..., i, :]
        dof_quat.append(temp)

    dof_pos = local_rotation_to_dof(dof_quat, dof_body_ids, num_dof, x_axis_joint, z_axis_joint)
    limited_joint_pos, joint_changed = limit_joint_pos2(dof_pos, joint_limit)
    limited_joint_quat_part = dof_to_local_rotation(limited_joint_pos, num_dof, x_axis_joint, z_axis_joint)

    limited_joint_quat = local_rotation.clone()
    for joint_id in range(len(dof_body_ids)):
        id = dof_body_ids[joint_id]
        limited_joint_quat[:, id, :] = limited_joint_quat_part[:, joint_id, :]

    return limited_joint_quat, joint_changed

def joint_quat_to_rad(local_rotation, dof_body_ids,
                      num_dof = 20,
                      x_axis_joint = [1, 7, 5, 11, 13, 17],
                      z_axis_joint = [0, 6, 14, 18]):
    dof_quat = []
    for i in range(local_rotation.shape[1]):
        temp = local_rotation[..., i, :]
        dof_quat.append(temp)
    dof_rad = local_rotation_to_dof(dof_quat, dof_body_ids, num_dof, x_axis_joint, z_axis_joint)

    return dof_rad

def joint_rad_to_quat(body_rotation, joint_rad, dof_body_ids,
                     num_dof = 20,
                     x_axis_joint = [1, 7, 5, 11, 13, 17],
                     z_axis_joint = [0, 6, 14, 18]):
    joint_quat = dof_to_local_rotation(joint_rad, num_dof, x_axis_joint, z_axis_joint)
    body_quat = body_rotation.clone()
    for joint_id in range(len(dof_body_ids)):
        id = dof_body_ids[joint_id]
        body_quat[:, id, :] = joint_quat[:, joint_id, :]
    return body_quat

def create_test_info_file(mocap_fz, desire_fz, play_speed, local_rotation, info_file_name, dof_body_ids):
    dof_quat = []
    for i in range(local_rotation.shape[1]):
        temp = local_rotation[..., i, :]
        dof_quat.append(temp)

    scale = mocap_fz // desire_fz

    info_file_path = "data/info/"
    if not os.path.exists(info_file_path):
        os.makedirs(info_file_path)

    info_file_path += info_file_name
    dof_pos = local_rotation_to_dof(dof_quat, dof_body_ids)
    numFrame = max(len(sublist) for sublist in dof_quat)

    joint_name = ["LL_HIP_Y",
                  "LL_HIP_R",
                  "LL_HIP_P",
                  "LL_KNEE_P",
                  "LL_ANKLE_P",
                  "LL_ANKLE_R",
                  "LR_HIP_Y",
                  "LR_HIP_R",
                  "LR_HIP_P",
                  "LR_KNEE_P",
                  "LR_ANKLE_P",
                  "LR_ANKLE_R",
                  "AL_SHOULDER_P",
                  "AL_SHOULDER_R",
                  "AL_SHOULDER_Y",
                  "AL_ELBOW_P",
                  "AR_SHOULDER_P",
                  "AR_SHOULDER_R",
                  "AR_SHOULDER_Y",
                  "AR_ELBOW_P"]

    with open(info_file_path, "w") as file:
        file.write(f"KeyFrameNum  {numFrame//scale:.0f}\n")
        file.write("KeyFrameTime\n")
        file.write("{\n")
        for i in range(numFrame):
            file.write(f"  ({i},0)  {1/desire_fz*i/play_speed}\n")
        file.write("}\n\n")

        frame_id = 1
        for i, frame in enumerate(dof_pos, start=0):
            if (i % scale == 0):
                file.write(f"KeyFrame{frame_id}\n")
                file.write("{\n")
                file.write("  jointPos\n")
                file.write("  {\n")
                for j, pos in enumerate(frame):
                    file.write(f"    ({j},0)  {pos:.2f}    ; {joint_name[j]}\n")
                file.write("  }\n")
                file.write("}\n\n")
                frame_id+=1

    print("Test info file created successfully!")

    ###### plot joint ######
    # dof_pos_filter = dof_pos.clone()
    # fillter = 0.4
    # for i in range(1, len(dof_pos_filter)):
    #     dof_pos_filter[i, :] = fillter * dof_pos_filter[i, :] + (1- fillter) * dof_pos_filter[i-1, :]

    # # for i in range(len(joint_name)):
    # # for i in [12,15,16,19]:
    # for i in [12,19]:
    #     plt.plot(dof_pos_filter[:, i], label=f'{joint_name[i]}')

    # plt.xlabel('Frame')
    # plt.ylabel('rad')
    # plt.title('Plot dof_pos')
    # plt.legend()
    # plt.show()

def create_special_pos_info_file(mocap_fz, local_rotation, root_translation, dof_body_ids, output_file_info):
    dof_quat = []
    for i in range(local_rotation.shape[1]):
        temp = local_rotation[..., i, :]
        dof_quat.append(temp)

    desire_fz = output_file_info["special_pos_info_desired_fz"]
    play_speed = output_file_info["special_pos_info_play_speed"]
    info_file_name = output_file_info["special_pos_info_file_name"]
    fix_base_pos = output_file_info["fix_base_pos"]
    fix_foot_pos = output_file_info["fix_foot_pos"]
    ypr_stance_upper_bound = output_file_info["ypr_stance_upper_bound"]
    ypr_stance_lower_bound = output_file_info["ypr_stance_lower_bound"]
    transition_time = output_file_info["transition_time"]
    set_mode = output_file_info["set_mode"]
    mode_frame = output_file_info["mode_frame"]
    mode = output_file_info["mode"]
    if (len(mode_frame) != len(mode) + 1):
        print("Error: mode_frame != mode + 1")
        exit()

    basePos = torch.zeros((local_rotation.shape[0], 6), dtype=torch.float)

    if (fix_base_pos):
        local_root_translation = torch.tensor([0, 0, 0.8706], dtype=torch.float32)
    else:
        zyx = quatToZyx(local_rotation[:, 0, :])
        first_yaw = torch.tensor([zyx[0, 0]]) # to do: change yaw to foot direction
        joint_axis = torch.tensor(np.array([[0.0, 0.0, 1.0]]))
        first_yaw_quat = quat_from_angle_axis(first_yaw, joint_axis, degree=False)
        first_base_pos = root_translation[0, :]

        local_root_translation = root_translation.clone()
        local_root_translation = local_root_translation[:, :3] # remove link pos
        local_root_translation[:, 0] -= first_base_pos[0]
        local_root_translation[:, 1] -= first_base_pos[1]
        local_root_translation = quat_rotate(first_yaw_quat, local_root_translation)

    fix_base_rpy = fix_base_pos
    if (fix_base_rpy):
        zyx = torch.zeros((local_rotation.shape[0], 3), dtype=torch.float)
    else:
        zyx = quatToZyx(local_rotation[:, 0, :])
        first_yaw = zyx[0, 0] # to do: change yaw to foot direction
        zyx[:, 0] -= first_yaw
        if (fix_foot_pos):
            for i in range(3):
                zyx[:, i] = torch.clamp(zyx[:, i], ypr_stance_lower_bound[i], ypr_stance_upper_bound[i])

    basePos[:, 0:3] = local_root_translation
    basePos[:, 3:6] = zyx

    info_file_path = "data/info/"
    if not os.path.exists(info_file_path):
        os.makedirs(info_file_path)

    info_file_path += info_file_name
    dof_pos = local_rotation_to_dof(dof_quat, dof_body_ids)
    numFrame = max(len(sublist) for sublist in dof_quat)
    basePosNode = ["x", "y", "z", "yaw", "pitch", "roll"]
    scale = mocap_fz // desire_fz

    joint_name = ["LL_HIP_Y",
                  "LL_HIP_R",
                  "LL_HIP_P",
                  "LL_KNEE_P",
                  "LL_ANKLE_P",
                  "LL_ANKLE_R",
                  "LR_HIP_Y",
                  "LR_HIP_R",
                  "LR_HIP_P",
                  "LR_KNEE_P",
                  "LR_ANKLE_P",
                  "LR_ANKLE_R",
                  "AL_SHOULDER_P",
                  "AL_SHOULDER_R",
                  "AL_SHOULDER_Y",
                  "AL_ELBOW_P",
                  "AR_SHOULDER_P",
                  "AR_SHOULDER_R",
                  "AR_SHOULDER_Y",
                  "AR_ELBOW_P"]

    with open(info_file_path, "w") as file:
        file.write("moiton_config\n")
        file.write("{\n")
        file.write(f"  KeyFrameNum  {(numFrame//scale + 1):.0f}\n")
        file.write("}\n\n")

        frame_id = 1
        mode_id = 0
        for i, frame in enumerate(dof_pos, start=0):
            if (i >= mode_frame[mode_id + 1]):
                print("mode_id: ", mode_id)
                print("len(mode) - 1: ", len(mode) - 1)
                mode_id = min(len(mode) - 1, mode_id + 1)
            if (i % scale == 0):
                file.write(f"KeyFrame{frame_id}\n")
                frame_id+=1
                file.write("{\n")

                if not set_mode:
                    file.write(f"  mode         STANCE\n")
                else:
                    print("                    mode_id: ", mode_id, "  i: ", i, "mode[mode_id]",mode[mode_id])
                    file.write(f"  mode         {mode[mode_id]}\n")

                if (i == 0):
                    file.write(f"  time         {transition_time}\n")
                else:
                    file.write(f"  time         {1/desire_fz/play_speed}\n")
                file.write(f"  oriPosMode   LR\n\n")

                file.write("  basePos\n")
                file.write("  {\n")
                for j in range(0, 6):
                    if (basePos[i, j] >= 0):
                        file.write(f"    ({j},0)  {basePos[i, j]:.2f}     ; {basePosNode[j]}\n")
                    else:
                        file.write(f"    ({j},0)  {basePos[i, j]:.2f}    ; {basePosNode[j]}\n")
                file.write("  }\n\n")

                file.write("  jointPos\n")
                file.write("  {\n")
                for j, pos in enumerate(frame):
                    space = ""
                    if (j < 10):
                        space += " "
                    if (pos >= 0):
                        space += " "
                    file.write(f"    ({j},0)  {pos:.2f}    " + space + ";" + f" {joint_name[j]}\n")
                file.write("  }\n")
                file.write("}\n\n")

        # last transition frame
        basePosTemp = [0, 0, 0.8706, 0, 0, 0]
        posTemp = [0.20, 0.20, -0.51, 1.13, -0.68, 0.00, 0.00, 0.00, -0.51, 1.13, -0.68, 0.00, 0.0, 0.30, 0.00, -0.50, 0.00, -0.30, 0.00, -0.50]

        file.write(f"KeyFrame{frame_id}\n")
        frame_id+=1
        file.write("{\n")

        file.write(f"  mode         STANCE\n")
        file.write(f"  time         {transition_time}\n")
        file.write(f"  oriPosMode   LR\n\n")

        file.write("  basePos\n")
        file.write("  {\n")
        for j in range(0, 6):
            if (basePos[i, j] >= 0):
                file.write(f"    ({j},0)  {basePosTemp[j]:.2f}     ; {basePosNode[j]}\n")
            else:
                file.write(f"    ({j},0)  {basePosTemp[j]:.2f}    ; {basePosNode[j]}\n")
        file.write("  }\n\n")

        file.write("  jointPos\n")
        file.write("  {\n")
        for j in range(len(posTemp)):
            space = ""
            if (j < 10):
                space += " "
            if (pos >= 0):
                space += " "
            file.write(f"    ({j},0)  {posTemp[j]:.2f}    " + space + ";" + f" {joint_name[j]}\n")
        file.write("  }\n")
        file.write("}\n\n")

    print("Special pos info file created successfully!")

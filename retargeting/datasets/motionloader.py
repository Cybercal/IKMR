from torch.utils.data import Dataset
import copy
# from datasets.motion_dataset import MotionData
import os
import numpy as np
import torch
from datasets.bvh_parser import BVH_file
from option_parser import get_std_bvh


class HRdataset0(Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, args, motions, skeleton_idx):
        super(HRdataset0, self).__init__()

        self.motions = motions
        # self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        if args.is_train:
            self.length = motions.shape[0]
        else:
            self.length = len(motions)
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # if self.args.data_augment == 0 or torch.rand(1) < 0.5:
        #     return [self.motions[item], self.skeleton_idx[item]]
        # else:
        #     return [self.motions_reverse[item], self.skeleton_idx[item]]
        return [self.motions[item], self.skeleton_idx[item]]


# human-robot dataset
class HRdataset(Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, args, datasets_groups):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.length = 0
        self.offsets = []
        self.joint_topologies = []
        self.ee_ids = []
        self.means = []
        self.vars = []
        dataset_num = 0
        seed = 114514 #19260817
        total_length = 10000000
        all_datas = []

        for j, datasets in enumerate(datasets_groups):
            offsets_group = []
            means_group = []
            vars_group = []
            dataset_num += len(datasets)
            tmp = []
            for i, dataset in enumerate(datasets):
                new_args = copy.copy(args)
                new_args.data_augment = 0   #no reverse
                new_args.dataset = dataset

                if args.test_data == 0:
                    tmp_data = MotionData(new_args)
                    print('Loading data from train sets')
                else:
                    tmp_data = MotionData(new_args).test_set
                    print('Loading data from test sets')

                tmp.append(tmp_data)

                # mean = np.load('./datasets/Mixamo/mean_var/{}_mean.npy'.format(dataset))
                # var = np.load('./datasets/Mixamo/mean_var/{}_var.npy'.format(dataset))
                # mean = torch.tensor(mean)
                # mean = mean.reshape((1,) + mean.shape)
                # var = torch.tensor(var)
                # var = var.reshape((1,) + var.shape)
                mean = tmp_data.mean
                var = tmp_data.var

                file = BVH_file(get_std_bvh(dataset=dataset))
                if i == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)

                means_group.append(mean)
                vars_group.append(var)
                offsets_group.append(new_offset)

                total_length = min(total_length, len(tmp[-1]))
            all_datas.append(tmp)
            offsets_group = torch.cat(offsets_group, dim=0)
            offsets_group = offsets_group.to(device)
            means_group = torch.cat(means_group, dim=0).to(device)
            vars_group = torch.cat(vars_group, dim=0).to(device)
            self.offsets.append(offsets_group)
            self.means.append(means_group)
            self.vars.append(vars_group)

        for j, datasets in enumerate(all_datas):
            pt = 0
            motions = []
            skeleton_idx = []
            for dataset in datasets:
                motions.append(dataset[:])
                skeleton_idx += [pt] * len(dataset)
                pt += 1
            if args.is_train == True:
                motions = torch.cat(motions, dim=0) # torch.Size([48842, 87, 64]) 
            else:
                motions = motions[0]
            # if self.length != 0 and self.length != len(skeleton_idx):
            #     self.length = min(self.length, len(skeleton_idx))
            # else:
            #     self.length = len(skeleton_idx)
            self.length = len(dataset)
            self.final_data.append(HRdataset0(args, motions, skeleton_idx))

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return data * var + means

    def denorm2(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return data * var[:, 4:, :] + means[:, 4:, :]
    
    def norm2(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return (data[:, 4:, :] - means) / var
    
    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res
    


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(MotionData, self).__init__()
        name = args.dataset

        file_path = './datasets/CMP/{}.npy'.format(name)
        
        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.std_bvh = get_std_bvh(args)
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)
        motions = list(motions)

        self.if_train = args.is_train

        if self.if_train == True:
            self.data = torch.cat(self.get_windows(name, motions, self.if_train))    # torch.Size([511, 64, 91])
            self.data = self.data.permute(0, 2, 1)  # torch.Size([511, 91, 64])

            if args.normalization == 1:
                self.mean = torch.mean(self.data, (0, 2), keepdim=True)
                self.var = torch.var(self.data, (0, 2), keepdim=True)
                self.var = self.var ** (1/2)
                idx = self.var < 1e-5
                self.var[idx] = 1
                self.data = (self.data - self.mean) / self.var
            else:
                self.mean = torch.mean(self.data, (0, 2), keepdim=True)
                self.mean.zero_()
                self.var = torch.ones_like(self.mean)

            # if False:   # only for first time
            # torch.save(self.mean, './datasets/CMP/mean_var/{}_mean.pth'.format(name))
            # torch.save(self.var, './datasets/CMP/mean_var/{}_var.pth'.format(name))
        else:
            self.data = self.get_windows(name, motions, self.if_train)   # list

        self.mean = torch.load('./datasets/CMP/mean_var/{}_mean.pth'.format(name))
        self.var = torch.load('./datasets/CMP/mean_var/{}_var.pth'.format(name))

        # if split dataset
        # train_len = self.data.shape[0] * 95 // 100
        # self.test_set = self.data[train_len:, ...]
        # self.data = self.data[:train_len, ...]
        # self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        self.reset_length_flag = 0
        # self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.if_train == True:
            return self.data.shape[0]
        else:
            return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        # if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
        #     return self.data[item]
        # else:
        #     return self.data_reverse[item]
        return self.data[item]

    def get_windows(self, name, motions, if_train):
        new_windows = []

        for motion_dict in motions:
            motion = motion_dict['rot'] 
            # size(frames, joints, 4), q = [x, y, z, w]
            # motion = motion[..., [3, 0, 1, 2]]    # q = [w, x, y, z]
            traj = motion_dict['trans'] # size(frames, 3)
            frames = motion.shape[0]

            if name == 'smpl':
                motion = np.concatenate((motion[:, :12, :], motion[:, 14:, :]), axis=1)

            # linear interpolate
            # Here, if 30fps True, if 50fps False
            if True:
                times = (frames - 1) / 30
                phase = np.around(np.arange(0, times, 0.02) * 30, decimals=2)
                idx_low, idx_high = np.floor(phase), np.ceil(phase)
                blend = np.around(phase - idx_low, decimals=2)
                blend = torch.tensor(blend, dtype=torch.float32).unsqueeze(1)

                motion_left = motion[idx_low.astype(int), :, :]
                motion_right = motion[idx_high.astype(int), :, :]

                motion_new = []
                for j in range(motion.shape[1]):
                    motion_new.append(self.slerp(motion_left[:, j, :], motion_right[:, j, :], blend)) 
                motion = torch.cat(motion_new, dim=1)

                traj_left = torch.tensor(traj[idx_low.astype(int), :], dtype=torch.float32)
                traj_right = torch.tensor(traj[idx_high.astype(int), :], dtype=torch.float32)
                traj = (1.0 - blend) * traj_left + blend * traj_right
            else:
                motion = torch.tensor(motion, dtype=torch.float32)
                traj = torch.tensor(traj, dtype=torch.float32)

            if if_train == 1:
                motion = torch.cat((motion[:, 1:, :].reshape(motion.shape[0], -1), traj), dim=1)
                self.total_frame += motion.shape[0]
                # motion = self.subsample(motion)
                self.motion_length.append(motion.shape[0])
                step_size = self.args.window_size // 2
                window_size = step_size * 2
                n_window = motion.shape[0] // step_size - 1
                for i in range(n_window):
                    begin = i * step_size
                    end = begin + window_size
                    
                    new = np.copy(motion[begin:end, :])
                    new[:,-3:-1] = new[:,-3:-1] - new[32,-3:-1]

                    # new = new.reshape(new.shape[0], -1)
                    new = new[np.newaxis, ...]

                    new_window = torch.tensor(new, dtype=torch.float32)
                    new_windows.append(new_window)
                    
            else:
                motion = torch.cat((motion.reshape(motion.shape[0], -1), traj), dim=1)
                self.total_frame += motion.shape[0]
                self.motion_length.append(motion.shape[0])
                new_windows.append(motion.unsqueeze(0).permute(0, 2, 1))

        
        return new_windows

    def subsample(self, motion):
        return motion[::2, :]

    def slerp(self, q0, q1, t):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        q0 = torch.tensor(q0, dtype=torch.float32)
        q1 = torch.tensor(q1, dtype=torch.float32)
        
        cos_half_theta = torch.sum(q0 * q1, dim=-1)

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
        ratioB = torch.sin(t * half_theta) / sin_half_theta
        
        new_q = ratioA * q0 + ratioB * q1

        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

        return new_q.unsqueeze(1)



    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans

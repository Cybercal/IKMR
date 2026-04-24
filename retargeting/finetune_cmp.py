import os
import sys

# if terminal
sys.path.append('./retargeting/')

# if debug
# current_working_dir = os.getcwd()
# sys.path.append(current_working_dir)
# print(f"当前工作目录: {current_working_dir}")
# new_working_dir = os.path.join(current_working_dir, 'retargeting/')
# os.chdir(new_working_dir)
# print(f"当前工作目录: {new_working_dir}")

from torch.utils.data.dataloader import DataLoader
from models import create_model
from datasets.motionloader import HRdataset
import option_parser
import os
from option_parser import try_mkdir
import time


def get_skeleton_names(args):
    characters = [['smpl'], ['g1']]
    return characters


def build_dataset(args, characters):
    # load human-robot datasets ['smpl', 'g1']
    dataset = HRdataset(args, characters)

    # get modified datasets topology part
    # dataset.joint_topologies[0] = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 11, 17, 18, 19, 20)
    # dataset.ee_ids[0] = [3, 7, 11, 16, 21]

    # dataset.joint_topologies[1] = (-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 15, 23, 24, 25, 26, 27, 28)
    # dataset.ee_ids[1] = [6, 12, 15, 22, 29]

    return dataset


def main():
    args = option_parser.get_args()
    characters = get_skeleton_names(args)

    log_path = os.path.join(args.save_dir, 'logs/')
    try_mkdir(args.save_dir)
    try_mkdir(log_path)

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))

    dataset = build_dataset(args, characters)    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = create_model(args, characters, dataset)
    model.load(epoch=20000)
    model.setup()

    start_time = time.time()
    
    args.epoch_num = 1000
    args.batch_size = 512
    for epoch in range(args.epoch_begin, args.epoch_num):
        for step, motions in enumerate(data_loader):
            model.set_input(motions)
            # model.optimize_parameters()
            model.finetune_parameters()
            if args.verbose:
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args.epoch_num, step, len(data_loader)), res)

        if epoch % 100 == 0 or epoch == args.epoch_num - 1:
            model.save()

        model.epoch()

    end_tiem = time.time()
    print('training time', end_tiem - start_time)


if __name__ == '__main__':
    main()

import os
from models import create_model
from datasets import create_dataset, get_character_names
import option_parser
import torch
from tqdm import tqdm


def eval(eval_seq, save_dir, test_device='cpu'):
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq
    args.save_dir = save_dir
    character_names = get_character_names(args)

    dataset = create_dataset(args, character_names)
    
    # get modified datasets topology part  
    new_topo = list(dataset.joint_topologies[0])
    new_topo = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19]
    dataset.joint_topologies[0] = tuple(new_topo)
    dataset.ee_ids[0] = [4, 8, 12, 16, 20]

    temp_list = list(dataset.joint_topologies[1])
    temp_list [0] = 0
    dataset.joint_topologies[1] = tuple(temp_list)
    dataset.ee_ids[1] = [6, 12, 15, 22, 29]

    model = create_model(args, character_names, dataset)
    model.load(epoch=20000)

    for i, motions in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(motions)
        model.test()


if __name__ == '__main__':
    parser = option_parser.get_parser()
    args = parser.parse_args()
    eval(args.eval_seq, args.save_dir, args.cuda_device)

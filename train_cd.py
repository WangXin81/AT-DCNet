from argparse import ArgumentParser
from models.trainer import *
import random
from datasets import dataloader_utils
from curve import curve

print(torch.cuda.is_available())
""" 
the main function for training the CD networks
"""

CUDA_LAUNCH_BLOCKING=1
def train(args_in):
    dataloaders = dataloader_utils.get_loaders(args_in)
    model = CDTrainer(args=args_in, dataloaders=dataloaders)
    model.train_models()


def test(args_in):
    from models.evaluator import CDEvaluator
    dataloader = dataloader_utils.get_loader(args_in.data_name, img_size=args_in.img_size,
                                             batch_size=args_in.batch_size, is_train=False,
                                             split='test')
    model = CDEvaluator(args=args_in, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='AT_DCNet-LEVIR-CD', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='data_levir', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    # model
    parser.add_argument('--embed_dim', default=64, type=int)

    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='AT_DCNet', type=str,
                        help='DMINet|MT_CDNet| '
                             'base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--loss', default='ce', type=str)
    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--weight1', default='6', type=float)
    parser.add_argument('--weight2', default='4', type=float)
    parser.add_argument('--gamma', default='2', type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str, help='linear | step')
    parser.add_argument('--lr_decay_iters', default=50, type=int)
    # parser.add_argument('--seed', default=2024, type=int)

    args = parser.parse_args()
    # seed_torch(args.seed)
    dataloader_utils.get_device(args)
    # print(args.gpu_ids)
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)

    curve(args)

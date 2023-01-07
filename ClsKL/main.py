import numpy as np
import argparse
import random
import torch
import os

# My functions
from ClsKL.train import train_kl
from ClsKL.test import test_kl
import ClsKL.utils.utils_functions as uf
from ClsKL.utils import plots


# Ignore randomness
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Fine-Tune')
    parser.add_argument('--net_type', type=str, default="vgg")
    parser.add_argument('--depth', type=str, default="19")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay_epoch', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--data_dir', type=str, default='proj/data/ClsKLData/kneeKL224')
    parser.add_argument('--model_dir', type=str, default='saving_models')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--wloss', type=int, default=1)  # 0- CE, 1- weighted_loss, 2- OCE
    parser.add_argument('--session', type=int, default=0)
    parser.add_argument('--diagonal', type=int, default=42)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--number_of_labels', type=int, default=5)
    parser.add_argument('--const_number', type=int, default=3)
    parser.add_argument('--labels_for_const', type=list, default=[3])
    parser.add_argument('--early_stopping', type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    set_seed(0)
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

    # cost_matrix = uf.cost_matrix_generator(5)

    args.cost_matrix = np.array([[1, 3, 5, 7, 9],
                                 [3, 1, 3, 5, 7],
                                 [5, 3, 1, 3, 5],
                                 [7, 5, 3, 1, 3],
                                 [9, 7, 5, 3, 1]])

    args.fault_price = uf.fault_price_generator(5)
    args.model_name = '{}-{}-{}-{}'.format(args.net_type, args.depth, args.optim, args.wloss)

    # Train NN (create weights):
    train_kl.train_build_models(args)

    # Validate and test the model over tha framework:
    best_epoch = train_kl.phases_build_all_criterions(args)
    print('The best epoch is: ' + str(best_epoch))

    # Test old printing:
    # args.phase = 'test'
    # args.batch_size = 16
    # test_kl.test_models(args, best_epoch)

    # Create more graphs
    # plots.create_external_graphs(args)

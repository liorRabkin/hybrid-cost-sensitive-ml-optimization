import os, sys, pdb
import argparse
import torch
from torchvision import models
import numpy as np

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

# My functions
from ClsKL.utils.loader import data_load
from ClsKL.utils.eval_eng import gen_vis_loc, gen_grad_cam
from ClsKL.utils.eval_eng import eval_test


def test_models(args, best_epoch):
    best_models_path = os.path.join(args.model_dir, args.model_name) #, str(0))
    assert os.path.exists(best_models_path), "Model does not exist"

    dset_loaders, dset_size, num_class = data_load(args)


    # Cost
    cost_path = os.path.join(best_models_path, 'cost')
    assert os.path.exists(cost_path), "Cost Models does not exist"
    best_epoch_cost_path = os.path.join(best_models_path, 'cost', 'model-epoch_number_' + str(best_epoch))

    model = torch.load(best_epoch_cost_path)
    model.cuda()
    model.eval()
    # Evaluate model
    print('---Evaluate Cost model : {}--'.format(args.phase))
    _, _, _, _ = eval_test(args, model, dset_loaders, dset_size, args.phase)


    # Mse
    mse_path = os.path.join(best_models_path, 'mse', os.listdir(os.path.join(best_models_path, 'mse'))[0])

    model = torch.load(mse_path)
    model.cuda()
    # Evaluate model
    print('---Evaluate Mse model : {}--'.format(args.phase))
    _, _, _, _ = eval_test(args, model, dset_loaders, dset_size, args.phase)



    # Generate saliency visulization
    # gen_vis_loc(args, phase, dset_loaders, dset_size, args.save_dir)
    # gen_grad_cam(args, phase, dset_loaders, dset_size, args.save_dir)
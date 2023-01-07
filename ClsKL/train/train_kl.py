import os, sys
import numpy as np
import torch
import pandas as pd
import shutil

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

# My functions
from ClsKL.utils.loader import data_load
from ClsKL.utils.model import cls_model
from ClsKL.utils.train_eng import train_model
from ClsKL.utils.train_eng import stopping_epoch
# from utils.model import resnet
from ClsKL.ml_and_or.ml_to_or import ml_and_or
import ClsKL.utils.utils_functions as uf
from ClsKL.utils import plots
from ClsKL.utils.eval_eng import eval_test


def train_build_models(args):
    """Train the model - create file with weights for each epoch"""

    print('--Phase 0: Argument settings--')

    print('--Phase 1: Data prepration--')
    dset_loaders, dset_size, num_class = data_load(args)
    args.num_class = num_class
    # uf.data_distribution(dset_loaders)

    print('--Phase 2: Model setup--')
    model = cls_model(args)
    model.cuda()

    print('--Phase 3: Model training--')
    train_model(args, model, dset_loaders, dset_size)
    # plots.plot_loss_results(args, model, dset_loaders, dset_size)
    # plots.plot_acc_func_epoch(args.num_epoch, acc_train_epochs, acc_val_epochs)


def phases_build_all_criterions(args): #, stopping_epoch):
    """Validate and Test the framework -
    create plot ofcost results as function of epochs,
     transition matrices, excel with detailed information,
    and find stopping epoch according to validation results"""

    if args.wloss == 0:
        loss = 'CE'
    elif args.wloss == 1:
        loss = 'weighted_loss'

    classes_for_const = ''
    for cons_class in args.labels_for_const:
        classes_for_const += '_' + str(cons_class)


    excel_title = str(loss) + '_' + str(args.num_epoch) + '_epochs_and_stop_after_' + str(args.early_stopping) + '_epochs_const_on_class' + classes_for_const + '_with_const_' + str(args.const_number) + '%'

    draw_path = os.path.join(args.model_dir, args.model_name, 'maps_const_'+str(args.const_number)+'%_on_class'+classes_for_const)
    os.makedirs(draw_path, exist_ok=True)

    dict_all_phases_cost = {}
    dict_all_phases_acc = {}
    best_epoch = 0

    for phase in ['train', 'val', 'test']:
        df_epochs = build_pd_from_model(args, phase)

        print('--Phase 3: Ml to Or--')
        dict_all_phases_cost[phase], dict_all_phases_acc[phase], ml_lab, or_lab = \
            ml_and_or(args, df_epochs, args.cost_matrix, args.fault_price, args.const_number, args.number_of_labels, excel_title, phase, args.labels_for_const)
        # dict_all_phases_cost[phase], dict_all_phases_acc[phase], ml_lab, or_lab = build_ml_and_or(args, phase, df_epochs, excel_title)

        if phase != 'train':

            if phase == 'val':
                print(f'--Phase 4: Find stopping epoch for val--')
                best_epoch = stopping_epoch(dict_all_phases_cost['val'], args.early_stopping)
                # best_epoch = 12
            print(f'--Phase 5: Mistakes matrix for {phase}--')
            print(f'--Phase 5: Mistakes matrix for {phase}--')
            mistakes_real_ml = plots.mistakes_matrix(df_epochs[phase + ' labels epoch' + str(best_epoch)],
                                                     ml_lab[phase + ' epoch' + str(best_epoch)], args.number_of_labels)
            mistakes_ml_or = plots.mistakes_matrix(ml_lab[phase + ' epoch' + str(best_epoch)],
                                                   or_lab[phase + ' epoch' + str(best_epoch)], args.number_of_labels)
            mistakes_real_or = plots.mistakes_matrix(df_epochs[phase + ' labels epoch' + str(best_epoch)],
                                                     or_lab[phase + ' epoch' + str(best_epoch)], args.number_of_labels)
            mistakes = [mistakes_real_ml, mistakes_ml_or, mistakes_real_or]


            plots.draw_maps(mistakes, args.number_of_labels, phase + '_' + excel_title, draw_path)
            print('end draw maps')

    print('--Phase 6: Plots--')
    plots_path = os.path.join(args.model_dir, args.model_name, 'plots_const_'+str(args.const_number)+'%_on_class'+classes_for_const)
    os.makedirs(plots_path, exist_ok=True)
    print('plot cost and acc as function of epochs')
    plots.crit_as_epochs(dict_all_phases_cost, 'cost', plots_path, excel_title, best_epoch)
    plots.crit_as_epochs(dict_all_phases_acc, 'acc', plots_path, excel_title, best_epoch)

    return best_epoch


def build_pd_from_model(args, phase):

    print('--Phase 1: Data prepration--')
    dset_loaders, dset_size, num_class = data_load(args)

    print('--Phase 2: Model loading--')
    best_models_path = os.path.join(args.model_dir, args.model_name)  # , str(0))
    assert os.path.exists(best_models_path), "Model does not exist"
    # list_path_epochs = os.listdir(os.path.join(best_models_path, 'cost'))

    df_epochs = pd.DataFrame()
    max_epoch = args.num_epoch

    for epoch in range(max_epoch):
        path = os.path.join(best_models_path, 'cost', 'model-epoch_number_' + str(epoch + 1))
        model = torch.load(path)
        model.cuda()
        model.eval()
        acc, mse, outputs_all, labels_all = eval_test(args, model, dset_loaders, dset_size, phase)
        df_epochs[phase + ' epoch' + str(epoch+1)] = pd.Series(outputs_all)
        df_epochs[phase + ' labels epoch' + str(epoch+1)] = pd.Series(labels_all)
        # print(f'{phase} epoch number {epoch+1}')

    print('df_epochs')
    print(df_epochs)
    return df_epochs
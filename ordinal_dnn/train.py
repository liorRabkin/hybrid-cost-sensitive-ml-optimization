import os, sys

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

# My functions
from ordinal_dnn.utils.loader import data_load
from ordinal_dnn.utils.model import cls_model
from ordinal_dnn.utils.train_eng import train_model
# from utils.model import resnet


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

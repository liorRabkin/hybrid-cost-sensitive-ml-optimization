# My functions
from ordinal_dnn.utils.loader import data_load
from ordinal_dnn.utils.model import cls_model
from ordinal_dnn.utils.train_eng import train_model


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

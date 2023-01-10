import os, sys, pdb
import shutil
import torch
from torch import nn, optim
from torch.autograd import Variable
import time

# My functions
from ordinal_dnn.utils.torch_util import LRScheduler
from ordinal_dnn.utils.loss_util import weighted_loss
from ordinal_dnn.utils.eval_eng import eval_test


def stopping_epoch(val_df, early_stopping):
    best_cost = 0
    for index, row in val_df.iterrows():

        if row['epoch'] == 1:
            best_cost = row['or cost']
            best_epoch = row['epoch']

        # find stopping epoch
        if best_cost - row['or cost'] >= best_cost * 0.025:  # 0.05:
            best_epoch = row['epoch']
            best_cost = row['or cost']

        if row['epoch'] - best_epoch == early_stopping:
            print(f'The best epoch is: {best_epoch}')
            return int(best_epoch)
    return int(best_epoch)


def train_model(args, model, dset_loaders, dset_size):
    cost_matrix = args.cost_matrix
    best_acc, best_epoch_for_mse, best_epoch_for_cost, best_mse, best_stop_cost = 0, 0, 0, 1., cost_matrix[
        0, cost_matrix.shape[0] - 1]
    best_models_path = os.path.join(args.model_dir, args.model_name)  # , str(args.session))

    # Clean best_models_path from old models
    if os.path.exists(best_models_path):
        shutil.rmtree(best_models_path)

    cost_path = os.path.join(best_models_path, 'cost')
    if os.path.exists(cost_path):
        shutil.rmtree(cost_path)
    os.makedirs(cost_path)

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, momentum=0.9)

    lr_scheduler = LRScheduler(args.lr, args.lr_decay_epoch)

    for epoch in range(args.num_epoch):
        since = time.time()
        print('Epoch {}/{}'.format(str(epoch + 1), args.num_epoch))
        for phase in ['train', 'val']:
            list_labels = []
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)
            else:
                model.train(False)

            running_loss, running_corrects = .0, .0

            for batch_num, data in enumerate(dset_loaders[phase]):
                # if batch_num >= 50:
                #     break
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                list_labels.append(labels.detach().cpu().tolist())

                optimizer.zero_grad()
                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, preds = torch.max(outputs.data, 1)

                if args.wloss == 0:
                    loss = criterion(outputs, labels)
                elif args.wloss == 1:
                    loss = weighted_loss(outputs, labels, args, cost_matrix)
                else:
                    raise NotImplementedError("choose wloss 0 or 1")

                if phase == 'train':
                    loss.backward()

                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            phase_acc, phase_mse, outputs_all, labels_all = eval_test(args, model, dset_loaders, dset_size, phase)

            # save best mse model
            if phase == "val" and phase_mse <= best_mse:
                best_acc = phase_acc  # epoch_acc
                best_epoch_for_mse = epoch
                best_mse = phase_mse

                mse_data = 'best_mse_at_epoch_' + str(best_epoch_for_mse)
                val_metric_str = "-" + str(epoch).zfill(2) + '-' + str(round(best_acc, 3))
                test_metric_str = "-" + str(round(phase_acc, 3)) + "-" + str(round(phase_mse, 3)) + ".pth"
                best_mse_path = os.path.join(best_models_path, 'mse')
                if os.path.exists(best_mse_path):
                    shutil.rmtree(best_mse_path)
                os.makedirs(best_mse_path)
                path = os.path.join(best_mse_path, mse_data + val_metric_str + test_metric_str)
                torch.save(model.cpu(), path)
                model.cuda()

            # end phase train vs val

        # Save model for each epoch
        metric_str = 'model-epoch_number_' + str(epoch + 1)
        path = os.path.join(cost_path, metric_str)
        torch.save(model.cpu(), path)
        model.cuda()

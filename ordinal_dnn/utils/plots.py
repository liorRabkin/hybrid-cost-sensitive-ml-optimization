import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.ticker as ticker

# My functions
from ordinal_dnn.utils.train_eng import stopping_epoch


def plot_acc_func_epoch(num_epochs, acc_train_epochs, acc_val_epochs):
    x = [i for i in range(num_epochs)]
    plt.plot(x, acc_train_epochs)
    plt.plot(x, acc_val_epochs)
    plt.legend(["Dataset 1", "Dataset 2"])
    plt.show()


def mistakes_matrix(vector1, vector2, num_of_labels):
    mistakes = np.zeros((num_of_labels, num_of_labels))
    for i in range(len(vector1)):
        mistakes[vector1[i], vector2[i]] += 1
    return mistakes


def draw_maps(mistakes, num_of_labels, name, draw_path):
    sum_instances = sum([sum(i) for i in mistakes[0]])

    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[0][i, j] != 0:
                if mistakes[0][i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[0][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='w')
                else:
                    ax.text(j, i, int(mistakes[0][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

    ax.imshow(mistakes[0], cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Actual class', fontsize=12)
    ax.set_xlabel('Predicted class', fontsize=12)
    # ax.set_title('True vs ML')
    fig.savefig(os.path.join(draw_path, 'true_ml_' + name + '.png'))

    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[1][i, j] != 0:
                if mistakes[1][i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[1][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='w')
                else:
                    ax.text(j, i, int(mistakes[1][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

    image = ax.imshow(mistakes[1], cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Predicted class by ML model', fontsize=12)
    ax.set_xlabel('Predicted class by OR model', fontsize=12)
    # ax.set_title('ML vs OR')
    fig.savefig(os.path.join(draw_path, 'ml_or_' + name + '.png'))

    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[2][i, j] != 0:
                if mistakes[2][i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[2][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='w')
                else:
                    ax.text(j, i, int(mistakes[2][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

    image = ax.imshow(mistakes[2], cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Actual class', fontsize=12)
    ax.set_xlabel('Predicted class', fontsize=12)
    # ax.set_title('True vs OR')
    fig.savefig(os.path.join(draw_path, 'true_or_' + name + '.png'))

    # plt.show()


def crit_as_epochs(dict_phases, crit, path, title, best_epoch):
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()

    # make a plot
    ax.plot(dict_phases['train']['epoch'], dict_phases['train']['ml ' + crit], color='#176ccd', label='Train CS_VGG-19')
    ax.plot(dict_phases['train']['epoch'], dict_phases['train']['or ' + crit], color='#176ccd', linestyle='dotted',
            label='Train Hyb_CS')

    # set x-axis label
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)

    ax.plot(dict_phases['val']['epoch'], dict_phases['val']['ml ' + crit], color='#cd7817', label='Val CS_VGG-19')
    ax.plot(dict_phases['val']['epoch'], dict_phases['val']['or ' + crit], color='#cd7817', linestyle='dotted',
            label='Val Hyb_CS')

    ax.axvline(x=best_epoch, color='black', linestyle='dashed', label=f'Selected epoch')  # {best_epoch}')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    ax.set_ylim([0.3, 0.85])

    # save the plot as a file
    fig.savefig(os.path.join(path, f'{crit}__{title}.png'))

    fig2, ax2 = plt.subplots()

    # make a plot
    ax2.plot(dict_phases['test']['epoch'], dict_phases['test']['ml ' + crit], color='red', label='test ml')
    ax2.plot(dict_phases['test']['epoch'], dict_phases['test']['or ' + crit], color='blue', label='test or')
    # set x-axis label
    ax2.set_xlabel("Epochs", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.legend()

    # save the plot as a file
    fig2.savefig(os.path.join(path, f'{crit}__test__{title}.png'))


def crit_as_epochs_all_constraints(paths, args, constraints, phase, crit, type):
    dict = {}

    for excel_path in paths:
        excel_files = os.listdir(excel_path)
        for file in excel_files:

            matches = [phase, "20"]

            if all(x in file for x in matches):
                print(file)
                constraint = file.split('const_')[2].split('%')[0]
                algo = file.split('_20')[0].split('val_')[1]

                path = os.path.join(excel_path, file)

                read_total_sheet = pd.read_excel(path, sheet_name='total')
                df = pd.DataFrame()
                df['or ' + crit] = read_total_sheet['OR real cost'][:20]
                df['ml ' + crit] = read_total_sheet['ML (max_likelihood) real cost'][:20]
                print(df)
                df['epoch'] = pd.Series([i + 1 for i in range(read_total_sheet.shape[1] - 2)])
                if constraint == '100':
                    best_epoch = stopping_epoch(df, args.early_stopping)

                dict[constraint + '%'] = df

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()

    if type == 'CE':
        colors = ['#dcbd9d', '#d88c64', '#966735', '#3e2b16']  # orange
    else:
        colors = ['#9de3e6', '#2baba3', '#1b5d6c', '#2b33ab']  # , '#2c2182'] #blue

    # make a plot
    for ind, name in enumerate(constraints):
        if name == '100%':
            ax.plot(dict[name]['epoch'], dict[name]['or ' + crit],
                    label='W/o constraints', color=colors[ind])
        else:
            ax.plot(dict[name]['epoch'], dict[name]['or ' + crit],
                    label=' $n_4$=' + constraints[ind], color=colors[ind])

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # set x-axis label
    ax.set_xlabel("Epochs", fontsize=12)
    # set y-axis label
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_ylim([1.8, 2.8])

    ax.legend()
    # save the plot as a file
    fig.savefig(os.path.join('/home/dsi/liorrabkin/projects/thesis_server/saving_models',
                             crit + ' as function of epochs ' + algo + ' algorithm.png'))


def moves_bars(paths, args, groups, phase, class_num):
    equal = []
    pos = []
    neg = []

    for excel_path in paths:
        excel_files = os.listdir(excel_path)
        for file in excel_files:

            matches = [phase, "20"]

            if all(x in file for x in matches):
                path = os.path.join(excel_path, file)

                read_total_sheet = pd.read_excel(path, sheet_name='total')
                df_cost = pd.DataFrame()
                df_cost['or cost'] = read_total_sheet['OR real cost']
                df_cost['ml cost'] = read_total_sheet['ML (max_likelihood) real cost']
                df_cost['epoch'] = pd.Series([i + 1 for i in range(read_total_sheet.shape[1])])
                best_epoch = stopping_epoch(df_cost, args.early_stopping)
                best_epoch_row_name = 'val epoch' + str(best_epoch)

                read_total_sheet = read_total_sheet.set_index('type')

                best_epoch_row = read_total_sheet.loc[best_epoch_row_name]

                equal.append(np.floor(best_epoch_row['equal - no moves'] * 1000) / 1000)
                pos.append(np.ceil(best_epoch_row['pos move'] * 1000) / 1000)
                neg.append(np.floor(best_epoch_row['neg move'] * 1000) / 1000)

    neg[0] = neg[0] + 0.001
    equal[3] = equal[3] + 0.001
    pos[3] = pos[3] - 0.001

    N = len(groups)
    print(F'N {N}')
    ind = np.arange(N)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, equal, width, label='Equal', color='#bdbdbd')
    p2 = ax.bar(ind, pos, width, bottom=equal, label='Pos', color='#ffffff')  # cd7817') #b5f38d')
    p3 = ax.bar(ind, neg, width, bottom=[x + y for x, y in zip(equal, pos)], label='Neg',
                color='#292929')  # 176ccd') #ef796a')

    ax.set_ylabel('Percentage of samples', fontsize=12)
    # ax.set_title('Percentage equal/pos/neg moves')
    ax.set_xticks(ind, labels=groups, fontsize=10)
    ax.legend()

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p2, label_type='center')
    ax.bar_label(p3, color='w', label_type='center')

    plt.show()
    fig.savefig(os.path.join('/home/dsi/liorrabkin/projects/thesis_server/saving_models',
                             'moves_bars_constraints_class_' + class_num + '.png'))


def steps_ml_or_bars(paths, args, names, class_num):
    steps_ce = []
    steps_ord = []
    for excel_path in paths:
        excel_files = os.listdir(excel_path)
        for file in excel_files:

            matches = ["val", "20"]

            if all(x in file for x in matches):
                path = os.path.join(excel_path, file)

                read_total_sheet = pd.read_excel(path, sheet_name='total')
                df_cost = pd.DataFrame()
                df_cost['or cost'] = read_total_sheet['OR real cost']
                df_cost['ml cost'] = read_total_sheet['ML (max_likelihood) real cost']
                df_cost['epoch'] = pd.Series([i + 1 for i in range(read_total_sheet.shape[1])])
                best_epoch = stopping_epoch(df_cost, args.early_stopping)
                best_epoch_row_name = 'val epoch' + str(best_epoch)

                read_total_sheet = read_total_sheet.set_index('type')

                best_epoch_row = read_total_sheet.loc[best_epoch_row_name]

                if 'vgg-19-SGD-0' in excel_path:
                    steps_ce.append(np.round(best_epoch_row['steps ML OR'], 3))

                else:
                    steps_ord.append(np.round(best_epoch_row['steps ML OR'], 3))

    N = len(names)
    print(F'N {N}')
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    p1 = plt.bar(ind - 0.2, steps_ce, width, label='CE', color='#cd7817')  # e9d6c3') #orange
    p2 = plt.bar(ind + 0.2, steps_ord, width, label='OL', color='#176ccd')  # c6eff0') #blue

    plt.xticks(ind, names, fontsize=12)
    ax.set_ylabel('Mean Steps Number', fontsize=12)
    ax.legend(loc=4)

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1)
    ax.bar_label(p2)

    plt.show()
    fig.savefig(os.path.join('/home/dsi/liorrabkin/projects/thesis_server/saving_models',
                             'Steps_ML_OR_constraints_class_' + class_num + '.png'))


def create_external_graphs(args):
    root_dir_CE = '/home/dsi/liorrabkin/projects/thesis_server/saving_models/vgg-19-SGD-0'
    root_dir_ord = '/home/dsi/liorrabkin/projects/thesis_server/saving_models/vgg-19-SGD-1'

    x_names = ['VGG-19', 'CS_VGG-19', r'VGG-19 $n_3=3\%$', 'CS_VGG-19 $n_3=3\%$']

    paths1 = [os.path.join(root_dir_CE, 'excels_const_100%_on_class_0'),
              os.path.join(root_dir_ord, 'excels_const_100%_on_class_0'),
              os.path.join(root_dir_CE, 'excels_const_1.5%_on_class_4'),
              os.path.join(root_dir_ord, 'excels_const_1.5%_on_class_4'),
              os.path.join(root_dir_CE, 'excels_const_1%_on_class_4'),
              os.path.join(root_dir_ord, 'excels_const_1%_on_class_4'),
              os.path.join(root_dir_CE, 'excels_const_0.5%_on_class_4'),
              os.path.join(root_dir_ord, 'excels_const_0.5%_on_class_4')]

    paths2 = [os.path.join(root_dir_CE, 'excels_const_100%_on_class_0'),
              os.path.join(root_dir_ord, 'excels_const_100%_on_class_0'),
              os.path.join(root_dir_CE, 'excels_const_3%_on_class_3'),
              os.path.join(root_dir_ord, 'excels_const_3%_on_class_3')]

    paths3 = [os.path.join(root_dir_CE, 'excels_const_100%_on_class_0'),
              os.path.join(root_dir_CE, 'excels_const_0.5%_on_class_4'),
              os.path.join(root_dir_CE, 'excels_const_1%_on_class_4'),
              os.path.join(root_dir_CE, 'excels_const_1.5%_on_class_4')]

    paths4 = [os.path.join(root_dir_ord, 'excels_const_100%_on_class_0'),
              os.path.join(root_dir_ord, 'excels_const_0.5%_on_class_4'),
              os.path.join(root_dir_ord, 'excels_const_1%_on_class_4'),
              os.path.join(root_dir_ord, 'excels_const_1.5%_on_class_4')]



    constraints = ['100%', '1.5%', '1%', '0.5%']
    phase = 'val'
    crit = 'cost'
    type = 'OL'

    moves_bars(paths2, args, x_names, phase, '3')
    steps_ml_or_bars(paths1, args, constraints, '4')
    crit_as_epochs_all_constraints(paths4, args, constraints, phase, crit, type)

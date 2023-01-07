from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score


def fault_price_generator(num_of_labels):
    fault_price = np.zeros(num_of_labels)
    for i in range(num_of_labels):
        fault_price[i] = 0
    return fault_price


def cost_matrix_generator(num_of_labels):
    cost_matrix = np.zeros((num_of_labels, num_of_labels))
    for i in range(num_of_labels):
        for j in range(num_of_labels):
                cost_matrix[i, j] = abs(j - i)-1
    return cost_matrix


def cost_pred_vector(cost_matrix, real_label, pred_label):
    cost_vector = np.zeros(len(real_label))
    for i in range(len(real_label)):
        cost_vector[i] = cost_matrix[real_label[i], pred_label[i]]
    return cost_vector


def cost_pred_matrix(cost_matrix, one_predict_vect, two_predict_matrix):
    cost_vector = np.zeros(len(one_predict_vect))
    for i in range(len(one_predict_vect)):
        for j in range(two_predict_matrix.shape[1]):
            cost_vector[i] += two_predict_matrix[i, j] * cost_matrix[one_predict_vect[i], j]
    return cost_vector


def min_cost_label(cost_matrix, NUM_OF_LABELS, ml_predict_prob):
    cost_eachSample_eachLabel = np.zeros((ml_predict_prob.shape[0], NUM_OF_LABELS))
    for s in range(ml_predict_prob.shape[0]):
        for i in range(NUM_OF_LABELS):
            for j in range(NUM_OF_LABELS):  # ml_predict_prob.shape[1]
                cost_eachSample_eachLabel[s, i] += ml_predict_prob[s, j] * cost_matrix[j, i]
    predict_labels = np.argmin(cost_eachSample_eachLabel, axis=1)

    return predict_labels, cost_eachSample_eachLabel


def indices(NUM_OF_LABELS: int, labels: np.ndarray, predict_hard: np.ndarray,
            cost_matrix: np.ndarray) -> Dict[str, Any]:
    # Cost
    cost_sum = cost_pred_vector(cost_matrix, labels, predict_hard)
    # AUC
    auc = np.zeros(NUM_OF_LABELS)
    for i in range(NUM_OF_LABELS):
        predict_labels_binary = predict_hard == i
        labels_binary = labels == i
        auc[i] = roc_auc_score(labels_binary, predict_labels_binary)
    # Accuracy
    acc = np.equal(labels, predict_hard).astype(int)
    return {'cost': cost_sum, 'auc': auc, 'acc': acc}


def data_distribution(dset_loaders):

    for phase in ['train', 'val', 'test']:

        print(f'{phase} Dataset lengh is: {len(dset_loaders[phase].dataset)}')

        labels = [x[1] for x in dset_loaders[phase].dataset.imgs]
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))
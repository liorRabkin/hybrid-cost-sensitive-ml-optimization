import torch
import numpy as np


def weighted_loss(outputs, labels, args, cost_matrix):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    cls_weights = np.array([[1, 3, 5, 7, 9],
                            [3, 1, 3, 5, 7],
                            [5, 3, 1, 3, 5],
                            [7, 5, 3, 1, 3],
                            [9, 7, 5, 3, 1]], dtype=np.float)

    batch_num, class_num = outputs.size()

    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot)
    class_hot = torch.autograd.Variable(class_hot).cuda()
    loss = torch.sum((prob_pred * class_hot) ** 2) / batch_num

    return loss

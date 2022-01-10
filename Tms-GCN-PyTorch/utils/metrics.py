import torch
import numpy as np
from numpy.core.fromnumeric import size


def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)


def mape(pred, y):
    pre = pred.cpu().numpy()
    y = y.cpu().numpy()
    sum = 0
    n = y.size
    for i in range(len(y)):
        for j in range(len(y[0])):
            if y[i][j] != 0:
                sum += np.absolute((pre[i][j] - y[i][j]) / y[i][j])
            else:
                n -= 1
    return sum / n

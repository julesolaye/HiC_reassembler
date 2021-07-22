import numpy as np

np.seterr(divide="ignore")


def count_0(matrix):

    counts = np.zeros(matrix.shape[0])
    for k in range(0, matrix.shape[0]):

        row = matrix[k, :]
        counts[k] = len(np.where(row == 0)[0])
    return counts


def white_index(matrix):

    size_mat = matrix.shape[0]
    percent = 0.9

    counts = count_0(matrix)

    return np.where(counts >= size_mat * percent)[0]

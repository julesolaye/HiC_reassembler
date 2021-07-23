# Several little functions which are useful.

import numpy as np
np.seterr(divide="ignore")

def count_0(matrix : "np.ndarray((N,N))") -> "np.ndarray(N)":
    """
    This function counts the number of zeros for each line/column of a symetric matrix.

    Parameters
    ----------
    matrix: "np.ndarray((N,N))"
        The symetric matrix where you want to count the number of zeros of each element
    """

    counts = np.zeros(matrix.shape[0])
    for k in range(0, matrix.shape[0]):

        row = matrix[k, :]
        counts[k] = len(np.where(row == 0)[0])
    return counts


def white_index(matrix : "np.ndarray((N,N))", percent = 0.9) -> np.ndarray():
    
    """
    This function returns the index of lign/column of a symetric matrix considered as "white index"
    (when the percent of zeros is high in this lign/column).

    Parameters
    ----------
    matrix: "np.ndarray((N,N))"
        The symetric matrix where you want to return the "white index".

    percent: 
        The percentage thresold when you consider that the percent of zeros is high.
    """

    size_mat = matrix.shape[0]
    counts = count_0(matrix)

    return np.where(counts >= size_mat * percent)[0]

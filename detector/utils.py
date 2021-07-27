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


def white_index(matrix : "np.ndarray((N,N))", percent = 0.9) -> np.ndarray:
    
    """
    This function returns the index of lign/column of a symetric matrix near element 
    considered as "white index" (when the percent of zeros is high in this lign/column).

    Parameters
    ----------
    matrix: "np.ndarray((N,N))"
        The symetric matrix where you want to return the "white index".

    percent: 
        The percentage thresold when you consider that the percent of zeros is high.
    """

    size_mat = matrix.shape[0]
    counts = count_0(matrix)

    index_near_white = np.concatenate((np.where(counts >= size_mat * percent)[0]-1, 
                                        np.where(counts >= size_mat * percent)[0],
                                            np.where(counts >= size_mat * percent)[0]+1)) # To take into account index near the white index

    index_near_white = np.sort(np.unique(index_near_white))
    
    return index_near_white


def delete_index(array_to_modify: np.ndarray, array_bad_element: np.ndarray) -> np.ndarray : 
    """
    This function allow to delete the element of the first array which are in the
    second array.

    Parameters
    ----------
    array_to_modify: np.ndarray
        Array where we want to delete elements.

    array_bad_element: 
        Array with the elements to delete.
    """

    for bad_element in array_bad_element:

        array_to_modify = np.delete(array_to_modify, np.where(array_to_modify == bad_element))
    return array_to_modify

# Functions linked to the deletion (updating and correction).

import numpy as np
from Bio import Seq
from typing import Iterable

def fragment_del(coord_del_1 : int, coord_del_2 : int):
    """
    Returns the coordinates of the local matrix we will take for our linear regression.

    Parameters
    ----------
    coord_del_1: int
        One coordinate of the deletion (may be the end or the beginning) in the Hi-C matrix.

    coord_del_2: int
        The other coordinate of the deletion (may be the end or the beginning) in the Hi-C matrix.

    coord_ins : int
        Coordinate of the insertion linked to the translocation in the Hi-C matrix.
    """
    # Sort values    
    coords_matrix_del = np.sort(np.array([coord_del_1, coord_del_2]))
    coord_matrix_start_del = coords_matrix_del[0]
    coord_matrix_end_del = coords_matrix_del[1]

    ## Add coordinates of one fragment modified for the linear regression
    fragment = np.array( ### NOT OPTIMAL FRAGMENT
    [
        [
            coord_matrix_start_del,
            coord_matrix_end_del,
        ],
        [
            coord_matrix_start_del,
            coord_matrix_end_del,
        ],
    ]
    )
    return fragment

def correct_deletion_matrix(start: int, end: int, matrix: "np.ndarray[N,N]") -> "np.ndarray[N,N]":
    """
    Correction of deletion in the Hi-C matrix.

    Parameters
    ----------
    start: int
        Coordinate of the beginning of the deletion.

    end: int
        Coordinate of the end of the deletion.

    matrix : np.ndarray
        Matrix where we want to correct the deletion.
    """

    matrix = np.concatenate((matrix[:start, :], matrix[end:, :],), axis=0)
    matrix = np.concatenate((matrix[:, :start], matrix[:, end:],), axis=1)

    return matrix

def correct_deletion_sequence(start: int, end: int, mutseq: Seq.MutableSeq) -> Seq.MutableSeq:
        """
        Correction of deletions in the sequence. 

        Parameters
        ----------
        start: int
            Coordinate of the beginning of the deletion.

        end: int
            Coordinate of the end of the deletion.

        mutseq: Seq.MutableSeq
            Sequence where we want to correct the deletion.   
        """

        mutseq = mutseq[0:start] + "N"*(end+1-start) + mutseq[end:]
        return mutseq


def update_coords_del(start: int, end: int, coords: Iterable[int]) -> "np.ndarray[int]":
    """
    Update coordinates after applying a deletion at specified positions.

    Parameters
    ----------
    start: int
        Coordinate of the beginning of the deletion.

    end: int
        Coordinate of the end of the deletion.

    coords: Iterable[int]
        Coordinates we want to update.

    Examples
    --------
    >>> update_coords_del(10, 15, [6, 18, 22])
    array([ 6, 13, 17])
    """
    # Shift coordinates on the right of DEL region
    del_size = end - start
    coords = np.array(coords)
    coords_edit = coords[coords > start]
    coords[coords > start] = coords_edit - np.minimum(del_size, coords_edit - start)
    coords[coords < 0] = 0

    return coords
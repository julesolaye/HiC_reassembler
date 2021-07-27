# Functions linked to the back translocation (updating and correction).

import numpy as np
from Bio import Seq
from typing import Iterable

def fragment_tra_back(coord_del_1 : int, coord_del_2 : int, coord_ins : int):
    """
    Returns the coordinates of the local matrix we will take for our linear regression.

    Parameters
    ----------
    coord_del_1: int
        One coordinate of the deletion (may be the end or the beginning) linked to the translocation in the Hi-C matrix.

    coord_del_2: int
        The other coordinate of the deletion (may be the end or the beginning) linked to the translocation in the Hi-C matrix.

    coord_ins : int
        Coordinate of the insertion linked to the translocation in the Hi-C matrix.
    """
    # Sort values    
    coords_matrix_del = np.sort(np.array([coord_del_1, coord_del_2]))
    coord_matrix_start_del = coords_matrix_del[0]
    coord_matrix_end_del = coords_matrix_del[1]

    ## Add coordinates of one fragment modified for the linear regression
    fragment = np.array(
        [
            [
                coord_ins
                - 2 * coord_matrix_end_del
                + 2 * coord_matrix_start_del,
                coord_ins
                - coord_matrix_end_del
                + coord_matrix_start_del,
            ],
            [
                coord_ins
                - coord_matrix_end_del
                + coord_matrix_start_del,
                coord_ins,
            ],
        ]
    )
    return fragment
def correct_back_translocation_scrambled(start: int, end: int, start_paste: int, matrix: "np.ndarray[N,N]"
) -> "np.ndarray[N,N]":
    """
    Correction of back translocation in the HiC-matrix.
    Parameters
    ----------
    start: int
        Coordinate of the beginning of the deletion linked to the translocation.

    end: int
        Coordinate of the end of the deletion linked to the translocation.

    start_paste: int
        Coordinate of the end of the deletion linked to the translocation.

    matrix : np.ndarray
        Matrix where we want to correct the back translocation.
    """

    size_insertion = end - start + 1
    fragment_to_modify_1 = np.concatenate(
        (
            matrix[start : end + 1, :start],
            matrix[start : end + 1, end + 1 : start_paste],
        ),
        axis=1,
    )  #  Fragment at the left of the DEL-separation
    fragment_to_modify_2 = matrix[
        start : end + 1, start : end + 1
    ]  # Square
    fragment_to_modify_3 = matrix[
        start : end + 1, start_paste:
    ]  # Fragment at the right of the DEL-separation
    # Delete rows/cols
    matrix = np.concatenate(
        (matrix[0:start, :], matrix[end + 1 :, :]), axis=0
    )
    matrix = np.concatenate(
        (matrix[:, 0:start], matrix[:, end + 1 :]), axis=1
    )
    # Insertion white spaces
    matrix = np.concatenate(
        (
            matrix[: start_paste + 1 - size_insertion, :],
            np.zeros((size_insertion, matrix.shape[1])),
            matrix[start_paste + 1 - size_insertion :, :],
        ),
        axis=0,
    )
    matrix = np.concatenate(
        (
            matrix[:, : start_paste + 1 - size_insertion],
            np.zeros((matrix.shape[0], size_insertion)),
            matrix[:, start_paste + 1 - size_insertion :],
        ),
        axis=1,
    )
    # Ceation fragment that we will insert in the white spaces
    insertion_1 = np.concatenate(
        (
            fragment_to_modify_1,
            np.zeros(
                (
                    size_insertion,
                    matrix.shape[1] - fragment_to_modify_1.shape[1],
                )
            ),
        ),
        axis=1,
    )
    insertion_2 = np.concatenate(
        (
            np.zeros((size_insertion, start_paste - size_insertion)),
            fragment_to_modify_2,
            np.zeros((size_insertion, matrix.shape[1] - start_paste)),
        ),
        axis=1,
    )
    insertion_3 = np.concatenate(
        (
            np.zeros(
                (
                    size_insertion,
                    matrix.shape[1] - fragment_to_modify_3.shape[1],
                )
            ),
            fragment_to_modify_3,
        ),
        axis=1,
    )
    insertion = np.concatenate(
        (
            np.zeros((start_paste - size_insertion, matrix.shape[1])),
            insertion_1 + insertion_2 + insertion_3,
            np.zeros(
                (matrix.shape[0] - start_paste, matrix.shape[1])
            ),
        ),
        axis=0,
    )
    insertion_transposed = np.concatenate(
        (
            np.zeros((start_paste - size_insertion, matrix.shape[1])),
            insertion_1 + insertion_3,
            np.zeros(
                (matrix.shape[0] - start_paste, matrix.shape[1])
            ),
        ),
        axis=0,
    ).T

    matrix = matrix + insertion + insertion_transposed

    # Correction of traces
    matrix[start_paste - size_insertion, :] = 0
    matrix[:, start_paste - size_insertion] = 0
    if (
        start_paste < matrix.shape[0]
    ):  #  Sometimes, the coord is exactly scrambled.shape so there is nothing to delete after.
        matrix[start_paste, :] = 0
    if start_paste < matrix.shape[1]:
        matrix[:, start_paste] = 0
    matrix[start, :] = 0
    matrix[:, start] = 0
    if end < matrix.shape[0]:
        matrix[end, :] = 0
    if end < matrix.shape[1]:
        matrix[:, end] = 0
    return matrix


def translocation(start_cut: int, end_cut: int, start_paste: int, genome: str) -> str:
    """
    Correction of insertions in the sequence. 

    Parameters
    ----------
    start_cut: int
        Coordinate of the beginning of the deletion linked to the translocation.

    end_cut: int
        Coordinate of the end of the deletion linked to the translocation.

    start_paste: int
        Coordinate of the insertion linked to the translocation.
    """

    tra_size = end_cut - start_cut

    seq_cut = genome[start_cut:end_cut]

    mutseq = genome[:start_cut] + genome[end_cut:]

    if start_paste >= end_cut:
        start_paste -= tra_size  # Update coords

    mutseq = mutseq[:start_paste] + seq_cut + mutseq[start_paste:]

    return mutseq


def update_coords_tra(
    start_cut: int, end_cut: int, start_paste: int, coords: Iterable[int]
) -> "np.ndarray[int]":
    """
    Update coordinates after applying a translocation at specified positions.

    Parameters
    ----------
    start_cut: int
        Coordinate of the beginning of the deletion linked to the translocation.

    end_cut: int
        Coordinate of the end of the deletion linked to the translocation.

    start_paste: int
        Coordinate of the insertion linked to the translocation.

    coords: Iterable[int]
        Coordinates we want to update.

    Examples
    --------
    >>> update_coords_tra(12, 20, 3, [6, 18, 22])
    array([14,  9, 22])
    """

    coords = np.array(coords)

    min_SV_breakpoint = min(start_cut, end_cut, start_paste)
    max_SV_breakpoint = max(start_cut, end_cut, start_paste)
    inter_SV_breakpoint = sorted([start_cut, end_cut, start_paste])[
        1
    ]  #  Intermediate breakpoint

    coords_before_inter = (coords < inter_SV_breakpoint) & (coords >= min_SV_breakpoint)
    coords_after_inter = (coords >= inter_SV_breakpoint) & (coords < max_SV_breakpoint)

    coords[coords_before_inter] += max_SV_breakpoint - inter_SV_breakpoint
    coords[coords_after_inter] -= inter_SV_breakpoint - min_SV_breakpoint

    return coords
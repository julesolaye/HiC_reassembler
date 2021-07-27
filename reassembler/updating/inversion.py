# Functions linked to the inversion (updating and correction).

import numpy as np
from Bio import Seq
from typing import Iterable

def fragment_inv(coord_1 : int, coord_2 : int, shape_matrix : int):
    """
    Returns the coordinates of the local matrix we will take for our linear regression.

    Parameters
    ----------
    coord_1: int
        One coordinate of the inversion (may be the end or the beginning) in the Hi-C matrix.

    end: int
        The other coordinate of the inversion (may be the end or the beginning) in the Hi-C matrix.

    shape_matrix : int
        Shape of the Hi-C matrix used for our linear regression.
    """
    # Sort values    
    coords_matrix = np.sort(np.array([coord_1, coord_2]))

    ## Add coordinates of one fragment modified for the linear regression
    if (3 * coords_matrix[0] - 2 * coords_matrix[1] > 0) & (
        2 * coords_matrix[1] - coords_matrix[0]
        <= shape_matrix
    ):

        fragment = np.array(
            [
                [
                    3 * coords_matrix[0] - 2 * coords_matrix[1],
                    coords_matrix[0],
                ],
                [
                    coords_matrix[0],
                    2 * coords_matrix[1] - coords_matrix[0],
                ],
            ]
        )

    elif (
        2 * coords_matrix[0] - coords_matrix[1] > 0
    ):  # Sometimes, values taken to create the fragment can be negative
        # (so the fragment doesn't exist)
        fragment = np.array(
            [
                [
                    2 * coords_matrix[0] - coords_matrix[1],
                    coords_matrix[0],
                ],
                [coords_matrix[0], coords_matrix[1]],
            ]
        )

    else:
        fragment = np.array(
            [
                [0, coords_matrix[1]],
                [coords_matrix[1], 2 * coords_matrix[1]],
            ]
        )

    return fragment

def correct_inversion_matrix(start: int, end: int, matrix: "np.ndarray[N,N]") -> "np.ndarray[N,N]":
    """
    Correction of inversions in the HiC-matrix.

    Parameters
    ----------
    start: int
        Coordinate of the beginning of the inversion.

    end: int
        Coordinate of the end of the inversion.

    matrix : np.ndarray
        Matrix where we want to correct the inversion.
    """

    matrix[start : end + 1, :] = matrix[start : end + 1, :][::-1, :]

    matrix[:, start : end + 1] = matrix[:, start : end + 1][:, ::-1]

    matrix[start, :] = 0
    matrix[:, start] = 0

    if end < matrix.shape[0]:
        matrix[end, :] = 0

    if end < matrix.shape[1]:
        matrix[:, end] = 0

    return matrix


def correct_inversion_sequence(start: int, end: int, mutseq: Seq.MutableSeq) -> Seq.MutableSeq:
        """
        Correction of inversions in the sequence. 

        Parameters
        ----------
        start: int
            Coordinate of the beginning of the inversion.

        end: int
            Coordinate of the end of the inversion.

        mutseq: Seq.MutableSeq
            Sequence where we want to correct the inversion    
        """

        mutseq[start:end] = Seq.reverse_complement(mutseq[start:end])
        return mutseq


def update_coords_inv(start: int, end: int, coords: Iterable[int]) -> "np.ndarray[int]":
    """Update coordinates after applying an inversion at specified positions

    Parameters
    ----------
    start: int
        Coordinate of the beginning of the inversion.

    end: int
        Coordinate of the end of the inevrsion.

    coords: Iterable[int]
        Coordinates we want to update.
    
    Examples
    --------
    >>> update_coords_inv(12, 20, [4, 18, 22])
    array([ 4, 14, 16])
    """
    mid = (end + start) // 2
    coords = np.array(coords)
    coords_edit = coords[(coords >= start) & (coords <= end)]
    coords[(coords >= start) & (coords <= end)] = mid + mid - coords_edit

    if (
        start + end
    ) % 2 == 1:  # During updating, if start+ end is impair, coords are shifted by one

        coords[(coords == start - 1) | (coords == end - 1)] += 1

    return coords

def update_sgn_inversion(
    start: int,
    end: int,
    sgn_start: str,
    sgn_end: str,
    coords: Iterable[int],
    sgns: Iterable[str],
) -> "np.ndarray[str]":
    """
    Update the signs of each SV breakpoint when an inversion has been applied.

    Parameters
    ----------
    start: int
        Coordinate of the beginning of the inversion.

    end: int
        Coordinate of the end of the inversion.

    sgn_start: str
        Signs at the beginning of the inversion.

    coords: Iterable[int]
        Coordinates of the signs we want to update.

    sgn_start: Iterable[str]
        Signs we want to update

    Examples
    --------
    >>> update_sgn_inversion(10, 15, "--", "++", np.array([10, 11, 14, 15]), np.array(["--", "-+", "+-", "++"]))
    array(['-+', '+-', '-+', '-+'])
    """

    coords_inside_fragment = (coords > start) & (coords < end)
    sgn_inside_fragment = sgns[np.argsort(coords[coords_inside_fragment])]

    sgn_inside_fragment = sgns[coords_inside_fragment]

    sgn_inside_fragment = np.array([sgn[::-1] for sgn in sgn_inside_fragment])

    sgns[coords_inside_fragment] = sgn_inside_fragment

    sgns[coords == start] = sgn_start[0] + sgn_end[0]
    sgns[coords == end] = sgn_start[1] + sgn_end[1]

    return sgns
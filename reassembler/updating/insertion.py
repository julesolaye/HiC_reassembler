# Functions linked to the insertion (updating and correction)

import numpy as np
from Bio import Seq
from typing import Iterable

def correct_insertion_matrix(pos: int, size: int, matrix: "np.ndarray[N,N]") -> "np.ndarray[N,N]":
    """
    Correction of insertion in the Hi-C matrix.

    Parameters
    ----------
    pos: int
        Coordinate of the insertion.

    size: int
        Coordinate of the end of the insertion.

    matrix : np.ndarray
        Matrix where we want to correct the insertion.
    """

    # Insertion white spaces
    matrix = np.concatenate(
        (
            matrix[:pos, :],
            np.zeros((size, matrix.shape[1])),
            matrix[pos:, :],
        ),
        axis=0,
    )

    matrix = np.concatenate(
            (
                matrix[:, : pos + 1],
                np.zeros((matrix.shape[0], size)),
                matrix[:, pos + 1 :],
            ),
            axis=1,
    )
    return matrix

def correct_insertion_sequence(pos: int, size: int, mutseq: Seq.MutableSeq) -> Seq.MutableSeq:
        """
        Correction of insertions in the sequence. 

        Parameters
        ----------
        pos: int
            Coordinate of the insertion.

        size: int
            Coordinate of the end of the insertion.

        mutseq: Seq.MutableSeq
            Sequence where we want to correct the insertion.    
        """

        mutseq = mutseq[:pos] + mutseq[pos+size:]
        return mutseq

def update_coords_ins(pos: int, size: int, coords: Iterable[int]) -> "np.ndarray[int]":
    """
    Update coordinates after applying a deletion at specified positions.

    Parameters
    ----------
    pos: int
        Coordinate of the insertion.

    size: int
        Coordinate of the end of the insertion.

    coords: Iterable[int]
        Size of the insertion.

    Examples
    --------
    >>> update_coords_ins(12, 8, [6, 18, 22])
    array([ 6, 26, 30])
    """

    coords = np.array(coords)

    coords_edit = coords[coords > pos]
    coords[coords > pos] = coords_edit + size

    return coords
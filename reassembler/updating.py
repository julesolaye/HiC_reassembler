# Utilities to apply SV to a genome and update the values linked to this.

from typing import Iterable
import numpy as np
import warnings
from Bio import Seq

warnings.filterwarnings("ignore")



def deletion(start: int, end: int, genome: Seq.MutableSeq) -> Seq.MutableSeq:
    """Apply deletion on input genome.

    Parameters
    ----------
    start: int
        Coordinate of the beginning of the deletion.

    end: int
        Coordinate of the end of the deletion.

    genome: Seq.MutableSeq
        Sequence of the genome.

    Examples
    --------
    >>> deletion(3, 8, "ACGTACGTACGT")
    'ACGACGT'
    """
    mutseq = genome[:start] + genome[end:]
    return mutseq


def inversion(genome: Seq.MutableSeq) -> Seq.MutableSeq:
    """Apply inversion on input genome.

    Parameters
    ----------
    genome: Seq.MutableSeq
        Sequence of the genome.

    Examples
    --------
    >>> inversion("ACGTACGTACGT")
    'TGCATGCATGCA'
    """
    mutseq = Seq.reverse_complement(genome)
    return mutseq


def translocation(start_cut: int, end_cut: int, start_paste: int, genome: Seq.MutableSeq) -> Seq.MutableSeq:
    """Apply translocation on input genome.

    Parameters
    ----------
    start_cut: int
        Coordinate of the beginning of the deletion linked to the translocation.

    end_cut: int
        Coordinate of the end of the deletion linked to the translocation.

    start_paste: int
        Coordinate of insertion linked to the translocation.

    genome: Seq.MutableSeq
        Sequence of the genome.

    Examples
    --------
    >>> translocation(2, 4, 7,"ACGTACGTACGT")
    'ACACGGTTACGT'
    """

    tra_size = end_cut - start_cut

    seq_cut = genome[start_cut:end_cut]

    mutseq = genome[:start_cut] + genome[end_cut:]

    if start_paste >= end_cut:
        start_paste -= tra_size  # Update coords

    mutseq = mutseq[:start_paste] + seq_cut + mutseq[start_paste:]

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


def update_coords_ins(pos: int, size: int, coords: Iterable[int]) -> "np.ndarray[int]":
    """
    Update coordinates after applying a deletion at specified positions.

    Parameters
    ----------
    pos: int
        Coordinate of the insertion.

    size: int
        Coordinate of the end of the deletion.

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




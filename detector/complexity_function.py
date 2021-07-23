# Some functions which are useful for the detection of the repeats.

from collections import OrderedDict
from Bio import SeqIO


def load_seq(seqfile: str, chrom_id: str, ind_beg: int, ind_end: int):
    """
    Return the sequence of a chromosome between two indexes.
  
    Parameters:
    ----------
    seqfile : str 
        Filename of the sequence.

    chrom_id : str
        Id of the chromosome we want.

    ind_beg : int
        Index of the begin of the part of the sequence we want. 
        
    ind_end : int
        Index of the end of the part of the sequence we want. 
    """

    records = SeqIO.parse(seqfile, format="fasta")

    for rec in records:
        if rec.id == chrom_id:

            seq_to_return = str(rec.seq)[ind_beg:ind_end]
            break
    return seq_to_return


def lempel_complexity(seq: str):
    """
    Compute the lempel complexity of a sequence.

    Parameters:
    ----------
    seq : str 
        The sequence which we want to compute its complexity.
    """

    words = OrderedDict()
    size_seq = len(seq)

    ind_seq = 0
    ind_end_word = 1

    while ind_end_word <= size_seq:

        while seq[ind_seq:ind_end_word] in words and ind_end_word <= size_seq:

            ind_end_word += 1
        words[seq[ind_seq:ind_end_word]] = 0
        ind_seq = ind_end_word
        ind_end_word = ind_seq + 1
    return len(words) / size_seq

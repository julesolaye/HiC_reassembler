# Script used to detect SV breakpoints and reassemble Hi-C maps.
import numpy as np
from Bio import SeqIO

from os.path import join

from os import mkdir
from shutil import rmtree
import click

from detector.matrixdetector import Matrixdetector
from detector.bamdetector import BAMdetector
from detector.combiner import Combiner
from reassembler.reassembler import Reassembler


@click.command()
@click.option(
    "--binsize", "-b", default=2000, help="Binsize used to create the Hi-C matrix.",
)
@click.argument("matrix", type=click.Path(exists=True))
@click.argument("seq", type=click.Path(exists=True))
@click.argument("bam", type=click.Path(exists=True))
@click.argument("chrom_name")
@click.argument("tmpdir", default="./tmpdir", type=click.Path(exists=False))
def reassembly(
    matrix: str, seq: str, bam: str, chrom_name: str, tmpdir: str, binsize: int
):

    # Create temporary drectory
    mkdir(tmpdir)

    # Detection on Hi-C
    MatDetect = Matrixdetector()
    MatDetect.load()
    MatDetect.predict(matrix)

    # Detection on BAM
    BamDetect = BAMdetector()
    BamDetect.load()
    BamDetect.predict(bam, seq, binsize)

    # Combine SV breakpoints in order to have SVs
    SVCombiner = Combiner(binsize, matrix, bam)
    info_sv = SVCombiner.combine()
    SVCombiner.save_sv_combined()

    # Reassembly
    reassembler = Reassembler(info_sv, matrix, seq, chrom_name, binsize)
    mat_reassembled, seq_reassembled = reassembler.reassembly()

    # Save
    save_dir = "data/output/reassembly"
    np.save(join(save_dir, "mat_reassembled.npy"), mat_reassembled)

    with open(join(save_dir, "seq_reassembled.fa"), "w") as fa_out:
        rec = SeqIO.SeqRecord(seq=seq_reassembled, id=chrom_name, description="")
        SeqIO.write(rec, fa_out, format="fasta")

    # Delete temporary directory
    rmtree(tmpdir)


if __name__ == "__main__":
    reassembly()

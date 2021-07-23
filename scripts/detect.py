# Script used to detect SV breakpoints.
import numpy as np
from Bio import SeqIO

from os.path import join

from os import mkdir
from shutil import rmtree
import click

from detector.matrixdetector import Matrixdetector
from detector.bamdetector import BAMdetector
from detector.combiner import Combiner


@click.command()
@click.option(
    "--binsize", "-b", default=2000, help="Binsize used to create the Hi-C matrix.",
)
@click.argument("matrix", type=click.Path(exists=True))
@click.argument("seq", type=click.Path(exists=True))
@click.argument("bam", type=click.Path(exists=True))
@click.argument("chrom_name")
@click.argument("tmpdir", default="./tmpdir", type=click.Path(exists=False))
def detect(matrix: str, seq: str, bam: str, chrom_name: str, tmpdir: str, binsize: int):

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

    # Delete temporary directory
    rmtree(tmpdir)


if __name__ == "__main__":
    detect()

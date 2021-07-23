# Script used to train the machine learning models which are used to detect SV breakpoints.

from detector.matrixdetector import Matrixdetector
from detector.bamdetector import BAMdetector
from detector.badmappedfinder import BadMappedFinder
from detector.repeatsfinder import RepeatsFinder

if __name__ == "__main__":

    MatDetect = Matrixdetector() # Train model which detects SV breakpoints on the Hi-C matrix.
    MatDetect.train()
    MatDetect.save()

    BamDetect = BAMdetector() # Train model which detects the coordinate of SV breakpoints on the BAMfiles.
    BamDetect.train()
    BamDetect.save()

    BMFinder = BadMappedFinder() # Train model which detects the coordinates on the BAMfiles 
                                    # which have a lot of reads badly mapped (sometimes they are considered as SV breakpoint).
    BMFinder.train()
    BMFinder.save()

    RFinder = RepeatsFinder()# Train model which detects the coordinates on the BAMfiles 
                                    # which are repeats (sometimes they are considered as SV breakpoint).
    RFinder.train()
    RFinder.save()

# Script used to train the machine learning models.

from detector.matrixdetector import Matrixdetector
from detector.bamdetector import BAMdetector
from detector.badmappedfinder import BadMappedFinder
from detector.repeatsfinder import RepeatsFinder

if __name__ == "__main__":

    MatDetect = Matrixdetector()
    MatDetect.train()
    MatDetect.save()

    BamDetect = BAMdetector()
    BamDetect.train()
    BamDetect.save()

    BMFinder = BadMappedFinder()
    BMFinder.train()
    BMFinder.save()

    RFinder = RepeatsFinder()
    RFinder.train()
    RFinder.save()

import numpy as np
import pysam as ps

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

from os.path import join
import joblib

import detector.bam_functions as bm


class BadMappedFinder(object):
    """
    Handles to find reads which are not correctly mapped. Sometimes, positions 
    with these elements can be considered as SV by the bamdetector. It is why it
    is important to have this class. It is based on a SVC.

    Attributes
    ----------
    size_win : str
        The model will find for each position if the reads near this position are
        correctly mapped. Size_win is the size of the window we will use to take the
        reads near this position.
    """

    def __init__(self, size_win : int =4):

        self.load_data()
        self.size_win = size_win
        self.create_model()

    def load_data(self, training_path : str ="data/training/mapping"):
        """
        It loads training set in order to train the model.
        """

        array_badly_mapped = np.load(join(training_path, "array_badly_mapped.npy"))
        array_SV = np.load(join(training_path, "array_SV.npy"))

        labels_SV = np.zeros(len(array_SV))
        labels_badly_mapped= np.ones(len(array_badly_mapped))

        features = np.concatenate((array_SV, array_badly_mapped)).reshape((-1, 2))
        labels = np.concatenate((labels_SV, labels_badly_mapped))

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            features, labels
        )

    def create_model(self, C : float =1):
        """
        Create SVC to detect badly mapped elements.
        """
        self.classifier = RandomForestClassifier(max_depth=5)

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

        print("Training Recall Score:")
        print(recall_score(self.y_train, self.classifier.predict(self.X_train)))
        print("Training Precision Score:")
        print(precision_score(self.y_train, self.classifier.predict(self.X_train)))

        print("Validation Recall Score:")
        print(recall_score(self.y_valid, self.classifier.predict(self.X_valid)))
        print("Validation Precision Score:")
        print(precision_score(self.y_valid, self.classifier.predict(self.X_valid)))


    def predict(self, coord : int, bam_file : str, chrom_id :str) -> int:
        """
        Detect badly mapped elements for one coord. It returns the label 
        detected by the model.

        Parameters
        --------
        coord: int
            Coordinate of the coord where we want to make the detection.
        
        bam_file: str
            Filename of the bam files.
        
        chrom_id: str
            Name of the chromosome where we will do the detection.
        """

        c_beg = coord - self.size_win // 2
        c_end = coord + self.size_win // 2

        region = chrom_id + ":" + str(c_beg) + "-" + str(c_end + 1)
        chrom_id, start, end = bm.parse_ucsc_region(region)
        bam = ps.AlignmentFile(bam_file, "rb")

        nb_little = 0
        nb_read = 0
        for read in bam.fetch(chrom_id, start, end):

            nb_read += 1

            if read.mapq <= 3:
                nb_little += 1

        if nb_read != 0:
            mapQ_ratio = nb_little / nb_read  # Proportion
        else:
            mapQ_ratio = -1

        array_mapQ = np.array([mapQ_ratio, nb_read]).reshape((-1, 2))
        map_prediction = self.classifier.predict(array_mapQ)


        return map_prediction[0]

    def save(self):
        """
        Save BadMappedFinder to joblib format.
        """
        joblib.dump(
            self.classifier, "data/models/badlymappedfinder/badlymappedfinder.joblib",
        )

    def load(self):
        """
        Load the SVC model.
        """
        self.classifier = joblib.load(
            "data/models/badlymappedfinder/badlymappedfinder.joblib"
        )

import numpy as np
import pysam as ps

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from os.path import join
import joblib

import detector.bam_functions as bm


class BadMappedFinder(object):
    """
    Handles to find element which are not correctly mapped. Sometimes, positions 
    with these elements can be considered as SV by the bamdetector. It is why it
    is important to have this class.
    """

    def __init__(self, size_win=4):

        self.load_data()
        self.size_win = size_win
        self.create_model()

    def load_data(self, training_path="data/training/mapping"):

        array_blank = np.load(join(training_path, "array_badly_mapped.npy"))
        array_SV = np.load(join(training_path, "array_SV.npy"))

        labels_SV = np.zeros(len(array_SV))
        labels_blank = np.ones(len(array_blank))

        features = np.concatenate((array_SV, array_blank)).reshape((-1, 2))
        labels = np.concatenate((labels_SV, labels_blank))

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            features, labels
        )

    def create_model(self, C=1):
        """
        Create SVC to detect badly mapped elements.
        """
        self.classifier = RandomForestClassifier(max_depth=5)

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, coords, bam_file, verbose=True, chrom="Sc_chr04"):
        """
        Detect badly mapped elements for one coord.
        """

        c_beg = coords - self.size_win // 2
        c_end = coords + self.size_win // 2

        region = chrom + ":" + str(c_beg) + "-" + str(c_end + 1)
        chrom, start, end = bm.parse_ucsc_region(region)
        bam = ps.AlignmentFile(bam_file, "rb")

        nb_little = 0
        nb_read = 0
        for read in bam.fetch(chrom, start, end):

            nb_read += 1

            if read.mapq <= 3:
                nb_little += 1

        if nb_read != 0:
            mapQ_ratio = nb_little / nb_read  # Proportion
        else:
            mapQ_ratio = -1

        array_mapQ = np.array([mapQ_ratio, nb_read]).reshape((-1, 2))
        map_prediction = self.classifier.predict(array_mapQ)
        if map_prediction[0] == 1:

            if verbose:
                print("Badly mapped")

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
        Load RandomForest model to detect reoeats.
        """
        self.classifier = joblib.load(
            "data/models/badlymappedfinder/badlymappedfinder.joblib"
        )

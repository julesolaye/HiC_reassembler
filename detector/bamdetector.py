# After the detection of SVs on the Hi-C matrix, we will use this class to have
#  the exact position of SVs.

import joblib
import numpy as np
import pandas as pd
import detector.bam_functions as bm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from alive_progress import alive_bar

from detector.badmappedfinder import BadMappedFinder
from detector.repeatsfinder import RepeatsFinder

from os.path import join


class BAMdetector(object):
    """
    After the detection of SV breakpoints on the Hi-C matrix, this class handles 
    to detect the exact coordinate of each SV with the help of BAM files. The
    detection works with the vote of three model: 
    - a RandomForestClassifier,
    - a MLPClassifier,
    - a GradientBoostingClassifier.

    Every model is trained to detect split-reads. When there is this type of reads, 
    it considers that it is a SV. So the models will detect if there is a big 
    number of reads which start/end to align at each position.
    
    Examples
    --------
        Detector = BPdetector() \n
        Detector.test() \n
    
    Attributes
    ----------
    size_win : int
        Size of the window used to detect if the number of reads which align at
        one position is grater than usual.

    tmpdir : str
        Path where the temporary directory is. There is inside this directory the
        coordinates of the matrix which are structural variations.
    """

    def __init__(self, tmpdir: str = "./tmpdir"):

        self.tmpdir = tmpdir

        self.BMFinder = BadMappedFinder()
        self.RFinder = RepeatsFinder()

        self.load_data()

        self.create_model()
        self.size_win = size_win # Size of the window.

    def load_data(self, training_path="data/training/bamdetection"):
        """
        Load training set to train the models.

        Parameters
        ----------
        training_path : str
            Path to the training set.
        """

        features = np.load(join(training_path, "bamfeatures.npy"))
        labels = np.load(join(training_path, "bamlabels.npy"))

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            features, labels
        )

    def load_detect(self):
        """
        Load zones which are be detected as structural variations on the Hi-C matrix.
        """
        self.delim = np.load(join(self.tmpdir, "coords_delim.npy"))
        self.INVzones = np.load(join(self.tmpdir, "INV_index.npy"))
        self.INSzones = np.load(join(self.tmpdir, "INS_index.npy"))
        self.DELzones = np.load(join(self.tmpdir, "DEL_index.npy"))
        self.white_inds = np.load(join(self.tmpdir, "index_not_used.npy"))

    def create_model(self):
        """
        Create RandomForest.
        """
        self.first_classifier = RandomForestClassifier()

        self.second_classifier = MLPClassifier(
            hidden_layer_sizes=(200, 3), activation="logistic"
        )

        self.third_classifier = GradientBoostingClassifier().fit(
            self.x_train, self.y_train
        )

    def train(self):
        """
        Train model.
        """
        self.first_classifier.fit(self.x_train, self.y_train)
        self.second_classifier.fit(self.x_train, self.y_train)
        self.third_classifier.fit(self.x_train, self.y_train)

    def find_features(
        self,
        coord: int,
        bam_file: str,
        binsize: int,
        chrom: str,
    ) -> tuple("np.ndarray(N)", "np.ndarray(N)", "np.ndarray(N)"):
        """
        Compute numbers of reads which start/end to align at each position near the coord. This 
        will be used to detect if a position is a SV or not because these are 
        the features of our models.

        Parameters
        ----------
        coord : int
            Index detected by Matrixdetector as a potential structural variation.

        bam_file : str
            Filename of the  bam file we will use for our prediction.

        binsize: int
            Binsize used for the creation of the Hi-C matrix.

        chrom: str
            Name of the chromosome where we will detect.

        Returns
        ---------- 
        """
        self.corrector = 5 # Without it, the detector can't detect for position at the 
                        # extremities (because of smooth). "self" because it will 
                        # be reused in self.prediction_for_each_coord().
        

        c_beg = int(
            coord * binsize
            - pos * size
            - self.corrector // 2
            - self.self.size_win  // 4
        )
        c_end = int(
            coord * binsize
            + (1 - pos) * size
            + self.corrector // 2
            + self.self.size_win  // 4
        )

        region = (
            chrom
            + ":"
            + str(c_beg)
            + "-"
            + str(c_end + 1)
        )
        start_reads, end_reads = bm.bam_region_read_ends(
            file=bam_file, region=region, side="both"
        )

        # Mean in a windows of 3 positions to smooth.
        mean_start_reads = (
            start_reads
            + np.concatenate((start_reads[1:], np.zeros(1)))
            + np.concatenate((np.zeros(1), start_reads[: len(start_reads) - 1]))
        )[1:-1] // 3
        mean_end_reads = (
            end_reads
            + np.concatenate((end_reads[1:], np.zeros(1)))
            + np.concatenate((np.zeros(1), end_reads[: len(end_reads) - 1]))
        )[1:-1] // 3
        coords_windows = np.arange(c_beg, c_end) 

        return mean_start_reads, mean_end_reads, coords_windows

    def prediction_for_each_coord(
        self, coord, bam_file, fileseq, binsize
    ):
        """
        For one coordinate detected by Matrixdetector, this method will detect
        for each position associated to this coordinate in the BAM files where 
        is the exact position of the SV (and will correct the false positives).

        Parameters
        ----------
        coord : int
            Coordinate 

        bam_file : str
            Filename of the  bam file we will use for our prediction.
        
        fileseq : str
            Filename of the fasta file where there is the sequence.

        binsize:
            Binsize used to genereate the Hi-C matrix.
        """

        start_reads, end_reads, coords_windows = self.create_coverage_starts(coord, bam_file, binsize)

        coord_find = False
        new_cluster = np.arange(0, binsize)

        probs = np.zeros(len(new_cluster))
        for i in range(0, len(new_cluster)):

            index = new_cluster[i] + self.corrector// 2 + self.self.size_win  // 4
            start_read = start_reads[
                (index - self.self.size_win  // 4) : (
                    index + self.self.size_win  // 4 + 1
                )
            ]
            end_read = end_reads[
                (index - self.self.size_win  // 4) : (
                    index + self.self.size_win  // 4 + 1
                )
            ]

            feature = np.concatenate((start_read, end_read)).reshape((1, -1))

            probs[i] = (
                3 * self.first_classifier.predict_proba(feature)[0, 1]
                + self.second_classifier.predict_proba(feature)[0, 1]
                + 3 * self.third_classifier.predict_proba(feature)[0, 1]
            ) / 7
            self.bar()

        while not coord_find:
          
            thresold_prob = (
                0.9  # Proba must be superior to that value (to avoid false positive)
            )
            if np.max(probs) >= thresold_prob:

                final_ind = coords_windows[
                    new_cluster[np.argmax(probs)]
                    + self.self.corrector// 2
                    + self.self.size_win  // 4
                ]

                index = (
                    new_cluster[np.argmax(probs)]
                    + self.self.corrector// 2
                    + self.self.size_win  // 4
                )
                start_read = start_reads[
                    (index - self.self.size_win  // 4) : (
                        index + self.self.size_win  // 4 + 1
                    )
                ]
                end_read = end_reads[
                    (index - self.self.size_win  // 4) : (
                        index + self.self.size_win  // 4 + 1
                    )
                ]

                feature = np.concatenate((start_read, end_read)).reshape((1, -1))

                badmapped = self.BMFinder.predict(coords=final_ind, bam_file=bam_file)

                if badmapped == 1:

                    new_cluster = np.delete(new_cluster, np.argmax(probs))
                    probs = np.delete(probs, np.argmax(probs))

                else:

                    repeat_pred = self.RFinder.predict(coord=final_ind, fileseq=fileseq)

                    if repeat_pred:
                        new_cluster = np.delete(new_cluster, np.argmax(probs))
                        probs = np.delete(probs, np.argmax(probs))

                    else:
                        coord_find = True
                        return final_ind, np.max(probs)

            else:

                return -1, -1

    def predict(self, bam_file: str, fileseq: str, binsize: int):
        """
        For each coordinate detected by Matrixdetector, this method computes
        the exact positions of SVs on the BAM files.

        Parameters

        bam_file: str
            Filename of the  bam file we will use for our prediction.
            
        fileseq : str
            Filename of the fasta file where there is the sequence.

        binsize : int
            Binsize used to create the Hi-C matrix.
        """

        self.load_detect()
        for coord in self.INVzones:
            for k in range(0, len(self.white_inds)):
                if (
                    ((coord + 1) == self.white_inds[k])
                    | (coord == self.white_inds[k])
                    | ((coord - 1) == self.white_inds[k])
                ):
                    self.INVzones = np.delete(
                        self.INVzones, np.where(self.INVzones == coord)
                    )

        for coord in self.INSzones:
            for k in range(0, len(self.white_inds)):
                if (
                    ((coord + 1) == self.white_inds[k])
                    | (coord == self.white_inds[k])
                    | ((coord - 1) == self.white_inds[k])
                ):
                    self.INSzones = np.delete(
                        self.INSzones, np.where(self.INSzones == coord)
                    )
        for coord in self.DELzones:
            for k in range(0, len(self.white_inds)):
                if (
                    ((coord + 1) == self.white_inds[k])
                    | (coord == self.white_inds[k])
                    | ((coord - 1) == self.white_inds[k])
                ):
                    self.DELzones = np.delete(
                        self.DELzones, np.where(self.DELzones == coord)
                    )

        INVcoordBP = list()
        INScoordBP = list()
        DELcoordBP = list()

        old_coord = -10
        self.size_zone = 0

        INVZone = list()
        INVproba_zone = list()
        INVProbas = list()

        n_pred = len(self.INVzones) + len(self.INSzones) + len(self.DELzones)

        print("DETECTION OF SVs ON BAM FILES:")
        with alive_bar(n_pred * binsize) as self.bar:
            for coord in self.INVzones:
                if coord - old_coord == 1:
                    self.size_zone += 1
                    if self.size_zone == 4:
                        self.size_zone = 0
                        INVcoordBP.append(INVZone)
                        INVProbas.append(INVproba_zone)
                        INVZone = list()
                        INVproba_zone = list()
                else:
                    self.size_zone = 0
                    INVcoordBP.append(INVZone)
                    INVProbas.append(INVproba_zone)
                    INVZone = list()
                    INVproba_zone = list()

                BP, prob = self.prediction_for_each_coord(
                    coord, bam_file, fileseq, size=2000, pos=0
                )

                if BP != -1:
                    INVZone.append(BP)
                    INVproba_zone.append(prob)
                old_coord = coord

            INVcoordBP.append(INVZone)
            INVProbas.append(INVproba_zone)

            old_coord = -10
            self.size_zone = 0

            INSZone = list()
            INSproba_zone = list()
            INSProbas = list()

            for coord in self.INSzones:
                if coord - old_coord == 1:
                    self.size_zone += 1
                    if self.size_zone == 4:
                        self.size_zone = 0
                        INScoordBP.append(INSZone)
                        INSProbas.append(INSproba_zone)
                        INSZone = list()
                        INSproba_zone = list()
                else:
                    self.size_zone = 0
                    INScoordBP.append(INSZone)
                    INSProbas.append(INSproba_zone)
                    INSZone = list()
                    INSproba_zone = list()

                BP, prob = self.prediction_for_each_coord(
                    coord, bam_file, fileseq, size=2000, pos=0
                )

                if BP != -1:
                    INSZone.append(BP)
                    INSproba_zone.append(prob)
                old_coord = coord

            INScoordBP.append(INSZone)
            INSProbas.append(INSproba_zone)

            old_coord = -10
            self.size_zone = 0

            DELZone = list()
            DELproba_zone = list()
            DELProbas = list()

            for coord in self.DELzones:
                if coord - old_coord == 1:
                    self.size_zone += 1
                    if self.size_zone == 4:
                        self.size_zone = 0
                        DELcoordBP.append(DELZone)
                        DELProbas.append(DELproba_zone)
                        DELZone = list()
                        DELproba_zone = list()
                else:
                    self.size_zone = 0
                    DELcoordBP.append(DELZone)
                    DELProbas.append(DELproba_zone)
                    DELZone = list()
                    DELproba_zone = list()

                BP, prob = self.prediction_for_each_coord(
                    coord, bam_file, fileseq, size=2000, pos=0
                )

                if BP != -1:
                    DELZone.append(BP)
                    DELproba_zone.append(prob)
                old_coord = coord

            DELcoordBP.append(DELZone)
            DELProbas.append(DELproba_zone)

        # Keep good element of each zone for INV
        INV_finalBP = list()
        INV_finalprob = list()
        for i in range(0, len(INVcoordBP)):

            if len(INVcoordBP[i]) > 0:
                array_probas = np.array(INVProbas[i])
                INV_finalBP.append(INVcoordBP[i][np.argmax(array_probas)])
                INV_finalprob.append(np.max(array_probas))

        # Keep good element of each zone for INS
        INS_finalBP = list()
        INS_finalprob = list()
        for i in range(0, len(INScoordBP)):

            if len(INScoordBP[i]) > 0:
                array_probas = np.array(INSProbas[i])
                INS_finalBP.append(INScoordBP[i][np.argmax(array_probas)])
                INS_finalprob.append(np.max(array_probas))

        # Keep good zones for DEL
        DEL_finalBP = list()
        DEL_finalprob = list()
        for i in range(0, len(DELcoordBP)):

            if len(DELcoordBP[i]) > 0:
                array_probas = np.array(DELProbas[i])
                DEL_finalBP.append(DELcoordBP[i][np.argmax(array_probas)])
                DEL_finalprob.append(np.max(array_probas))

        INV_coords_BAM = list(INV_finalBP)
        INS_coords_BAM = list(INS_finalBP)
        DEL_coords_BAM = list(DEL_finalBP)

        INV_sgns = np.array(
            [" "] * len(INV_coords_BAM)
        )  #  Implement something to detect them
        INS_sgns = np.array([" "] * len(INS_coords_BAM))
        DEL_sgns = np.array([" "] * len(DEL_coords_BAM))

        INV_DataFrame = pd.DataFrame(
            np.array([INV_coords_BAM, INV_sgns]).T, columns=["BAM", "SGNS"],
        )
        INS_DataFrame = pd.DataFrame(
            np.array([INS_coords_BAM, INS_sgns]).T, columns=["BAM", "SGNS"],
        )
        DEL_DataFrame = pd.DataFrame(
            np.array([DEL_coords_BAM, DEL_sgns]).T, columns=["BAM", "SGNS"],
        )

        INV_DataFrame.to_csv(
            join(self.tmpdir, "INV_detected_info.tsv"), index=False, sep="\t"
        )
        INS_DataFrame.to_csv(
            join(self.tmpdir, "INS_detected_info.tsv"), index=False, sep="\t"
        )
        DEL_DataFrame.to_csv(
            join(self.tmpdir, "DEL_detected_info.tsv"), index=False, sep="\t"
        )

    def save(self):
        """
        Save all the models to joblib format.
        """
        joblib.dump(self.first_classifier, "data/models/bamdetector/bamclassif1.joblib")
        joblib.dump(
            self.second_classifier, "data/models/bamdetector/bamclassif2.joblib"
        )
        joblib.dump(self.third_classifier, "data/models/bamdetector/bamclassif3.joblib")

    def load(self):
        """
        Load all the models to detect BP and others models.
        """
        self.first_classifier = joblib.load(
            "data/models/bamdetector/bamclassif1.joblib"
        )
        self.second_classifier = joblib.load(
            "data/models/bamdetector/bamclassif2.joblib"
        )
        self.third_classifier = joblib.load(
            "data/models/bamdetector/bamclassif3.joblib"
        )
        self.BMFinder.load()
        self.RFinder.load()

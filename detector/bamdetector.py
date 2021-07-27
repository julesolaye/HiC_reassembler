# After the detection of SVs on the Hi-C matrix, we will use this class to have
#  the exact position of SVs.

import joblib
import numpy as np
import pandas as pd
import detector.bam_functions as bm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score

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

    def __init__(self, size_win = 100, tmpdir: str = "./tmpdir"):

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
        self.INVdetected = np.load(join(self.tmpdir, "INV_index.npy"))
        self.INSdetected = np.load(join(self.tmpdir, "INS_index.npy"))
        self.DELdetected = np.load(join(self.tmpdir, "DEL_index.npy"))


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
        
        print("Validation Recall Score first model:")
        print(recall_score(self.y_valid, self.first_classifier.predict(self.x_valid)))
        print("Validation Recall Score second model:")
        print(recall_score(self.y_valid, self.second_classifier.predict(self.x_valid)))
        print("Validation Recall Score third model:")
        print(recall_score(self.y_valid, self.third_classifier.predict(self.x_valid)))
        
        print("Validation Precision Score first model:")
        print(precision_score(self.y_valid, self.first_classifier.predict(self.x_valid)))
        print("Validation Precision Score second model:")
        print(precision_score(self.y_valid, self.second_classifier.predict(self.x_valid)))
        print("Validation Precision Score third model:")
        print(precision_score(self.y_valid, self.third_classifier.predict(self.x_valid)))

    def find_features(
        self,
        coord: int,
        bam_file: str,
    ) -> ["np.ndarray(N)", "np.ndarray(N)", "np.ndarray(N)"]:
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

        """
        self.corrector = 6 # Without it, the detector can't detect for position at the 
                        # extremities (because of smooth). "self" because it will 
                        # be reused in self.prediction_for_each_coord().
        
        # Compute start/read for each position
        c_beg = int(
            coord * self.binsize
            - self.corrector // 2
            - self.size_win  // 2
        )
        c_end = int(
            coord * self.binsize
            + self.binsize
            + self.corrector // 2
            + self.size_win  // 2
        )

        region = (
            self.chrom
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
        
        # Coordinate associated in the BAM files associated to each position in the
        # array of start/end
        coords_windows = np.arange(c_beg, c_end) 

        return mean_start_reads, mean_end_reads, coords_windows



    def prediction_for_one_coord(
        self, coord:int, bam_file:str, fileseq:str
    ) -> [int, float]:
        """
        For one coordinate detected by Matrixdetector, this method will detect
        for each position associated to this coordinate in the BAM files where 
        is the exact position of the SV (and will correct the false positives).

        It returns the coordinate which is a SV on the BAM files and the probability 
        of be a structural variation given by the model.

        Parameters
        ----------
        coord : int
            Coordinate 

        bam_file : str
            Filename of the  bam file we will use for our prediction.
        
        fileseq : str
            Filename of the fasta file where there is the sequence.
        """

        start_reads, end_reads, coords_windows = self.find_features(coord, bam_file)

        # Will test for each position linked to the coord if it is a structural variation or not
        coord_find = False
        new_cluster = np.arange(0, self.binsize)
        probs = np.zeros(len(new_cluster))

        for i in range(0, len(new_cluster)):
            
            # The cluster and coord_windows doesn't have the same size linked to the correction
            # so their index are shifted.
            index = new_cluster[i] + self.corrector// 2 + self.size_win  // 2

            # Feature associated to the position
            start_read = start_reads[
                (index - self.size_win  // 2) : (
                    index + self.size_win  // 2 + 1
                )
            ]
            end_read = end_reads[
                (index - self.size_win  // 2) : (
                    index + self.size_win  // 2 + 1
                )
            ]
            feature = np.concatenate((start_read, end_read)).reshape((1, -1))
            
            # Vote
            probs[i] = (
                3 * self.first_classifier.predict_proba(feature)[0, 1]
                + self.second_classifier.predict_proba(feature)[0, 1]
                + 3 * self.third_classifier.predict_proba(feature)[0, 1]
            ) / 7
            
            self.bar() # Bar progression


        # If the position has a big probability, and is not a repeat/badly mapped
        # position, it returns it as SV breakpoint.
        while not coord_find:
        
            thresold_prob = (
                0.9  # Proba must be superior to that value (to avoid false positive)
            )
            if np.max(probs) >= thresold_prob:

                final_coord = coords_windows[
                    new_cluster[np.argmax(probs)]
                    + self.corrector// 2
                    + self.size_win  // 2
                ] # Coordinate in the bamfile

                is_repeat_or_bad_mapped = self.test_repeats_badmapped(final_coord, bam_file, fileseq)

                if is_repeat_or_bad_mapped:
                    probs = np.delete(probs, np.argmax(probs))

                else:
                    coord_find = True
                    return final_coord, max(probs)
            else:
                return -1, -1

        return -1, -1
    
    def prediction_for_all_coords(self, all_coords : "np.ndarray[int]", bam_file : str, fileseq : str) -> "List[int]": 
        """
        For each coordinate detected by Matrixdetector, this method will detect
        for each position associated to this coordinate in the BAM files where 
        is the exact position of the SV (and will correct the false positives).

        It returns a list with all the coordinates on the BAMfile detected.

        Parameters
        ----------
        all_coords : int
            All coordinates detected by matrixdetector (all for each type of structural variation).

        bam_file : str
            Filename of the  bam file we will use for our prediction.
        
        fileseq : str
            Filename of the fasta file where there is the sequence.
        """

        predictions = list()

        for coord in all_coords:
            
            bam_coord_predicted, proba_prediction = self.prediction_for_one_coord(coord, bam_file, fileseq)
            predictions.append(bam_coord_predicted)
        
        predictions = np.array(predictions)
        predictions = np.delete(predictions, np.where(predictions == -1))

        return predictions



    def test_repeats_badmapped(self, coord : int, bam_file: str, fileseq : str):
        """
        Test at a position detected as SV breakpoint if it is a repeat or a 
        coordinate with a lot of reads badly mapped.

        Parameters
        --------
        coord: int
            Coordinate of the coord where we want to make the detection.
        
        bam_file: str
            Filename of the bam files.

        fileseq : str
            Filename of the fasta file where there is the sequence.
        """

        test_repeat =  self.RFinder.predict(coord = coord, fileseq=fileseq, chrom_id=self.chrom)
        test_badmapped = self.BMFinder.predict(coord = coord, bam_file=bam_file, chrom_id=self.chrom)

        return (test_badmapped)

    def predict(self, bam_file: str, fileseq: str, binsize : int, chrom : str):
        """
        For each coordinate detected by Matrixdetector, this method computes
        the exact positions of SVs on the BAM files.

        Parameters

        bam_file: str
            Filename of the  bam file we will use for our prediction.
            
        fileseq : str
            Filename of the fasta file where there is the sequence.

        binsize: int
            Binsize used to create the matrix.
        
        chrom: str
            Name of the chromosome linked to the Hi-C matrix.
        """
        self.binsize = binsize
        self.chrom = chrom
        self.load_detect()

        n_pred = len(self.INVdetected) + len(self.INSdetected) + len(self.DELdetected)

        print("DETECTION OF SVs ON BAM FILES:")
        with alive_bar(n_pred * self.binsize) as self.bar:

            # Keep the best element of each zones detected
            INV_coords_BAM = self.prediction_for_all_coords(self.INVdetected, bam_file, fileseq)
            INS_coords_BAM = self.prediction_for_all_coords(self.INSdetected, bam_file, fileseq)
            DEL_coords_BAM = self.prediction_for_all_coords(self.DELdetected, bam_file, fileseq)

        ## Save coordinates detected
        INV_sgns = np.array(
            [" "] * len(INV_coords_BAM)
        )  #  Need to implement something to detect signs
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

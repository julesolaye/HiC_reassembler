# Can detect if there is a repeat near the position or not.

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score

from os.path import join
import joblib

import detector.complexity_function as cf


class RepeatsFinder(object):
    """
    Handles to detect repeats with the help of the Lempel-Ziv complexity. This 
    model is a DecisionTreeClassifier of depth 1 so it can be considered simply 
    as a thresold of complexity that the model find with some examples.

    To detect if there is a repeat near a position, we will use a sliding window
    near this position and compute the complexity of a sequence for each element 
    of the sliding window. If one of this complexity is low, it is that this is a 
    repeat.

    This class is important because some repeats can be detected as SV by the model.

    Attributes
    ----------
    size_seq : str
        Size of the sequence used to compute the complexity.

    size_win : str
        Size of the sliding window.
    """

    def __init__(self, size_seq: int = 40, size_win: int = 30):

        self.load_data()
        self.create_model()
        self.size_seq = size_seq
        self.size_win = size_win


    def load_data(self, training_path: str = "data/training/repeats"):
        """
        Load data to train the model.

        Parameters
        ----------
        training_path : str
            Path where there is the training set in order to train the model.
        """
        array_repeats = np.load(join(training_path, "complexity_repeats.npy")) # Complexity for repeats.
        array_SV = np.load(join(training_path, "complexity_SV.npy")) # Complexity for SV.

        labels_repeats = np.ones(len(array_repeats))
        labels_SV = np.zeros(len(array_SV))

        features = np.concatenate((array_repeats, array_SV)).reshape((-1, 1))
        labels = np.concatenate((labels_repeats, labels_SV))

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            features, labels
        )

    def create_model(self):
        """
        Create DecisionTreeClassifier to detect repeats.
        """
        self.classifier = DecisionTreeClassifier(max_depth=1)

    def train(self):
        """
        Train the DecisionTreeClassifier.
        """
        self.classifier.fit(self.X_train, self.y_train)

        print("Training Recall Score:")
        print(recall_score(self.y_train, self.classifier.predict(self.X_train)))
        print("Training Precision Score:")
        print(precision_score(self.y_train, self.classifier.predict(self.X_train)))

        print("Validation Recall Score:")
        print(recall_score(self.y_valid, self.classifier.predict(self.X_valid)))
        print("Validation Precision Score:")
        print(precision_score(self.y_valid, self.classifier.predict(self.X_valid)))



    def predict(self, coord: int, fileseq: str, chrom_id: str):
        """
        Predict if there is a repeat near a coordinate or not.

        Parameters
        ----------
        coord: int
            Coordinate of the position where we want to detect if there is a 
            repeat near this position or not.

        fileseq: str
            Filename of the genome file.
        
        chrom_id: str
            Id of the chromosome linked to the coordinate.
        """

        complexity = np.zeros(2 * self.size_win + 1)
        for k in range(-self.size_win, self.size_win + 1):
            ind_beg = coord + k - self.size_seq // 2
            ind_end = coord + k + self.size_seq // 2
            seq = cf.load_seq(
                fileseq, chrom_id=chrom_id, ind_beg=ind_beg, ind_end=ind_end
            )
            complexity[k] = cf.lempel_complexity(seq)

        min_complexity = np.min(complexity)

        label_predicted = self.classifier.predict(
            np.array([min_complexity]).reshape(-1, 1)
        )[0]

        if label_predicted == 1:
            return True
        else:
            return False

    def save(self):
        """
        Save the DecisionTreeClassifier to joblib format.
        """
        joblib.dump(
            self.classifier, "data/models/repeatsfinder/repeatsfinder.joblib",
        )

    def load(self):
        """
        Load the DecisionTreeClassifier to detect repeats.
        """
        self.classifier = joblib.load("data/models/repeatsfinder/repeatsfinder.joblib")

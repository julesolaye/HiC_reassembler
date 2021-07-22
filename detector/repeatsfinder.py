import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from os.path import join
import joblib

import detector.complexity_function as cf


class RepeatsFinder(object):
    """
    Handles to detect repeats with the help of Lempel-Ziv complexity.
    """

    def __init__(self, size_win: int = 40, size_tab: int = 30, chrom: str = "Sc_chr04"):

        self.load_data()
        self.create_model()
        self.size_win = size_win
        self.size_tab = size_tab
        self.chrom = chrom

    def load_data(self, training_path: str = "data/training/repeats"):
        """
        Load data to train model.
        """
        array_repeats = np.load(join(training_path, "complexity_repeats.npy"))
        array_SV = np.load(join(training_path, "complexity_SV.npy"))

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

    def predict(self, coord: int, fileseq: str, verbose: bool = True):

        complexity = np.zeros(2 * self.size_tab + 1)
        for k in range(-self.size_tab, self.size_tab + 1):
            ind_beg = coord + k - self.size_win // 2
            ind_end = coord + k + self.size_win // 2
            seq = cf.load_seq(
                fileseq, chrom_id="Sc_chr04", ind_beg=ind_beg, ind_end=ind_end
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
        Save BadMappedFinder to joblib format.
        """
        joblib.dump(
            self.classifier, "data/models/repeatsfinder/repeatsfinder.joblib",
        )

    def load(self):
        """
        Load RandomForest model to detect reoeats.
        """
        self.classifier = joblib.load("data/models/repeatsfinder/repeatsfinder.joblib")

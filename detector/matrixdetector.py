# Trying a simple keras NN to predict SVs in a Hi-C matrix.

from os.path import join
import numpy as np
from time import time
import matplotlib.pyplot as plt

import os  # To remove warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
)

import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import model_from_json

from alive_progress import alive_bar

import joblib
from detector.utils import white_index, delete_index


import warnings

warnings.filterwarnings("ignore")


class Matrixdetector(object):

    """
    Handles to detect SV on a dataset with pictures of HiC matrix. A CNN will 
    be used for this detection. Firstly, we must train the model with "imgs.npy" 
    and "imgslabels.npy" which represent pictures and label. After that we can 
    make predictions.
    
    Examples
    --------
        Detector = Matrixdetector() \n
        Detector.train() \n
        Detector.test()
    
    Attributes
    ----------
    n_neurons : int
        Number of neurons in the two last layers of the CNN we will create for our detector.
        
    training_path : str
        Path to the npy files to load to create the training dataset.
    tmpdir : str
        Path where the temporary directory is.
    """

    def __init__(
        self,
        n_neurons: int = 40,
        training_path: str = "./data/training/matrixdetection",
        tmpdir: str = "./tmpdir",
    ):
        self.load_data(training_path)
        self.img_size = self.xtrain.shape[1]
        self.n_labels = len(np.unique(self.ytrain))
        self.matrixdetector = self.create_CNN(n_neurons)
        self.tmpdir = tmpdir

    def load_data(self, training_path: str):

        """
        Loads npy files, and split data into train and validation set.

        Parameters
        ----------
        training_path : str
            Path to the npy files to load.
        """

        x_data = np.load(join(training_path, "imgs.npy"))
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
        y_data = np.load(join(training_path, "imgslabels.npy"))

        (self.xtrain, self.xvalid, self.ytrain, self.yvalid,) = train_test_split(
            x_data, y_data, train_size=0.8
        )

        self.scaler = MinMaxScaler()
        self.xtrain = self.scaler.fit_transform(
            self.xtrain.reshape(-1, self.xtrain.shape[-1])
        ).reshape(self.xtrain.shape)

        self.xvalid = self.scaler.transform(
            self.xvalid.reshape(-1, self.xvalid.shape[-1])
        ).reshape(self.xvalid.shape)

    def create_CNN(self, n_neurons: int):

        """
        Builds model from scratch for training.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in the two last layers.
        Returns
        -------
        keras.Model :
            Return the CNN model.
        """

        # Initializes a sequential model (i.e. linear stack of layers)
        model = tf.keras.models.Sequential()
        model.add(keras.Input(shape=(self.img_size, self.img_size, 1), name="Input"))
        # Need to start w/ some conv layers to use neighbourhood info
        # conv2d(n_output_channels, kernel_size, ...)
        # 128x128 - (k-1) -> 126x126
        model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
        # Dropout to reduce overfitting
        model.add(tf.keras.layers.Dropout(0.4))
        # 126x126 / 2 -> 62x62x32
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.4))
        # 62x62 / 2 -> 30x30x64
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # Finish up by flattening and feeding to a dense layer
        # 63x63 -> 63**2x1
        model.add(tf.keras.layers.Flatten())  # Flattens input matrix

        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(
            tf.keras.layers.Dense(self.n_labels, activation="softmax", name="Output")
        )

        # Optimizer
        learning_rate = 1e-3  # Paramètres
        optimizer = Adam(learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, n_epochs: int = 70):

        """
        Train model with training set.

        Parameters
        ----------
        n_epochs : int
            Number of epochs for the training.
        """

        print(
            "TRAIN detector ON DATASET WITH {n_pic} PICTURES.".format(
                n_pic=len(self.ytrain)
            )
        )

        # Stopper
        stopper = EarlyStopping(
            monitor="val_loss",
            patience=24,
            mode="min",
            verbose=2,
            restore_best_weights=True,
        )
        # Reducelr
        reducelr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, verbose=2, patience=12
        )
        time_begin = time()

        # Training
        self.matrixdetector.fit(
            self.xtrain,
            self.ytrain,
            validation_data=(self.xvalid, self.yvalid),
            verbose=True,
            epochs=n_epochs,
            callbacks=[stopper, reducelr],
        )
        time_end = time()

        time_tot = time_end - time_begin
        print(
            "Training time is {hour_} h {min_} min {sec_} s.".format(
                hour_=time_tot // 3600,
                min_=(time_tot % 3600) // 60,
                sec_=(time_tot % 3600) % 60,
            )
        )

        # Print results of the training
        print(
            "Validation recall score:",
            recall_score(
                self.yvalid,
                np.argmax(self.matrixdetector.predict(self.xvalid), axis=1),
                average=None,
            ),
            "Validation precision score: ",
            precision_score(
                self.yvalid,
                np.argmax(self.matrixdetector.predict(self.xvalid), axis=1),
                average=None,
            ),
        )

    def predict(self, file_scrambled: str):

        """
        Find indexes where the detector detects a SV.

        Parameters
        ----------
        file_scrambled : str
            File where is the scrambled Hi-C matrix which we want to detect structural variations.
        """

        scrambled = np.load(file_scrambled)
        
        

        half_img_size = self.img_size // 2

        ind_beg = half_img_size  # Because the pictures has a size N*N,
        # SVs before the coordinate N//2 can't be detected (same at the end).
        ind_end = scrambled.shape[0] - half_img_size

        index_used = np.arange(ind_beg, ind_end) # Index used for our detection (we remove the index linked white bands).
        index_not_used = white_index(scrambled) # Detect white bands. We will not 
                                                # take index of white bands for our detection
        index_used = delete_index(index_used, index_not_used)

        inds_INV_detected = list()
        probs_INV_detected = list()

        inds_INS_detected = list()
        probs_INS_detected = list()

        inds_DEL_detected = list()
        probs_DEL_detected = list()

        print("DETECTION OF SVs ON HI-C:")
        with alive_bar(len(index_used)) as bar:  #  Allow to show progression bar.

            for i in index_used: # We do a sliding window to detect the SVs on the Hi-C map.

                slice_scrambled = scrambled[
                    i - half_img_size : i + half_img_size,
                    i - half_img_size : i + half_img_size,
                ]

                slice_scrambled = self.scaler.transform(slice_scrambled).reshape(
                    (-1, slice_scrambled.shape[0], slice_scrambled.shape[1], 1,)
                )

                others_labels = np.argmax(
                    self.matrixdetector.predict(slice_scrambled), axis=1
                )

                if others_labels == 1:
                    inds_INV_detected.append(i)
                    probs_INV_detected.append(
                        self.matrixdetector.predict(slice_scrambled)[0, 1]
                    )

                if others_labels == 2:
                    inds_INS_detected.append(i)
                    probs_INS_detected.append(
                        self.matrixdetector.predict(slice_scrambled)[0, 2]
                    )
                if others_labels == 3:
                    inds_DEL_detected.append(i)
                    probs_DEL_detected.append(
                        self.matrixdetector.predict(slice_scrambled)[0, 3]
                    )
                bar()

        inds_INV_detected = np.array(inds_INV_detected)
        inds_INS_detected = np.array(inds_INS_detected)
        inds_DEL_detected = np.array(inds_DEL_detected)

        # Save to a temporary directory the index detected. BAMdetector will take
        # it.
        np.save(join(self.tmpdir, "INV_index.npy"), inds_INV_detected)
        np.save(join(self.tmpdir, "INS_index.npy"), inds_INS_detected)
        np.save(join(self.tmpdir, "DEL_index.npy"), inds_DEL_detected)

        # Save also the index of white bands, and index of the beginning and the
        # end of the sliding window.
        coords_delim = np.array([ind_beg, ind_end])
        np.save(join(self.tmpdir, "coords_delim.npy"), coords_delim)

    def confusion_matrix(self):
        """
        Return the confusion matrices of each detector for the validation set.
        """
        return confusion_matrix(
            self.yvalid, np.argmax(self.matrixdetector.predict(self.xvalid), axis=1),
        )

    def plot(self, history: dict):
        """
        Plot the evolution of loss, val_loss, accuracy, val_accuracy during the training.
        
        history: dict
            Dictionnary created during the training where there are the informations
            of the training (evolution of accuracy, loss ...).
        """

        # Plot training & validation accuracy values
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(history.history["accuracy"], label="Train")
        ax[0].plot(history.history["val_accuracy"], label="Test")
        ax[0].set_title("Model accuracy")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylim(0, 1)
        # Plot training & validation loss values
        ax[1].plot(history.history["loss"], label="Train")
        ax[1].plot(history.history["val_loss"], label="Test")
        ax[1].set_title("Model loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylim(0, 2)
        plt.legend(loc="upper left")
        plt.show()

    def save(self, model_dir: str = "data/models"):
        """
        Saves models configuration and weights to disk.

        Parameters
        ----------
        model_dir : str
            Directory where the model will be saved.
        """
        model_json = self.matrixdetector.to_json()
        with open(join(model_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)
        self.matrixdetector.save_weights(join(model_dir, "weights.h5"))
        joblib.dump(self.scaler, join(model_dir, "scaler.gz"))

    def load(self, model_dir="data/models/matrixdetector"):
        """
        Loads a trained neural network from a json file and a 
        RandomForestClassifier.
            
        Parameters
        ----------
        model_dir : str
            Directory where the model will be loaded.
        """

        with open(join(model_dir, "model.json"), "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(join(model_dir, "weights.h5"))
        loaded_model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            optimizer="adam",
        )
        self.matrixdetector = loaded_model
        self.scaler = joblib.load(join(model_dir, "scaler.gz"))

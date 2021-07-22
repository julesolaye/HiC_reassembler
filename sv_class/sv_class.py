import numpy as np


class SVs(object):
    def __init__(
        self,
        sv_name: np.ndarray,
        sv_type: np.ndarray,
        coordsBP1: np.ndarray,
        coordsBP2: np.ndarray,
        coordsBP3: np.ndarray,
        size: np.ndarray,
        sgnsBP1: np.ndarray = None,
        sgnsBP2: np.ndarray = None,
        sgnsBP3: np.ndarray = None,
    ):

        self.sv_name = sv_name
        self.sv_type = sv_type
        self.coordsBP1 = coordsBP1
        self.coordsBP2 = coordsBP2
        self.coordsBP3 = coordsBP3
        self.size = size

        # We have not implemented yet how to find sign. They are by defaut None
        # for the moment but they can be useful in the future.

        self.sgnsBP1 = sgnsBP1
        self.sgnsBP2 = sgnsBP2
        self.sgnsBP3 = sgnsBP3

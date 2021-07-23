# Class used to represent the SVs detected. 

import numpy as np

class SVs(object):
    """
    This class allows to represent SVs detected with its principal characterics.
    It will be really useful during the reassembly. The characteristrics we give to 
    the class are:
    
        - the name of the SV (each SV has a different name linked to its type, 
        we do that to differentiate each SV during the reassembly),

        - the type of the SV (INV, INS, DEL, TRA_forward and TRA_back). We differentiate
        TRA_forward and TRA_back (which represents if the sequence has been translocated 
        to the back or not) because the reassembly will be different,

        - the coordinates of each breakpoint linked to the SV on the BAMFILE (there is only one 
        coordinate linked to an insertion, we will put -1 in coordsBP2 and coordsBP3 in this case, same for deletion or inversion which have only two),

        - the size of the SV,

        - the signs for each SV breakpoint (not implemented the detection).
    """
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

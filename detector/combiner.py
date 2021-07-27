# Class which combine the different SV breakpoints in order to have a SV.

import numpy as np
import pandas as pd
import pysam as ps

import detector.bam_functions as bm
from svs.svs import SVs

from os.path import join

from typing import Iterable


np.seterr(divide="ignore", invalid="ignore")


class Combiner(object):
    """
    After all the coords has been detected, we must find which breakpoints form 
    an SV when we combine the two of its (or the three of its for a translocation).

    Examples
    --------
    SVCombiner = Combiner(binsize, matrix, bam)
    SVCombiner.combine()
    SVCombiner.save_sv_combined()

    Attributes
    ----------
    binsize : int
        Binsize used to create the Hi-C maps.

    chrom : str
        Name of the chromosome where we want to detect the structural variations.

    file_scrambled : str
        Name of the file where there is the scrambled Hi-C matrix.
    
    bamfile : str
        Name of the bamfile where the alignments are.

    tmpdir : str
        Name of the temporary directory is. There is inside the coordinates detected 
        as SV breakpoints by bamdetector.
    """

    def __init__(
        self, binsize: int, chrom: str, file_scrambled: str, bamfile: str, tmpdir: str = "./tmpdir",
    ):

        self.binsize = binsize
        self.chrom = chrom

        self.scrambled = np.load(file_scrambled)
        self.bamfile = bamfile

        self.INV_info = pd.read_csv(join(tmpdir, "INV_detected_info.tsv"), sep="\t")
        self.INS_info = pd.read_csv(join(tmpdir, "INS_detected_info.tsv"), sep="\t")
        self.DEL_info = pd.read_csv(join(tmpdir, "DEL_detected_info.tsv"), sep="\t")

        self.tmpdir = tmpdir

        self.col_bam = 0  # Column where there are coords detected before as SV
        self.col_sgns = 1  # Column where there are BAM coords

    def combine(self) -> SVs:
        """
        This is the function which combine every SV breakpoint in order to have
        a SV. It will returns an element of the "SVs" which have every information
        about the structural variants.
        """

        # All infos for sv_class
        self.sv_name = list()
        self.sv_type = list()

        self.coordsBP1 = list()
        self.coordsBP2 = list()
        self.coordsBP3 = list()

        self.sgnsBP1 = (
            list()
        )  #  WE DON'T IMPLEMENT HOW TO FIND SGNS SO WE WILL NOT UPDATE THE LIST
        self.sgnsBP2 = list()
        self.sgnsBP3 = list()

        self.size = list()

        # Add_element to the list for each sv

        self.add_INVs()
        self.add_TRA_DEL_INS()

        self.info_sv = SVs(
            np.array(self.sv_name),
            np.array(self.sv_type),
            np.array(self.coordsBP1),
            np.array(self.coordsBP2),
            np.array(self.coordsBP3),
            np.array(self.size),
        )  #  Without the signs because the detection of signs is not implemented yet

        return self.info_sv

    def find_mate(self, coord : int, allcoords : Iterable[int]) -> int:
        """
        This function allow to find a the mate of a SV breakpoint, among an array 
        with a lot of others breakpoints. It will use the SA tag of the alignement
        and see if there is an element of the array where on read is aligned on 
        the coord, and on the element of the array. In this case, there are mates 
        and they are combined as a structural variation. This method returns the 
        index of the mate on the dataframe(-1 if no mate has been found).

        Parameters 
        --------
        coord : int
            Coordinate that we want to find the mate.

        allcoords : Iterable[int]
            An array with all the element which are potential mate (if coord is 
            an inversion, it will be an array with all the others SV breakpoints
            considered as inversion).
        """

        bam = ps.AlignmentFile(self.bamfile, "rb")

        win = 4
        start = coord - win
        end = coord + win

        ind_coord = 1
        ind_sgn = 2

        coords_reads = list()
        sgns_reads = list()

        for read in bam.fetch(self.chrom, start, end):

            try:
                read_info = read.get_tag("SA").split(",")
                coords_reads.append(int(read_info[ind_coord]))  # str to int

            except:
                pass

        if len(coords_reads) == 0:
            return -1  #  No mate in this case

        thresold_dis = 5

        mate_found = False
        index_candidate = 0

        while (not mate_found) and (
            index_candidate < len(allcoords.iloc[:, self.col_bam])
        ):

            dis_to_coords = abs(
                allcoords.iloc[index_candidate, self.col_bam] - coords_reads
            )

            if np.min(dis_to_coords) < thresold_dis:

                mate_found = True

                return index_candidate

            index_candidate += 1

        return -1

    def find_all_mates(self, allcoords : Iterable[int]) -> np.ndarray:
        """
        For one type of structural variation, it will combine every breakpoints
        in order to have all the mates. For instance, if we give the array with 
        all the SV breakpoints detected as SV breakpoints for an inversion to this 
        method, it will returns all the pairs of SV breakpoints which are inversion.

        Parameters 
        --------
        allcoords : Iterable[int]
            An Iterable with all the SVs breakpoints detected for a structural 
            variation by the detectors.
        """
        # Search for each coord detected a mate.
        all_mates = list()
        for index_bp in allcoords.index:

            coord_bam = allcoords.iloc[index_bp, self.col_bam]
            index_other_bp = self.find_mate(coord_bam, allcoords)

            if (
                index_other_bp != -1
            ):  #  When find_mate returns -1, it is that there is no mate found.
                all_mates.append([index_bp, index_other_bp])

        if len(all_mates) > 0:

            all_mates = np.array(all_mates)
            all_mates = np.sort(all_mates, axis=1)

            mates_indexes = np.unique(all_mates, axis=0)
        else:
            mates_indexes = np.array(list())

        return mates_indexes

    def add_INVs(self):
        """
        This is a method used during the combinaison. It add every inversion
        combinated in order to make the class SVs with all the informations.
        """

        self.INV_mates_indexes = self.find_all_mates(self.INV_info)

        count_inv = 1
        for indexbp1, indexbp2 in self.INV_mates_indexes:

            self.sv_name.append("INV" + str(count_inv))
            self.sv_type.append("INV")

            self.coordsBP1.append(self.INV_info.iloc[indexbp1, self.col_bam])
            self.coordsBP2.append(self.INV_info.iloc[indexbp2, self.col_bam])
            self.coordsBP3.append(-1)  # No third BP for INV

            ### ADD SOMETHING FOR SIGNS AFTER ###

            self.size.append(
                abs(
                    self.INV_info.iloc[indexbp1, self.col_bam]
                    - self.INV_info.iloc[indexbp2, self.col_bam]
                )
            )

            count_inv += 1

    def find_TRA(self):
        """
        Allows to detect if the insertions arelinked to a translocation 
        (so if there is a deletion associated) or not. 
        """

        self.DEL_mates_indexes = self.find_all_mates(
            self.DEL_info
        )  # self because we will re-use this value after

        self.is_TRA = list()

        for DEL_mates in self.DEL_mates_indexes:

            other_index_test_1 = self.find_mate(
                self.DEL_info.iloc[DEL_mates[0], self.col_bam], self.INS_info
            )
            other_index_test_2 = self.find_mate(
                self.DEL_info.iloc[DEL_mates[1], self.col_bam], self.INS_info
            )

            if (other_index_test_1 != -1) | (other_index_test_2 != -1):

                self.is_TRA.append(
                    max(np.max(other_index_test_1), np.max(other_index_test_2))
                )  # Max because one can be -1, the other the true index.

            else:
                self.is_TRA.append(-1)

    def add_TRA_DEL_INS(self):
        """
        This is a method used during the combinaison. It add every deletion, 
        translocation and insertion which have been combinated in order to make 
        the class SVs with all the informations.
        """

        self.find_TRA()

        count_tra = 1
        count_del = 1
        count_ins = 1

        for index_DEL_mates in range(0, self.DEL_mates_indexes.shape[0]):

            DEL_mates = self.DEL_mates_indexes[index_DEL_mates]

            if self.is_TRA[index_DEL_mates] != -1:

                INS_index = self.is_TRA[index_DEL_mates]  # INS index linked to TRA

                self.sv_name.append("TRA" + str(count_tra))

                if (
                    min(
                        self.DEL_info.iloc[DEL_mates[0], self.col_bam],
                        self.DEL_info.iloc[DEL_mates[1], self.col_bam],
                    )
                    >= self.INS_info.iloc[INS_index, self.col_bam]
                ):  # When we delete something to put it somewhere after.
                    self.sv_type.append("TRA_forward")
                else:  # When we delete something to put it somewhere before.
                    #  These translocations are different so we separate them.

                    self.sv_type.append("TRA_back")

                self.coordsBP1.append(self.DEL_info.iloc[DEL_mates[0], self.col_bam])
                self.coordsBP2.append(self.INS_info.iloc[INS_index, self.col_bam])
                self.coordsBP3.append(self.DEL_info.iloc[DEL_mates[1], self.col_bam])

                ### ADD SOMETHING FOR SIGNS AFTER ###

                self.size.append(
                    abs(
                        self.DEL_info.iloc[DEL_mates[0], self.col_bam]
                        - self.DEL_info.iloc[DEL_mates[1], self.col_bam]
                    )
                )

                self.INS_info = self.INS_info.drop(
                    INS_index
                )  # drop in order to have after this loop a
                # dataframe with all INS which are not breakpoints.

                count_tra += 1

            else:

                self.sv_name.append("DEL" + str(count_del))
                self.sv_type.append("DEL")

                self.coordsBP1.append(self.DEL_info.iloc[DEL_mates[0], self.col_bam])
                self.coordsBP2.append(self.DEL_info.iloc[DEL_mates[1], self.col_bam])
                self.coordsBP3.append(-1)  # No third BP for DEL

                ### ADD SOMETHING FOR SIGNS AFTER ###

                self.size.append(
                    abs(
                        self.DEL_info.iloc[DEL_mates[0], self.col_bam]
                        - self.DEL_info.iloc[DEL_mates[1], self.col_bam]
                    )
                )

                count_del += 1

        # We delete the rows of INS associated to TRA. We have atin self.INSinfo
        # only INS not linked to TRA.

        self.INS_info.index = np.arange(
            0, len(self.INS_info.index)
        )  #  To have indexes from 0 to 1 (not the cas before because we drop some rows).

        for INS_index in self.INS_info.index:

            self.sv_name.append("INS" + str(count_ins))
            self.sv_type.append("INS")

            self.coordsBP1.append(self.INS_info.iloc[INS_index, self.col_bam])
            self.coordsBP2.append(-1)  # No second BP for INS
            self.coordsBP3.append(-1)  # No third BP for INS

            ### ADD SOMETHING FOR SIGNS AFTER ###

            self.size.append(0)  # IMPLEMENT SOMETHING WHICH FIND SIZE AFTER

            count_ins += 1

    def save_sv_combined(self):
        """
        After the combinaison, we save every SVs we have detected in the output
        file.
        """
        n_final_INV = len(self.info_sv.coordsBP1[self.info_sv.sv_type == "INV"])
        final_INV_detected = np.sort(
            np.concatenate(
                (
                    self.info_sv.coordsBP1[self.info_sv.sv_type == "INV"],
                    self.info_sv.coordsBP2[self.info_sv.sv_type == "INV"],
                )
            ).reshape((n_final_INV, 2))
            // self.binsize,
            axis=1,
        )

        n_final_INS = len(self.info_sv.coordsBP1[self.info_sv.sv_type == "INS"])
        final_INS_detected = np.sort(
            self.info_sv.coordsBP1[self.info_sv.sv_type == "INS"] // self.binsize,
            axis=0,
        )

        n_final_DEL = len(self.info_sv.coordsBP1[self.info_sv.sv_type == "DEL"])
        final_DEL_detected = np.sort(
            np.concatenate(
                (
                    self.info_sv.coordsBP1[self.info_sv.sv_type == "DEL"],
                    self.info_sv.coordsBP2[self.info_sv.sv_type == "DEL"],
                )
            ).reshape((n_final_DEL, 2))
            // self.binsize,
            axis=1,
        )

        n_final_TRA =  len(self.info_sv.coordsBP1[(self.info_sv.sv_type == "TRA_back") | (self.info_sv.sv_type == "TRA_forward")])
        final_TRA_detected = np.sort(
            np.concatenate(
                (
                    self.info_sv.coordsBP1[
                        (self.info_sv.sv_type == "TRA_back")
                        | (self.info_sv.sv_type == "TRA_forward")
                    ],
                    self.info_sv.coordsBP2[
                        (self.info_sv.sv_type == "TRA_back")
                        | (self.info_sv.sv_type == "TRA_forward")
                    ],
                    self.info_sv.coordsBP3[
                        (self.info_sv.sv_type == "TRA_back")
                        | (self.info_sv.sv_type == "TRA_forward")
                    ],
                )
            ).reshape((n_final_TRA,3))
            // self.binsize,
            axis=1,
        )

        np.save("data/output/detection/INV_detected.npy", final_INV_detected)
        np.save("data/output/detection/INS_detected.npy", final_INS_detected)
        np.save("data/output/detection/DEL_detected.npy", final_DEL_detected)
        np.save("data/output/detection/TRA_detected.npy", final_TRA_detected)

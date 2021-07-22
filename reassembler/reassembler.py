import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import has_path, all_simple_paths

from sklearn.linear_model import LinearRegression

from alive_progress import alive_bar

from Bio import Seq, SeqIO
from Bio.Seq import MutableSeq

from typing import Generator, List, Tuple

import reassembler.updating as upd
from sv_class.sv_class import SVs


class Reassembler(object):
    """
    This class handles to reassemble a scrambled HiC-matrix. The framework needs 
    the coord of each SV that has been detected, the HiC-matrix file and the genome 
    file and with that, it will reassemble the matrix.

    Examples
    --------
    reassembler = Reassembler(SV_dico, file_scrambled, file_seq, chrom_name)
    rec_reassembled = reassembler.reassembly()
    
    Attributes
    ----------
    info_sv : SVs
        Information of every SV.
        
    file_scrambled: str
        Filename of the npy file where the scrambled matrix is.

    file_seq: str
        Filename of the genome file.

    chrom_name: str
        Name of the chromosome associated to the scrambled matrix.

    binsize: int
        Binsize used in HiC-matrix.
    """

    def __init__(
        self,
        info_sv: SVs,
        file_scrambled: str,
        file_seq: str,
        chrom_name: str,
        binsize: int = 2000,
    ):

        self.svs = info_sv
        self.file_scrambled = file_scrambled  #  We will use that for plot
        self.scrambled = np.load(file_scrambled)
        self.chrom_name = chrom_name
        self.binsize = binsize
        self.seq_scrambled = MutableSeq(self.load_seq(file_seq, self.chrom_name))
        self.sgns_exist = self.svs.sgnsBP1 is not None

    def load_seq(self, filename: str, chrom_id: str) -> Generator:
        """
        Return the sequence of a chromosome between two indexes.
    
        Parameters:
        ----------
        path : str 
            Path where the sequences are.

        chrom_id : str
            Id of the chromosome we want.

        ind_beg : int
            Index of the begin of the part of the sequence we want. 
            
        ind_end : int
            Index of the end of the part of the sequence we want. 
        """

        records = SeqIO.parse(filename, format="fasta")

        for rec in records:
            if rec.id == chrom_id:

                seq_to_return = str(rec.seq)
                break
        return seq_to_return

    def check_overlap(self, coords_1: List[int], coords_2: List[int]) -> int:
        """
        Check if the two SVs are overlapped or not. Return the overlap ratio and
        the size of the overlap.

        Parameters:
        ----------
        coords_1: List[int] 
            List of the coordinates of the first SV.

        coords_2: List[int] 
            List of the coordinates of the first SV.

        chrom_id : str
            List of the coordinates of the second SV.

        """

        if (coords_2[0] < coords_1[1]) & (coords_1[0] < coords_2[1]):

            size_overlap = coords_1[1] - coords_2[0]
            overlap_ratio = size_overlap / (coords_2[1] - coords_1[0])

            return overlap_ratio
        else:

            return 0

    def connect(self, index_sv1: int, index_sv2: int) -> float:

        """
        Check if two SVs must me connected in the graph of the complex SVs or not.
        It returns +inf if they are not connected.

        Parameters:
        ----------
        sv1:
            Tuple with the index of the first sv.

        sv2:
            Tuple with the index of the second sv.
        """

        other_1 = False  # Two tests if translocation because there are two fragments.
        # I specify if it is for coords 1 or coords2.

        other_2 = False
        ### Compute coords for SV1
        if self.svs.sv_type[index_sv1] == "INV":

            coords_1 = np.array(
                [self.svs.coordsBP1[index_sv1], self.svs.coordsBP2[index_sv1],]
            )

        elif (self.svs.sv_type[index_sv1] == "TRA_forward") or (
            self.svs.sv_type[index_sv1] == "TRA_back"
        ):
            other_1 = True

            coords_1 = np.array(
                [self.svs.coordsBP1[index_sv1], self.svs.coordsBP3[index_sv1],]
            )
            coords_1_other = np.array(
                [
                    self.svs.coordsBP2[index_sv1],
                    self.svs.coordsBP2[index_sv1]
                    + abs(
                        self.svs.coordsBP3[index_sv1] - self.svs.coordsBP1[index_sv1]
                    ),
                ]
            )

        elif self.svs.sv_type[index_sv1] == "INS":

            coords_1 = np.array(
                [
                    self.svs.coordsBP1[index_sv1],
                    self.svs.coordsBP1[index_sv1] + self.svs.size[index_sv1],
                ]
            )

        elif self.svs.sv_type[index_sv1] == "DEL":

            coords_1 = np.array(
                [self.svs.coordsBP1[index_sv1], self.svs.coordsBP2[index_sv1],]
            )
        else:
            raise NotImplementedError("SV type not implemented yet.")

        ### Compute coords for SV2
        if self.svs.sv_type[index_sv2] == "INV":

            coords_2 = np.array(
                [self.svs.coordsBP1[index_sv2], self.svs.coordsBP2[index_sv2],]
            )

        elif (self.svs.sv_type[index_sv2] == "TRA_forward") or (
            self.svs.sv_type[index_sv2] == "TRA_back"
        ):
            other_2 = True
            coords_2 = np.array(
                [self.svs.coordsBP1[index_sv2], self.svs.coordsBP3[index_sv2],]
            )

            coords_2_other = np.array(
                [
                    self.svs.coordsBP2[index_sv2],
                    self.svs.coordsBP2[index_sv2]
                    + abs(
                        self.svs.coordsBP3[index_sv2] - self.svs.coordsBP1[index_sv2]
                    ),
                ]
            )

        elif self.svs.sv_type[index_sv2] == "INS":

            coords_2 = np.array(
                [
                    self.svs.coordsBP1[index_sv2],
                    self.svs.coordsBP1[index_sv2] + self.svs.size[index_sv2],
                ]
            )

        elif self.svs.sv_type[index_sv2] == "DEL":

            coords_2 = np.array(
                [self.svs.coordsBP1[index_sv2], self.svs.coordsBP2[index_sv2],]
            )
        else:
            raise NotImplementedError("SV type not implemented yet.")

        # To have sorted coords
        coords_1 = np.sort(coords_1)
        coords_2 = np.sort(coords_2)

        ratio = self.check_overlap(coords_1, coords_2)

        if other_1:
            ratio = max(ratio, self.check_overlap(coords_1_other, coords_2))

        if other_2:
            ratio = max(ratio, self.check_overlap(coords_1, coords_2_other))

        if other_1 & other_2:
            ratio = max(ratio, self.check_overlap(coords_1_other, coords_2_other))

        if ratio > 0:

            return 1 / ratio

        return np.inf

    def build_graph(self) -> nx.DiGraph:
        """
        Build a graph with every complex SV. Two SVs are connected in the graph when
        they are candidates to be complex SVs.
        """

        complexSV_graph = nx.DiGraph()

        for sv_name1 in self.svs.sv_name:
            for sv_name2 in self.svs.sv_name:

                if sv_name1 != sv_name2:

                    index_sv1 = np.where(self.svs.sv_name == sv_name1)[0][
                        0
                    ]  # [0][0] to have only the "int" (not tuple or arrays)
                    index_sv2 = np.where(self.svs.sv_name == sv_name2)[0][0]
                    dis = self.connect(index_sv1, index_sv2)

                    if dis < np.inf:
                        complexSV_graph.add_weighted_edges_from(
                            [(sv_name1, sv_name2, dis)]
                        )

        for component in nx.strongly_connected_components(complexSV_graph):

            subgraph = complexSV_graph.subgraph(component)

            for node1 in subgraph.nodes:
                for node2 in subgraph.nodes:

                    if node1 != node2:

                        complexSV_graph.add_weighted_edges_from([(node1, node2, 1)])

        return complexSV_graph

    def plot_graph(self):
        """
        Plot the graph created.
        """
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        plt.show()

    def containloop(self, path: str) -> bool:
        """
        Check if a fragment overlap two differents fragments in the path or not.

        Parameters:
        ----------
        path: str 
            Path in the graph (not a filename) you want to check if there is a 
            loop or not.
        """

        for i in range(0, len(path)):
            for j in range(i + 1, len(path)):

                if path[i] == path[j]:
                    return True

        else:
            return False

    def candidate_paths(self) -> List[str]:
        """
        Find which path can be a candidate for each subgraph of the graph. Each
        subgraph represents a connected component of the graph. Return a list with
        all the path which are possible.
        """

        pool_for_each_subgraph = list()
        for component in nx.strongly_connected_components(self.graph):

            subgraph = self.graph.subgraph(component).copy()

            pool = []
            nodes = subgraph.nodes()

            for n1 in nodes:
                for n2 in nodes:
                    if n1 == n2:
                        continue
                    pre = has_path(subgraph, n1, n2)
                    if pre:
                        # paths = list(all_shortest_paths(subgraph, n1, n2, weight=None))
                        paths = list(all_simple_paths(subgraph, n1, n2))
                        for p in paths:
                            if not self.containloop(p) and len(p) == len(nodes):
                                pool.append((len(p), p))

            pool.sort(reverse=True)
            pool_for_each_subgraph.append(pool)

        return pool_for_each_subgraph

    def find_best_path(self) -> Tuple[List[str], List[float]]:

        """
        Test every path and returns the best path. For each path, the programs reassemble
        and make a linear regression between the local average contact and the global
        average contact. Keep the path with the best Rsquared associated. It returns the
        best paths with the scores associated.
        """

        paths_for_each_subgraph = self.candidate_paths()

        best_paths = list()
        best_scores = list()

        scrambled_old = np.copy(
            self.scrambled
        )  # Keep old value because we want to test all combinaisons

        coordsBP1_old = np.copy(self.svs.coordsBP1)
        coordsBP2_old = np.copy(self.svs.coordsBP2)
        coordsBP3_old = np.copy(self.svs.coordsBP3)

        sgns_BP1_old = np.copy(self.svs.sgnsBP1)
        sgns_BP2_old = np.copy(self.svs.sgnsBP2)
        sgns_BP3_old = np.copy(self.svs.sgnsBP3)

        print("REASSEMBLY OF COMPLEX SVs:")
        n_combinaisons_tested = sum([len(paths) for paths in paths_for_each_subgraph])
        with alive_bar(n_combinaisons_tested) as bar:
            for k in range(0, len(paths_for_each_subgraph)):

                paths = paths_for_each_subgraph[k]
                Scores = list()  # Score = R² + nb_element which are used to compute R²
                path_with_good_sign = list()

                for path in paths:

                    self.scrambled = np.copy(scrambled_old)

                    self.svs.coordsBP1 = np.copy(coordsBP1_old)
                    self.svs.coordsBP2 = np.copy(coordsBP2_old)
                    self.svs.coordsBP3 = np.copy(coordsBP3_old)

                    self.svs.sgnsBP1 = np.copy(sgns_BP1_old)
                    self.svs.sgnsBP2 = np.copy(sgns_BP2_old)
                    self.svs.sgnsBP3 = np.copy(sgns_BP3_old)

                    path_ = path[1:][
                        0
                    ]  # The first element is the number of SVs in the path. [0] because it is a tuple.

                    fragments = list()

                    for sv_name in path_:

                        index_sv = np.where(self.svs.sv_name == sv_name)[0][0]

                        if self.svs.sv_type[index_sv] == "INV":

                            coordsBP = np.array(
                                [
                                    self.svs.coordsBP1[index_sv],
                                    self.svs.coordsBP2[index_sv],
                                ]
                            )

                            if self.sgns_exist:
                                sgns = np.array(
                                    [
                                        self.svs.sgnsBP1[index_sv],
                                        self.svs.sgnsBP2[index_sv],
                                    ]
                                )

                                sgns = sgns[
                                    np.argsort(coordsBP)
                                ]  #  to have sorted coords and sgn associated
                            coordsBP = np.sort(coordsBP)

                            coords_matrix = coordsBP // self.binsize

                            self.correct_inversion_scrambled(
                                coords_matrix[0], coords_matrix[1]
                            )

                            # Update coords
                            self.svs.coordsBP1 = upd.update_coords_inv(
                                coordsBP[0], coordsBP[1], self.svs.coordsBP1
                            )
                            self.svs.coordsBP2 = upd.update_coords_inv(
                                coordsBP[0], coordsBP[1], self.svs.coordsBP2
                            )
                            self.svs.coordsBP3 = upd.update_coords_inv(
                                coordsBP[0], coordsBP[1], self.svs.coordsBP3
                            )
                            if self.sgns_exist:
                                # Update signs
                                self.svs.sgnsBP1 = upd.update_sgn_inversion(
                                    coordsBP[0],
                                    coordsBP[1],
                                    sgns[0],
                                    sgns[1],
                                    self.svs.coordsBP1,
                                    self.svs.sgnsBP1,
                                )
                                self.svs.sgnsBP2 = upd.update_sgn_inversion(
                                    coordsBP[0],
                                    coordsBP[1],
                                    sgns[0],
                                    sgns[1],
                                    self.svs.coordsBP2,
                                    self.svs.sgnsBP2,
                                )

                                self.svs.sgnsBP3 = upd.update_sgn_inversion(
                                    coordsBP[0],
                                    coordsBP[1],
                                    sgns[0],
                                    sgns[1],
                                    self.svs.coordsBP3,
                                    self.svs.sgnsBP3,
                                )

                            ## Add one fragment modified for the test
                            if (3 * coords_matrix[0] - 2 * coords_matrix[1] > 0) & (
                                2 * coords_matrix[1] - coords_matrix[0]
                                <= self.scrambled.shape[1]
                            ):

                                fragment = np.array(
                                    [
                                        [
                                            3 * coords_matrix[0] - 2 * coords_matrix[1],
                                            coords_matrix[0],
                                        ],
                                        [
                                            coords_matrix[0],
                                            2 * coords_matrix[1] - coords_matrix[0],
                                        ],
                                    ]
                                )

                            elif (
                                2 * coords_matrix[0] - coords_matrix[1] > 0
                            ):  # Sometimes, values taken to create the fragment can be negative
                                # (so the fragment doesn't exist)
                                fragment = np.array(
                                    [
                                        [
                                            2 * coords_matrix[0] - coords_matrix[1],
                                            coords_matrix[0],
                                        ],
                                        [coords_matrix[0], coords_matrix[1]],
                                    ]
                                )

                            else:
                                fragment = np.array(
                                    [
                                        [0, coords_matrix[1]],
                                        [coords_matrix[1], 2 * coords_matrix[1]],
                                    ]
                                )

                            fragments.append(fragment)

                        elif self.svs.sv_type[index_sv] == "TRA_forward":

                            coordsBP_ins = np.array(
                                [
                                    self.svs.coordsBP1[index_sv],
                                    self.svs.coordsBP3[index_sv],
                                ]
                            )

                            if self.sgns_exist:

                                sgns_ins = np.array(
                                    [
                                        self.svs.sgnsBP1[index_sv],
                                        self.svs.sgnsBP3[index_sv],
                                    ]
                                )

                                sgns_ins = sgns_ins[
                                    np.argsort(coordsBP_ins)
                                ]  #  to have sorted coords and sgn associated

                            coordsBP_ins = np.sort(coordsBP_ins)
                            beg_seq_ins = coordsBP_ins[0]
                            end_seq_ins = coordsBP_ins[1]

                            beg_seq_del = self.svs.coordsBP2[index_sv]

                            if self.sgns_exist:
                                sgns_del = self.svs.sgnsBP2[index_sv]

                            if min(beg_seq_ins, end_seq_ins) < beg_seq_del:

                                break

                            coord_matrix_start_ins = beg_seq_ins // self.binsize
                            coord_matrix_end_ins = end_seq_ins // self.binsize
                            coord_matrix_start_del = beg_seq_del // self.binsize

                            self.seq_scrambled = self.correct_translocation(
                                self.seq_scrambled,
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                            )

                            self.correct_forward_translocation_scrambled(
                                coord_matrix_start_ins,
                                coord_matrix_end_ins,
                                coord_matrix_start_del,
                            )

                            # Update sgns
                            if self.sgns_exist:

                                new_sgn_start = sgns_del[0] + sgns_ins[0][1]
                                new_sgn_end = sgns_ins[0][0] + sgns_ins[1][1]

                                new_sgn_bp2 = (
                                    sgns_ins[1][0] + sgns_del[1]
                                )  # First sign of the end of sequence deleted + Last sign for del

                            index_BP1 = np.where(
                                coordsBP_ins == self.svs.coordsBP1[index_sv]
                            )[0][
                                0
                            ]  # If BP1 correspond to start or end

                            index_BP3 = np.where(
                                coordsBP_ins == self.svs.coordsBP3[index_sv]
                            )[0][
                                0
                            ]  # If BP3 correspond to start or end

                            if self.sgns_exist:
                                self.svs.sgnsBP1[index_sv] = [
                                    new_sgn_start,
                                    new_sgn_end,
                                ][index_BP1]

                                self.svs.sgnsBP2[index_sv] = new_sgn_bp2

                                self.svs.sgnsBP3[index_sv] = [
                                    new_sgn_start,
                                    new_sgn_end,
                                ][index_BP3]

                            # Update coords

                            self.svs.coordsBP1 = upd.update_coords_tra(
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                                self.svs.coordsBP1,
                            )
                            self.svs.coordsBP2 = upd.update_coords_tra(
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                                self.svs.coordsBP2,
                            )
                            self.svs.coordsBP3 = upd.update_coords_tra(
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                                self.svs.coordsBP3,
                            )

                            ## Add one fragment modified for the test

                            fragment = np.array(
                                [
                                    [
                                        coord_matrix_start_del,
                                        coord_matrix_start_del
                                        + coord_matrix_end_ins
                                        - coord_matrix_start_ins,
                                    ],
                                    [
                                        coord_matrix_start_del,
                                        coord_matrix_start_del
                                        + coord_matrix_end_ins
                                        - coord_matrix_start_ins,
                                    ],
                                ]
                            )

                            fragments.append(fragment)

                        elif self.svs.sv_type[index_sv] == "TRA_back":

                            coordsBP_ins = np.array(
                                [
                                    self.svs.coordsBP1[index_sv],
                                    self.svs.coordsBP3[index_sv],
                                ]
                            )
                            if self.sgns_exist:

                                sgns_ins = np.array(
                                    [
                                        self.svs.sgnsBP1[index_sv],
                                        self.svs.sgnsBP3[index_sv],
                                    ]
                                )

                                sgns_ins = sgns_ins[
                                    np.argsort(coordsBP_ins)
                                ]  #  to have sorted coords and sgn associated

                            coordsBP_ins = np.sort(coordsBP_ins)
                            beg_seq_ins = coordsBP_ins[0]
                            end_seq_ins = coordsBP_ins[1]

                            beg_seq_del = self.svs.coordsBP2[index_sv]
                            sgns_del = self.svs.sgnsBP2[index_sv]

                            if max(beg_seq_ins, end_seq_ins) > beg_seq_del:

                                break

                            coord_matrix_start_ins = beg_seq_ins // self.binsize
                            coord_matrix_end_ins = end_seq_ins // self.binsize
                            coord_matrix_start_del = beg_seq_del // self.binsize

                            self.seq_scrambled = self.correct_translocation(
                                self.seq_scrambled,
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                            )

                            self.correct_back_translocation_scrambled(
                                coord_matrix_start_ins,
                                coord_matrix_end_ins,
                                coord_matrix_start_del,
                            )

                            # Update sgns

                            new_sgn_start = sgns_del[0] + sgns_ins[0][1]
                            new_sgn_end = sgns_ins[0][0] + sgns_ins[1][1]

                            new_sgn_bp2 = (
                                sgns_ins[1][0] + sgns_del[1]
                            )  # First sign of the end of sequence deleted + Last sign for del

                            index_BP1 = np.where(
                                coordsBP_ins == self.svs.coordsBP1[index_sv]
                            )[0][
                                0
                            ]  # If BP1 correspond to start or end

                            index_BP3 = np.where(
                                coordsBP_ins == self.svs.coordsBP3[index_sv]
                            )[0][
                                0
                            ]  # If BP3 correspond to start or end

                            self.svs.sgnsBP1[index_sv] = [new_sgn_start, new_sgn_end][
                                index_BP1
                            ]

                            self.svs.sgnsBP2[index_sv] = new_sgn_bp2

                            self.svs.sgnsBP3[index_sv] = [new_sgn_start, new_sgn_end][
                                index_BP3
                            ]

                            # Update coords

                            self.svs.coordsBP1 = upd.update_coords_tra(
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                                self.svs.coordsBP1,
                            )
                            self.svs.coordsBP2 = upd.update_coords_tra(
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                                self.svs.coordsBP2,
                            )
                            self.svs.coordsBP3 = upd.update_coords_tra(
                                beg_seq_ins,
                                end_seq_ins,
                                beg_seq_del,
                                self.svs.coordsBP3,
                            )

                            ## Add one fragment modified for the test

                            fragment = np.array(
                                [
                                    [
                                        coord_matrix_start_del
                                        - 2 * coord_matrix_end_ins
                                        + 2 * coord_matrix_start_ins,
                                        coord_matrix_start_del
                                        - coord_matrix_end_ins
                                        + coord_matrix_start_ins,
                                    ],
                                    [
                                        coord_matrix_start_del
                                        - coord_matrix_end_ins
                                        + coord_matrix_start_ins,
                                        coord_matrix_start_del,
                                    ],
                                ]
                            )

                            fragments.append(fragment)

                        elif self.svs.sv_type[index_sv] == "INS":

                            pos = self.svs.coordsBP1[index_sv]
                            size = self.svs.size[index_sv]

                            coord_pos = self.svs.coordsBP1[index_sv] // self.binsize
                            coord_size = size // self.binsize
                            self.seq_scrambled = self.correct_insertion(
                                self.seq_scrambled, pos, size
                            )

                            self.correct_insertion_scrambled(coord_pos, coord_size)

                            self.svs.coordsBP1 = upd.update_coords_ins(
                                pos, size, self.svs.coordsBP1
                            )
                            self.svs.coordsBP2 = upd.update_coords_ins(
                                pos, size, self.svs.coordsBP2
                            )
                            self.svs.coordsBP3 = upd.update_coords_ins(
                                pos, size, self.svs.coordsBP3
                            )

                    indexes_path = np.nonzero(
                        np.isin(self.svs.sv_name, np.array(path_))
                    )  # Index of element in the path in sv_name

                    indexes_path_TRA = np.nonzero(
                        (self.svs.sv_type[indexes_path] == "TRA_forward")
                        | (self.svs.sv_type[indexes_path] == "TRA_back")
                    )

                    if self.sgns_exist:

                        full_sgns = np.concatenate(
                            (
                                self.svs.sgnsBP1[indexes_path],
                                self.svs.sgnsBP2[indexes_path],
                                np.concatenate(
                                    (
                                        self.svs.sgnsBP3[indexes_path][
                                            indexes_path_TRA
                                        ],
                                        self.svs.sgnsBP3[indexes_path][
                                            indexes_path_TRA
                                        ],
                                    )
                                ),
                            )
                        )

                        good_sgns = np.all(full_sgns == "+-")
                    else:

                        good_sgns = True  # To statisfy the condition

                    if good_sgns:  # A good path MUST have all sgns of BP to "+-"

                        path_with_good_sign.append(path_)

                        sv_name_enough_values = (
                            list()
                        )  # SV name where linear regression has been done
                        R_squareds_path = list()
                        for j in range(0, len(fragments)):

                            fragment = fragments[j]
                            local_scrambled = self.scrambled[
                                fragment[0, 0] : fragment[0, 1],
                                fragment[1, 0] : fragment[1, 1],
                            ]

                            nb_row = local_scrambled.shape[0]
                            nb_col = local_scrambled.shape[1]

                            mat_row_local = (
                                np.matrix(
                                    np.arange(
                                        fragment[0, 0],
                                        fragment[0, 0] + local_scrambled.shape[0],
                                    ).tolist()
                                    * nb_col
                                )
                                .reshape((nb_row, nb_col))
                                .T
                            )

                            mat_col_local = np.matrix(
                                np.arange(
                                    fragment[1, 0],
                                    fragment[1, 0] + local_scrambled.shape[1],
                                ).tolist()
                                * nb_row
                            ).reshape((nb_row, nb_col))

                            average_global = list()
                            average_local = list()

                            kmax = np.max(mat_col_local - mat_row_local)
                            kmin = 5  # Begin at 5 to have enough value for the mean.
                            thresold_local = (
                                35  # Allows to have enough value for local matrix
                            )

                            if kmax > thresold_local:

                                for k in range(kmin, kmax):

                                    local_values = local_scrambled[
                                        np.where(mat_col_local - mat_row_local == k)
                                    ]
                                    local_values = np.delete(
                                        local_values, np.where(local_values == 0)
                                    )

                                    global_values = np.diag(self.scrambled, k)
                                    global_values = np.delete(
                                        global_values, np.where(global_values == 0)
                                    )

                                    if not (np.isnan(np.mean(local_values))):

                                        average_global.append(np.mean(global_values))
                                        average_local.append(np.mean(local_values))

                                average_global = np.array(average_global).reshape(
                                    (-1, 1)
                                )
                                average_local = np.array(average_local).reshape((-1, 1))

                                thresold_lin = 15  # If there is enough value to make a linear regression
                                if len(average_local) > thresold_lin:

                                    LR = LinearRegression()
                                    LR.fit(average_global, average_local)

                                    sv_name_enough_values.append(path_[j])
                                    R_squareds_path.append(
                                        LR.score(average_global, average_local)
                                    )

                        Scores.append(
                            (
                                np.mean(np.array(R_squareds_path))
                                + len(sv_name_enough_values)
                            )
                            / (len(path_) + 1)
                        )

                        bar()

                Scores = np.array(Scores)
                if len(Scores) > 0:
                    best_path = path_with_good_sign[np.argmax(Scores)]
                    best_score = np.max(Scores)
                    best_paths.append(best_path)
                    best_scores.append(best_score)

        self.scrambled = np.copy(scrambled_old)

        self.svs.coordsBP1 = np.copy(coordsBP1_old)
        self.svs.coordsBP2 = np.copy(coordsBP2_old)
        self.svs.coordsBP3 = np.copy(coordsBP3_old)

        self.svs.sgnsBP1 = np.copy(sgns_BP1_old)
        self.svs.sgnsBP2 = np.copy(sgns_BP2_old)
        self.svs.sgnsBP3 = np.copy(sgns_BP3_old)

        return best_paths, best_scores

    def create_pipeline(self) -> List[str]:
        """
        Create the pipeline which will be used for the reassembly. It creates the graph,
        tests all combinations of complex SVs and keep the best combinaison.
        """

        self.graph = self.build_graph()

        # Correction of simple SV before correct complex SV.
        nodes_graph = np.array(self.graph.nodes)

        print("REASSEMBLY OF SIMPLE SV:")
        n_simple_sv = len(self.svs.sv_name) - len(nodes_graph)
        with alive_bar(n_simple_sv) as bar:  # Create progresing bar
            for sv_name in self.svs.sv_name:

                if (
                    len(np.where(nodes_graph == sv_name)[0]) == 0
                ):  # If the SV is not in the graph, it is a simple SV.

                    index_sv = np.where(self.svs.sv_name == sv_name)[0][
                        0
                    ]  # [0][0] to have the "int", not tuple or array.

                    if self.svs.sv_type[index_sv] == "INV":

                        BPs_INV = np.array(
                            [self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]]
                        )
                        INV_start = np.min(BPs_INV)
                        INV_end = np.max(BPs_INV)

                        coords_matrix = BPs_INV // self.binsize

                        if self.sgns_exist:
                            sgns = np.array(
                                [self.svs.sgnsBP1[index_sv], self.svs.sgnsBP2[index_sv]]
                            )

                            sgn_start = sgns[np.argmin(coords_matrix)]
                            sgn_end = sgns[np.argmax(coords_matrix)]

                        coord_matrix_start = np.min(coords_matrix)
                        coord_matrix_end = np.max(coords_matrix)

                        self.seq_scrambled = self.correct_inversion(
                            self.seq_scrambled, INV_start, INV_end
                        )

                        self.correct_inversion_scrambled(
                            coord_matrix_start, coord_matrix_end
                        )

                        # Update coords
                        self.svs.coordsBP1 = upd.update_coords_inv(
                            INV_start, INV_end, self.svs.coordsBP1
                        )
                        self.svs.coordsBP2 = upd.update_coords_inv(
                            INV_start, INV_end, self.svs.coordsBP2
                        )

                        self.svs.coordsBP3 = upd.update_coords_inv(
                            INV_start, INV_end, self.svs.coordsBP3
                        )

                        # Update signs

                        if self.sgns_exist:
                            self.svs.sgnsBP1 = upd.update_sgn_inversion(
                                INV_start,
                                INV_end,
                                sgn_start,
                                sgn_end,
                                self.svs.coordsBP1,
                                self.svs.sgnsBP1,
                            )
                            self.svs.sgnsBP2 = upd.update_sgn_inversion(
                                INV_start,
                                INV_end,
                                sgn_start,
                                sgn_end,
                                self.svs.coordsBP2,
                                self.svs.sgnsBP2,
                            )

                            self.svs.sgnsBP3 = upd.update_sgn_inversion(
                                INV_start,
                                INV_end,
                                sgn_start,
                                sgn_end,
                                self.svs.coordsBP3,
                                self.svs.sgnsBP3,
                            )

                    elif self.svs.sv_type[index_sv] == "TRA_forward":

                        beg_seq_ins = min(
                            self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                        )
                        end_seq_ins = max(
                            self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                        )
                        beg_seq_del = self.svs.coordsBP2[index_sv]

                        coord_matrix_start_ins = beg_seq_ins // self.binsize
                        coord_matrix_end_ins = end_seq_ins // self.binsize
                        coord_matrix_start_del = beg_seq_del // self.binsize

                        self.seq_scrambled = self.correct_translocation(
                            self.seq_scrambled, beg_seq_ins, end_seq_ins, beg_seq_del
                        )

                        self.correct_forward_translocation_scrambled(
                            coord_matrix_start_ins,
                            coord_matrix_end_ins,
                            coord_matrix_start_del,
                        )

                        # Update coords

                        self.svs.coordsBP1 = upd.update_coords_tra(
                            beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP1
                        )
                        self.svs.coordsBP2 = upd.update_coords_tra(
                            beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP2
                        )
                        self.svs.coordsBP3 = upd.update_coords_tra(
                            beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP3
                        )

                    elif self.svs.sv_type[index_sv] == "TRA_back":

                        beg_seq_ins = min(
                            self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                        )
                        end_seq_ins = max(
                            self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                        )
                        beg_seq_del = self.svs.coordsBP2[index_sv]

                        coord_matrix_start_ins = beg_seq_ins // self.binsize
                        coord_matrix_end_ins = end_seq_ins // self.binsize
                        coord_matrix_start_del = beg_seq_del // self.binsize

                        self.seq_scrambled = self.correct_translocation(
                            self.seq_scrambled, beg_seq_ins, end_seq_ins, beg_seq_del
                        )

                        self.correct_back_translocation_scrambled(
                            coord_matrix_start_ins,
                            coord_matrix_end_ins,
                            coord_matrix_start_del,
                        )

                        # Update coords

                        self.svs.coordsBP1 = upd.update_coords_tra(
                            beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP1
                        )
                        self.svs.coordsBP2 = upd.update_coords_tra(
                            beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP2
                        )
                        self.svs.coordsBP3 = upd.update_coords_tra(
                            beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP3
                        )
                    elif self.svs.sv_type[index_sv] == "INS":

                        pos = self.svs.coordsBP1[index_sv]
                        size = self.svs.size[index_sv]

                        coord_pos = self.svs.coordsBP1[index_sv] // self.binsize
                        coord_size = size // self.binsize

                        self.seq_scrambled = self.correct_insertion(
                            self.seq_scrambled, pos, size
                        )

                        self.correct_insertion_scrambled(coord_pos, coord_size)

                        self.svs.coordsBP1 = upd.update_coords_ins(
                            pos, size, self.svs.coordsBP1
                        )
                        self.svs.coordsBP2 = upd.update_coords_ins(
                            pos, size, self.svs.coordsBP2
                        )

                        self.svs.coordsBP3 = upd.update_coords_ins(
                            pos, size, self.svs.coordsBP3
                        )

                    elif self.svs.sv_type[index_sv] == "DEL":

                        beg_seq_del = min(
                            self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]
                        )
                        end_seq_del = max(
                            self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]
                        )

                        coord_matrix_start_del = beg_seq_del // self.binsize
                        coord_matrix_end_del = end_seq_del // self.binsize

                        self.seq_scrambled = self.correct_deletion(
                            self.seq_scrambled, beg_seq_del, end_seq_del
                        )

                        self.correct_deletion_scrambled(
                            coord_matrix_start_del, coord_matrix_end_del
                        )

                        self.svs.coordsBP1 = upd.update_coords_del(
                            beg_seq_del, end_seq_del, self.svs.coordsBP1
                        )
                        self.svs.coordsBP2 = upd.update_coords_del(
                            beg_seq_del, end_seq_del, self.svs.coordsBP2
                        )

                        self.svs.coordsBP3 = upd.update_coords_del(
                            beg_seq_del, end_seq_del, self.svs.coordsBP3
                        )

                    bar()
        # After that find the best combinaisons of complex SV

        best_paths, best_scores = self.find_best_path()

        final_path = []

        for i in range(0, len(best_paths)):
            final_path += best_paths[i]

        return final_path

    def correct_inversion_scrambled(self, start: int, end: int):
        """
        Correction of inversions in the HiC-matrix.
        """

        self.scrambled[start : end + 1, :] = self.scrambled[start : end + 1, :][::-1, :]

        self.scrambled[:, start : end + 1] = self.scrambled[:, start : end + 1][:, ::-1]

        self.scrambled[start, :] = 0
        self.scrambled[:, start] = 0

        if end < self.scrambled.shape[0]:
            self.scrambled[end, :] = 0

        if end < self.scrambled.shape[1]:
            self.scrambled[:, end] = 0

    def correct_forward_translocation_scrambled(
        self, start: int, end: int, start_paste: int
    ):
        """
        Correction of forward translocation in the HiC-matrix.
        """

        size_insertion = end - start + 1

        fragment_to_modify_1 = self.scrambled[
            0 : start_paste + 1, start : end + 1
        ]  #  Fragment at the top of the DEL-separation
        fragment_to_modify_2 = self.scrambled[
            start : end + 1, start : end + 1
        ]  # Square
        fragment_to_modify_3 = np.concatenate(
            (
                self.scrambled[start_paste + 1 : start, start : end + 1],
                self.scrambled[start + size_insertion :, start : end + 1],
            ),
            axis=0,
        )  #  Fragment at the bottom of the DEL-separation

        # Delete rows/cols
        self.scrambled = np.concatenate(
            (self.scrambled[0:start, :], self.scrambled[end + 1 :, :]), axis=0
        )
        self.scrambled = np.concatenate(
            (self.scrambled[:, 0:start], self.scrambled[:, end + 1 :]), axis=1
        )

        # Insertion white spaces
        self.scrambled = np.concatenate(
            (
                self.scrambled[: start_paste + 1, :],
                np.zeros((size_insertion, self.scrambled.shape[1])),
                self.scrambled[start_paste + 1 :, :],
            ),
            axis=0,
        )
        self.scrambled = np.concatenate(
            (
                self.scrambled[:, : start_paste + 1],
                np.zeros((self.scrambled.shape[0], size_insertion)),
                self.scrambled[:, start_paste + 1 :],
            ),
            axis=1,
        )

        # Create matrix for each fragment to insert
        insertion_1 = np.concatenate(
            (
                fragment_to_modify_1,
                np.zeros(
                    (
                        self.scrambled.shape[0] - fragment_to_modify_1.shape[0],
                        size_insertion,
                    )
                ),
            ),
            axis=0,
        )

        insertion_2 = np.concatenate(
            (
                np.zeros((start_paste + 1, size_insertion)),
                fragment_to_modify_2,
                np.zeros(
                    (
                        self.scrambled.shape[0] - (start_paste + 1 + size_insertion),
                        size_insertion,
                    )
                ),
            )
        )

        insertion_3 = np.concatenate(
            (
                np.zeros((start_paste + 1 + size_insertion, size_insertion)),
                fragment_to_modify_3,
                np.zeros(
                    (
                        self.scrambled.shape[0]
                        - fragment_to_modify_3.shape[0]
                        - (start_paste + 1 + size_insertion),
                        size_insertion,
                    )
                ),
            ),
            axis=0,
        )

        insertion = np.concatenate(
            (
                np.zeros((self.scrambled.shape[0], start_paste)),
                insertion_1 + insertion_2 + insertion_3,
                np.zeros(
                    (
                        self.scrambled.shape[0],
                        self.scrambled.shape[1] - start_paste - size_insertion,
                    )
                ),
            ),
            axis=1,
        )

        insertion_transposed = np.concatenate(
            (
                np.zeros((self.scrambled.shape[0], start_paste)),
                insertion_1 + insertion_3,
                np.zeros(
                    (
                        self.scrambled.shape[0],
                        self.scrambled.shape[1] - start_paste - size_insertion,
                    )
                ),
            ),
            axis=1,
        ).T

        self.scrambled = self.scrambled + insertion + insertion_transposed

        if (
            start_paste < self.scrambled.shape[0]
        ):  #  Sometimes, the coord is exactly scrambled.shape so there is nothing to delete after.
            self.scrambled[start_paste, :] = 0
        if start_paste < self.scrambled.shape[1]:
            self.scrambled[:, start_paste] = 0

        self.scrambled[start, :] = 0
        self.scrambled[:, start] = 0

        if end < self.scrambled.shape[0]:
            self.scrambled[end, :] = 0
        if end < self.scrambled.shape[1]:
            self.scrambled[:, end] = 0

    def correct_back_translocation_scrambled(
        self, start: int, end: int, start_paste: int
    ):
        """
        Correction of back translocation in the HiC-matrix.
        """

        size_insertion = end - start + 1

        fragment_to_modify_1 = np.concatenate(
            (
                self.scrambled[start : end + 1, :start],
                self.scrambled[start : end + 1, end + 1 : start_paste],
            ),
            axis=1,
        )  #  Fragment at the left of the DEL-separation
        fragment_to_modify_2 = self.scrambled[
            start : end + 1, start : end + 1
        ]  # Square
        fragment_to_modify_3 = self.scrambled[
            start : end + 1, start_paste:
        ]  # Fragment at the right of the DEL-separation

        # Delete rows/cols
        self.scrambled = np.concatenate(
            (self.scrambled[0:start, :], self.scrambled[end + 1 :, :]), axis=0
        )
        self.scrambled = np.concatenate(
            (self.scrambled[:, 0:start], self.scrambled[:, end + 1 :]), axis=1
        )

        # Insertion white spaces
        self.scrambled = np.concatenate(
            (
                self.scrambled[: start_paste + 1 - size_insertion, :],
                np.zeros((size_insertion, self.scrambled.shape[1])),
                self.scrambled[start_paste + 1 - size_insertion :, :],
            ),
            axis=0,
        )
        self.scrambled = np.concatenate(
            (
                self.scrambled[:, : start_paste + 1 - size_insertion],
                np.zeros((self.scrambled.shape[0], size_insertion)),
                self.scrambled[:, start_paste + 1 - size_insertion :],
            ),
            axis=1,
        )

        # Create matrix for each fragment to insert
        insertion_1 = np.concatenate(
            (
                fragment_to_modify_1,
                np.zeros(
                    (
                        size_insertion,
                        self.scrambled.shape[1] - fragment_to_modify_1.shape[1],
                    )
                ),
            ),
            axis=1,
        )

        insertion_2 = np.concatenate(
            (
                np.zeros((size_insertion, start_paste - size_insertion)),
                fragment_to_modify_2,
                np.zeros((size_insertion, self.scrambled.shape[1] - start_paste)),
            ),
            axis=1,
        )
        insertion_3 = np.concatenate(
            (
                np.zeros(
                    (
                        size_insertion,
                        self.scrambled.shape[1] - fragment_to_modify_3.shape[1],
                    )
                ),
                fragment_to_modify_3,
            ),
            axis=1,
        )

        insertion = np.concatenate(
            (
                np.zeros((start_paste - size_insertion, self.scrambled.shape[1])),
                insertion_1 + insertion_2 + insertion_3,
                np.zeros(
                    (self.scrambled.shape[0] - start_paste, self.scrambled.shape[1])
                ),
            ),
            axis=0,
        )

        insertion_transposed = np.concatenate(
            (
                np.zeros((start_paste - size_insertion, self.scrambled.shape[1])),
                insertion_1 + insertion_3,
                np.zeros(
                    (self.scrambled.shape[0] - start_paste, self.scrambled.shape[1])
                ),
            ),
            axis=0,
        ).T

        self.scrambled = self.scrambled + insertion + insertion_transposed

        self.scrambled[start_paste - size_insertion, :] = 0
        self.scrambled[:, start_paste - size_insertion] = 0

        if (
            start_paste < self.scrambled.shape[0]
        ):  #  Sometimes, the coord is exactly scrambled.shape so there is nothing to delete after.
            self.scrambled[start_paste, :] = 0
        if start_paste < self.scrambled.shape[1]:
            self.scrambled[:, start_paste] = 0

        self.scrambled[start, :] = 0
        self.scrambled[:, start] = 0

        if end < self.scrambled.shape[0]:
            self.scrambled[end, :] = 0
        if end < self.scrambled.shape[1]:
            self.scrambled[:, end] = 0

    def correct_insertion_scrambled(self, pos: int, size: int):
        """
        Correction of insertion in the Hi-C matrix.
        """

        # Insertion white spaces
        self.scrambled = np.concatenate(
            (
                self.scrambled[:pos, :],
                np.zeros((size, self.scrambled.shape[1])),
                self.scrambled[pos:, :],
            ),
            axis=0,
        )
        self.scrambled = np.concatenate(
            (
                self.scrambled[:, : pos + 1],
                np.zeros((self.scrambled.shape[0], size)),
                self.scrambled[:, pos + 1 :],
            ),
            axis=1,
        )

    def correct_deletion_scrambled(self, start: int, end: int):
        """
        Correction of deletion in the Hi-C matrix.
        """

        self.scrambled = np.concatenate(
            (self.scrambled[:start, :], self.scrambled[end:, :],), axis=0,
        )
        self.scrambled = np.concatenate(
            (self.scrambled[:, :start], self.scrambled[:, end:],), axis=1,
        )

    def correct_inversion(
        self, mutseq: Seq.MutableSeq, start: int, end: int
    ) -> Seq.MutableSeq:
        """
        Correction of inversions in the sequence.        
        """

        mutseq[start:end] = upd.inversion(mutseq[start:end])
        return mutseq

    def correct_insertion(
        self, mutseq: Seq.MutableSeq, pos: int, size: int
    ) -> Seq.MutableSeq:
        """
        Correction of insertions in the sequence.        
        """

        mutseq = mutseq[:pos] + "N" * size + mutseq[pos:]
        return mutseq

    def correct_deletion(
        self, mutseq: Seq.MutableSeq, deb: int, end: int
    ) -> Seq.MutableSeq:
        """
        Correction of deletions in the sequence.        
        """
        mutseq = mutseq[:deb] + mutseq[end:]

        return mutseq

    def correct_translocation(
        self, mutseq: Seq.MutableSeq, start_ins, end_ins, start_cut
    ) -> Seq.MutableSeq:
        """
        Correction of forward translocation in the sequence.        
        """

        mutseq = upd.translocation(start_ins, end_ins, start_cut, mutseq)

        return mutseq

    def reassembly(self, plot=False):
        """
        Create the pipeline and reassembly the sequence.
        """

        pipeline = self.create_pipeline()

        for sv_name in pipeline:

            index_sv = np.where(self.svs.sv_name == sv_name)[0][
                0
            ]  # [0][0] to have the "int", not tuple or array.

            if self.svs.sv_type[index_sv] == "INV":

                BPs_INV = np.array(
                    [self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]]
                )

                INV_start = np.min(BPs_INV)
                INV_end = np.max(BPs_INV)

                coords_matrix = BPs_INV // self.binsize

                if self.sgns_exist:
                    sgns = np.array(
                        [self.svs.sgnsBP1[index_sv], self.svs.sgnsBP2[index_sv]]
                    )
                    sgn_start = sgns[np.argmin(coords_matrix)]
                    sgn_end = sgns[np.argmax(coords_matrix)]

                coord_matrix_start = np.min(coords_matrix)
                coord_matrix_end = np.max(coords_matrix)

                self.seq_scrambled = self.correct_inversion(
                    self.seq_scrambled, INV_start, INV_end
                )

                self.correct_inversion_scrambled(coord_matrix_start, coord_matrix_end)

                # Update coords
                self.svs.coordsBP1 = upd.update_coords_inv(
                    INV_start, INV_end, self.svs.coordsBP1
                )
                self.svs.coordsBP2 = upd.update_coords_inv(
                    INV_start, INV_end, self.svs.coordsBP2
                )

                self.svs.coordsBP3 = upd.update_coords_inv(
                    INV_start, INV_end, self.svs.coordsBP3
                )

                # Update signs
                if self.sgns_exist:
                    self.svs.sgnsBP1 = upd.update_sgn_inversion(
                        INV_start,
                        INV_end,
                        sgn_start,
                        sgn_end,
                        self.svs.coordsBP1,
                        self.svs.sgnsBP1,
                    )
                    self.svs.sgnsBP2 = upd.update_sgn_inversion(
                        INV_start,
                        INV_end,
                        sgn_start,
                        sgn_end,
                        self.svs.coordsBP2,
                        self.svs.sgnsBP2,
                    )

                    self.svs.sgnsBP3 = upd.update_sgn_inversion(
                        INV_start,
                        INV_end,
                        sgn_start,
                        sgn_end,
                        self.svs.coordsBP3,
                        self.svs.sgnsBP3,
                    )

            elif self.svs.sv_type[index_sv] == "TRA_forward":

                beg_seq_ins = min(
                    self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                )
                end_seq_ins = max(
                    self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                )
                beg_seq_del = self.svs.coordsBP2[index_sv]

                coord_matrix_start_ins = beg_seq_ins // self.binsize
                coord_matrix_end_ins = end_seq_ins // self.binsize
                coord_matrix_start_del = beg_seq_del // self.binsize

                self.seq_scrambled = self.correct_translocation(
                    self.seq_scrambled, beg_seq_ins, end_seq_ins, beg_seq_del
                )

                self.correct_forward_translocation_scrambled(
                    coord_matrix_start_ins,
                    coord_matrix_end_ins,
                    coord_matrix_start_del,
                )

                # Update coords

                self.svs.coordsBP1 = upd.update_coords_tra(
                    beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP1
                )

                self.svs.coordsBP2 = upd.update_coords_tra(
                    beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP2
                )

                self.svs.coordsBP3 = upd.update_coords_tra(
                    beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP3
                )

            elif self.svs.sv_type[index_sv] == "TRA_back":

                beg_seq_ins = min(
                    self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                )
                end_seq_ins = max(
                    self.svs.coordsBP1[index_sv], self.svs.coordsBP3[index_sv]
                )
                beg_seq_del = self.svs.coordsBP2[index_sv]

                coord_matrix_start_ins = beg_seq_ins // self.binsize
                coord_matrix_end_ins = end_seq_ins // self.binsize
                coord_matrix_start_del = beg_seq_del // self.binsize

                self.seq_scrambled = self.correct_translocation(
                    self.seq_scrambled, beg_seq_ins, end_seq_ins, beg_seq_del
                )

                self.correct_back_translocation_scrambled(
                    coord_matrix_start_ins,
                    coord_matrix_end_ins,
                    coord_matrix_start_del,
                )

                # Update coords
                self.svs.coordsBP1 = upd.update_coords_tra(
                    beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP1
                )
                self.svs.coordsBP2 = upd.update_coords_tra(
                    beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP2
                )
                self.svs.coordsBP3 = upd.update_coords_tra(
                    beg_seq_ins, end_seq_ins, beg_seq_del, self.svs.coordsBP3
                )

            elif self.svs.sv_type[index_sv] == "INS":

                pos = self.svs.coordsBP1[index_sv]
                size = self.svs.size[index_sv]

                coord_pos = self.svs.coordsBP1[index_sv] // self.binsize
                coord_size = size // self.binsize

                self.seq_scrambled = self.correct_insertion(
                    self.seq_scrambled, pos, size
                )

                self.correct_insertion_scrambled(coord_pos, coord_size)

                self.svs.coordsBP1 = upd.update_coords_ins(
                    pos, size, self.svs.coordsBP1
                )

                self.svs.coordsBP2 = upd.update_coords_ins(
                    pos, size, self.svs.coordsBP2
                )

                self.svs.coordsBP3 = upd.update_coords_ins(
                    pos, size, self.svs.coordsBP3
                )

            elif self.svs.sv_type[index_sv] == "DEL":

                beg_seq_del = min(
                    self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]
                )
                end_seq_del = max(
                    self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]
                )

                coord_matrix_start_del = beg_seq_del // self.binsize
                coord_matrix_end_del = end_seq_del // self.binsize

                self.seq_scrambled = self.correct_deletion(
                    self.seq_scrambled, beg_seq_del, end_seq_del
                )

                self.correct_deletion_scrambled(
                    coord_matrix_start_del, coord_matrix_end_del
                )

                self.svs.coordsBP1 = upd.update_coords_del(
                    beg_seq_del, end_seq_del, self.svs.coordsBP1
                )
                self.svs.coordsBP2 = upd.update_coords_del(
                    beg_seq_del, end_seq_del, self.svs.coordsBP2
                )

                self.svs.coordsBP3 = upd.update_coords_del(
                    beg_seq_del, end_seq_del, self.svs.coordsBP3
                )

        self.plot_difference()
        return self.scrambled, self.seq_scrambled

    def plot_difference(self,):

        fig, ax = plt.subplots(2)

        # Before reassembly
        img_scrambled = ax[0].imshow(np.load(self.file_scrambled), cmap="afmhot_r")
        ax[0].set_title("Scrambled matrix")

        # After reassembly
        img_reassembled = ax[1].imshow(self.scrambled, cmap="afmhot_r")
        ax[1].set_title("Reassembled matrix")

        plt.colorbar(img_scrambled, ax=ax[0])
        plt.colorbar(img_reassembled, ax=ax[1])

        fig.savefig("data/output/reassembly/difference.png")

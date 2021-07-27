# Class used for the reassembly of the Hi-C map.

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import has_path, all_simple_paths

from sklearn.linear_model import LinearRegression

from alive_progress import alive_bar

from Bio import Seq, SeqIO
from Bio.Seq import MutableSeq

from typing import Generator, List, Tuple

import reassembler.updating.inversion as upd_inv
import reassembler.updating.deletion as upd_del
import reassembler.updating.insertion as upd_ins
import reassembler.updating.tra_for as upd_for
import reassembler.updating.tra_back as upd_back

from svs.svs import SVs


class Reassembler(object):
    """
    This class handles to reassemble a scrambled HiC-matrix. The framework needs 
    the coord of each SV that has been detected, the HiC-matrix file and the genome 
    file and with that, it will reassemble the matrix.

    Examples
    --------
    reassembler = Reassembler(info_sv, matrix, seq, chrom_name, binsize)
    mat_reassembled, seq_reassembled = reassembler.reassembly()
    
    Attributes
    ----------
    info_sv : SVs
        Informations about all the SVs.
        
    file_scrambled: str
        Filename of the npy file where the scrambled matrix is.

    file_seq: str
        Filename of the genome file.

    chrom_name: str
        Name of the chromosome associated to the scrambled matrix.

    binsize: int
        Binsize used to generate the HiC-matrix.
    """

    def __init__(
        self,
        info_sv: SVs,
        file_scrambled: str,
        file_seq: str,
        chrom_name: str,
        binsize: int,
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
        """

        records = SeqIO.parse(filename, format="fasta")

        for rec in records:
            if rec.id == chrom_id:

                seq_to_return = str(rec.seq)
                break
        return seq_to_return

    def check_overlap(self, coords_1: List[int], coords_2: List[int]) -> int:
        """
        Check if the two SVs are overlapped or not. It eturns the overlap ratio.

        Parameters:
        ----------
        coords_1: List[int] 
            List of the coordinates of the first SV.

        coords_2: List[int] 
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
        It returns +inf if they are not connected, the inverse of the overlap ratio 
        if two SVs are connected.

        Parameters:
        ----------
        sv1:
            Tuple with the index of the first sv.

        sv2:
            Tuple with the index of the second sv.
        """

        other_1 = False  # other_1 and other_2 are important for the translocation because there are two fragments, 
                        # so we must check the overlap for the two fragments.
        
        other_2 = False

        ### Compute coords for SV1
        if self.svs.sv_type[index_sv1] == "INV": # It is different for each SV

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
        Build a graph with every complex SV. The graph has different strongly connected component.
        Each strongly connected component represents SVs which are overlapped. We will 
        test every combinaision of nodes for each strongly connected component.
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

    def correct_inversion(self, index_sv):

        # To have coordinates linked to the sv
        coordsBP = np.array(
            [
                self.svs.coordsBP1[index_sv],
                self.svs.coordsBP2[index_sv],
            ])
    
        if self.sgns_exist:
            sgns = np.array(
                [
                    self.svs.sgnsBP1[index_sv],
                    self.svs.sgnsBP2[index_sv],
                ])
            sgns = sgns[
                np.argsort(coordsBP)
            ]  #  to have sorted coords and sgn associated

        coordsBP = np.sort(coordsBP) 
        coords_matrix = coordsBP // self.binsize


        # Correction in the matrix and in the sequence
        self.scrambled = upd_inv.correct_inversion_matrix(
            coords_matrix[0],
            coords_matrix[1],
            self.scrambled)
        self.seq_scrambled = upd_inv.correct_inversion_sequence(coords_matrix[0], coords_matrix[1], self.seq_scrambled)
    
        # Update coords
        self.svs.coordsBP1 = upd_inv.update_coords_inv(
            coordsBP[0], coordsBP[1], self.svs.coordsBP1
        )
        self.svs.coordsBP2 = upd_inv.update_coords_inv(
            coordsBP[0], coordsBP[1], self.svs.coordsBP2
        )
        self.svs.coordsBP3 = upd_inv.update_coords_inv(
            coordsBP[0], coordsBP[1], self.svs.coordsBP3
        )

        # Update signs
        if self.sgns_exist:
            
            self.svs.sgnsBP1 = upd_inv.update_sgn_inversion(
                coordsBP[0],
                coordsBP[1],
                sgns[0],
                sgns[1],
                self.svs.coordsBP1,
                self.svs.sgnsBP1,
            )
            self.svs.sgnsBP2 = upd_inv.update_sgn_inversion(
                coordsBP[0],
                coordsBP[1],
                sgns[0],
                sgns[1],
                self.svs.coordsBP2,
                self.svs.sgnsBP2,
            )
    
            self.svs.sgnsBP3 = upd_inv.update_sgn_inversion(
                coordsBP[0],
                coordsBP[1],
                sgns[0],
                sgns[1],
                self.svs.coordsBP3,
                self.svs.sgnsBP3,
            )

    def correct_forward_translocation(self,index_sv):

        # To have coordinates linked to the sv
        coordsBP_del = np.array(
            [
                self.svs.coordsBP1[index_sv],
                self.svs.coordsBP3[index_sv],
            ]
        ) # coords of deletion linked to translocation
        if self.sgns_exist:
            sgns_del = np.array(
                [
                    self.svs.sgnsBP1[index_sv],
                    self.svs.sgnsBP3[index_sv],
                ]
            )
            sgns_del = sgns_del[
                np.argsort(coordsBP_del)
            ]  #  to have sorted coords and sgn associated
        coordsBP_del = np.sort(coordsBP_del)
        beg_seq_del = coordsBP_del[0]
        end_seq_del = coordsBP_del[1]
        beg_seq_ins = self.svs.coordsBP2[index_sv] # coords of insertion linked to translocation

        if self.sgns_exist:
            sgns_ins = self.svs.sgnsBP2[index_sv]

        coord_matrix_start_del = beg_seq_del // self.binsize
        coord_matrix_end_del = end_seq_del // self.binsize
        coord_matrix_start_ins = beg_seq_ins // self.binsize

        # Corrections in the matrix and in the sequence                 
        self.scrambled = upd_for.correct_forward_translocation_matrix(
            coord_matrix_start_del,
            coord_matrix_end_del,
            coord_matrix_start_ins,
            self.scrambled
        )
        self.seq_scrambled = upd_for.correct_translocation_sequence(coord_matrix_start_del, coord_matrix_end_del, coord_matrix_start_ins, self.seq_scrambled)

        # Update sgns
        if self.sgns_exist:

            new_sgn_start = sgns_ins[0] + sgns_del[0][1]
            new_sgn_end = sgns_del[0][0] + sgns_del[1][1]

            new_sgn_bp2 = (
                sgns_del[1][0] + sgns_ins[1]
            )  # First sign of the end of sequence deleted + Last sign for del

            index_BP1 = np.where(
            coordsBP_del == self.svs.coordsBP1[index_sv]
            )[0][0]  # If BP1 correspond to start or end

            index_BP3 = np.where(
                coordsBP_del == self.svs.coordsBP3[index_sv]
            )[0][
                0
            ]  # If BP3 correspond to start or end
        
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
            beg_seq_del,
            end_seq_del,
            beg_seq_ins,
            self.svs.coordsBP1,
        )
        self.svs.coordsBP2 = upd.update_coords_tra(
            beg_seq_del,
            end_seq_del,
            beg_seq_ins,
            self.svs.coordsBP2,
        )
        self.svs.coordsBP3 = upd.update_coords_tra(
            beg_seq_del,
            end_seq_del,
            beg_seq_ins,
            self.svs.coordsBP3,
        )

    def correct_back_translocation(self, index_sv):
        
        # To have coordinates linked to the sv
        coordsBP_del = np.array(
            [
                self.svs.coordsBP1[index_sv],
                self.svs.coordsBP3[index_sv],
            ]
        ) # coords of deletion linked to translocation
        if self.sgns_exist:
            sgns_del = np.array(
                [
                    self.svs.sgnsBP1[index_sv],
                    self.svs.sgnsBP3[index_sv],
                ]
            )
            sgns_del = sgns_del[
                np.argsort(coordsBP_del)
            ]  #  to have sorted coords and sgn associated
        coordsBP_del = np.sort(coordsBP_del)
        beg_seq_del = coordsBP_del[0]
        end_seq_del = coordsBP_del[1]
        beg_seq_ins = self.svs.coordsBP2[index_sv] # coords of insertion linked to translocation

        if self.sgns_exist:
            sgns_ins = self.svs.sgnsBP2[index_sv]

        coord_matrix_start_del = beg_seq_del // self.binsize
        coord_matrix_end_del = end_seq_del // self.binsize
        coord_matrix_start_ins = beg_seq_ins // self.binsize

        # Correction in the matrix and the sequence
        self.scrambled = upd_back.correct_back_translocation_matrix(
            coord_matrix_start_del,
            coord_matrix_end_del,
            coord_matrix_start_ins,
            self.scrambled
        )
        self.seq_scrambled = upd_back.correct_translocation_sequence(
            coord_matrix_start_del, 
            coord_matrix_end_del, 
            coord_matrix_start_ins, 
            self.seq_scrambled
        )

        # Update sgns
        if self.sgns_exist:
            new_sgn_start = sgns_ins[0] + sgns_del[0][1]
            new_sgn_end = sgns_del[0][0] + sgns_del[1][1]

            new_sgn_bp2 = (
                sgns_del[1][0] + sgns_ins[1]
            )  # First sign of the end of sequence deleted + Last sign for del

            index_BP1 = np.where(
                coordsBP_del == self.svs.coordsBP1[index_sv]
            )[0][0]  # If BP1 correspond to start or end

            index_BP3 = np.where(
                coordsBP_del == self.svs.coordsBP3[index_sv]
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
        self.svs.coordsBP1 = upd_back.update_coords_tra(
            beg_seq_del,
            end_seq_del,
            beg_seq_ins,
            self.svs.coordsBP1,
        )
        self.svs.coordsBP2 = upd_back.update_coords_tra(
            beg_seq_del,
            end_seq_del,
            beg_seq_ins,
            self.svs.coordsBP2,
        )
        self.svs.coordsBP3 = upd_back.update_coords_tra(
            beg_seq_del,
            end_seq_del,
            beg_seq_ins,
            self.svs.coordsBP3,
        )


    def correct_insertion(self, index_sv):
        
        # To have coordinates linked to the sv
        pos = self.svs.coordsBP1[index_sv]
        size = self.svs.size[index_sv]

        coord_pos = self.svs.coordsBP1[index_sv] // self.binsize
        coord_size = size // self.binsize

        # Correction in the matrix and the sequence
        self.scrambled = upd_ins.correct_insertion_matrix(coord_pos, coord_size, self.scrambled)
        self.seq_scrambled = upd_ins.correct_insertion_sequence(coord_pos, coord_size, self.seq_scrambled)

        # Updating
        self.svs.coordsBP1 = upd_ins.update_coords_ins(
            pos, size, self.svs.coordsBP1
                            )
        self.svs.coordsBP2 = upd_ins.update_coords_ins(
            pos, size, self.svs.coordsBP2
        )
        self.svs.coordsBP3 = upd_ins.update_coords_ins(
            pos, size, self.svs.coordsBP3
        )


    def correct_deletion(self, index_sv):

        # To have coordinates linked to the sv
        beg_seq_del = min(
            self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]
        )
        end_seq_del = max(
            self.svs.coordsBP1[index_sv], self.svs.coordsBP2[index_sv]
        )
        coord_matrix_start_del = beg_seq_del // self.binsize
        coord_matrix_end_del = end_seq_del // self.binsize

        # Correction in the matrix and in the sequence
        self.scrambled = upd_del.correct_deletion_matrix(
            coord_matrix_start_del, coord_matrix_end_del, self.scrambled
        )
        self.seq_scrambled = upd_ins.correct_insertion_seq(beg_seq_del, end_seq_del, self.seq_scrambled)
                
        # Updating
        self.svs.coordsBP1 = upd_del.update_coords_del(
            beg_seq_del, end_seq_del, self.svs.coordsBP1
        )
        self.svs.coordsBP2 = upd_del.update_coords_del(
            beg_seq_del, end_seq_del, self.svs.coordsBP2
        )

        self.svs.coordsBP3 = upd_del.update_coords_del(
            beg_seq_del, end_seq_del, self.svs.coordsBP3
        )

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

        # Keep old value because we want to test all combinaisons
        scrambled_old = np.copy(
            self.scrambled
        )  
        coordsBP1_old = np.copy(self.svs.coordsBP1)
        coordsBP2_old = np.copy(self.svs.coordsBP2)
        coordsBP3_old = np.copy(self.svs.coordsBP3)
        sgns_BP1_old = np.copy(self.svs.sgnsBP1)
        sgns_BP2_old = np.copy(self.svs.sgnsBP2)
        sgns_BP3_old = np.copy(self.svs.sgnsBP3)

        print("REASSEMBLY OF COMPLEX SVs:")
        n_combinaisons_tested = sum([len(paths) for paths in paths_for_each_subgraph])
        with alive_bar(n_combinaisons_tested) as bar: # Progression bar
            for k in range(0, len(paths_for_each_subgraph)):

                paths = paths_for_each_subgraph[k]
                Scores = list()  # Score = R² + nb_element which are used to compute R²
                path_with_good_sign = list()
                

                # We will test all paths for each subgraph
                for path in paths:
                    
                    # Because we update it at each iteration, we take the old value at the beginning
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

                    fragments = list() # Fragment we will use to make the linear_regression (fragment = local matrix)

                    for sv_name in path_:

                        index_sv = np.where(self.svs.sv_name == sv_name)[0][0]

                        if self.svs.sv_type[index_sv] == "INV":

                            self.correct_inversion(index_sv)
                            ## Add coordinates of one fragment modified for the linear regression
                            fragments.append(upd_inv.fragment_inv(self.svs.coordsBP1[index_sv]//self.binsize, 
                                                                self.svs.coordsBP2[index_sv]//self.binsize, 
                                                                self.scrambled.shape[0]))

                        elif self.svs.sv_type[index_sv] == "TRA_forward":

                            self.correct_forward_translocation(index_sv)

                            ## Add coordinates of one fragment modified for the linear regression
                            fragments.append(udp_for.fragment_for(self.svs.coordsBP1[index_sv]//self.binsize, 
                                                                  self.svs.coordsBP3[index_sv]//self.binsize, 
                                                                  self.svs.coordsBP2[index_sv]))

                        elif self.svs.sv_type[index_sv] == "TRA_back":

                            self.correct_back_translocation(index_sv)
                            ## Add coordinates of one fragment modified for the linear regression
                            fragments.append(udp_back.fragment_back(self.svs.coordsBP1[index_sv]//self.binsize, 
                                                                    self.svs.coordsBP3[index_sv]//self.binsize, 
                                                                    self.svs.coordsBP2[index_sv]//self.binsize))

                        elif self.svs.sv_type[index_sv] == "INS":

                            self.correct_insertion(index_sv)
                            # No fragment here because we just add lines

                        elif self.svs.sv_type[index_sv] == "DEL":

                            self.correct_deletion(index_sv)
                            ## Add coordinates of one fragment modified for the linear regression
                            fragments.append(udp_del.fragment_del(self.svs.coordsBP1[index_sv]//self.binsize, 
                                                                    self.svs.coordsBP3[index_sv]//self.binsize))

                    # We have the coordinates of each local matrix where we will make the linear regression after that.

                    # Test condition on sgns if sgns exist
                    if self.sgns_exist:

                        indexes_path = np.nonzero(
                            np.isin(self.svs.sv_name, np.array(path_))
                        )  # Index of element in the path in sv_name

                        indexes_path_TRA = np.nonzero(
                            (self.svs.sv_type[indexes_path] == "TRA_forward")
                            | (self.svs.sv_type[indexes_path] == "TRA_back")
                        )
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
                        good_sgns = np.all(full_sgns == "+-") # Condition is that every sign is "+-"

                    else:

                        good_sgns = True  # To satisfy the condition

                    ## Linear regression for each local matrix, mean of R² will be done
                    if good_sgns:  

                        path_with_good_sign.append(path_)
                        sv_name_enough_values = (
                            list()
                        )  # SV name where linear regression has been done
                        R_squareds_path = list()

                        ### Make the linear regression between each fragment modified and the global matrix
                        for j in range(0, len(fragments)):

                            fragment = fragments[j]

                            # We take the local matrix with the coordinates which are in fragment
                            local_scrambled = self.scrambled[
                                fragment[0, 0] : fragment[0, 1],
                                fragment[1, 0] : fragment[1, 1],
                            ]

                            nb_row = local_scrambled.shape[0]
                            nb_col = local_scrambled.shape[1]
                            

                            # To know at which col/row belongs each element of the local matrix
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
                                    
                                    # Compute local values for each sub_diag
                                    local_values = local_scrambled[
                                        np.where(mat_col_local - mat_row_local == k)
                                    ]
                                    local_values = np.delete(
                                        local_values, np.where(local_values == 0)
                                    )
                                    # Compute global values for each sub_diag
                                    global_values = np.diag(self.scrambled, k)
                                    global_values = np.delete(
                                        global_values, np.where(global_values == 0)
                                    )

                                    if not (np.isnan(np.mean(local_values))):
                                        # Compute the mean of values for each sub_diag
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

                        # Compute the score of the linear regression + number of svs where we make the linear regression 
                        # (sometimes, fragment is too small and we don't make the linear regression in this case)
                        Scores.append(
                            (
                                np.mean(np.array(R_squareds_path))
                                + len(sv_name_enough_values)
                            )
                            / (len(path_) + 1)
                        )

                        bar()
                # Choose path with the best score
                Scores = np.array(Scores)
                if len(Scores) > 0:
                    best_path = path_with_good_sign[np.argmax(Scores)]
                    best_score = np.max(Scores)
                    best_paths.append(best_path)
                    best_scores.append(best_score)

        # Scrambled like before
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

        # Correction of simple SV before correcting complex SV.
        print("REASSEMBLY OF SIMPLE SV:")
        nodes_graph = np.array(self.graph.nodes)
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
                        self.correct_inversion(index_sv)

                    elif self.svs.sv_type[index_sv] == "TRA_forward":
                        self.correct_forward_translocation(index_sv)

                    elif self.svs.sv_type[index_sv] == "TRA_back":
                        self.correct_back_translocation(index_sv)

                    elif self.svs.sv_type[index_sv] == "INS":
                        self.correct_insertion(index_sv)

                    elif self.svs.sv_type[index_sv] == "DEL":
                        self.correct_deletion(index_sv)

                    bar()

        # After that, we find the best combinaisons of complex SV
        best_paths, best_scores = self.find_best_path()

        final_path = []

        for i in range(0, len(best_paths)):
            final_path += best_paths[i]

        return final_path


    def reassembly(self):
        """
        Create the pipeline and reassembly the sequence.
        """

        pipeline = self.create_pipeline()

        for sv_name in pipeline:

            index_sv = np.where(self.svs.sv_name == sv_name)[0][
                0
            ]  # [0][0] to have the "int", not tuple or array.

            if self.svs.sv_type[index_sv] == "INV":
                self.correct_inversion(index_sv)

            elif self.svs.sv_type[index_sv] == "TRA_forward":
                self.correct_forward_translocation(index_sv)

            elif self.svs.sv_type[index_sv] == "TRA_back":
                self.correct_back_translocation(index_sv)

            elif self.svs.sv_type[index_sv] == "INS":

                self.correct_insertion(index_sv)

            elif self.svs.sv_type[index_sv] == "DEL":

                self.correct_deletion(index_sv)

        self.plot_difference()
        return self.scrambled, self.seq_scrambled

    def plot_difference(self):
        """
        Save a figure where there is on this figure the scrambled matrix before 
        reassembly and after reassembly.
        """
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

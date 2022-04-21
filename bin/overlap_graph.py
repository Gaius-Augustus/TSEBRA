#!/usr/bin/env python3
# ==============================================================
# Lars Gabriel
#
# Graph for transcripts of multiple genome annotations.
# It can detect overlapping transcripts.
# Add a feature vector to each node.
# Compare nodes with the 'decision rule'.
# ==============================================================
from features import Node_features
import numpy as np
import tensorflow as tf


class BatchSizeError(Exception):
    pass

class Edge:
    """
        Class handling an edge in the overlap graph.
    """
    def __init__(self, n1_id, n2_id):
        """
            Args:
                n1_id (str): Node ID from overlap graph
                n2_id (str): Node ID from overlap graph
        """
        self.node1 = n1_id
        self.node2 = n2_id
        self.node_to_remove = None
        self.__numb_features__ = 230

        ### feature vectors for directed edge from node n_i to node n_j
        ### features:

        # numb introns in union
         #### For src in E, P, C and M, all:
            # number of introns in n_i / numb introns in union of n_i, n_j
            # number of introns in n_j / numb introns in union of n_i, n_j !
            # number of introns in n_i and in n_j / numb introns in union of n_i, n_j
        # fraction of CDS of n_i that is also in n_j
        # fraction of CDS of n_j that is also in n_i !
        # start codon position difference of nj to ni if they agree on first coding DSS
        # stop codon position difference of nj to ni if they agree on last coding ASS

        self.feature_vector_n1_to_n2 = np.zeros(self.__numb_features__, float)
        self.feature_vector_n2_to_n1 = np.zeros(self.__numb_features__, float)

    def add_features(self, tx1, tx2, n1, n2, evi):
        k=0
        ### for codon in start/stop_codon
            # 1, if codon of ni and nj matches
            # 1, if n_i is starting upstream(start_codon) or ending downstream(stop_codon)
            # for src in P, C, M
                # 1, if codon of ni is supported and not of nj
                # 1, if codon of nj is supported and not of ni
            # start codon position difference of nj to ni if they agree on first coding DSS
            # stop codon position difference of nj to ni if they agree on last coding ASS
        for i, type in enumerate(['start_codon', 'stop_codon']):
            if n1.coords[type] == n2.coords[type]:
                self.feature_vector_n1_to_n2[k] = 1.0
                self.feature_vector_n2_to_n1[k] = 1.0
            elif n1.coords[type] and n2.coords[type]:
                if (tx1.strand == '+' and \
                    n1.coords[type][0][0] < n2.coords[type][0][0]) \
                    or (tx1.strand == '-' and \
                    n1.coords[type][0][0] > n2.coords[type][0][0]):
                    self.feature_vector_n1_to_n2[k+1] = 1.0
                else :
                    self.feature_vector_n2_to_n1[k+1] = 1.0
            k+=2
            for src in ['P', 'C', 'M']:
                if n1.hints[type] and not n2.hints[type]:
                    self.feature_vector_n1_to_n2[k] = 1.0
                    self.feature_vector_n2_to_n1[k+1] = 1.0
                elif n2.hints[type] and not n1.hints[type]:
                    self.feature_vector_n1_to_n2[k+1] = 1.0
                    self.feature_vector_n2_to_n1[k] = 1.0
                k+=2
            index = [0,-1]
            if tx1.strand == '-':
                index = [-1, 0]
            for i in index:
                j = abs(i)
                if n1.coords['CDS'][i][i+1] == n2.coords['CDS'][i][i+1]:
                    self.feature_vector_n1_to_n2[k] = n2.coords['CDS'][i][j] - n1.coords['CDS'][i][j]
                    self.feature_vector_n2_to_n1[k] = n1.coords['CDS'][i][j] - n2.coords['CDS'][i][j]
                k += 1

        set_tx1 = set([f'{i[0]}_{i[1]}' for i in n1.coords['intron']])
        set_tx2 = set([f'{i[0]}_{i[1]}' for i in n2.coords['intron']])
        union_size = len(set_tx1.union(set_tx2))

        # numb introns in union
        # numb introns in n_i
        # numb introns in n_j
        self.feature_vector_n1_to_n2[k] = self.feature_vector_n2_to_n1[k] = union_size
        self.feature_vector_n1_to_n2[k+1] = self.feature_vector_n2_to_n1[k+2] \
            = union_size / len(n1.coords['intron'])
        self.feature_vector_n1_to_n2[k+2] = self.feature_vector_n2_to_n1[k+1] \
            = union_size / len(n2.coords['intron'])
        k += 3


        ### for introns in union, in tx1 and not in tx2, in tx2 and not in tx1:
            ### for src in E, P, C, M, and any:
                # rel support of introns
                # abs support of introns (log10)
                # intron with min support (log10)
                # intron with max support (log10)
                # std of intron support (log10)
                # entropy of intron support (log10)
        intron_sets = [set_tx1.union(set_tx2), set_tx1.difference(set_tx2),
            set_tx2.difference(set_tx1)]
        for set, hints in zip(intron_sets, [n1.hints, n1.hints, n2.hints]):
            for evi_src in ['E', 'P', 'C', 'M', 'all']:
                current_hints = [hints['intron'][s][evi_src] for s in set]
                self.feature_vector_n1_to_n2[k] = len(current_hints) / len(set)
                self.feature_vector_n1_to_n2[k+1] = np.log10(np.sum(current_hints))
                self.feature_vector_n1_to_n2[k+2] = np.log10(np.min(current_hints))
                self.feature_vector_n1_to_n2[k+3] = np.log10(np.max(current_hints))
                self.feature_vector_n1_to_n2[k+4] = np.log10(np.std(current_hints))
                p = np.array(current_hints)/np.sum(current_hints)
                self.feature_vector_n1_to_n2[k+5] = np.sum(p * np.log10(p) / np.log10(len(set)))
                k+=6
        self.feature_vector_n2_to_n1[k-90 : k-60] = self.feature_vector_n1_to_n2[k-90 : k-60]
        self.feature_vector_n2_to_n1[k-60 : k-30] = self.feature_vector_n1_to_n2[k-30 : k]
        self.feature_vector_n2_to_n1[k-30 : k] = self.feature_vector_n1_to_n2[k-60 : k-30]

        i = 0
        j = 0
        overlap_size = 0
        while i<len(n1.coords['CDS']) and j<len(n2.coords['CDS']):
            overlap_size += max(0, min(n1.coords['CDS'][i][1], \
                n2.coords['CDS'][j][1]) - max(n1.coords['CDS'][i][0], \
                    n2.coords['CDS'][j][0]) + 1)
            if n1.coords['CDS'][i][1] < n2.coords['CDS'][j][1]:
                i += 1
            else:
                j += 1
        # fraction of CDS of n_i that is also in n_j
        # fraction of CDS of n_j that is also in n_i
        self.feature_vector_n1_to_n2[k] = overlap_size / sum([c[1]-c[0]+1 \
            for c in n1.coords['CDS']])
        self.feature_vector_n1_to_n2[k+1] = overlap_size / sum([c[1]-c[0]+1 \
            for c in n2.coords['CDS']])
        self.feature_vector_n2_to_n1[k+1] = self.feature_vector_n1_to_n2[k]
        self.feature_vector_n2_to_n1[k] = self.feature_vector_n1_to_n2[k+1]
        #self.feature_vector_n2_to_n1[k] = overlap_size / sum([c[1]-c[0]+1 \
            #for c in coords_tx2['CDS']])

        k += 2
        self.feature_vector_n1_to_n2[k:] = self.feature_vector_n1_to_n2[:k]
        self.feature_vector_n2_to_n1[k:] = self.feature_vector_n2_to_n1[:k]

class Node:
    """
        Class handling a node that represents a transcript in the overlap graph.
    """
    def __init__(self, a_id, t_id):
        """
            Args:
                a_id (str): Annotation ID of the transcript from Anno object
                t_id (str): Transcript ID from Transcrpt object
        """
        self.id = '{};{}'.format(a_id, t_id)
        self.transcript_id = t_id
        # ID of original annotation/gene prediction
        self.anno_id = a_id
        # unique ID for a cluster of overlapping transcripts
        self.component_id = None

        # dict of edge_ids of edges that are incident
        # self.edge_to[id of incident Node] = edge_id
        self.edge_to = {}
        self.is_in_ref_anno = 0
        self.ref_anno_cds_acc = 0.
        # feature vector
        # features in order:
        ### for type in "intron", "CDS", "3'-UTR", "5'-UTR":
            # numb of exons with type or if type == intron: number of introns
            # total len (in bp) of type
            # min len of type
            # max len of type
        #### For type in intron, start, stop:
            #### For src in E, P C and M:
                # rel intron hint support for src
            #### For src in E, P C and M:
                # abs intron hint support for src
        ### relative fraction of introns supported by any hint source
        # single exon tx?
        # for type in intron, CDS
            # max (numb. of type matching a protein chain / numb. of type in chain)
            # max (numb. of type matching a protein chain / numb. of type in tx)
        self.__numb_features__ = 162#39#46
        self.feature_vector = np.zeros(self.__numb_features__, float)
        self.dup_sources = {}
        self.evi_support = False
        # self.hints[type][startcoord_endcoord][src] = hint_multiplicity
        self.hints = {}
        # self.hints_sorted_by_src[type][src] = [log10(hint_multiplicity)]
        self.hints_sorted_by_src = {}
        # self.coords[type] = [[starcoord, endcoord]]
        # self.cdspart_hints[groupID][cds_index] = [[overlap_start, overlap_end]]
        self.cdspart_hints = {}
        self.coords = {}
        self.best_chain = ''
        # list of group IDs of matching protein chains
        self.matching_chains = []

    def add_evidence(self, tx, evi):
        for type in ["intron", "CDS", 'start_codon', 'stop_codon']:
            self.coords.update({type : tx.get_type_coords(type, frame=False)})

        for type in ['intron', 'start_codon', 'stop_codon']:
            if type not in self.hints: self.hints[type] = {}
            if type not in self.hints_sorted_by_src: self.hints_sorted_by_src[type] = {}
            for line in tx.transcript_lines[type]:
                hint = evi.get_hint(line[0], line[3], line[4], line[2], \
                    line[6])
                if hint:
                    all = 0
                    coord_key = f'{line[3]}_{line[4]}'
                    self.hints[type][coord_key] = hint
                    all = 0
                    for key in hint:
                        if key not in self.hints_sorted_by_src[type]:
                            self.hints_sorted_by_src[type][key] = []
                        self.hints_sorted_by_src[type][key].append(np.log10(hint[key]))
                        all +=hint[key]
                    if type == 'intron':
                        if not 'all' in self.hints_sorted_by_src[type]:
                            self.hints_sorted_by_src[type]['all'] = []
                        self.hints_sorted_by_src[type]['all'].append(np.log10(all))

        self.matching_chains = evi.get_matching_chains[tx.chr, self.coords, tx.strand]
        intron_keys = [f"{tx.chr}_{c[0]}_{c[1]}_intron_{tx.strand}"
            for c in self.coords['intron']]
        start_stop_keys = [f"{tx.chr}_{c[0]}_{c[1]}_start_{tx.strand}"
            for c in self.coords['start_codon']]
        start_stop_keys += [f"{tx.chr}_{c[0]}_{c[1]}_stop_{tx.strand}"
            for c in self.coords['stop_codon']]
        self.best_chain = ''
        best_chain_match = 0
        best_border_match = 0
        for group in matching_chains:
            chain_match = len(set(evi.group_chains[g][type]).intersection(intron_keys))
            border_match = len(set(evi.group_chains[g]['start'] +
                evi.group_chains[g]['stop']).intersection(start_stop_keys))
            if chain_match < best_chain_match:
                continue
            elif chain_match == best_chain_match and border_match < best_border_match:
                    continue
            best_chain_match = chain_match
            best_border_match = border_match
            self.best_chain = group

        for group in matching_chains:
            chain = sorted(self.group_chains[g]['CDSpart'])
            chain_match = 0
            j=0
            self.cdspart_hints[group] = []
            for c1 in self.coords['CDS']:
                self.cdspart_hints[group].append([])
                for i in range(j, len(chain)):
                    if chain[i][1] > c1[1]:
                        j = i
                        break
                    self.cdspart_hints[group].append([max(chain[i][0], c1[0]), min(chain[i][1], c1[1])])

    def add_features(self, tx, evi):
        """
            Compute for all nodes the feature vector based on the evidence support by evi.

            Args:
                evi (Evidence): Evidence class object with all hints from any source.
        """
        #coords = {}
        k = 0
        ### for type in "intron", "CDS":
            # numb of exons with type or if type == intron: number of introns
            # total len (in bp) of type
            # min len of type
            # max len of type
            # std of lengths of type
        for i, type in enumerate(["intron", "CDS"]):#, "3'-UTR", "5'-UTR"]):
            #coords.update({type : tx.get_type_coords(type, frame=False)})
            self.feature_vector[i*6] = 1.0 * len(self.coords[type])
            if self.feature_vector[i*6] > 0:
                len_type = [c[1]-c[0]+1 for c in self.coords[type]]
                self.feature_vector[i*6+1] = np.log10(np.sum(len_type))
                self.feature_vector[i*6+2] = np.log10(np.min(len_type))
                self.feature_vector[i*6+3] = np.log10(np.max(len_type))
                self.feature_vector[i*6+4] = np.log10(np.std(len_type))
            k+=5

        """
        # CDS predicted by BRAKER1?
        # CDS predicted by BRAKER2?
        # CDS predicted by long-read protocol?
        if 'anno1' in self.dup_sources:
            self.feature_vector[16] = 1.0
        if 'anno2' in self.dup_sources:
            self.feature_vector[17] = 1.0
        if 'anno3' in self.dup_sources:
            self.feature_vector[18] = 1.0"""

        # numb. input gene sets that include tx
        self.feature_vector[k] =  len(self.dup_sources)
        k+=1

        ### relative fraction of introns supported by any hint source
        self.feature_vector[k] = len(self.hints['intron'])/len(self.coords['intron'])
        k += 1

        #### For type in intron, start, stop:
            #### For src in E, P C and M:
                # rel type hint support for src
                # abs type hint support for src

        for type, abs_numb in zip(['intron', 'start_codon', 'stop_codon'], \
            [len(self.coords['intron']), 1, 1]) :
            for evi_src in ['E', 'P', 'C', 'M']:
                if abs_numb > 0:
                    self.feature_vector[k] = \
                        1.0*len(self.hints_sorted_by_src[type][evi_src])/abs_numb
                    self.feature_vector[k+1] = \
                        np.log10(np.sum(10**self.hints_sorted_by_src[type][evi_src]))
                k+=2

        #### For src in E, P, C, M, and any:
            # Entropy H for intron hint of source src
            # min absolute type hint support for src
            # max absolute type hint support for src
            # std absolute type hint support for src
        for evi_src in ['E', 'P', 'C', 'M', 'all']:
            if len(self.coords['intron']) > 0:
                p = np.array(self.hints_sorted_by_src['intron'][evi_src])/np.sum(self.hints_sorted_by_src['intron'][evi_src])
                self.feature_vector[k] = np.sum(p * np.log10(p) / np.log10(len(coords['intron'])))
                self.feature_vector[k+1] = np.min(self.hints_sorted_by_src['intron'][evi_src])
                self.feature_vector[k+2] = np.max(self.hints_sorted_by_src['intron'][evi_src])
                self.feature_vector[k+3] = np.std(self.hints_sorted_by_src['intron'][evi_src])
            k+=4

        # 1, if tx is intronless
        if len(self.coords['intron']) == 0:
            self.feature_vector[k] = 1
        k += 1

        # number of matching_chains
        self.feature_vector[k] = len(self.matching_chains)
        k += 1

        ### start/stop_codon match best_chain
        for s in ['start', 'stop']:
            if [f"{tx.chr}_{c[0]}_{c[1]}_start_{tx.strand}"
                for c in self.coords['start_codon']] ==
                evi.group_chains[self.best_chain]['start']:
                self.feature_vector[k] = 1.0
            k+=1

        # number of introns matching best_chain / number of introns in tx
        # number of introns matching best_chain / number of introns in chain
        intron_keys = [f"{tx.chr}_{c[0]}_{c[1]}_intron_{tx.strand}"
            for c in self.coords['intron']]
        chain_match = len(set(evi.group_chains[g][type]).intersection(intron_keys))
        self.feature_vector[k] = chain_match / len(self.coords['intron'])
        self.feature_vector[k+1] = chain_match / len(self.group_chains[g]['intron']))
        k+=2

        # log10 len of CDSparts of best chain matching CDS in tx / number of CDS in tx
        # log10 len of CDSparts of best chain matching CDS in tx / number of CDS in chain
        # number of CDS in tx that dont have any supported by CDSparts from best chain
        total_length_chain = np.log10(sum([c[1]-c[0]+1 for c in self.group_chains[self.best_chain_match][type]]))
        chain_match = np.log10(sum([c[1]-c[0] + 1 for c in self.cdspart_hints[self.best_chain_match]]))
        #total_len_cds = np.log10(sum([c[1]-c[0] + 1 for c in self.coords['CDS']]))
        #self.feature_vector[k] = chain_match - total_len_cds)
        self.feature_vector[k] = chain_match - total_length_chain)
        self.feature_vector[k+1] = np.sum([1 for c in
            self.group_chains[self.best_chain_match][type] if not c])
        k+=2

        # list of best_chain support for each CDS
        # list of relativ support
        # list of absolute support
        cds_support = [self.cdspart_hints[self.best_chain_match], [], []]

        for i in range(len(self.coords['CDS'])):
            cds_support[1].append([])
            cds_support[2].append([])
            for group in self.cdspart_hints.values()
                for c in group[i]:
                    cds_support[2][i].append(c)
                    if not cds_support[1][i] or cds_support[1][i][-1][1] >= c[0]:
                        cds_support[1][i][-1][1] = max(cds_support[1][i][-1][1], c[1])
                    else:
                        cds_support[1][i].append(c)

        # number of CDS with no CDSpart support
        self.feature_vector[k] = np.sum([1 for c in
            cds_support[1] if not c])
        k+=1

        ### for best chain, relative support by any chain, absolute support by any chain
            # sum support / len(cds)
            # support of maximal supported cds
            # support of minimal supported cds
            # std of support of supported cds
            # entropy of support of supported cds
        for cds in cds_support:
            cds_sum = np.array([np.sum([c2[1] - c2[0] + 1 for c2 in c1]) for c1 in cds])
            cds_len = np.array([c[1] - c[0] + 1 for c in self.coords['CDS']])
            self.feature_vector[k] = np.log10(np.sum(cds_sum)) - np.log10(np.sum(cds_len))
            cds_norm = np.log10(cds_sum) - np.log10(cds_len)
            self.feature_vector[k+1] = np.max(cds_norm)
            self.feature_vector[k+2] = np.min(cds_norm)
            self.feature_vector[k+3] = np.std(cds_norm)
            p = cds_norm / np.sum(cds_norm)
            self.feature_vector[k+4] = np.sum(p * np.log10(p) / np.log10(len(coords['CDS'])))
            k+=5

        # number of overlapping transcripts
        self.feature_vector[k] = len(self.edge_to)

        self.feature_vector[k:] = self.feature_vector[:k]

class Graph_component:
    """
        Connected component of Graph object.
    """
    def __init__(self):
        # list of node IDs in graph component
        self.nodes = []
        # False if new node has been added and edges, incidence_matrix,...
        # haven't been updated
        self.up_to_date = True
        self.incidence_matrix_sender = None
        self.incidence_matrix_receiver = None
        # list Edge() objects of all edges in component
        self.edges = []
        # edges (i,j) in component, where i,j are the indices from self.nodes of adjacent nodes
        self.edge_path = []

    def add_node(self, node):
        self.nodes.append(node.id)
        for e_id in node.edge_to.values():
            if e_id not in self.edges:
                self.edges.append(e_id)
        self.up_to_date = False

    def add_nodes(self, node_list):
        for n in node_list:
            self.add_node(n)

    def update_all(self, edge_dict):
        if not self.up_to_date and self.edges:
            self.__update_edges__(edge_dict)
            self.__update_incidence_matrix__()
        self.up_to_date = True

    def __update_edges__(self, edge_dict):
        if not self.up_to_date:
            self.edge_path = []
            for e_id in self.edges:
                self.edge_path.append([self.nodes.index(edge_dict[e_id].node1),
                    self.nodes.index(edge_dict[e_id].node2)])
                self.edge_path.append([self.nodes.index(edge_dict[e_id].node2),
                    self.nodes.index(edge_dict[e_id].node1)])

    def __update_incidence_matrix__(self):
        n = len(self.nodes)
        m = len(self.edge_path)
        self.incidence_matrix_sender = np.zeros((n,m), bool)
        self.incidence_matrix_receiver = np.zeros((n,m), bool)
        self.incidence_matrix_sender[np.array(self.edge_path)[:,0], \
            range(m)] = True
        self.incidence_matrix_receiver[np.array(self.edge_path)[:,1], \
            range(m)] = True

    def get_incidence_matrix_sender(self):
        if not self.up_to_date:
            self.__update_incidence_matrix__()
        return self.incidence_matrix_sender

    def get_incidence_matrix_receiver(self):
        if not self.up_to_date:
            self.__update_incidence_matrix__()
        return self.incidence_matrix_receiver

    def get_node_features(self, node_dict):
        return np.array([node_dict[n].feature_vector for n in self.nodes])

    def get_edge_features(self, edge_dict):
        edge_features = np.array([[edge_dict[e].feature_vector_n1_to_n2, \
            edge_dict[e].feature_vector_n2_to_n1] for e in self.edges])
        return edge_features.reshape(-1, edge_features.shape[-1])

    def get_target_label(self, node_dict):
        target = np.zeros((len(self.nodes),1))
        #target[[i for i,n in enumerate(self.nodes) if node_dict[n].is_in_ref_anno], 0] = 1.0
        target[:,0] = np.array([node_dict[n].ref_anno_cds_acc for n in self.nodes])
        return target

class Graph:
    """
        Overlap (undirected) graph that can detect and filter overlapping transcripts.
    """
    def __init__(self, genome_anno_lst, verbose=0):
        """
            Args:
                genome_anno_lst (list(Anno)): List of Anno class objects
                                              containing genome annotations.
                para (dict(float)): Dictionary for parameter used for filtering of transcripts.
                verbose (int): Verbose mode if verbose >0 .
        """
        # self.nodes['anno;txid'] = Node(anno, txid)
        self.nodes = {}
        # self.edges['ei'] = Edge()
        self.edges = {}

        # self.anno[annoid] = Anno()
        self.anno = {}

        # list of connected graph components
        self.component_list = {}

        # subset of all transcripts that weren't removed by the transcript comparison rule
        self.decided_graph = []

        # dict of duplicate genome annotation ids to new ids
        self.duplicates = {}

        # variables for verbose mode
        self.v = verbose
        self.ties = 0
        self.rng = np.random.RandomState(5253242)


        # list of Graph_component(), each component concists of a number of
        # connected component, all nodes and edges of one connected component is
        # always in the same batch and one connected component is only in one batch
        self.batches = []
        self.batches_no_edge = []
        self.__features_to_norm__ = np.concatenate((np.arange(16) , \
            np.arange(23,27), np.arange(31,35), np.arange(39, 43)))

        # numb nodes
        # numb edges
        # numb components
        self.global_bias_features = np.zeros(30, float)


        # init annotations, check for duplicate ids
        self.init_anno(genome_anno_lst)

    def init_anno(self, genome_anno_lst):
        # make sure that the genome_anno ids are unique
        counter = 0
        for ga in genome_anno_lst:
            if ga.id in self.anno.keys():
                counter += 1
                new_id = "duplicate.anno.{}".format(counter)
                self.duplicates.update({new_id : ga.id})
                ga.change_id(new_id)
            self.anno.update({ga.id : ga})

    def __tx_from_key__(self, key):
        """
            Gets a transcript of a node.

            Args:
                key (str): ID of a node as 'anno_id;tx_id'

            Returns:
                (Transcript): Transcript class object with id = tx_id
                              from Anno() with id = anno_id
        """
        anno_id, tx_id = key.split(';')
        return self.anno[anno_id].transcripts[tx_id]

    def build(self):
        """
            Builds the overlap graph for >=1 Anno() objects.
            Each node of the graph represents a unique transcript from any annotation.
            Two nodes have an edge if their transcripts overlap.
            Two transcripts overlap if they share at least 3 adjacent protein coding nucleotides.
        """
        # tx_start_end[chr] = [tx_id, coord, id for start or end]
        # for every tx one element for start and one for end
        # this dict is used to check for overlapping transcripts
        tx_start_end = {}
        # check for duplicate txs, list of ['start_end_strand']
        unique_tx_keys = {}
        numb_dup = {}
        for k in self.anno.keys():
            for tx in self.anno[k].get_transcript_list():
                if tx.chr not in tx_start_end.keys():
                    tx_start_end.update({tx.chr : []})
                    unique_tx_keys.update({tx.chr : {}})
                unique_key = f"{tx.start}_{tx.end}_{tx.strand}"
                dup = {tx.source_anno}
                if unique_key in unique_tx_keys[tx.chr].keys():
                    check = False
                    coords = tx.get_type_coords('CDS')
                    for t in unique_tx_keys[tx.chr][unique_key]:
                        if coords == t.get_type_coords('CDS'):
                            if tx.utr and not t.utr:
                                unique_tx_keys[tx.chr][unique_key].remove(t)
                                dup.add(t.source_anno)
                            elif tx.utr and t.utr and tx.utr_len  > t.utr_len:
                                unique_tx_keys[tx.chr][unique_key].remove(t)
                                dup.add(t.source_anno)
                            else:
                                numb_dup[f"{t.source_anno};{t.id}"].add(tx.source_anno)
                                check = True
                                break
                    if check:
                            continue
                else:
                    unique_tx_keys[tx.chr].update({unique_key : []})
                numb_dup.update({f"{tx.source_anno};{tx.id}" : dup})
                unique_tx_keys[tx.chr][unique_key].append(tx)

        for chr in unique_tx_keys.keys():
            for tx_list in unique_tx_keys[chr].values():
                for tx in tx_list:
                    key = f"{tx.source_anno};{tx.id}"
                    self.nodes.update({key : Node(tx.source_anno, \
                        tx.id)})
                    self.nodes[key].dup_sources = numb_dup[key]
                    tx_start_end[tx.chr].append([key, tx.start, 0])
                    tx_start_end[tx.chr].append([key, tx.end, 1])

        # detect overlapping nodes
        edge_count = 0
        for chr in tx_start_end.keys():
            tx_start_end[chr] = sorted(tx_start_end[chr], key=lambda t:(t[1], t[2]))
            open_intervals = []
            for interval in tx_start_end[chr]:
                if interval[2] == 0:
                    open_intervals.append(interval[0])
                else:
                    open_intervals.remove(interval[0])
                    for match in open_intervals:
                        tx1 = self.__tx_from_key__(interval[0])
                        tx2 = self.__tx_from_key__(match)
                        if self.__compare_tx_cds__(tx1, tx2):
                            new_edge_key = f"e{edge_count}"
                            edge_count += 1
                            self.edges.update({new_edge_key : Edge(interval[0], match)})
                            self.nodes[interval[0]].edge_to.update({match : new_edge_key})
                            self.nodes[match].edge_to.update({interval[0] : new_edge_key})

    def __compare_tx_cds__(self, tx1, tx2):
        """
            Check if two transcripts share at least 3 adjacent protein
            coding nucleotides on the same strand and reading frame.

            Args:
                tx1 (Transcript): Transcript class object of first transcript
                tx2 (Transcript): Transcript class object of second transcript

            Returns:
                (boolean): TRUE if they overlap and FALSE otherwise
        """
        if not tx1.strand == tx2.strand:
            return False
        tx1_coords = tx1.get_type_coords('CDS')
        tx2_coords = tx2.get_type_coords('CDS')
        for phase in ['0', '1', '2', '.']:
            coords = []
            coords += tx1_coords[phase]
            coords += tx2_coords[phase]
            coords = sorted(coords, key=lambda c:c[0])
            for i in range(1, len(coords)):
                if coords[i-1][1] - coords[i][0] > 1:
                    return True
        return False

    def print_nodes(self):
        # prints all nodes of the graph (only used for development)
        for k in self.nodes.keys():
            print(self.nodes[k].id)
            print(self.nodes[k].transcript_id)
            print(self.nodes[k].anno_id)
            print(self.nodes[k].edge_to.keys())
            print('\n')

    def __find_component__(self, node_id, comp_id):
        self.nodes[node_id].component_id = comp_id
        for next_node_id in self.nodes[node_id].edge_to:
            if not self.nodes[next_node_id].component_id:
                self.__find_component__(next_node_id, comp_id)

    def connected_components(self):
        """
            Compute all clusters of connected transcripts.
            A cluster is connected component of the graph.
            Adds component IDs to nodes.

            Returns:
                (list(list(str))): Lists of list of all node IDs of a component.
        """
        if self.component_list:
            return self.component_list

        self.component_list = {}
        component_index = 1
        for key in self.nodes:
            if not self.nodes[key].component_id:
                c_id = f'g_{component_index}'
                self.component_list.update({c_id : []})
                self.__find_component__(key, f'g_{component_index}')
                component_index += 1
            self.component_list[self.nodes[key].component_id].append(key)
        return self.component_list


    def create_batch(self, numb_batches, batch_size, repl=False):

        if not self.component_list:
            self.connected_components()
        """if not repl and numb_batches*batch_size>len(self.nodes):
            raise BatchSizeError('ERROR: numb_batches*batch_size has to be smaller '\
                + f'than number_connected_components. numb_batches={numb_batches} '\
                + f'batch_size={batch_size} number_connected_components={len(self.component_list)}.')"""
        components = [i for i in list(self.component_list.values()) if len(i)>0]
        self.batches = [Graph_component()]

        end = len(components)
        if repl:
            end = numb_batches*batch_size
        #print(len(self.nodes), len(components), end, numb_batches)
        for k, i in enumerate(np.random.choice(len(components), \
            end, replace=repl)):
            self.batches[-1].add_nodes([self.nodes[n] for n in components[i]])
            if len(self.batches[-1].nodes) >=  batch_size:
                if len(self.batches[-1].edges) == 0:
                    print('#### No edges')
                    self.batches.pop()
                else:
                    self.batches[-1].update_all(self.edges)
                if len(self.batches) <= numb_batches:
                    self.batches.append(Graph_component())
                else:
                    break
        self.batches[-1].update_all(self.edges)


    def create_batch_no_edges(self, numb_batches, batch_size, repl=False):

        if not self.component_list:
            self.connected_components()
        """if not repl and numb_batches*batch_size>len(self.nodes):
            raise BatchSizeError('ERROR: numb_batches*batch_size has to be smaller '\
                + f'than number_connected_components. numb_batches={numb_batches} '\
                + f'batch_size={batch_size} number_connected_components={len(self.component_list)}.')"""
        components = [i for i in list(self.component_list.values()) if len(i)==1]
        self.batches_no_edge = [Graph_component()]

        end = len(components)
        if repl:
            end = numb_batches*batch_size
        #print('NOEDGE', len(self.nodes), len(components), end, numb_batches, len(self.rng.choice(len(components), \
            #end, replace=repl)))
        for k, i in enumerate(self.rng.choice(len(components), \
            end, replace=repl)):
            self.batches_no_edge[-1].add_nodes([self.nodes[n] for n in components[i]])
            if len(self.batches_no_edge[-1].nodes) >=  batch_size:
                self.batches_no_edge[-1].update_all(self.edges)
                if len(self.batches_no_edge) <= numb_batches:
                    self.batches_no_edge.append(Graph_component())
                else:
                    print(k, i)
                    break
        self.batches_no_edge[-1].update_all(self.edges)


    def get_batches_as_input_target(self, val_size=0.1):
        # val size as fraction of numb_batches
        input_target_train = []
        input_target_val = []
        input_target_train_no_edge = []
        input_target_val_no_edge = []
        numb_batches = len(self.batches)
        val_indices = np.random.choice(numb_batches, int(val_size*numb_batches), \
            replace=False)
        for i, batch in enumerate(self.batches):
            input_bias = np.ones((len(batch.nodes), self.global_bias_features.shape[0]), float)
            input_bias *= self.global_bias_features
            new_batch = [
                {
                    "input_nodes" : np.expand_dims(batch.get_node_features(self.nodes) ,0),
                    "input_edges" : np.expand_dims(batch.get_edge_features(self.edges) ,0),
                    #"input_bias" : np.expand_dims(input_bias ,0),
                    "incidence_matrix_sender" : tf.expand_dims(batch.get_incidence_matrix_sender() ,0),
                    "incidence_matrix_receiver" : tf.expand_dims(batch.get_incidence_matrix_receiver() ,0)
                },
                {
                    "target_label" : np.expand_dims(batch.get_target_label(self.nodes),0)
                }
            ]
            if i in val_indices:
                input_target_val.append(new_batch)
            else:
                input_target_train.append(new_batch)

        """numb_batches = len(self.batches_no_edge)
        val_indices = self.rng.choice(numb_batches, int(val_size*numb_batches), \
            replace=False)
        for i, batch in enumerate(self.batches_no_edge):
            input_bias = np.ones((len(batch.nodes), self.global_bias_features.shape[0]), float)
            input_bias *= self.global_bias_features
            new_batch = [
                {
                    "input_nodes" : np.expand_dims(batch.get_node_features(self.nodes) ,0),
                },
                {
                    "target_label" : np.expand_dims(batch.get_target_label(self.nodes),0)
                }
            ]
            if i in val_indices:
                input_target_val_no_edge.append(new_batch)
            else:
                input_target_train_no_edge.append(new_batch)"""
        return input_target_train, input_target_val, input_target_train_no_edge, input_target_val_no_edge

    def add_reference_anno_label(self, ref_anno):
        """
            Sets the value of is_in_ref_anno for each node to 1
            if the coding sequence of the corresponding transcript matches the
            coding sequence of a transcript in the reference anno

            Args:
                ref_anno (Anno): Anno() obeject of reference annotation
        """
        def get_cds_keys(tx):
            keys = [tx.chr, tx.strand] + [str(c[0]) + '_' + str(c[1]) \
                for c in tx.get_type_coords('CDS', frame=False)]
            return keys
        ref_anno_keys = []
        ref_anno_cds = []
        for tx in ref_anno.transcripts.values():
            cds_keys = get_cds_keys(tx)
            ref_anno_cds += cds_keys
            ref_anno_keys.append('_'.join(cds_keys))
        ref_anno_cds = set(ref_anno_cds)
        ref_anno_keys = set(ref_anno_keys)

        for n in self.nodes:
            c_keys = get_cds_keys(self.__tx_from_key__(n))
            if '_'.join(c_keys) in ref_anno_keys:
                self.nodes[n].is_in_ref_anno = 1
            self.nodes[n].ref_anno_cds_acc = len(set(c_keys).intersection(ref_anno_cds)) / len(c_keys)
            if self.nodes[n].ref_anno_cds_acc == 1.0 and self.nodes[n].is_in_ref_anno == 0:
                self.nodes[n].ref_anno_cds_acc = 0.99999999

    def add_node_features(self, evi):
        """
            Compute for all nodes the feature vector based on the evidence support by evi.

            Args:
                evi (Evidence): Evidence class object with all hints from any source.
        """
        #mean = sigma = np.zeros(46, float)

        #ma = np.zeros(self.__features_to_norm__.shape[-1], float)
        #ma = np.zeros(46, float)
        epsi = 1e-17
        n_f = []
        for node_key in self.nodes.keys():
            #def add_node_f:
            tx = self.__tx_from_key__(node_key)
            self.nodes[node_key].add_features(tx, evi)
            #mean += self.nodes[node_key].feature_vector
            n_f.append(self.nodes[node_key].feature_vector)
            #ma = np.max(np.array([ma, \
                #self.nodes[node_key].feature_vector[self.__features_to_norm__]]), 0)
            #ma = np.max(np.array([ma, \
                #self.nodes[node_key].feature_vector]), 0)
        #mean /= len(self.nodes)
        #ma/=2
        #ma += epsi
        n_f = np.array(n_f)
        m = np.max(n_f, axis=0)
        m = np.maximum(m, np.ones(m.shape[0]) * epsi)
        n_f /= m
        std = np.std(n_f, axis=0)
        std = np.maximum(std, np.ones(std.shape[0]) * epsi)
        mean = np.mean(n_f, axis=0)
        self.global_bias_features = mean
        print('NODES\nMEAN:  ', mean, '\nSTD:  ', std,
                  '\nMIN:  ', np.min(n_f, axis=0), '\nMAX:  ',
                  np.max(n_f, axis=0))

        for node_key in self.nodes.keys():
            #print(self.nodes[node_key].feature_vector)
            self.nodes[node_key].feature_vector /= m
            self.nodes[node_key].feature_vector -= mean
            self.nodes[node_key].feature_vector /= std
            #print(self.nodes[node_key].feature_vector, '\n')
            #self.nodes[node_key].feature_vector[self.__features_to_norm__] /= np.linalg.norm(self.nodes[node_key].feature_vector[self.__features_to_norm__])

    def add_edge_features(self, evi):
        """
            Compute for all edges the feature vector based on the evidence support by evi.
        """
        epsi = 1e-17
        e_f = []
        for edge in self.edges.values():
            #def add_node_f:
            tx1 = self.__tx_from_key__(edge.node1)
            tx2 = self.__tx_from_key__(edge.node2)
            edge.add_features(tx1, tx2, evi)
            e_f.append(edge.feature_vector_n1_to_n2)
            e_f.append(edge.feature_vector_n2_to_n1)

        e_f = np.array(e_f)
        m = np.max(e_f, axis=0)
        m = np.maximum(m, np.ones(m.shape[0]) * epsi)
        e_f /= m
        std = np.std(e_f, axis=0)
        std = np.maximum(std, np.ones(std.shape[0]) * epsi)
        mean = np.mean(e_f, axis=0)

        print('EDGES\nMEAN:  ', mean, '\nSTD:  ', std,
                  '\nMIN:  ', np.min(e_f, axis=0), '\nMAX:  ',
                  np.max(e_f, axis=0))

        for edge_key in self.edges.keys():
            #print(self.nodes[node_key].feature_vector)
            self.edges[edge_key].feature_vector_n1_to_n2 /= m
            self.edges[edge_key].feature_vector_n2_to_n1 /= m
            self.edges[edge_key].feature_vector_n1_to_n2 -= mean
            self.edges[edge_key].feature_vector_n1_to_n2 /= std
            self.edges[edge_key].feature_vector_n2_to_n1 -= mean
            self.edges[edge_key].feature_vector_n2_to_n1 /= std

    def __decide_edge__(self, edge):
        """
            Apply transcript comparison rule to two overlapping transcripts

            Args:
                edge (Edge): edge between two transcripts

            Returns:
                (str): node ID of the transcript that is marked for removal
        """
        n1 = self.nodes[edge.node1]
        n2 = self.nodes[edge.node2]
        for i in range(0,4):
            diff = n1.feature_vector[i] - n2.feature_vector[i]
            if diff > 0:
                return n2.id
            elif diff < 0:
                return n1.id
        return None

    def decide_component(self, component):
        """
            Applies transcript comparison rule to all transcripts of one component
            and returns the node IDs of all transcripts that are not removed by
            a comparison.

            Args:
                component (list(str)): List of node IDs

            Returns:
                (list(str)): Filtered subset of component list.
        """
        # return all ids of vertices of a graph component, that weren't excluded by the decision rule
        result = component.copy()
        for node_id in component:
            for e_id in self.nodes[node_id].edge_to.values():
                node_to_remove = self.edges[e_id].node_to_remove
                if node_to_remove:
                    if node_to_remove in result:
                        result.remove(node_to_remove)
        return result

    def decide_graph(self):
        """
            Create list of connected components of the graph and apply the
            transcript comparison rule to all components.
        """
        for key in self.edges.keys():
            self.edges[key].node_to_remove = self.__decide_edge__(self.edges[key])
        self.decided_graph = []
        if not self.component_list:
            self.connected_components()
        for component in self.component_list.values():
            if len(component) > 1:
                self.decided_graph += self.decide_component(component)
            else:
                self.decided_graph += component

    def get_decided_graph(self):
        """
            Filter graph with the transcript comparison rule.
            Then, remove all transcripts with low evidence support and
            compute the subset of transcripts that are included in the
            combined gene prediciton.

            Returns:
                (dict(list(list(str))): Dictionary with transcript IDs and new
                gene IDs of all transcripts included in the combined gene prediciton
                for all input annotations

        """
        if not self.decided_graph:
            self.decide_graph()
        # result[anno_id] = [[tx_ids, new_gene_id]]
        result = {}
        for key in self.anno.keys():
            result.update({key : []})
        for node in self.decided_graph:
            if self.nodes[node].evi_support:
                anno_id, tx_id = node.split(';')
                result[anno_id].append([tx_id, self.nodes[node].component_id])
        return result

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
        self.__numb_features__ = 23

        ### feature vectors for directed edge from node n_i to node n_j
        ### features:
        # matching start_codon?
        # is n_i starting before n_j?
        # matching stop codon?
        # is n_i ending before n_j?
         #### For src in E, P C and M, none:
            # number of introns in n_i / numb introns in union of n_i, n_j
            # number of introns in n_j / numb introns in union of n_i, n_j
            # number of introns in n_i and in n_j / numb introns in union of n_i, n_j
        # fraction of CDS of n_i that is also in n_j
        # fraction of CDS of n_j that is also in n_i
        # start codon position difference of nj to ni if they agree on first coding DSS
        # stop codon position difference of nj to ni if they agree on first coding ASS

        self.feature_vector_n1_to_n2 = np.zeros(self.__numb_features__, float)
        self.feature_vector_n2_to_n1 = np.zeros(self.__numb_features__, float)

    def add_features(self, tx1, tx2, evi):
        coords_tx1 = {}
        coords_tx2 = {}
        for i, type in enumerate(["intron", "CDS", "start_codon", "stop_codon"]):
            coords_tx1.update({type : tx1.get_type_coords(type, frame=False)})
            coords_tx2.update({type : tx2.get_type_coords(type, frame=False)})

        for i, type in enumerate(['start_codon', 'stop_codon']):
            if coords_tx1[type] == coords_tx1[type]:
                self.feature_vector_n1_to_n2[2*i] = 1.0
                self.feature_vector_n2_to_n1[2*i] = 1.0
            else:
                if (tx1.strand == '+' and \
                    coords_tx1[type][0][0] < coords_tx1[type][0][0]) \
                    or (tx1.strand == '-' and \
                    coords_tx1[type][0][0] > coords_tx1[type][0][0]):
                    self.feature_vector_n1_to_n2[2*i+1] = 1.0
                else :
                    self.feature_vector_n2_to_n1[2*i+1] = 1.0

        k = 4
        hints_tx1 = {'E' : [], 'P': [], 'C': [], 'M': [],  'none' : []}
        hints_tx2 = {'E' : [], 'P': [], 'C': [], 'M': [],  'none' : []}
        for t, h in zip([tx1, tx2], [hints_tx1, hints_tx2]):
            for line in t.transcript_lines[type]:
                hint = evi.get_hint(line[0], line[3], line[4], line[2], \
                    line[6])
                if hint:
                    for key in hint:
                        h[key].append([line[3], line[4]])
                else:
                    h['none'].append([line[3], line[4]])
        set_tx1 = set([f'{i[0]}_{i[1]}' for i in coords_tx2['intron']])
        set_tx2 = set([f'{i[0]}_{i[1]}' for i in coords_tx2['intron']])
        union_size = len(set_tx1.union(set_tx2))
        for c1, c2 in zip(hints_tx1.values(), hints_tx2.values()):
            set_tx1 = set([f'{i[0]}_{i[1]}' for i in c1])
            set_tx2 = set([f'{i[0]}_{i[1]}' for i in c2])
            if union_size > 0:
                intron_set_sizes = [len(set_tx1)/union_size, \
                    len(set_tx2)/union_size,
                    len(set_tx1.intersection(set_tx2))/union_size]
                self.feature_vector_n1_to_n2[k:k+3] = intron_set_sizes
                self.feature_vector_n2_to_n1[[k+1,k,k+2]] = intron_set_sizes
            k += 3

        k = 19
        i = 0
        j = 0
        overlap_size = 0
        while i<len(coords_tx1['CDS']) and j<len(coords_tx2['CDS']):
            overlap_size += max(0, min(coords_tx1['CDS'][i][1], \
                coords_tx2['CDS'][j][1]) - max(coords_tx1['CDS'][i][0], \
                    coords_tx2['CDS'][j][0]) + 1)
            if coords_tx1['CDS'][i][1] < coords_tx2['CDS'][j][1]:
                i += 1
            else:
                j += 1
        self.feature_vector_n1_to_n2[k] = overlap_size / sum([c[1]-c[0]+1 \
            for c in coords_tx1['CDS']])
        self.feature_vector_n1_to_n2[k+1] = overlap_size / sum([c[1]-c[0]+1 \
            for c in coords_tx2['CDS']])
        self.feature_vector_n2_to_n1[k+1] = self.feature_vector_n1_to_n2[k]
        self.feature_vector_n2_to_n1[k] = self.feature_vector_n1_to_n2[k+1]

        k += 2
        index = [0,-1]
        if tx1.strand == '-':
            index = [-1, 0]
        for i in index:
            j = abs(i)
            if coords_tx1['CDS'][i][j] == coords_tx2['CDS'][i][j]:
                self.feature_vector_n1_to_n2[k] = coords_tx2['CDS'][i][j] - coords_tx1['CDS'][i][j]
                self.feature_vector_n2_to_n1[k] = coords_tx1['CDS'][i][j] - coords_tx2['CDS'][i][j]
                k += 1


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
        # feature vector
        # features in order:
        ### for type in "intron", "CDS", "3'-UTR", "5'-UTR":
            # numb of exons with type or if type == intron: number of introns
            # total len (in bp) of type
            # min len of type
            # max len of type

        # CDS predicted by BRAKER1?
        # CDS predicted by BRAKER2?
        # CDS predicted by long-read protocol?

        #### For type in intron, start, stop:
            #### For src in E, P C and M:
                # rel intron hint support for src
            #### For src in E, P C and M:
                # abs intron hint support for src
        ### relative fraction of introns supported by any hint source
        # single exon tx?
        # number of neighbours
        self.__numb_features__ = 46
        self.feature_vector = np.zeros(self.__numb_features__, float)
        self.dup_sources = {}
        self.evi_support = False

    def add_features(self, tx, evi):
        """
            Compute for all nodes the feature vector based on the evidence support by evi.

            Args:
                evi (Evidence): Evidence class object with all hints from any source.
        """
        coords = {}

        for i, type in enumerate(["intron", "CDS", "3'-UTR", "5'-UTR"]):
            coords.update({type : tx.get_type_coords(type, frame=False)})
            self.feature_vector[i*4] = 1.0 * len(coords[type])
            if self.feature_vector[i*4] > 0:
                len_type = [c[1]-c[0]+1 for c in coords[type]]
                self.feature_vector[i*4+1] = 1.0 * sum(len_type)
                self.feature_vector[i*4+2] = 1.0 * min(len_type)
                self.feature_vector[i*4+3] = 1.0 * max(len_type)

        if 'anno1' in self.dup_sources:
            self.feature_vector[16] = 1.0
        if 'anno2' in self.dup_sources:
            self.feature_vector[17] = 1.0
        if 'anno3' in self.dup_sources:
            self.feature_vector[18] = 1.0
        k = 19
        evi_list = {'intron' : {'E' : [], 'P': [], 'C': [], 'M': []}, \
            'start_codon' : {'E' : [], 'P': [], 'C': [], 'M': []}, \
            'stop_codon': {'E' : [], 'P': [], 'C': [], 'M': []}}
        for type in ['intron', 'start_codon', 'stop_codon']:
            for line in tx.transcript_lines[type]:
                hint = evi.get_hint(line[0], line[3], line[4], line[2], \
                    line[6])
                if hint:
                    if type == 'intron':
                        self.feature_vector[k+24] += 1/len(coords['intron'])
                    for key in hint.keys():
                        if key not in evi_list[type].keys():
                            evi_list[type].update({key : []})
                        evi_list[type][key].append(hint[key])
        for type, i, abs_numb in zip(['intron', 'start_codon', 'stop_codon'], \
            range(3), [len(coords['intron']), 1, 1]) :
            for evi_src, j in zip(['E', 'P', 'C', 'M'], range(4)):
                if abs_numb == 0:
                    self.feature_vector[k + i * 8 + j] = 0.0
                else:
                    self.feature_vector[k + i * 8 + j] = \
                        1.0*len(evi_list[type][evi_src])/abs_numb
                self.feature_vector[k+4 + i * 8 + j] = \
                    sum(evi_list[type][evi_src]) * 1.0
        k += 25
        if len(coords['intron']) == 0:
            self.feature_vector[k] = 1
        k = 45
        self.feature_vector[k] = len(self.edge_to)

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
        target = np.zeros((len(self.nodes),2))
        target[np.arange(len(self.nodes)),[node_dict[n].is_in_ref_anno for n in self.nodes]] = 1
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
        np.random.seed(5)
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

        # list of Graph_component(), each component concists of a number of
        # connected component, all nodes and edges of one connected component is
        # always in the same batch and one connected component is only in one batch
        self.batches = []

        self.__features_to_norm__ = np.concatenate((np.arange(16) , \
            np.arange(23,27), np.arange(31,35), np.arange(39, 43)))

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
        if not repl and numb_batches*batch_size>len(self.component_list):
            raise BatchSizeError('ERROR: numb_batches*batch_size has to be smaller '\
                + f'than number_connected_components. numb_batches={numb_batches} '\
                + f'batch_size={batch_size} number_connected_components={len(self.component_list)}.')
        components = list(self.component_list.values())
        self.batches = [Graph_component()]
        print(len(components))
        for k, i in enumerate(np.random.choice(len(self.component_list), \
            batch_size*numb_batches, replace=repl)):
            self.batches[-1].add_nodes([self.nodes[n] for n in components[i]])
            if ((k+1) % batch_size) == 0:
                if len(self.batches[-1].edges) == 0:
                    self.batches.pop()
                else:
                    self.batches[-1].update_all(self.edges)
                if k+1 < batch_size*numb_batches:
                    self.batches.append(Graph_component())


    def get_batches_as_input_target(self, val_size=0.1):
        # val size as fraction of numb_batches
        input_target_train = []
        input_target_val = []
        numb_batches = len(self.batches)
        val_indices = np.random.choice(numb_batches, int(val_size*numb_batches), \
            replace=False)
        for i, batch in enumerate(self.batches):
            new_batch = [
                {
                    "input_nodes" : np.expand_dims(batch.get_node_features(self.nodes) ,0),
                    "input_edges" : np.expand_dims(batch.get_edge_features(self.edges) ,0),
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
        return input_target_train, input_target_val

    def add_reference_anno_label(self, ref_anno):
        """
            Sets the value of is_in_ref_anno for each node to 1
            if the coding sequence of the corresponding transcript matches the
            coding sequence of a transcript in the reference anno

            Args:
                ref_anno (Anno): Anno() obeject of reference annotation
        """
        def create_cds_key(tx):
            return '_'.join([tx.chr, tx.strand] + [str(c[0]) + '_' + str(c[1]) \
                for c in tx.get_type_coords('CDS', frame=False)])
        ref_anno_keys = []
        for tx in ref_anno.transcripts.values():
            ref_anno_keys.append(create_cds_key(tx))

        for n in self.nodes:
            if create_cds_key(self.__tx_from_key__(n)) in ref_anno_keys:
                self.nodes[n].is_in_ref_anno = 1

    def add_node_features(self, evi):
        """
            Compute for all nodes the feature vector based on the evidence support by evi.

            Args:
                evi (Evidence): Evidence class object with all hints from any source.
        """
        max = np.zeros(self.__features_to_norm__.shape[-1], float)
        epsi = 0.000000000001
        for node_key in self.nodes.keys():
            #def add_node_f:
            tx = self.__tx_from_key__(node_key)
            self.nodes[node_key].add_features(tx, evi)
            max = np.max(np.array([max, \
                self.nodes[node_key].feature_vector[self.__features_to_norm__]]), 0)
        for node_key in self.nodes.keys():
            self.nodes[node_key].feature_vector[self.__features_to_norm__] /= np.linalg.norm(self.nodes[node_key].feature_vector[self.__features_to_norm__])

    def add_edge_features(self, evi):
        """
            Compute for all edges the feature vector based on the evidence support by evi.
        """
        for edge in self.edges.values():
            #def add_node_f:
            tx1 = self.__tx_from_key__(edge.node1)
            tx2 = self.__tx_from_key__(edge.node2)
            edge.add_features(tx1, tx2, evi)

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
            #print(diff)
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

#!/usr/bin/env python3
# ==============================================================
# author: Lars Gabriel
#
# features.py: Handles the features for a transcript
# ==============================================================

class Node_features:
    """
        Class handling the features for a transcripts.
        Features are scores that characterize the support of the transcript
        by extrinsic evidence in different ways.
    """
    def __init__(self, tx, evi, hint_source_weight={'P' : 0.1, 'E' : 10, 'C' : 5,  'M' : 1}):
        """
            Args:
                tx (Transcript): Transcript class object containing a transcript.
                evi (Evidence): Evidence class object containing all extrinsic evidence.
                hint_source_weight (dict(int)): Weights for each evidence source.
        """
        self.sw = hint_source_weight
        self.evi_list = {'intron' : [], 'start_codon' : [], 'stop_codon': []}
        self.numb_introns = 0
        self.__init_hints__(tx, evi)
        # feature vector specifies the support of
        # introns, start/stop codons for a transcript
        # self.feature_vector[0] : (supported introns by evidence of tx) / (number of introns in tx)
        # self.feature_vector[1] : (supported start/stop codons by evidence of tx) / 2
        # self.feature_vector[2] : sum of multiplicities of intron evidence for tx
        # self.feature_vector[3] : sum of multiplicities of start/stop codon evidence for tx
        # self.feature_vector[4] : 1 if tx is from anno_pref, 0 otherwise
        self.feature_vector = self.create_feature_vec()

    def __init_hints__(self, tx, evi):
        """
            Collect hints from evi that support tx.

            Args:
                tx (Transcript): Transcript class object containing a transcript.
                evi (Evidence): Evidence class object containing all extrinsic evidence.
        """
        cds_len = 0
        for type in ['intron', 'start_codon', 'stop_codon']:            
            for line in tx.transcript_lines[type]:
                hint = evi.get_hint(line[0], line[3], line[4], line[2], \
                    line[6])
                if hint:
                    self.evi_list[type].append(hint)
        if tx.transcript_lines['intron']:
            self.numb_introns = len(tx.transcript_lines['intron'])

    def create_feature_vec(self):
        """
            Compute all features.

            Returns:
                (list(float)): List of feature scores.
        """
        return [self.relative_support(['intron'], self.numb_introns), \
                self.relative_support(['stop_codon'], 1.0), \
                self.relative_support(['start_codon'], 1.0), \
                self.absolute_support(['intron']), \
                self.absolute_support(['stop_codon']),
                self.absolute_support(['start_codon'])]

    def relative_support(self, gene_feature_types, abs_numb):
        """
            Compute relative support of introns or start/stop-codons.

            Args:
                gene_feature_types (str): Either introns or start/stop-codons
                abs_numb (int): absolute number of gene_feature_type in tx
                                (e.g. number of introns in tx)

            Returns:
                (float): Relative support in [0,1].
        """
        if abs_numb > 0:
            hint_numb = 0
            for type in gene_feature_types:
                hint_numb += len(self.evi_list[type])
            return hint_numb / abs_numb
        return 1

    def absolute_support(self, gene_feature_types):
        """
            Compute absolute support of introns or start/stop-codons.

            Args:
                gene_feature_types (str): Either introns or start/stop-codons

            Returns:
                (float): Multiplicity*weight of supporting hints for gene_feature_types.
        """
        score = 0.0
        for type in gene_feature_types:
            for hint in self.evi_list[type]:
                for src in hint.keys():
                    score += self.sw[src] * hint[src]
        return score

    def get_features(self):
        """
            Returns:
                (list(float)): List of feature scores.
        """
        return self.feature_vector

#!/usr/bin/env python3
# ==============================================================
# Lars Gabriel
#
# genome_anno.py: Handles the data structure for a genome annotation file
# ==============================================================

import os
import sys
import csv
import numpy as np

class NotGtfFormat(Exception):
    pass

class Transcript:
    """
        Class handling the data structures and methods for a transcript
    """
    def __init__(self, id, gene_id, chr, source_anno, strand):
        """
            Args:
                id (str): Transcript ID
                gene_id (str): Gene ID
                chr (str): Chromosome/Sequence name where the transcript is located
                source_anno (str): Anno ID
                strand (str): Strand (+/-) on which the transctipt is located
        """
        self.id = id
        self.chr = chr
        self.gene_id = gene_id
        # self.transcript_lines[segment_type] = [lines of segment type]
        self.transcript_lines = {}
        self.gtf = []
        self.source_anno = source_anno
        self.start = -1
        self.end = -1
        self.utr_start = -1
        self.utr_end = -1
        # number of bases in UTR
        self.utr_len = 0
        self.cds_len = 0
        self.strand = strand
        self.source_method = ''
        self.utr = False

        # # self.hints[type][startcoord_endcoord][src] = hint_multiplicity
        # self.hints = {'intron' : {}, 'start_codon' : {}, 'stop_codon' : {}}

    def add_line(self, line):
        """
            Add a single line from the gtf file to the transcript data structure.

            Args:
                line (list): List of all elements of a line from a gtf file
        """
        if not (line[0] == self.chr or line[6] == self.strand):
            raise NotGtfFormat('File is not in gtf format. ' \
                + 'Error in line {}\n'.format('\t'.join(map(str, line)))
                + 'Transcript ID is not unique')

        if line[2] not in self.transcript_lines.keys():
            self.transcript_lines.update({line[2] : []})

        self.source_method = line[1]

        line[3] = int(line[3])
        line[4] = int(line[4])
        if 'cds' in line[2].lower():
            if self.start < 0 or line[3] < self.start:
                self.start = line[3]
            if self.end < 0 or line[4] > self.end:
                self.end = line[4]
        if self.gene_id == '' and not line[2] == 'transcript':
            self.gene_id = line[8].split('gene_id "')[1].split('";')[0]
        self.transcript_lines[line[2]].append(line)

    def set_gene_id(self, new_gene_id):
        self.gene_id = new_gene_id

#     def add_evidence(self, evi):
#         for type in ['intron', 'start_codon', 'stop_codon']:
#             if type not in self.hints: self.hints[type] = {}
#             for line in self.transcript_lines[type]:
#                 hint = evi.get_hint(line[0], line[3], line[4], line[2], \
#                     line[6])
#                 if hint:
#                     all = 0
#                     coord_key = f'{line[3]}_{line[4]}'
#                     self.hints[type][coord_key] = hint'

    def get_type_coords(self, type, frame=True):
        """
            Get the coordinates and reading frame of the coding regions

            Returns:
                (dict(list(list(int)))): Dictionary with list of CDS coords for
                                        each each frame phase (0,1,2)
        """
        # returns dict of cds_coords[phase] = [start_coord, end_coord] of all CDS

        if frame:
            coords = {'0' : [], '1' : [], '2' : [], '.' : []}
        else:
            coords = []
        if type == 'CDS' and type not in self.transcript_lines.keys():
            type = 'exon'
        if type not in self.transcript_lines.keys():
            return coords
        for line in self.transcript_lines[type]:
            if frame:
                coords[line[7]].append([line[3], line[4]])
            else:
                coords.append([line[3], line[4]])
        if frame:
            for k in coords.keys():
                coords[k].sort(key=lambda c: (c[0],c[1]))
        else:
            coords.sort(key=lambda c: (c[0],c[1]))
        return coords

    def find_start_end(self):
        coords = []
        codons = ['start_codon', 'stop_codon']
        if 'CDS' in self.transcript_lines.keys():
            type = 'CDS'
        else:
            type = 'exon'
        if self.strand == '-':
            codons.reverse()
        if codons[0] in self.transcript_lines.keys():
            self.start = self.transcript_lines[codons[0]][0][3]
        else:
            coords = self.get_type_coords(type)
            self.start = min([c[0] for coord in coords.values() for c in coord])
        if codons[1] in self.transcript_lines.keys():
            self.end = self.transcript_lines[codons[1]][0][4]
        else:
            if not coords:
                coords = self.get_type_coords(type)
            self.end = max([c[1] for coord in coords.values() for c in coord])

        self.utr_len = 0
        if "3'-UTR" in self.transcript_lines.keys():
            self.utr=True
            self.utr_len += sum([i[4] - i[3] + 1 for i in self.transcript_lines["3'-UTR"]])
            utr_start = min([i[3] for i in self.transcript_lines["3'-UTR"]])
        else:
            utr_start = self.start
        if "5'-UTR" in self.transcript_lines.keys():
            self.utr=True
            self.utr += sum([i[4] - i[3] + 1 for i in self.transcript_lines["5'-UTR"]])
            utr_end = max([i[4] for i in self.transcript_lines["5'-UTR"]])
        else:
            utr_end = self.end
        if self.utr:
            self.utr_start = utr_start
            self.utr_end = utr_end

        if 'CDS' in self.transcript_lines.keys():
            self.cds_len = sum([i[4] - i[3] + 1 for i in self.transcript_lines["CDS"]])

    def add_missing_lines(self):
        """
            Add transcript, intron, CDS, exon coordinates if they were not
            included in the gtf file

            Returns:
                (boolean): FALSE if no cds were found for the tx, TRUE otherwise
        """
        # check if tx has cds or exon
        if not self.check_cds_exons():
            return False
        # add intron lines
        self.find_introns()
        # add start, end coords of transcript
        self.find_start_end()
        # add transcript line
        self.find_transcript()
        # add start/stop codon line
        self.find_start_stop_codon()
        for type in ['intron', 'CDS']:
            _ = self.get_type_coords(type, frame=False)
        return True

    def check_cds_exons(self):
        """
            Check if tx has CDS or exons.
        """
        if 'CDS' not in self.transcript_lines.keys() and 'exon' not in self.transcript_lines.keys():
            sys.stderr.write('Skipping transcript {}, no CDS nor exons in {}\n'.format(self.id, self.id))
            return False
        return True

    def find_introns(self):
        """
            Add intron lines.
        """
        if not 'intron' in self.transcript_lines.keys():
            self.transcript_lines.update({'intron' : []})
            keys = ["3'-UTR"]
            if 'CDS' in self.transcript_lines.keys():
                keys.append('CDS')
            elif 'exon' in self.transcript_lines.keys():
                keys.append('exon')
            keys.append("5'-UTR")
            exon_lst = []
            for key in keys:
                for coords in self.get_type_coords(key).values():
                    exon_lst += coords
            exon_lst = sorted(list(exon_lst), key=lambda e:e[0])
            for i in range(1, len(exon_lst)):
                intron = [self.chr, self.source_method, 'intron', exon_lst[i-1][1] + 1, \
                exon_lst[i][0] - 1, '.', self.strand, '.', "gene_id \"{}\"; transcript_id \"{}\";".format(\
                self.gene_id, self.id)]
                if intron[4] - intron[3] > 0:
                    self.transcript_lines['intron'].append(intron)
        if self.id == 'PB.27149.4':
            print (self.transcript_lines['intron'])

    def find_transcript(self):
        """
            Add transcript lines.
        """
        if not 'transcript' in self.transcript_lines.keys():
            if self.start < 0  or self.end < 0:
                self.find_start_end()
            tx_line = [self.chr, self.source_method, 'transcript', self.start, self.end, \
            '.', self.strand, '.', self.id]
            self.add_line(tx_line)

    def find_start_stop_codon(self):
        """
            Add start/stop codon lines.
        """

        if not 'start_codon' in self.transcript_lines.keys():
            self.transcript_lines.update({'start_codon' : []})
        if not 'stop_codon' in self.transcript_lines.keys():
            self.transcript_lines.update({'stop_codon' : []})

        key = ''
        if 'CDS' in self.transcript_lines.keys():
            key = 'CDS'
        elif 'exon' in self.transcript_lines.keys():
            key = 'exon'
        if key:
            self.transcript_lines[key].sort(key = lambda x : x[3])
            tx = self.transcript_lines[key][0]
            line1 = [self.chr, tx[1], '', tx[3], tx[3] + 2, \
            '.', self.strand, '0', "gene_id \"{}\"; transcript_id \"{}\";".format(\
            self.gene_id, self.id)]
            tx = self.transcript_lines[key][-1]
            line2 = [self.chr, tx[1], '', tx[4] - 2, tx[4], \
            '.', self.strand, '0', "gene_id \"{}\"; transcript_id \"{}\";".format(\
            self.gene_id, self.id)]

            fragmented_transcript = True
            if tx[6] == '+':
                line1[2] = 'start_codon'
                line2[2] = 'stop_codon'
                if self.transcript_lines[key][0][7] == 0:
                    fragmented_transcript = False
                start = line1
                stop = line2
            else:
                line1[2] = 'stop_codon'
                line2[2] = 'start_codon'
                if self.transcript_lines[key][-1][7] == 0:
                    fragmented_transcript = False
                stop = line1
                start = line2
            if not self.transcript_lines['start_codon'] and not fragmented_transcript:
                if not fragmented_transcript:
                    self.add_line(start)
                else:
                    self.transcript_lines.update({'start_codon' : []})
            if not self.transcript_lines['stop_codon']:
                self.add_line(stop)

    def get_gtf(self, prefix=''):
        """
            Creates gtf output for the transcript.

            Returns:
                (list(list(str))): List of lines in gtf format as lists
        """
        gtf = []
        if prefix:
            prefix += '.'
        tx_line = []
        for k in self.transcript_lines.keys():
            for g in self.transcript_lines[k]:
                if k == 'transcript':
                    tx_line  = g
                    tx_line[8] = prefix + self.id
                    continue
                else:
                    g[8] = 'transcript_id \"{}\"; gene_id \"{}";'.format(\
                        prefix + self.id, self.gene_id)
                gtf.append(g)
        gtf = sorted(gtf, key=lambda g: (g[3],g[4]))
        if tx_line:
            gtf = [tx_line] + gtf
        return gtf

class Anno:
    """
        Class handling the data structures and methods for a one genome annotation file
    """
    def __init__(self, path, id):
        """
            Args:
                path (str): Path to the annotation/gene prediction file in gtf format.
                id (str): Annotation ID
        """
        self.id = id
        self.genes = {'None' : []}
        self.gene_gtf = {}
        self.transcripts = {}
        self.path = path
        self.translation_tab = []

    def addGtf(self):
        """
            Read a gtf file and create a dictionary of Transcript objects for
            all transcript in the file
        """
        with open (self.path, 'r') as file:
            file_lines = csv.reader(file, delimiter='\t')
            for line in file_lines:
                line = [l.strip(' ') for l in line]
                if line[0][0] ==  '#':
                    continue
                line[3] = int(line[3])
                line[4] = int(line[4])
                if line[2] == 'gene':
                    gene_id = line[8]
                    self.genes_update(gene_id)
                    if not gene_id in self.gene_gtf.keys():
                        self.gene_gtf.update({gene_id : line})
                    else:
                        sys.stderr.write('ERROR, gene_id not unique: {}\n'.format(gene_id))
                elif line[2] == 'transcript':
                    transcript_id = line[8]
                    gene_id = ''
                    self.transcript_update(transcript_id, gene_id, line[0], line[6])
                    self.transcripts[transcript_id].add_line(line)
                else:
                    transcript_id = line[8].split('transcript_id "')
                    if len(transcript_id) > 1:
                        transcript_id = transcript_id[1].split('";')[0]
                    else:
                        raise NotGtfFormat('File: "{}" is not in gtf format. \n'.format(\
                            self.path) + 'Error in line {}\n'.format('\t'.join(map(str, line))))

                    gene_id = line[8].split('gene_id "')
                    if len(gene_id) > 1:
                        gene_id = gene_id[1].split('";')[0]
                    else:
                        gene_id = 'None'
                        for key, value in self.genes.items():
                            if value == transcript_id:
                                gene_id = key

                    self.transcript_update(transcript_id, gene_id, line[0], line[6])
                    self.genes_update(gene_id, transcript_id)
                    self.transcripts[transcript_id].add_line(line)

        for tx_id in self.genes['None']:
            gene_id = tx_id + '_g'
            self.genes_update(gene_id, tx_id)

    def norm_tx_format(self):
        """
            Add to all Transcript objects transcript, intron, CDS, exon
            coordinates if they were not included in the gtf file.
            Delete all transripts that have no exons or CDS
        """
        tx_no_cds = []
        # add missing lines to all tx
        for k in self.transcripts.keys():
            if not self.transcripts[k].add_missing_lines():
                tx_no_cds.append(k)
        for k in tx_no_cds:
            del self.transcripts[k]

    def genes_update(self, gene_id, transcript_id=''):
        """
            Update gene ID dict.
            Args:
                gene_id (str): Gene ID
                transcript_id (str): Transcript ID
        """
        # update gene ids
        if not gene_id in self.genes.keys():
            self.genes.update({ gene_id : []})
        if transcript_id and transcript_id not in self.genes[gene_id]:
            self.genes[gene_id].append(transcript_id)
        if transcript_id in self.genes['None'] and not gene_id == 'None':
            self.genes['None'].remove(transcript_id)
            self.transcripts[transcript_id].gene_id = gene_id

    def transcript_update(self, t_id, g_id, chr, strand):
        """
            Update transcript ID dict.
            Args:
                t_id (str): Transcript ID
                g_id (str): Gene ID
                chr (str): Chromosome name
                strand (str): Strand (+/-)
        """
        if not t_id in self.transcripts.keys():
            self.transcripts.update({ t_id : Transcript(t_id, g_id, chr, self.id, strand)})

    def find_genes(self):
        """
            Find all genes in the annotation and find the transcripts that
            belong to each gene. Also, cretae a dict with the gtf lines for each gene.
        """
        self.gene_gtf = {}
        self.genes = {}
        for tx in self.transcripts.values():
            if tx.gene_id in self.genes.keys():
                if not (tx.chr == self.gene_gtf[tx.gene_id][0] and \
                    tx.strand == self.gene_gtf[tx.gene_id][6]):
                    sys.stderr.write('ERROR, gene_id not unique: {}.'.format(tx.gene_id))
                    tx.gene_id = tx.gene_id + '.' + tx.chr + '.' + tx.strand
                    sys.stderr.write(' Adding new gene: {}\n'.format(tx.gene_id))
                else:
                    self.genes[tx.gene_id].append(tx.id)
                    self.gene_gtf[tx.gene_id][3] = min(self.gene_gtf[tx.gene_id][3], \
                        tx.start)
                    self.gene_gtf[tx.gene_id][4] = max(self.gene_gtf[tx.gene_id][4], \
                        tx.end)
                    continue
            self.genes.update({tx.gene_id : [tx.id]})
            self.gene_gtf.update({tx.gene_id : [tx.chr, tx.source_method, 'gene', \
                tx.start, tx.end, '.', tx.strand, '.', tx.gene_id]})

    def get_gtf(self):
        """
            Get annotaion file as gtf list.
            Returns:
                list(list(str)): Gtf file as list of lists
        """
        gtf = []
        gene_gtf = sorted(self.gene_gtf.values(), key=lambda g: (g[0],g[3],g[4]))
        for gene in gene_gtf:
            gtf.append(gene)
            for tx_id in self.genes[gene[8]]:
                gtf += self.transcripts[tx_id].get_gtf()
        return gtf

    def add_transcripts(self, txs, id_prefix=''):
        """
            Adds a dict of transcripts to the transcripts of the annotation.
            Args:
                dict(Transcript()): dictionary of Transcripts added to the annotation
        """
        if not id_prefix:
            self.transcripts.update({txs})
        else:
            for tx in txs.values():
                tx.id = id_prefix + tx.id
                self.transcripts.update({tx.id : tx})

    def get_subset(self, tx_list):
        """
            Get annotaion file for a subset of transcripts.
            Args:
                tx_list (list(str)): List of transcript IDs
            Returns:
                list(list(str)): Gtf file as list of lists
        """
        tx_subset = {}
        for tx in tx_list:
            tx_subset.update({tx : self.transcripts[tx]})
        return tx_subset

    def change_id(self, new_id):
        """
            Change annotation file ID.
        """
        self.id = new_id
        for k in self.transcripts.keys():
            self.transcripts[k].source_anno = self.id

    def get_transcript_list(self):
        """
            Returns:
                (List(Transcript)): List of all transcripts.
        """
        return list(self.transcripts.values())

    def rename_tx_ids(self, prefix=''):
        """
            Renames all tx and genes and returns translation table for old tx id to new tx id.
            Args:
                prefix (string): String added before each tx and gene ID.
            Returns:
                translation_tab (list(str, str)): Translation table for old tx id to new tx id.
        """
        self.translation_tab = []
        gene_numb = 1
        old_gene_gtf = sorted(self.gene_gtf.values(), key=lambda g: (g[0],g[3],g[4]))
        self.gene_gtf = {}
        old_genes = self.genes
        self.genes = {}
        old_txs = self.transcripts
        self.transcripts = {}
        if prefix:
            prefix += '_'
        for gene in old_gene_gtf:
            tx_numb = 1
            old_gene_id = gene[8]
            new_gene_id = "{}g{}".format(prefix, gene_numb)
            gene[8] = new_gene_id
            self.genes.update({new_gene_id : []})
            self.gene_gtf.update({new_gene_id : gene})
            for old_tx_id in old_genes[old_gene_id]:
                new_tx_id = "{}g{}.t{}".format(prefix, gene_numb, tx_numb)
                self.transcripts.update({new_tx_id : old_txs[old_tx_id]})
                self.transcripts[new_tx_id].id = new_tx_id
                self.transcripts[new_tx_id].gene_id = new_gene_id
                self.genes[new_gene_id].append(new_tx_id)
                tx_numb +=1
                self.translation_tab.append([new_tx_id, old_tx_id])
            gene_numb += 1
        return self.translation_tab

    def write_anno(self, out_path):
        """
            Write Annotation in gtf format to out_path.
            Args:
                (str) : path to the output file
        """
        with open(out_path, 'w+') as file:
            out_writer = csv.writer(file, delimiter='\t', quotechar = "|", lineterminator = '\n')
            for line in self.get_gtf():
                out_writer.writerow(line)

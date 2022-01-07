#!/usr/bin/env python3
# ==============================================================
# author: Lars Gabriel
#
# TSEBRA: Transcript Selector for BRAKER
# ==============================================================
import argparse
import sys
import os
import csv
import numpy as np
from tabulate import tabulate

from genome_anno import Anno
from overlap_graph import Graph
from evidence import Evidence

class ConfigFileError(Exception):
    pass

gtf = []
gene_sets = []
hintfiles = []
graph = None
out = ''
evi = None
v = 0
quiet = False
parameter = {'intron_support' : 0, 'stop_support' : 0, 'start_support' : 0, \
    'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0, 'e_5' : 0, 'e_6' : 0}
numb_features = 7
def main():
    """
        Overview:

        1. Read gene predicitions from .gtf files.
        2. Read Evidence from .gff files.
        3. Detect overlapping transcripts.
        4. Create feature vector (for a list of all features see features.py)
           for all transcripts.
        5. Compare the feature vectors of all pairs of overlapping transcripts.
        6. Exclude transcripts based on the 'transcript comparison rule' and 5.
        7. Remove Transcripts with low evidence support.
        8. Create combined gene predicitions (all transcripts that weren't excluded).
    """


    global graph, parameter, evi

    args = parseCmd()
    init(args)


    if v > 0:
        print(gtf)
    # read hintfiles
    evi = Evidence()
    for h in hintfiles:
        if not quiet:
            sys.stderr.write('### READING EXTRINSIC EVIDENCE: [{}]\n'.format(h))
        evi.add_hintfile(h)
    for src in evi.src:
        if src not in parameter.keys():
            sys.stderr.write('ConfigError: No weight for src={}, it is set to 1\n'.format(src))
            parameter.update({src : 1})
    anno_keys = []
    anno = Anno(args.anno, 'reference_annotation')
    anno.addGtf()
    anno.norm_tx_format()
    for tx in anno.transcripts.values():
        anno_keys.append(get_tx_key(tx))

    print_summary([anno], 'reference_annotation', anno_keys)
    # read gene prediciton files
    c = 0
    for g in gtf:
        if not quiet:
            sys.stderr.write('### READING GENE PREDICTION: [{}]\n'.format(g))
        gene_set = Anno(g, 'anno{}'.format(c))
        gene_set.addGtf()
        gene_set.norm_tx_format()
        gene_sets.append(gene_set)
        print_summary([gene_set], 'anno{}'.format(c), anno_keys)
        c += 1
    print_summary(gene_sets, 'all_gene_sets', anno_keys)



def print_summary(g, name, anno_keys):
    graph = Graph(g, para=parameter, verbose=v)
    if not quiet:
        sys.stderr.write('### BUILD OVERLAP GRAPH\n')
    graph.build()
    if not quiet:
        sys.stderr.write('### ADD FEATURES TO TRANSCRIPTS\n')
    graph.add_node_features(evi)

    features = []
    features_in_anno = []
    features_not_in_anno = []
    for node in graph.nodes.values():
        tx = graph.__tx_from_key__(node.id)
        if get_tx_key(tx) in anno_keys:
            features_in_anno.append(node.feature_vector)
        else:
            features_not_in_anno.append(node.feature_vector)
        features.append(node.feature_vector)
    print(name)
    table = []
    header = ["MIN","MAX","Q1","Q2","Q3","MEAN", "MIN","MAX","Q1","Q2","Q3","MEAN",\
        "MIN","MAX","Q1","Q2","Q3","MEAN"]
    for line in zip(summary(features), summary(features_in_anno), summary(features_not_in_anno)):
        s = []
        for k in line:
            s += [str(round(l, 2)) for l in k]
        table.append(s)
    print(tabulate(table, headers=header))

def get_tx_key(tx):
    coords = []
    for c in tx.get_type_coords('CDS').values():
        coords += c
    coords.sort()
    return '_'.join([tx.chr, tx.strand] + [str(c[0]) + '_' + str(c[1]) for c in coords])



def summary(list):
    res = []
    if not list:
        return [[0]*3]*6
    np_list = np.array(list)
    mins = np.amin(np_list, axis=0)
    maxs = np.amax(np_list, axis=0)
    q1 = np.quantile(np_list, q=0.25, axis=0)
    means = np.mean(np_list, axis=0)

    q2 = np.quantile(np_list, q=0.5, axis=0)
    q3 = np.quantile(np_list, q=0.75, axis=0)
    return zip(mins, maxs, q1, q2, q3, means)

def set_parameter(cfg_file):
    """
        read parameters from the cfg file and store them in parameter.

        Args:
            cfg_file (str): Path to configuration file.
    """
    global parameter
    with open(cfg_file, 'r') as file:
        cfg = csv.reader(file, delimiter=' ')
        for line in cfg:
            if not line[0][0] == '#':
                if line[0] not in parameter.keys():
                    parameter.update({line[0] : None})
                parameter[line[0]] = float(line[1])

def init(args):
    global gtf, hintfiles, threads, hint_source_weight, out, v, quiet
    if args.gtf:
        gtf = args.gtf.split(',')
    if args.hintfiles:
        hintfiles = args.hintfiles.split(',')
    if args.cfg:
        cfg_file = args.cfg
    else:
        cfg_file = os.path.dirname(os.path.realpath(__file__)) + '/../config/default.cfg'
    set_parameter(cfg_file)
    if args.out:
        out = args.out
    if args.verbose:
        v = args.verbose
    if args.quiet:
        quiet = True

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='TSEBRA: Transcript Selector for BRAKER\n\n' \
        + 'TSEBRA combines gene predictions by selecing ' \
        + 'transcripts based on their extrisic evidence support.')
    parser.add_argument('-g', '--gtf', type=str, required=True,
        help='List (separated by commas) of gene prediciton files in gtf.\n' \
            + '(e.g. gene_pred1.gtf,gene_pred2.gtf,gene_pred3.gtf)')
    parser.add_argument('-a', '--anno', type=str, required=True,
        help='')
    parser.add_argument('-e', '--hintfiles', type=str, required=True,
        help='List (separated by commas) of files containing extrinsic evidence in gff.\n' \
            + '(e.g. hintsfile1.gff,hintsfile2.gtf,3.gtf)')
    parser.add_argument('-c', '--cfg', type=str, required=True,
        help='Configuration file that sets the parameter for TSEBRA. ' \
            + 'You can find the recommended parameter at config/default.cfg.')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    parser.add_argument('-v', '--verbose', type=int,
        help='')
    return parser.parse_args()

if __name__ == '__main__':
    main()

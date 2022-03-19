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
from datetime import datetime
import matplotlib.pyplot as plt

class ConfigFileError(Exception):
    pass

gtf = []
gene_sets = []
hintfiles = []
anno = ''
graph = None
out = ''
v = 0
quiet = False

weight_class_one = 90.
#parameter = {'intron_support' : 0, 'stasto_support' : 0, \
    #'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0}
numb_node_features = 46
numb_edge_features = 23

numb_batches = 150
batch_size = 1
val_size = 0

def main():
    from genome_anno import Anno
    from overlap_graph import Graph

    global gene_sets, graph, input_test#, parameter

    args = parseCmd()
    init(args)

    # read gene prediciton files
    c = 1
    for g in gtf:
        if not quiet:
            sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] READING GENE PREDICTION: [{g}]\n')
        gene_sets.append(Anno(g, f'anno{c}'))
        gene_sets[-1].addGtf()
        gene_sets[-1].norm_tx_format()
        c += 1

    # create graph with an edge for each unique transcript
    # and an edge if two transcripts overlap
    # two transcripts overlap if they share at least 3 adjacent protein coding nucleotides
    graph = Graph(gene_sets, verbose=v)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] BUILD OVERLAP GRAPH\n')
    graph.build()
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] ADD NODE FEATURES\n')

    graph.connected_components()
    numb_batches = int(len(graph.component_list)/batch_size)

    combined_anno = Anno('', 'combined_annotation')
    k = 0
    keys = list(graph.component_list.keys())
    for i in np.random.choice(len(keys), numb_batches, replace=False):
        for id in graph.component_list[keys[i]]:
            k += 1
            tx = graph.__tx_from_key__(id)
            tx.id = tx.source_anno + '.' + tx.id
            tx.set_gene_id(graph.nodes[id].component_id)
            combined_anno.transcripts.update({tx.id : tx})
        if k > numb_train:
            break
    combined_anno.find_genes()
    combined_anno.write_anno(args.out)


def init(args):
    global gtf, numb_train, out,quiet
    if args.gtf:
        gtf = args.gtf.split(',')
    if args.numb_train:
        numb_train = args.numb_train
    if args.out:
        out = args.out
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
    parser.add_argument('-n', '--numb_train', type=int, required=True,
        help='Number of transcripts in the training set.')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    return parser.parse_args()

if __name__ == '__main__':
    main()

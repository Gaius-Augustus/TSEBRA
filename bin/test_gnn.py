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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
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
batch_size = 100
val_size = 0

def main():
    from genome_anno import Anno
    from overlap_graph import Graph
    from evidence import Evidence
    from gnn import GNN

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


    # read hintfiles
    evi = Evidence()
    for h in hintfiles:
        if not quiet:
            sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] READING EXTRINSIC EVIDENCE: [{h}]\n')
        evi.add_hintfile(h)

    # create graph with an edge for each unique transcript
    # and an edge if two transcripts overlap
    # two transcripts overlap if they share at least 3 adjacent protein coding nucleotides
    graph = Graph(gene_sets, verbose=v)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] BUILD OVERLAP GRAPH\n')
    graph.build()
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] ADD NODE FEATURES\n')
    graph.add_node_features(evi)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] ADD EDGE FEATURES\n')
    graph.add_edge_features(evi)
    graph.connected_components()
    numb_batches = int(len(graph.nodes)/batch_size)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] CREATE ANNO LABEL\n')
    graph.create_batch(numb_batches, batch_size)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] TRANSFORM BATCHES TO INPUT TARGETS\n')
    input_test, _,_,_ = graph.get_batches_as_input_target(val_size)
    #print(input_test[0][0].shape)
    #print(input_val[0][0])
    gnn = GNN(weight_class_one=weight_class_one)
    gnn.compile(args.model)

    combined_anno = Anno('', 'combined_annotation')
    for i in range(len(input_test)):
        predictions = gnn.predict(input_test[i][0])
        if i < 1:
            print(predictions[0][:5])
        for p, id in zip(np.array(predictions[-1][0]), graph.batches[i].nodes):
            if p >= 0.5:
                tx = graph.__tx_from_key__(id)
                tx.id = tx.source_anno + '.' + tx.id
                tx.set_gene_id(graph.nodes[id].component_id)
                combined_anno.transcripts.update({tx.id : tx})
    combined_anno.find_genes()
    combined_anno.write_anno(args.out)


def init(args):
    global gtf, hintfiles, out, v, quiet, anno
    if args.gtf:
        gtf = args.gtf.split(',')
    if args.hintfiles:
        hintfiles = args.hintfiles.split(',')
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
    parser.add_argument('-m', '--model', type=str, required=True,
        help='')
    parser.add_argument('-e', '--hintfiles', type=str, required=True,
        help='List (separated by commas) of files containing extrinsic evidence in gff.\n' \
            + '(e.g. hintsfile1.gff,hintsfile2.gtf,3.gtf)')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    return parser.parse_args()

if __name__ == '__main__':
    main()

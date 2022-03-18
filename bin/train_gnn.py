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
#parameter = {'intron_support' : 0, 'stasto_support' : 0, \
    #'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0}
numb_node_features = 46
numb_edge_features = 23

numb_batches = 1000
batch_size = 5
val_size = 0.2
NUM_EPOCHS = 40
weight_class_one = 30.
def main():
    from genome_anno import Anno
    from overlap_graph import Graph
    from evidence import Evidence
    from gnn import GNN

    global gene_sets, graph, input_train, input_val#, parameter

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

    ref_anno = Anno(anno, 'reference')
    ref_anno.addGtf()
    ref_anno.norm_tx_format()

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
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] ADD REF ANNO LABEL\n')
    graph.add_reference_anno_label(ref_anno)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] CREATE ANNO LABEL\n')
    graph.create_batch(numb_batches, batch_size, repl=True)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] TRANSFORM BATCHES TO INPUT TARGETS\n')
    input_train, input_val = graph.get_batches_as_input_target(val_size)


    train_gen = SampleGenerator(0, 80)
    val_gen = SampleGenerator(1, 20)


    gnn = GNN(weight_class_one=weight_class_one)
    gnn.compile()
    gnn.train(train_gen, val_gen, NUM_EPOCHS, args.out)
    """history.history["last_iteration_binary_accuracy"][-1]
    _, ax = plt.subplots(ncols = 2, figsize = (15, 6))

    ax[0].plot(np.arange(NUM_EPOCHS), history.history["loss"], 'b', label = 'Training loss')
    ax[0].plot(np.arange(NUM_EPOCHS), history.history["val_loss"], 'g', label = 'Validation loss')
    ax[0].set_title('Training and validation loss')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()

    ax[1].plot(np.arange(NUM_EPOCHS), history.history["last_iteration_binary_accuracy"], 'b', label = 'Training accuracy')
    ax[1].plot(np.arange(NUM_EPOCHS), history.history["val_last_iteration_binary_accuracy"], 'g', label = 'Validation accuracy')
    ax[1].set_title('Training and validation accuracy of the last message passing iteration')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend()

    plt.show()"""

class SampleGenerator(keras.utils.Sequence):
    def __init__(self, val, epoch_len):
        self.val = val
        self.epoch_len = epoch_len

    def __len__(self):
        return self.epoch_len #number of gradient descent steps per epoch

    def __getitem__(self, _index):
        if self.val:
            return input_val[_index][0], input_val[_index][1]
        else:
            return input_train[_index][0], input_train[_index][1]

def init(args):
    global gtf, hintfiles, out, v, quiet, anno
    if args.gtf:
        gtf = args.gtf.split(',')
    if args.hintfiles:
        hintfiles = args.hintfiles.split(',')
    if args.out:
        out = args.out
    if args.anno:
        anno = args.anno
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
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    return parser.parse_args()

if __name__ == '__main__':
    main()

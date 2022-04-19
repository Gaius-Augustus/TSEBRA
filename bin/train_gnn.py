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
val_sets = []
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

numb_batches_train = 10000
numb_batches_val = 3000
batch_size = 128
val_size = 0.1
NUM_EPOCHS = 25
weight_class_one = 1.

def main():
    global gene_sets, graph, input_train, input_val, Graph, out
    from genome_anno import Anno
    from overlap_graph import Graph
    from evidence import Evidence
    from gnn import GNN

    

    args = parseCmd()
    #init(args)
    out = args.out
    parent_dir = args.dir
    # read gene prediciton files
    c = 1
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] READING GENE SETS\n')
    for b in ['braker1', 'braker2']:        
        gene_sets.append(Anno(f'{parent_dir}/{b}_train.gtf', f'anno{c}'))
        gene_sets[-1].addGtf()
        gene_sets[-1].norm_tx_format()
        val_sets.append(Anno(f'{parent_dir}/{b}_val.gtf', f'anno{c}'))
        val_sets[-1].addGtf()
        val_sets[-1].norm_tx_format()
        
        c += 1

    ref_anno = Anno(f'{parent_dir}/annot_train.gtf', 'reference')
    ref_anno.addGtf()
    ref_anno.norm_tx_format()
    ref_anno_val = Anno(f'{parent_dir}/annot_val.gtf', 'reference')
    ref_anno_val.addGtf()
    ref_anno_val.norm_tx_format()
     
    # read hintfiles
    evi = Evidence()
    for h in ['hints1', 'hints2']:
        if not quiet:
            sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] READING EXTRINSIC EVIDENCE\n')
        evi.add_hintfile(f'{parent_dir}/{h}_train.gff')
    evi_val = Evidence()
    for h in ['hints1', 'hints2']:        
        evi_val.add_hintfile(f'{parent_dir}/{h}_val.gff')
    
    input_train, _ = get_batches(gene_sets, evi, ref_anno, numb_batches_train, batch_size, 0, True)
    input_val, _ = get_batches(val_sets, evi_val, ref_anno_val, numb_batches_val, batch_size, 0)    
    
    set_weight()
    print(len(input_train), len(input_val))
    train_gen = SampleGenerator(0, len(input_train))#int(numb_batches * (1-val_size)))
    val_gen = SampleGenerator(1, len(input_val))#int(numb_batches * val_size))


    gnn = GNN(weight_class_one=weight_class_one)
    gnn.compile()
    history = gnn.train(train_gen, val_gen, NUM_EPOCHS, args.out)
    
    history.history["last_iteration_binary_accuracy"][-1]
    _, ax = plt.subplots(ncols = 2)

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

    plt.savefig(args.out + '.png', dpi=200)
    
def get_batches(tx_sets, evi, ref_anno, numb_b, b_size, v_size, des=False):
    graph = Graph(tx_sets, verbose=v)
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
    numb_b = int(len(graph.nodes)/b_size) 
    graph.create_batch(numb_b, b_size, repl=True)
    if not quiet:
        sys.stderr.write(f'### [{datetime.now().strftime("%H:%M:%S")}] TRANSFORM BATCHES TO INPUT TARGETS\n')
    input_train, input_val = graph.get_batches_as_input_target(v_size)
    describe_features(graph)
    return input_train, input_val

def set_weight():
    global weight_class_one
    a=0
    b=0
    for i in input_train:
        for k in list(i[1].values())[0][0]:
            a+=1
            b+=k[0]
    weight_class_one = (a-b)/b
    print(a, b, weight_class_one)

def describe_features(graph):
    n_f = []
    e_f = []
    for node in graph.nodes.values():
        n_f.append(node.feature_vector)
    for edge in graph.edges.values():
        e_f.append(edge.feature_vector_n1_to_n2)
        e_f.append(edge.feature_vector_n2_to_n1)
    n_f = np.array(n_f)
    e_f = np.array(e_f)
    _, ax = plt.subplots(ncols = 1, nrows=4)
    i = int(n_f.shape[1] / 4 + 0.5)
    for j in range(4):
        ax[j].boxplot([n_f[:,k] for k in range(i*j, min(i*(j+1), n_f.shape[1]))])
        ax[j].set_ylim([-3,3])
    plt.savefig(out + 'node_features.png', dpi=200)
    
    _, ax = plt.subplots(ncols = 1, nrows=4)
    i = int(e_f.shape[1] / 4 + 0.5)
    for j in range(4):
        ax[j].boxplot([e_f[:,k] for k in range(i*j, min(i*(j+1), e_f.shape[1]))])
        ax[j].set_ylim([-3,3])
    plt.savefig(out + 'edge_features.png', dpi=200)
    
    
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
    """parser.add_argument('-g', '--gtf', type=str, required=True,
        help='List (separated by commas) of gene prediciton files in gtf.\n' \
            + '(e.g. gene_pred1.gtf,gene_pred2.gtf,gene_pred3.gtf)')
    parser.add_argument('-a', '--anno', type=str, required=True,
        help='')
    parser.add_argument('-e', '--hintfiles', type=str, required=True,
        help='List (separated by commas) of files containing extrinsic evidence in gff.\n' \
            + '(e.g. hintsfile1.gff,hintsfile2.gtf,3.gtf)')"""
    parser.add_argument('-d', '--dir', type=str, required=True,
        help='')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    return parser.parse_args()

if __name__ == '__main__':
    main()

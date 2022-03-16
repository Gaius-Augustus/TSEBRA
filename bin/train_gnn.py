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

config = {
    "message_passing_iterations" : 1,
    "latent_dim" : 32
}

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

numb_batches = 10000
batch_size = 200
val_size = 0.2
weight_class_one = 30.
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

    from genome_anno import Anno
    from overlap_graph import Graph
    from evidence import Evidence

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
    #print(input_val[0][0])
    print(len(input_train), len(input_val))
    train_gen = SampleGenerator(0, 8000)
    val_gen = SampleGenerator(1, 2000)

    # x_train = [[i[0]] for i in input_train]
    # print(len(x_train), x_train[0])
    # y_train = [[i[1]] for i in input_train]
    # x_val = [[i[0]] for i in input_val]
    # y_val = [[i[1]] for i in input_val]

    NUM_EPOCHS = 40

    GNN = make_GNN(config)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

    cee = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    acc = tf.keras.metrics.BinaryAccuracy()

    def all_iterations_cee(y_true, y_pred):
        loss = 0
        weights = (y_true[0][:,1] * (weight_class_one-1)) + 1.
        for i in range(config["message_passing_iterations"]):
            loss += tf.reduce_mean(cee(y_true, y_pred[i]) * weights) #compute loss for all iterations
        return loss / config["message_passing_iterations"]

    def last_iteration_binary_accuracy(y_true, y_pred):
        weights = (y_true[0][:,1] * (weight_class_one-1)) + 1.
        return acc(y_true, y_pred[-1])*weights

    GNN.compile(loss=last_iteration_binary_accuracy,#all_iterations_cee,
                optimizer=optimizer,
                metrics={"target_label" : last_iteration_binary_accuracy})
    
    #history = 
    GNN.fit(train_gen,
                    validation_data=val_gen,
                        epochs = NUM_EPOCHS,
                        verbose = 1)
    GNN.save_weights(args.out)
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


#define a feedforward layer
def make_ff_layer(config):
    phi = keras.Sequential([
            layers.Dense(config["latent_dim"], activation="relu",                         kernel_regularizer=tf.keras.regularizers.l1_l2(0.1)),
            layers.Dense(config["latent_dim"])])
    return phi

def make_GNN(config):

    #define the inputs
    #we use a batch size of 1 since we implemented batching
    #by using the compound graph
    V = keras.Input(shape=(None, numb_node_features), name="input_nodes", batch_size=1)
    E = keras.Input(shape=(None, numb_edge_features), name="input_edges", batch_size=1)
    Is = keras.Input(shape=(None,None), name="incidence_matrix_sender",
                      batch_size=1)
    Ir = keras.Input(shape=(None,None), name="incidence_matrix_receiver",
                      batch_size=1)

    node_encoder = make_ff_layer(config)
    edge_encoder = make_ff_layer(config)
    node_updater = make_ff_layer(config)
    edge_updater = make_ff_layer(config)
    node_decoder = layers.Dense(2, activation="sigmoid")

    #step 1: encode
    #transform each node (dim=3) and edge (dim=1) to a latent embedding of "latent_dim"
    V_enc = node_encoder(V)
    E_enc = edge_encoder(E)

    #V_enc.shape = (1, num_nodes, latent_dim)
    #E_enc.shape = (1, num_edges, latent_dim)

    target_label = []

    #step 2: message passing
    for _ in range(config["message_passing_iterations"]):

        #for each node, sum outgoing and incoming edges (separately)
        V_Es = tf.matmul(Is, E_enc)
        V_Er = tf.matmul(Ir, E_enc)
        #update all nodes based on current state and aggregated edge states
        V_concat = tf.concat([V_enc, V_Es, V_Er], axis=-1)
        V_enc = node_updater(V_concat)

        #get states of respective sender and receiver nodes for each edge
        E_Vs = tf.matmul(Is, V_enc, transpose_a=True)
        E_Vr = tf.matmul(Ir, V_enc, transpose_a=True)
        #update all edges
        E_concat = tf.concat([E_enc, E_Vs, E_Vr], axis=-1)
        E_enc = edge_updater(E_concat)

        #step 3: decode
        #in this case, we want to predict for each edge the probability of lying on a shortest path
        #we use sigmoid as activation
        target_label.append(node_decoder(V_enc))

    model = keras.Model(inputs=[V, E, Is, Ir],
                        outputs=[layers.Lambda(lambda x: x, name="target_label")(tf.stack(target_label))])
    return model

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

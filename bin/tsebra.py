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

class ConfigFileError(Exception):
    pass

gtf = []
gene_sets = []
hintfiles = []
graph = None
out = ''
v = 0
quiet = False
#parameter = {'intron_support' : 0, 'stasto_support' : 0, \
    #'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0}

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

    global gene_sets, graph#, parameter

    args = parseCmd()
    init(args)

    if v > 0:
        print(gtf)

    # read gene prediciton files
    c = 1
    for g in gtf:
        if not quiet:
            sys.stderr.write('### READING GENE PREDICTION: [{}]\n'.format(g))
        gene_sets.append(Anno(g, 'anno{}'.format(c)))
        gene_sets[-1].addGtf()
        gene_sets[-1].norm_tx_format()
        c += 1

    anno = Anno(args.anno, 'reference')
    anno.addGtf()
    anno.norm_tx_format()

    # read hintfiles
    evi = Evidence()
    for h in hintfiles:
        if not quiet:
            sys.stderr.write('### READING EXTRINSIC EVIDENCE: [{}]\n'.format(h))
        evi.add_hintfile(h)


    # create graph with an edge for each unique transcript
    # and an edge if two transcripts overlap
    # two transcripts overlap if they share at least 3 adjacent protein coding nucleotides
    graph = Graph(gene_sets, verbose=v)
    if not quiet:
        sys.stderr.write('### BUILD OVERLAP GRAPH\n')
    graph.build()

    # add features
    if not quiet:
        sys.stderr.write('### ADD FEATURES TO TRANSCRIPTS\n')
    graph.add_node_features(evi)


    anno_keys = []
    for tx in anno.transcripts.values():
        anno_keys.append(get_tx_key(tx))

    numb_nodes = len(graph.nodes)
    x = np.zeros((numb_nodes, 30))
    y = np.zeros((numb_nodes, 2))
    tx_keys = []
    for key, i in zip(graph.nodes.keys(), range(numb_nodes)):
        node = graph.nodes[key]
        tx = graph.__tx_from_key__(key)
        tx_keys.append(get_tx_key(tx))
        if tx_keys[-1] in anno_keys:
            y[i][1] = 1
        else:
            y[i][0] = 1
        x[i] = node.feature_vector

    test_indices = np.random.choice(numb_nodes, 3000, replace=False)
    mask = np.ones(numb_nodes, bool)
    mask[test_indices] = False
    x_train = x[test_indices,]
    y_train = y[test_indices,]
    x_test = x[mask]
    y_test = y[mask]

    model = keras.Sequential([
    keras.layers.Reshape(target_shape=(30,), input_shape=(30,)),
    keras.layers.Dense(units=30, activation='relu'),
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dense(units=2, activation='softmax')
    ])

    data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(len(y_train)) \
    .batch(128)

    model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.AUC()])
    history = model.fit(
        data.repeat(),
        epochs=500,
        steps_per_epoch=600
    )
    predictions = model.predict(x_test)
    predictions_all = model.predict(x)
    correct = 0
    false = 0
    print(np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)))
    #predictions[np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)]
    keys_test = []
    keys_train = []
    for i in range(numb_nodes):
        if i in test_indices and np.argmax(predictions_all[i]) == 1:
            keys_train.append(tx_keys[i])
        elif i not in test_indices and np.argmax(predictions_all[i]) == 1:
            keys_test.append(tx_keys[i])
    true = 0
    for k in keys_train:
        if k in anno_keys:
            true+=1
    print(np.shape(y_test))
    print(true, len(keys_train), len(anno_keys))
    anno_keys = set(anno_keys)
    true_total = true
    true = 0
    for k in keys_test:
        if k in anno_keys:
            true+=1
    print(true, len(keys_test), len(anno_keys))

    print(true + true_total, len(keys_train) + len(keys_test),(true + true_total) / (len(keys_train) + len(keys_test)),\
    (true + true_total)/len(anno_keys))
    #model.save(args.out)

    print(np.shape(y_test), len(set(tx_keys).intersection(anno_keys)))
    # apply decision rule to exclude a set of transcripts
    """if not quiet:
        sys.stderr.write('### SELECT TRANSCRIPTS\n')
    combined_prediction = graph.get_decided_graph()

    if v > 0:
        sys.stderr.write(str(combined_prediction.keys()) + '\n')
        for a in anno:
            sys.stderr.write('Numb_tx in {}: {}\n'.format(a.id, len(combined_prediction[a.id])))

    # write result to output file
    if not quiet:
        sys.stderr.write('### WRITE COMBINED GENE PREDICTION\n')
    combined_anno = Anno('', 'combined_annotation')
    for a in anno:
        txs = a.get_subset([t[0] for t in combined_prediction[a.id]])
        for id, new_gene_id in combined_prediction[a.id]:
            txs[id].set_gene_id(new_gene_id)
        combined_anno.add_transcripts(txs, a.id + '.')
    combined_anno.find_genes()
    combined_anno.write_anno(out)

    if not quiet:
        sys.stderr.write('### FINISHED\n\n')
        sys.stderr.write('### The combined gene prediciton is located at {}.\n'.format(\
            out))"""

def get_tx_key(tx):
    coords = []
    for c in tx.get_type_coords('CDS').values():
        coords += c
    coords.sort()
    return '_'.join([tx.chr, tx.strand] + [str(c[0]) + '_' + str(c[1]) for c in coords])

def init(args):
    global gtf, hintfiles, threads, hint_source_weight, out, v, quiet
    if args.gtf:
        gtf = args.gtf.split(',')
    if args.hintfiles:
        hintfiles = args.hintfiles.split(',')
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
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    parser.add_argument('-v', '--verbose', type=int,
        help='')
    return parser.parse_args()

if __name__ == '__main__':
    main()

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
numb_features = 69
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
    numb_test = 1700
    numb_val = 100
#x, y, mask_train, mask_val = split_data_set_by_nodes(graph, anno_keys, 2000, 500)
    x, y, mask_train, mask_val, txs = split_data_set_by_component(graph, anno_keys, numb_test, numb_val)
    x_max = x[:,:38].max(axis=0)+ 0.000000000001
    x[:,:38] /= x_max
    x[:,38:] /= x_max[5:36]

    x_train = x[mask_train]
    y_train = y[mask_train]
    x_val = x[mask_val]
    y_val = y[mask_val]
    x_test = x[(mask_train == False) & (mask_val == False)]
    y_test = y[(mask_train == False) & (mask_val == False)]
    print(x[0], y[0])
    print(x.shape, x_train.shape, x_val.shape, x_test.shape)

    if args.model:
        model = keras.models.load_model(args.model)
    else:
        model = keras.Sequential([
        keras.layers.Reshape(target_shape=(numb_features,), input_shape=(numb_features,)),
        keras.layers.Dense(units=int((numb_features+2)/2), activation='relu'),
        keras.layers.Dense(units=2, activation='softmax')
        ])
        model.compile(optimizer='adam',
          loss=tf.losses.CategoricalCrossentropy(),
          metrics=['accuracy']
              #metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
         # metrics=[keras.metrics.AUC(), keras.metrics.Accuracy(), keras.metrics.Precision(), \
                #keras.metrics.Recall()]\
        )

    if not args.load:
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(len(y_train)) \
            .batch(100)
        if numb_val > 0:
            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
                .shuffle(len(y_val)) \
                .batch(100)



        history = model.fit(
            train_data.repeat(),
            epochs=5000,
            steps_per_epoch=50,
            validation_data=val_data.repeat(),
            validation_steps=10
        )

        model.save(args.out)
    predictions = model.predict(x_test)
    correct = 0
    false = 0
    if numb_val > 0:
        list = [[x_train,  y_train], [x_val,  y_val], [x_test,  y_test], [x,y]]
    else:
        list = [[x_train,  y_train], [x_test,  y_test], [x,y]]
    for x_it, y_it in list:
        predictions = model.predict(x_it)

        pred_argmax = np.zeros(len(predictions))
        pred_argmax[predictions[:,1] > 0.5] = 1
        y_argmax = np.argmax(y_it, axis=1)
        print(np.sum( \
            pred_argmax == y_argmax) / \
            len(anno_keys))
        print(np.sum( \
            (pred_argmax == y_argmax)[pred_argmax == 1]) / \
            len(anno_keys))
        print(np.sum(pred_argmax == y_argmax) / \
            len(x_it))
        print(np.sum((pred_argmax == y_argmax)[pred_argmax == 1]) / \
            len(y_argmax[y_argmax == 1]))
        print('----------------')
    model.evaluate(x_train,  y_train, verbose=2)
    model.evaluate(x_val,  y_val, verbose=2)
    model.evaluate(x_test,  y_test, verbose=2)

    predictions = model.predict(x)
    pred_argmax = np.zeros(len(predictions))
    pred_argmax[predictions[:,1] > 0.5] = 1

    combined_anno = Anno('', 'combined_annotation')
    for i, tx in enumerate(txs):
        if pred_argmax[i] == 1:
            tx[0].id = tx[0].source_anno + '.' + tx[0].id
            tx[0].set_gene_id(tx[1])
            combined_anno.transcripts.update({tx[0].id : tx[0]})
    combined_anno.find_genes()
    combined_anno.write_anno(args.out + '.gtf')

    """
    #predictions[np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)]
    numb_nodes = len(graph.nodes)
    keys_test = []
    keys_train = []
    for i in range(numb_nodes):
        if (mask_train[i] == False) & (mask_val[i] == False) and np.argmax(predictions_all[i]) == 1:
            keys_train.append(tx_keys[i])
        elif (mask_train[i] == False) & (mask_val[i] == False) and np.argmax(predictions_all[i]) == 1:
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
    """
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


def split_data_set_by_component(graph, anno_keys, numb_train_components, numb_val_components=0):
    numb_nodes = len(graph.nodes)
    if not graph.component_list:
        graph.connected_components()
    numb_components = len(graph.component_list)
    print(f'### Numb. of components in graph: {numb_components}')
    train_val_indices = np.random.choice(numb_components, numb_train_components+numb_val_components, \
        replace=False)

    val_sub_indices = np.random.choice(numb_train_components+numb_val_components, \
        numb_val_components, replace=False)
    mask_sub_val = np.zeros(numb_train_components+numb_val_components, bool)
    mask_sub_val[val_sub_indices] = True

    val_indices = train_val_indices[mask_sub_val]
    train_indices = train_val_indices[mask_sub_val == False]

    mask_components = np.ones(numb_components, bool)
    mask_components[train_indices] = False

    x = np.zeros((numb_nodes, numb_features), float)
    y = np.zeros((numb_nodes, 2))
    mask_train = np.zeros(numb_nodes, bool)
    mask_val = np.zeros(numb_nodes, bool)
    k = 0
    txs = []
    for i, component in enumerate(graph.component_list):
        for node_key in component:
            x[k] = graph.nodes[node_key].feature_vector
            tx = graph.__tx_from_key__(node_key)
            txs.append([tx, graph.nodes[node_key].component_id])
            if get_tx_key(tx) in anno_keys:
                y[k][1] = 1
            else:
                y[k][0] = 1
            if i in train_indices:
                mask_train[k] = True
            elif i in val_indices:
                mask_val[k] = True
            k += 1
    return x, y, mask_train, mask_val, txs



def split_data_set_by_nodes(fraph, anno_keys, numb_train_nodes):

    numb_nodes = len(graph.nodes)
    x = np.zeros((numb_nodes, numb_features), float)
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

    train_val_indices = np.random.choice(len(x), numb_train_components+numb_val_components, \
        replace=False)

    val_sub_indices = np.random.choice(numb_train_components+numb_val_components, \
        numb_val_components, replace=False)
    mask_sub_val = np.zeros(numb_train_components+numb_val_components, bool)
    mask_sub_val[val_sub_indices] = True

    val_indices = train_val_indices[mask_sub_val]
    train_indices = train_val_indices[mask_sub_val == False]


    print(f'### Numb. of nodes in graph: {len(x)}')
    mask_train = np.zeros(len(x), bool)
    mask_train[train_indices] = True
    mask_val = np.zeros(len(x), bool)
    mask_val[val_indices] = True
    return x, y, mask_train, mask_val

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
    parser.add_argument('-m', '--model', type=str,
        help='')
    parser.add_argument('-l', '--load', action='store_true',
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

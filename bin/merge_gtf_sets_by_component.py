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
import subprocess as sp

class ConfigFileError(Exception):
    pass

gtf = []
hintfiles = []
graph = None
out = ''
v = 0
quiet = False
#parameter = {'intron_support' : 0, 'stasto_support' : 0, \
    #'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0}
numb_features = 69
species = ['Danio_rerio', 'Drosophila_melanogaster', 'Arabidopsis_thaliana']

numb_tx_per_s = 20000

def main():
    from genome_anno import Anno
    from overlap_graph import Graph
    from evidence import Evidence

    global gene_sets, graph#, parameter

    args = parseCmd()
    init(args)

    if v > 0:
        print(gtf)

    # read gene prediciton files
    for inp, outp in zip([['braker1.gtf', 'braker2.gtf'], ['annot.gtf']], \
                        ['braker.gtf', 'annot.gtf']):
        combined_anno = Anno('', 'combined_annotation')
        gene_sets = {}
        for s in species:
            gene_sets.update({s : []})
            c = 1
            for b in inp:            
                gene_sets[s].append(Anno(f'{args.dir}/{s}/{b}', f'{s}.{c}'))
                gene_sets[s][-1].addGtf()
                gene_sets[s][-1].norm_tx_format()
                c += 1
            graph = Graph(gene_sets[s], verbose=v)
            if not quiet:
                sys.stderr.write('### BUILD OVERLAP GRAPH\n')
            graph.build()

            txs = split_data_set_by_component(graph, s)
            for tx in txs:
                combined_anno.transcripts.update({tx.id : tx}) 
            
        combined_anno.find_genes()
        combined_anno.write_anno(f'{args.out}/{outp}')
    for s in species:
        cmd = f'cat {s}/hints1.gff {s}/hints2.gff >> {args.out}/hints.gff'
        sp.call(cmd, shell=True)

def split_data_set_by_component(graph, s):
    numb_nodes = len(graph.nodes)
    if not graph.component_list:
        graph.connected_components()
    numb_components = len(graph.component_list)
    print(f'### Numb. of components in graph: {numb_components}')
    indices = np.random.choice(numb_components, numb_components, \
        replace=False)

    txs = []
    for component in np.array(graph.component_list)[indices]:
        for node_key in component:
            tx = graph.__tx_from_key__(node_key)            
            tx.id = tx.source_anno + '.' + tx.id
            tx.set_gene_id(f'{s}_{graph.nodes[node_key].component_id}')
            txs.append(tx)
        if len(txs) > numb_tx_per_s:
            break
    print('Number TX: ', len(txs), s)
    return txs




def get_tx_key(tx):
    coords = []
    for c in tx.get_type_coords('CDS').values():
        coords += c
    coords.sort()
    return '_'.join([tx.chr, tx.strand] + [str(c[0]) + '_' + str(c[1]) for c in coords])

def init(args):
    global gtf, hintfiles, threads, hint_source_weight, out, v, quiet
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
    parser.add_argument('-d', '--dir', type=str, required=True,
        help='List (separated by commas) of gene prediciton files in gtf.\n' \
            + '(e.g. gene_pred1.gtf,gene_pred2.gtf,gene_pred3.gtf)')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    parser.add_argument('-v', '--verbose', type=int,
        help='')
    return parser.parse_args()

if __name__ == '__main__':
    main()

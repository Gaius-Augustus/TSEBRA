#!/usr/bin/env python3
# ==============================================================
# author: Lars Gabriel
#
# get_longest_isoform.py: combines gene sets into one that 
#      consists only of the longest isoform from each gene loci
# ==============================================================
import argparse
import sys
import os
import csv

class ConfigFileError(Exception):
    pass

class GeneSetMissing(Exception):
    pass

gtf = []
anno = []
hintfiles = []
graph = None
out = ''
v = 0
quiet = False
parameter = {'intron_support' : 0, 'stasto_support' : 0, \
    'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0}

def main():
    from genome_anno import Anno
    from overlap_graph import Graph

    global anno, graph, parameter

    args = parseCmd()
    init(args)

    if v > 0:
        print(gtf)

    # read gene prediciton files
    c = 1
    for c, g in enumerate(gtf):
        if not quiet:
            sys.stderr.write(f'### READING GENE PREDICTION: [{g}]\n')
        anno.append(Anno(g, f'anno{c+1}'))
        anno[-1].addGtf()
        anno[-1].norm_tx_format()

    # create graph with an edge for each unique transcript
    # and an edge if two transcripts overlap
    # two transcripts overlap if they share at least 3 adjacent protein coding nucleotides
    graph = Graph(anno, para=parameter, verbose=v)
    if not quiet:
        sys.stderr.write('### BUILD OVERLAP GRAPH\n')
    graph.build()
    
    combined_anno = Anno('', 'combined_annotation')
    # for each gene locus, choose the transcript with longes coding sequence
    if not quiet:
        sys.stderr.write('### CHOOSE LONGEST ISOFORM FOR EACH GENE\n')
    for i, comp in enumerate(graph.connected_components()):
        tx_longest = sorted([graph.__tx_from_key__(n) for \
                   n in comp], key=lambda t:t.get_cds_len())[-1]
        tx_longest.set_gene_id(f'g_{i+1}')
        tx_longest.id = f'{tx_longest.source_anno}.{tx_longest.id}'
        combined_anno.transcripts.update({tx_longest.id : tx_longest})    
    combined_anno.find_genes()
    combined_anno.write_anno(out)

    if not quiet:
        sys.stderr.write('### FINISHED\n\n')
        sys.stderr.write('### The longest isoforms are located at {}.\n'.format(\
            out))

def init(args):
    global gtf, out, v, quiet
    if args.gtf:
        gtf = args.gtf.split(',')    
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
    parser = argparse.ArgumentParser(description='Combine gene sets by choosing ' \
                 'the isoform with the longes coding sequence for each gene locus.')
    parser.add_argument('-g', '--gtf', type=str, required=True,
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

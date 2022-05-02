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

class ConfigFileError(Exception):
    pass

class GeneSetMissing(Exception):
    pass
    
gtf = []
keep_all = []
keep_long_reads = False
long_reads = []
anno = []
hintfiles = []
graph = None
out = ''
v = 0
quiet = False
filter = False
parameter = {'intron_support' : 0, 'stop_support' : 0, 'start_support' : 0, \
    'e_1' : 0, 'e_2' : 0, 'e_3' : 0, 'e_4' : 0, 'e_5' : 0, 'e_6' : 0}

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

    global anno, graph, parameter

    args = parseCmd()
    init(args)

    if v > 0:
        print(gtf)

    # read gene prediciton files
    keep = []
    c = 1
    for g in gtf:
        if not quiet:
            sys.stderr.write(f'### READING GENE PREDICTION: [{g}]\n')
        anno.append(Anno(g, f'anno{c}'))
        anno[-1].addGtf()
        anno[-1].norm_tx_format()
        c += 1
    for g in keep_all:
        if not quiet:
            sys.stderr.write(f'### READING GENE PREDICTION: [{g}]\n')
        anno.append(Anno(g, f'anno{c}'))
        anno[-1].addGtf()
        anno[-1].norm_tx_format()
        keep.append(f'anno{c}')
        c += 1
    c = 1
    for l in long_reads:
        if not quiet:
            sys.stderr.write(f'### READING LONG-READS: [{l}]\n')
        anno.append(Anno(l, f'long_reads{c}'))
        anno[-1].addGtf()
        anno[-1].norm_tx_format()
        if keep_long_reads:
            keep.append(f'long_reads{c}')
        c += 1


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

    # create graph with an edge for each unique transcript
    # and an edge if two transcripts overlap
    # two transcripts overlap if they share at least 3 adjacent protein coding nucleotides
    graph = Graph(anno, para=parameter, filter_short=filter, keep_tx=keep, verbose=v)
    if not quiet:
        sys.stderr.write('### BUILD OVERLAP GRAPH\n')
    graph.build()

    # add features
    if not quiet:
        sys.stderr.write('### ADD FEATURES TO TRANSCRIPTS\n')
    graph.add_node_features(evi)

    # apply decision rule to exclude a set of transcripts
    if not quiet:
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
            out))

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
    global gtf, hintfiles, threads, hint_source_weight, out, v, \
        filter, long_reads, quiet, keep_all, keep_long_reads
    if args.gtf:
        gtf = args.gtf.split(',')
    if args.keep_gtf:
        keep_all = args.keep_gtf.split(',')
    if not args.keep_gtf and not args.gtf:
        raise GeneSetMissing('At least one gene set has to be provided '\
            + 'either with --gtf or --kepp_all!')
    if args.hintfiles:
        hintfiles = args.hintfiles.split(',')
    if args.keep_long_reads:
        keep_long_reads = args.keep_long_reads
    if args.cfg:
        cfg_file = args.cfg
    else:
        cfg_file = os.path.dirname(os.path.realpath(__file__)) + '/../config/default.cfg'
    set_parameter(cfg_file)
    if args.out:
        out = args.out
    if args.verbose:
        v = args.verbose
    if args.filter_short:
        filter = args.filter_short
    if args.long_reads:
        long_reads = args.long_reads.split(',')
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
    parser.add_argument('-g', '--gtf', type=str,
        help='List (separated by commas) of gene prediciton files in gtf.\n' \
            + '(e.g. gene_pred1.gtf,gene_pred2.gtf,gene_pred3.gtf)')
    parser.add_argument('-k', '--keep_gtf', type=str,
        help='List (separated by commas) of gene prediciton files in gtf.\n' \
            + 'These gene sets are used the same way as other inputs, but TSEBRA '\
            + 'ensures that all transcripts from these gene sets are included in the output.')
    parser.add_argument('-l', '--long_reads', type=str,
        help='List (separated by commas) of transcript sets inferred from long-reads.\n' \
            + '(e.g. long_read1.gtf,long_read2.gtf,long_read3.gtf)')
    parser.add_argument('-e', '--hintfiles', type=str, required=True,
        help='List (separated by commas) of files containing extrinsic evidence in gff.\n' \
            + '(e.g. hintsfile1.gff,hintsfile2.gtf,3.gtf)')
    parser.add_argument('-kl', '--keep_long_reads', action='store_true',
        help='Set this flag if you want to keepl all transcripts from the long-read set.')
    parser.add_argument('-c', '--cfg', type=str, required=True,
        help='Configuration file that sets the parameter for TSEBRA. ' \
            + 'You can find the recommended parameter at config/default.cfg.')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the combined gene prediciton in gtf.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode.')
    parser.add_argument('-f', '--filter_short', action='store_true',
        help='Have a strict filter for short transcripts.')
    parser.add_argument('-v', '--verbose', type=int,
        help='')
    return parser.parse_args()

if __name__ == '__main__':
    main()

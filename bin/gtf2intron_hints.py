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

gtf = None

def main():
    from genome_anno import Anno
    from overlap_graph import Graph
    from evidence import Evidence

    args = parseCmd()
    src = 'M'
    if args.src and len(args.src) > 1:
        sys.stderr.write('### ERROR: Please provide only a single character ' \
            + 'as hint source with --src!')
        exit()
    if args.src:
        src = args.src
    # read gene prediciton file
    sys.stderr.write('### READING GENE SET: [{}]\n'.format(args.gtf))
    gtf = Anno(args.gtf, '')
    gtf.addGtf()
    gtf.norm_tx_format()
    # dict keys: 'chr_strand_start_end' values: multiplicity
    intron_keys = {}

    for tx in gtf.transcripts.values():
        introns = tx.get_type_coords('intron')
        for i in introns.values():
            for intron in i:
                key = "{};{};{};{}".format(tx.chr, tx.strand, intron[0], intron[1])
                if not key in intron_keys:
                    intron_keys.update({key : 0})
                intron_keys[key] += 1
    output = []
    for intron in intron_keys:
        key = intron.split(';')
        output.append([key[0], 'tsebra_hints', 'intron', int(key[2]), int(key[3]), \
            intron_keys[intron], key[1], '.', 'mult={};pri=4;src={}'.\
            format(intron_keys[intron], src)])
    output.sort(key = lambda i:(i[0], i[3], i[4], i[6]))

    with open(args.out, 'w+') as file:
        out_writer = csv.writer(file, delimiter='\t', quotechar = "|", lineterminator = '\n')
        for line in output:
            out_writer.writerow(line)
            
def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='Detects all introns in the '\
        + 'input gtf file and reports them as intron hints with a "src=" and a "mult=" attribute.')
    parser.add_argument('-g', '--gtf', type=str, required=True,
        help='Input gtf file with a set of transcripts.')
    parser.add_argument('-s', '--src', type=str, required=False,
        help='Source (single character) of the intron hints, default is "M" form manual hints.')
    parser.add_argument('-o', '--out', type=str, required=True,
        help='Outputfile for the intron hints in gtf.')

    return parser.parse_args()

if __name__ == '__main__':
    main()

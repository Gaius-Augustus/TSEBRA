#!/bin/sh
# if this file is not executable run: chmod +x run_prevco_example.sh

c="${0%/*}"
# prediciton and hint files that are included in the standard output of a BRAKER run
b1=$c/braker1_results/braker.gtf
b2=$c/braker2_results/braker.gtf
h1=$c/braker1_results/hintsfile.gff
h2=$c/braker2_results/hintsfile.gff

# create working directory
d=$c/tsebra_workdir/
mkdir -p $d

# Make sure that the transcript IDs of the BRAKER predicitons are in order
# This step is OPTIONAL and not necassary for a succefull combination

echo "\n*** Fix possible ID errors in *.gtf files ***\n"

new_b1=$d/braker1.gtf
new_b2=$d/braker2.gtf
$c/../bin/fix_gtf_ids.py --gtf $b1 --out $new_b1
$c/../bin/fix_gtf_ids.py --gtf $b2 --out $new_b2
b1=$new_b1
b2=$new_b2

# Combine BRAKER1 and BRAKER2 predicitons

o=$d/braker1+2.gtf

echo "*** Running TSEBRA ***\n"

$c/../bin/tsebra.py -g $b1,$b2 -e $h1,$h2 -o $o

echo "\n*** Finished. Result at: $o ***\n"

# TSEBRA: Transcript Selector for BRAKER

<p align="center">
<img src="docs/TSEBRA_Logo.png" alt="drawing" width="700"/>
</p>

### Introduction
[TSEBRA](https://doi.org/10.1186/s12859-021-04482-0) is a combiner tool that selects transcripts from gene predictions based on the support by extrisic evidence in form of introns and start/stop codons. It was developed to combine BRAKER1<sup name="a1">[1](#ref1)</sup> and BRAKER2<sup name="a2">[2](#ref2)</sup> predicitons to increase their accuracies.

## Prerequisites
Python 3.5.2 or higher is required.

## Installation
Download TSEBRA:
```console
git clone https://github.com/Gaius-Augustus/TSEBRA
```

## Usage
The main script is ```./bin/tsebra.py```. For usage information run ```./bin/tsebra.py --help```.

## Input Files
TSEBRA takes a list of gene prediciton files, a list of hintfiles and a configuration file as mandatory input.

#### Gene Predictions
The gene prediction files have to be in gtf format. This is the standard output format of a BRAKER or AUGUSTUS<sup name="a3">[3,](#ref3)</sup><sup name="a4">[4](#ref4)</sup> gene prediciton.

Example:
```console
2L      AUGUSTUS        gene    83268   87026   0.88    -       .       g5332
2L      AUGUSTUS        transcript      83268   87026   0.88    -       .       g5332.t1
2L      AUGUSTUS        intron  84278   87019   1       -       .       transcript_id "file_1_file_1_g5332.t1"; gene_id "file_1_file_1_g5332";
2L      AUGUSTUS        CDS     87020   87026   0.88    -       0       transcript_id "file_1_file_1_g5332.t1"; gene_id "file_1_file_1_g5332";
2L      AUGUSTUS        exon    87020   87026   .       -       .       transcript_id "file_1_file_1_g5332.t1"; gene_id "file_1_file_1_g5332";
```

#### Hint Files
The hints files have to be in gff format, the last column must include an attribute for the source for the hint with 'src=' and can include the number of hints supporting the gene structure segment with 'mult='. This is the standard file format of the ```hintsfile.gff``` in a BRAKER working directory.

Example:
```console
2L      ProtHint        intron  279806  279869  2       +       .       src=P;mult=25;pri=4;al_score=0.437399;
2L      ProtHint        intron  275252  275318  2       -       .       src=P;mult=19;pri=4;al_score=0.430006;
2L      ProtHint        stop    293000  293002  1       +       0       grp=7220_0:002b08_g42;src=C;pri=4;
2L      ProtHint        intron  207632  207710  1       +       .       grp=7220_0:002afa_g26;src=C;pri=4;
2L      ProtHint        start   207512  207514  1       +       0       grp=7220_0:002afa_g26;src=C;pri=4;
```

#### Configuration File
The configuration file has to include three different sets of parameter:
1. Weights for all sources of hints. The source of a hint is specified by the mandatory 'src=' attribute in the last column of the ```hintsfile.gff``` (see section 'Hint Files'). See section 'Transcript scores' in [TSEBRA](https://doi.org/10.1101/2021.06.07.447316) for more information on how these weigths are used.  
A weight is set to 1, if the weight for a hint source is not specified in the configuration file.  

   * *Notes on adjusting these parameters: Increase the weight of the hint sources that have the highest quality. For example, if the protein database includes only species that are remotely related to the target species, the hints produced by BRAKER2 might be less accurate than the RNA-seq evidence. Then, you should increase the weight of the source related to the RNA-seq hints.*    


2. Required fractions of supported introns or supported start/stop-codons for a transcript. A transcript is not included in the TSEBRA result if the fractions of introns and start/stop codons supported by extrinsic evidence are lower than the thresholds.  

   * *Notes on adjusting these parameters: The low evidence support thresholds for low evidence support are quite strict in the default configuration file. In this configuration, only transcripts with very high evidence support are allowed in the TSBERA result. In some cases, the default setting might be too strict, so that too many transcripts are filtered out. In this case, you should reduce the threshold of 'intron_support' (e.g., to 0.2).*  


3. Allowed difference between two overlapping transcripts for the six transcript scores. TSEBRA compares transcripts via their transcript scores and removes the one with the lower score if their difference exceeds the respective threshold.  
Note that it is recommended to choose thesholds between [0,2], since the transcript scores are normalized to [-1,1]. 

   * *Notes on adjusting these parameters: The higher the thresholds are set the less transcripts are filtered by the respective rule. With these thresholds one can adjust the effect of each filtering rule of TSEBRA. As these thresholds are increased, more transcripts are included in the TSEBRA result, in particular, more alternatively spliced isoforms per gene are contained in the result.*  



The name and the value of a parameter are separated by a space, and each parameter is listed in a different line.  
Example:
```console
# Weight for each hint source
# Values have to be >= 0
P 1
E 1
C 1
M 1
# Required fraction of supported introns 
# or supported start/stop-codons for a transcript
# Values have to be in [0,1]
intron_support 0.8
stasto_support 1
# Allowed difference for each feature 
# Values have to be in [0,2]
e_1 0.0
e_2 0.5
e_3 0.096
e_4 0.02
e_5 0.18
e_6 0.18
```
Description of evidence sources in default BRAKER1 and BRAKER2 outputs:
```
E = RNA-seq hints
M = manual hints, these are hints that are enforced during the prediction step of BRAKER,
C = protein hints from proteins with a 'high' spliced alignment score.
P = protein hints from proteins that have a 'good' spliced alignment score, 
     but that is lower than the score from the ones in 'C'. 
```

## Use Case
The recommended and most common usage for TSEBRA is to combine the resulting ```augustus.hints.gtf``` files of a BRAKER1 and a BRAKER2 run using the hintsfile.gff from both working directories. However, TSEBRA can be applied to any number (>1) of gene predictions and hint files as long as they are in the correct format.

A common case might be that a user wants to annotate a novel genome with BRAKER and has:
* a novel genome with repeats masked: ```genome.fasta.masked```,
* hints for intron positions from RNA-seq reads```rna_seq_hints.gff```,
* database of homologous proteins: ```proteins.fa```.

1. Run BRAKER1 and BRAKER2 for example with
```console
### BRAKER1
braker.pl --genome=genome.fasta.masked --hints=rna_seq_hints.gff \
            --softmasking --species=species_name --workingdir=braker1_out
### BRAKER2
braker.pl --genome=genome.fasta.masked --prot_seq=proteins.fa \
    --softmasking --species=species_name --epmode \
    --workingdir=braker2_out
```
2. Combine predicitons with TSEBRA
```console
./bin/tsebra.py -g braker1_out/augustus.hints.gtf,braker2_out/augustus.hints.gtf -c default.cfg \
    -e braker1_out/hintsfile.gff,braker2_out/hintsfile.gff \
    -o braker1+2_combined.gtf
```
The combined gene prediciton is ```braker1+2_combined.gtf```.

## Example
A small example is located at ```example/```. Run ```./example/run_prevco_example.sh``` to execute the example and to check if TSEBRA runs properly.

## Enforcing a gene set
A gene set can be enforced in the TSEBRA output, i.e. all transcript are guaranteed to be included in the output, with the `--keep_gtf` option. The transcripts of enforced gene sets are still compared to all gene sets and used to evaluate them.
Example:
 ```console
./bin/tsebra.py -g gene_set1,gene_set2 -c default.cfg \
    -k enforced_set1,enforced_set2 -e hintsfile1.gff,braker2_out/hintsfile2.gff \
    -o tsebra.gtf
```
To merge two gene sets, simply omit the `-g` option.


## Filter single-exon genes out
In default mode, TSEBRA is conservative in filtering single exon genes out. In some cases BRAKER predicts a lot of false positive single exon genes. In these cases, it is recommended to run TSBERA using the `--filter_single_exon_genes`. In this mode, TSBERA filters additonally all single-exon genes out that have no support by a start or stop codon hint. 

## Print transcript scores
The transcript scores play a very improtant role in TSEBRA. These are used for pairwise comparison of all transcripts isoforms that have overlapping coding regions. You can print the scores as table to a file with the option `--score_tab /path/to/output/file.tab`.

## Ignore Frame
By default, TSEBRA groups all transcript isoforms that have overlapping coding regions in the same open reading frame (phase column in gtf) to candidates of the same gene. However, in some cases, it might be desired to consider already all transcripts with overlapping conding regions (regardless of the reading frame) as candidates for a gene. In this case add the `--ignore_tx_phase` to the TSEBRA commmand. 

## Other scripts in the TSEBRA repository

### Renaming transcripts from a TSEBRA output
The IDs of the transcripts and genes in the TSEBRA output can be renamed such that the gene and transcript ID match.
Genes and transcript are numbered consecutively and for example, the second transcript of gene "g12" has the ID "g12.t2".
If a prefix is set then it will be added before all IDs, for example, the transcript ID is "dmel_g12.t2" if the prefix is set to "dmel".
Additionally, a translation table can be produced that provides the mapping from old to new transcript IDs.

Example for renaming ```tsebra_result.gtf```:
```console
./bin/rename_gtf.py --gtf tsebra_result.gtf --prefix dmel --translation_tab translation.tab --out tsebra_result_renamed.gtf
```
The arguments ```--prefix``` and ```--translation_tab``` are optional.

### Fixing the formatting issue of `braker.gtf`
A BRAKER run produces a second complete gene set named `braker.gtf`, besides the official output `augustus.hints.gtf`. The `braker.gtf` is the result of merging `augustus.hints.gtf` with some 'high-confidents' genes from the GeneMark prediction. However, the merging process leads to a formatting issue in `braker.gtf`. 
A quick fix for this formatting issue is the script `fix_gtf_ids.py`, e.g.:
```console
./bin/fix_gtf_ids.py --gtf braker_out/braker.gtf --out braker1_fixed.gtf
```
Take note that the `braker.gtf` and `fix_gtf_ids.py` haven't been tested sufficently and there is no guarantee that this gene set is superior to `augustus.hints.gtf`.

### Getting the longest isoform of each gene loci from different gene sets
Combines multiple gene sets and reports the transcript with the longest coding region for each cluster of overlapping transcripts (one transcript per gene loci), e.g.
```console
./bin/get_longest_isoform.py --gtf gene_set1.gtf,gene_set2.gtf --out longest_insoforms.gtf
```

## Licence
All source code, i.e. `bin/*.py` are under the [Artistic License](bin/LICENSE.txt) (see <https://opensource.org/licenses/Artistic-2.0>).

## Citing TSEBRA
Gabriel, L., Hoff, K.J., Brůna, T. *et al.* TSEBRA: transcript selector for BRAKER. *BMC Bioinformatics* **22**, 566 (2021). https://doi.org/10.1186/s12859-021-04482-0

## References
<b id="ref1">[1]</b> Hoff, Katharina J, Simone Lange, Alexandre Lomsadze, Mark Borodovsky, and Mario Stanke. 2015. “BRAKER1: Unsupervised Rna-Seq-Based Genome Annotation with Genemark-et and Augustus.” *Bioinformatics* 32 (5). Oxford University Press: 767--69.[↑](#a1)

<b id="ref2">[2]</b> Tomas Bruna, Katharina J. Hoff, Alexandre Lomsadze, Mario Stanke and Mark Borodvsky. 2021. “BRAKER2: automatic eukaryotic genome annotation with GeneMark-EP+ and AUGUSTUS supported by a protein database." *NAR Genomics and Bioinformatics* 3(1):lqaa108.[↑](#a2)

<b id="ref3">[3]</b> Stanke, Mario, Mark Diekhans, Robert Baertsch, and David Haussler. 2008. “Using Native and Syntenically Mapped cDNA Alignments to Improve de Novo Gene Finding.” *Bioinformatics* 24 (5). Oxford University Press: 637--44.[↑](#a3)

<b id="ref4">[4]</b> Stanke, Mario, Oliver Schöffmann, Burkhard Morgenstern, and Stephan Waack. 2006. “Gene Prediction in Eukaryotes with a Generalized Hidden Markov Model That Uses Hints from External Sources.” *BMC Bioinformatics* 7 (1). BioMed Central: 62.[↑](#a4)

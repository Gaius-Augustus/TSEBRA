#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess
from inspect import currentframe, getframeinfo
import shutil
import re
from datetime import datetime

__author__ = "Katharina J. Hoff"
__copyright__ = "Copyright 2023. All rights reserved."
__credits__ = ""
__license__ = "Artistic License 2.0"
__version__ = "1.0.0"
__email__ = "katharina.hoff@uni-greifswald.de"
__status__ = "development"

argparser = argparse.ArgumentParser(description = 'Find or build the best gene set generated with BRAKER and ' + 
                                    'TSEBRA minimizing missing BUSCOs using compleasm. Will only compute a ' +
                                    'new BRAKER gene set if the percentage of missing BUSCOs exceeds 5%.')
argparser.add_argument('-m', '--tmp_dir', type=str, required = True,
                          help = 'Temporary directory where intermediate files will be written')
argparser.add_argument('-c', '--compleasm_bin', type=str, required = False,
                          help = 'Location of compleasm.py on system')
argparser.add_argument('-d', '--input_dir', type=str, required = True,
                            help = 'Output directory of BRAKER or GALBA run')
argparser.add_argument('-y', '--tsebra', type=str, required = False,
                            help = 'Location of tesbra.py on system')
argparser.add_argument('-f', '--getanno', type=str, required = False,
                            help = 'Location of getAnnoFastaFromJoingenes.py on system')
argparser.add_argument('-g', '--genome', type=str, required = True,
                            help = 'Genome FASTA file')
argparser.add_argument('-t', '--threads', type=str, required = False, default = 1,
                       help = 'Number of threads to use for running compleasm')
argparser.add_argument('-p', '--busco_db', type=str, required = True,
                            help = 'BUSCO lineage for running compleasm')

args = argparser.parse_args()

def parse_compleasm(file):
    """
    Parse the compleasm statistics from the specified file.

    Args:
        file (str): Path to the file containing compleasm statistics.

    Returns:
        float: percentage of missing BUSCOs.

    Raises:
        SystemExit: If the file cannot be opened.

    """
    missing = 0
    stat_pattern = r'M:(\d+\.\d+)\%, \d+'
    try:
        with open(file, "r") as f:
            for line in f:
                if re.search(stat_pattern, line):
                    missing = float(re.search(stat_pattern, line).group(1))
    except IOError:
        print("ERROR: Could not open file: " + file)
        sys.exit(1)
    return missing


def find_input_files(args):
    """
    Find all the required input files for the process.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        dict: Dictionary containing the file paths.

    """
    file_paths = {}
    file_paths["braker_aa"] = check_file(os.path.join(args.input_dir, "braker.aa"))
    file_paths["galba_aa"] = check_file(os.path.join(args.input_dir, "galba.aa"))
    file_paths["augustus_aa"] = check_file(os.path.join(args.input_dir, "Augustus", "augustus.hints.aa"))
    file_paths["genome"] = check_file(args.genome)  # required to generate genemark protein file
    file_paths["hints"] = check_file(os.path.join(args.input_dir, "hintsfile.gff"))  # required to possibly run TSEBRA
    file_paths["braker_gtf"] = check_file(os.path.join(args.input_dir, "braker.gtf"))
    file_paths["galba_gtf"] = check_file(os.path.join(args.input_dir, "galba.gtf"))
    file_paths["augustus_gtf"] = check_file(os.path.join(args.input_dir, "Augustus", "augustus.hints.gtf"))
    file_paths["miniprot_trainingGenes.gtf"] = check_file(os.path.join(args.input_dir, "miniprot_trainingGenes.gtf"))
    return file_paths

def find_genemark_gtf(input_dir):
    """
    Find the genemark.gtf file and corresponding training.gtf file in the specified directory.

    Args:
        input_dir (str): Path to the BRAKER working directory.

    Returns:
        tuple: Tuple containing the paths to the genemark.gtf and training.gtf files.
        tuple consists of False,False if no files are found.

    """
    genemark_directories = ["GeneMark-ETP", "GeneMark-EP", "GeneMark-ET", "GeneMark-ES"]

    for genemark_dir in genemark_directories:
        genemark_gtf_file = os.path.join(input_dir, genemark_dir, "genemark.gtf")
        training_gtf_file = os.path.join(input_dir, genemark_dir, "training.gtf")
        if not os.path.isfile(training_gtf_file):
            training_gtf_file = os.path.join(input_dir, "traingenes.gtf")
        if os.path.isfile(genemark_gtf_file) and os.path.isfile(training_gtf_file):
            return check_file(genemark_gtf_file), check_file(training_gtf_file)

    return False, False


def run_compleasm(protein_files, threads, busco_db, tmp_dir):
    """
    Run compleasm on the specified protein files

    Args:
        protein_files (list): List of protein files to run compleasm on.
        threads (int): Number of threads to use.
        busco_db (str): Path to the BUSCO database.
        tmp_dir (str): Temporary directory to store the output.

    Returns:
        dict: Dictionary containing the paths to the compleasm result files.

    Raises:
        SystemExit: If there is an error in execting compleasm.

    """
    # download the BUSCO database if not present
    if not os.path.exists("mb_downloads/" + args.busco_db):
        # cut off the _odb10 suffix from args.busco_db
        # if it exists
        if args.busco_db.endswith("_odb10"):
            busco_db = args.busco_db[:-6]
        else:
            busco_db = args.busco_db
        compleasm_cmd = [args.compleasm_bin, "download", busco_db]
        run_simple_process(compleasm_cmd)

    # run compleasm on the protein files
    for protein_file in protein_files:
        # this currently only works with branch 0.2.3 from github
        # create a tool-specific output subdirectory
        tool = re.search(r'^([^.]+)\.', os.path.basename(protein_file)).group(1)
        tool_out_dir = args.tmp_dir + "/" + tool
        compleasm_cmd = [args.compleasm_bin, "protein", "-p", protein_file, "-l", busco_db, "-t", str(args.threads), "-o", tool_out_dir]
        run_simple_process(compleasm_cmd)

    result_dict = {}
    for protein_file in protein_files:
        # identify the gene prediction program from the protein file name
        tool = re.search(r'^([^.]+)\.', os.path.basename(protein_file)).group(1)
        result_dict[tool] = check_file(tmp_dir + "/" + tool + "/summary.txt")

    return result_dict

def run_getanno(annobin, genome_file, gtf, output_dir):
    """
    Run the getAnnoFastaFromJoingenes.py tool to process the Genemark GTF file and generate protein sequences.

    Args:
        annobin: getAnnoFastaFromJoingenes.py script.
        genome_file (str): Path to the genome file.
        gtf (str): Path to the GTF file.
        output_dir (str): Output directory to store the generated protein sequences

    """
    tool = re.search(r'^([^.]+)\.', os.path.basename(gtf)).group(1)
    cmd = [annobin, '-g', genome_file, '-f', gtf, '-o', output_dir + "/" + tool]
    run_simple_process(cmd)
    return check_file(output_dir + "/" + tool + ".aa")

def run_simple_process(args_lst):
    """
    Execute a subprocess command with the provided arguments.

    Args:
        args_lst (list): List of command-line arguments.

    Returns:
        CompletedProcess: Result of the subprocess execution.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.

    """
    try:
        print(" ".join(args_lst))
        result = subprocess.run(
            args_lst, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if(result.returncode == 0):
            return(result)
        else:
            frameinfo = getframeinfo(currentframe())
            print('Error in file ' + frameinfo.filename + ' at line ' +
                  str(frameinfo.lineno) + ': ' + "Return code of subprocess was " +
                  str(result.returncode) + str(result.args))
            sys.exit(1)
    except subprocess.CalledProcessError as grepexc:
        frameinfo = getframeinfo(currentframe())
        print('Error in file ' + frameinfo.filename + ' at line ' +
              str(frameinfo.lineno) + ': ' + "Failed executing: ",
              " ".join(grepexc.args))
        print("Error code: ", grepexc.returncode, grepexc.output)
        sys.exit(1)


def check_binary(binary, name):
    """
    Check if the provided binary is executable or available in the system's PATH.

    Args:
        binary (str): Path to the binary file or None.
        name (str): Name of the binary/tool.

    Returns:
        str: Path to the binary if found and executable.

    Raises:
        SystemExit: If the binary is not found, not executable, or not in the PATH.

    """        # check if binary is in PATH
    if binary is not None:
        if os.path.isfile(binary):
            # check if binary is executable
            if os.access(binary, os.X_OK):
                return binary
            else:
                print("ERROR: " + name + " binary is not executable: " + binary)
                sys.exit(1)
        else:
            print("ERROR: " + name + " binary not found at " + binary)
            sys.exit(1)
    else:
        # check if binary is in PATH
        if shutil.which(name) is not None:
            return shutil.which(name)
        else:
            print("ERROR: " + name + " binary not found in PATH")
            sys.exit(1)


def check_file(file):
    """
    Check if the specified file exists.

    Args:
        file (str): Path to the file.

    Returns:
        str: Path to the file if it exists, otherwise False

    """
    if os.path.isfile(file):
        return file
    else:
        return False


def determine_mode(path_dir):
    """
    Determine whether BRAKER or GALBA was run, and whether the files are complete.

    Args:
        path_dir (dict): Dictionary containing the file paths.

    Returns:
        str: Mode of the run (BRAKER or GALBA).

    Raises:
        SystemExit: If the files are not complete for either of the runs

    """
    if path_dir["braker_aa"] and path_dir["braker_gtf"] and path_dir["hints"] and path_dir["genome"] and path_dir["augustus_aa"] and path_dir["augustus_gtf"] and path_dir["genemark_gtf"] and path_dir["hints"]:
        return "BRAKER"
    elif path_dir["galba_aa"] and path_dir["galba_gtf"] and path_dir["miniprot_trainingGenes.gtf"] and path_dir["genome"] and path_dir["hints"]:
        return "GALBA"
    else:
        print("ERROR: The specified directory does not contain all required files.")
        print("These are the files that were found:")
        print(path_dir)
        print("We require the following key files for a BRAKER run: braker.aa, braker.gtf, hintsfile.gff, genome.fa, augustus.hints.aa, augustus.hints.gtf, genemark.gtf, hintsfile.gff")
        print("We require the following key files for a GALBA run: galba.aa, galba.gtf, miniprot_trainingGenes.gtf, genome.fa, hintsfile.gff")
        sys.exit(1)


def check_dir(dir):
    """
    Check if the specified directory exists.

    Returns:
        str: Path to the directory if it exists.

    Raises:
        SystemExit: If the directory is not found.

    """
    if os.path.isdir(dir):
        return dir
    else:
        print("ERROR: Directory not found: " + dir)
        sys.exit(1)

def main():
    """
    Execute workflow
    """
    
    # Step 1: Find all input files
    file_paths = find_input_files(args)

    # Step 2: find out which GeneMark was used: ETP or EP or ET or ES or none
    file_paths["genemark_gtf"], file_paths["training_gtf"] = find_genemark_gtf(args.input_dir)

    # Step 3: determine whether BRAKER was run or GALBA, and whether files are complete
    w_mode = determine_mode(file_paths)

    # Step 4: Check if all dependencies are available
    args.tsebra = check_binary(args.tsebra, "tsebra.py")
    args.getanno = check_binary(args.getanno, "getAnnoFastaFromJoingenes.py")
    args.compleasm_bin = check_binary(args.compleasm_bin, "compleasm.py")

    # Step 5: Check if temporary directory exists
    # if not, create it
    if not os.path.exists(args.tmp_dir):
        try:
            os.makedirs(args.tmp_dir)
        except OSError:
            print("ERROR: Creation of the directory %s failed" % args.tmp_dir)
            sys.exit(1)

    if w_mode == "BRAKER":
        # Step 6: Create protein sequene file for GeneMark gtf file
        file_paths["genemark_aa"] = run_getanno(args.getanno, args.genome, file_paths["genemark_gtf"], args.tmp_dir)
    elif w_mode == "GALBA":
        file_paths["miniprot_trainingGenes_aa"] = run_getanno(args.getanno, args.genome, file_paths["miniprot_trainingGenes.gtf"], args.tmp_dir)

    # Step 7: run compleasm
    if w_mode == "BRAKER":
        protein_file_list = [file_paths["genemark_aa"], file_paths["braker_aa"], file_paths["augustus_aa"]]
    elif w_mode == "GALBA":
        protein_file_list = [file_paths["miniprot_trainingGenes_aa"], file_paths["galba_aa"]]

    compleasm_out_dict = run_compleasm(protein_file_list, args.threads, args.busco_db, args.tmp_dir)

    # Step 8: parse and compare the number of missing BUSCOs

    if w_mode == "BRAKER":
        genemark_missing = parse_compleasm(compleasm_out_dict["genemark"])
        augustus_missing = parse_compleasm(compleasm_out_dict["augustus"])
        braker_missing = parse_compleasm(compleasm_out_dict["braker"])
    elif w_mode == "GALBA":
        galba_missing = parse_compleasm(compleasm_out_dict["galba"])
        miniprot_trainingGenes_missing = parse_compleasm(compleasm_out_dict["miniprot_trainingGenes"])

    
    # Step 9: Decide whether the provided final gene set is good enough
    if w_mode == "BRAKER":
        print("BRAKER is missing " + str(braker_missing) + " BUSCOs.")
        print("GeneMark is missing " + str(genemark_missing) + " BUSCOs.")
        print("Augustus is missing " + str(augustus_missing) + " BUSCOs.")
        if braker_missing <= 5 and braker_missing <= augustus_missing and braker_missing <= genemark_missing:
            print("The BRAKER gene set " + file_paths["braker_gtf"] + " is the best one. It lacks " + str(braker_missing) + "% BUSCOs.")
            sys.exit(0)
        elif augustus_missing >= genemark_missing:
            tsebra_force = file_paths["genemark_gtf"]
            not_tsebra_force = file_paths["augustus_gtf"]
        else:
            tsebra_force = file_paths["augustus_gtf"]
            not_tsebra_force = file_paths["genemark_gtf"]
    elif w_mode == "GALBA":
        print("GALBA is missing " + str(galba_missing) + " BUSCOs.")
        print("miniprot_trainingGenes is missing " + str(miniprot_trainingGenes_missing) + " BUSCOs.")

        if galba_missing <= miniprot_trainingGenes_missing:
            print("The GALBA gene set " + file_paths["galba_gtf"] + " is the best one. It lacks " + str(galba_missing) + "% BUSCOs.")
            sys.exit(0)
        else:
            tsebra_force = file_paths["miniprot_trainingGenes_gtf"]
            not_tsebra_force = file_paths["galba_gtf"]

    # Step 10: Run TSEBRA and enforce the best gene set
    if w_mode == "BRAKER":
        if file_paths['training_gtf']:
            tsebra_cmd = [args.tsebra, "-k", file_paths["training_gtf"] + "," + tsebra_force,
                      "-g", not_tsebra_force, "-e", file_paths["hints"], "-o", args.tmp_dir + "/better.gtf"]
        else:
            print("Warning: No training.gtf file found. TSEBRA will be run without it, you may be using an older version of BRAKER.")
            tsebra_cmd = [args.tsebra, "-k", tsebra_force,
                      "-g", not_tsebra_force, "-e", file_paths["hints"], "-o", args.tmp_dir + "/better.gtf"]
    elif w_mode == "GALBA":
        tsebra_cmd = [args.tsebra, "-k", tsebra_force,
                      "-g", not_tsebra_force, "-e", file_paths["hints"], "-o", args.tmp_dir + "/better.gtf"]
    run_simple_process(tsebra_cmd)

    # Step 11: generate protein sequence file for the new BRAKER gene set
    better_gtf = check_file(args.tmp_dir + "/better.gtf")
    bb_aa = run_getanno(args.getanno, args.genome, better_gtf, args.tmp_dir)

    # Step 12: Run compleasm on the new gene set
    secondary_compleasm_out = run_compleasm([bb_aa], args.threads, args.busco_db, args.tmp_dir)

    # Step 13: parse compleasm output and report numbers
    better_missing = parse_compleasm(list(secondary_compleasm_out.values())[0])
    if w_mode == "BRAKER":
        if better_missing < braker_missing:
            print("The new best BRAKER gene set is " + better_gtf + ".")
            print("It is missing " + str(better_missing) + "% BUSCOs.")
        else:
            # if the new gene set is not superior, produce an output that tells the user what
            # of the previously existing gene sets had the lowest percentage of missing BUSCOs
            print("WARNING: The new BRAKER gene set is not better than the original one!")
            print("The best gene set produced by the original BRAKER run is:")
            gene_sets = [file_paths["braker_gtf"], file_paths["genemark_gtf"], file_paths["augustus_gtf"]]
            # Create a dictionary to map gene set names to their respective missing values
            gene_set_values = {
                file_paths["braker_gtf"]: braker_missing,
                file_paths["genemark_gtf"]: genemark_missing,
                file_paths["augustus_gtf"]: augustus_missing
            }
            # Find the minimum value among the three
            min_value = min(gene_set_values.values())
            # Find the gene set name associated with the minimum value
            min_gene_set = None
            for gene_set, value in gene_set_values.items():
                if value == min_value:
                    min_gene_set = gene_set
                    break
            # Print the name of the gene set with the lowest number of missing BUSCOs
            print(min_gene_set)
    elif w_mode == "GALBA":
        if better_missing < galba_missing:
            print("The new best GALBA gene set is " + better_gtf + ".")
        else:
            # if the new gene set is not superior, produce an output that tells the user what
            # of the previously existing gene sets had the lowest percentage of missing BUSCOs
            print("WARNING: The new GALBA gene set is not better than the original one :-(")
            print("The best gene set produced by the original GALBA run is:")
            gene_sets = [file_paths["galba_gtf"], file_paths["miniprot_trainingGenes_gtf"]]
            # Create a dictionary to map gene set names to their respective missing values
            gene_set_values = {
                file_paths["galba_gtf"]: galba_missing,
                file_paths["miniprot_trainingGenes_gtf"]: miniprot_trainingGenes_missing
            }
            # Find the minimum value among the three
            min_value = min(gene_set_values.values())
            # Find the gene set name associated with the minimum value
            min_gene_set = None
            for gene_set, value in gene_set_values.items():
                if value == min_value:
                    min_gene_set = gene_set
                    break
            # Print the name of the gene set with the lowest number of missing BUSCOs
            print(min_gene_set) 

if __name__ == "__main__":
    main()
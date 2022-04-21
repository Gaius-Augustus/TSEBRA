#!/usr/bin/env python3
# ==============================================================
# author: Lars Gabriel
#
# evdence.py: Handles the extrinsic evidence from the hintfiles
# ==============================================================
import csv

class NotGtfFormat(Exception):
    pass

class AttributeMissing(Exception):
    pass

class Hint:
    """
        Class handling the data structures and methods for a hint
    """
    def __init__(self, line):
        """
            Create a hint from a gff line. The line has to include 'src=' as
            an attribute in the last column. Only introns, start/stop codons
            are used.

            Args:
                line (list(str)): GFF line for one hint from extrinsic evidence.
        """
        if not len(line) == 9:
            raise NotGtfFormat('File not in gtf Format. Error at line: {}'.format(line))
        self.chr, self.source_program, self.type, self.start, self.end, \
            self.score, self.strand, self.phase, attribute = line
        self.start = int(self.start)
        self.end = int(self.end)

        try:
            self.src = attribute.split('src=')[1].split(';')[0]
        except IndexError:
            raise AttributeMissing('Source of Hint is missing in line {}.'.format(line))

        self.mult = ''
        if 'mult=' in attribute:
            self.mult = attribute.split('mult=')[1].split(';')[0]
        else:
            self.mult= '1'
        if 'al_score=' in attribute:
            self.al_score = attribute.split('al_score=')[1].split(';')[0]
            self.mult = str(int(self.mult) * float(self.al_score))
        self.grp = ''
        if 'grp=' in attribute:
            self.grp = attribute.split('grp=')[1].split(';')[0]
        self.pri = ''
        if 'pri=' in attribute:
            self.pri = attribute.split('pri=')[1].split(';')[0]

        if self.type == 'stop_codon':
            self.type = 'stop'
        elif self.type == 'start_codon':
            self.type = 'start'

    def hint2list(self):
        """
            Returns:
                line (list(str)): GFF line for the hint.
        """
        attribute = ['src=' + self.src]
        if int(self.mult) > 1:
            attribute.append('mult={}'.format(self.mult))
        if self.pri:
            attribute.append('pri={}'.format(self.pri))
        return [self.chr, self.source_program, self.type, self.start, self.end, \
            self.score, self.strand, self.phase, ';'.join(attribute)]

class Hintfile:
    """
        Class handling the data structures and methods for a hintfile
    """
    def __init__(self, path):
        """
            Args:
                path (str): Path to the hintfile.
        """
        # dictonary containing evidence
        # self.hints[chromosom_id] = [Hints()]
        self.hints = {}
        self.src = set()
        self.read_file(path)

    def read_file(self, path):
        """
            Read a gff file with intron or start/stop codon hints
            and create a dict of Hints.
        """
        #
        with open(path, 'r') as file:
            hints_csv = csv.reader(file, delimiter='\t')
            for line in hints_csv:
                if line[0][0] == '#':
                    continue
                new_hint = Hint(line)
                if not new_hint.chr in self.hints.keys():
                    self.hints.update({new_hint.chr : []})
                self.hints[new_hint.chr].append(new_hint)
                self.src.add(new_hint.src)

class Evidence:
    """
        Class handling the data structures and methods for extrinsic evidence
        from one or more hintfiles.
    """
    def __init__(self):
        # hint_keys[chr][start_end_type_strand][src] = multiplicity
        self.hint_keys = {}
        self.src = set()
        # coord2group[type][coord_key] = group_name
        self.coord2group = {}

        # group_chain[group_name][intron or CDSpart] = [hint_keys]
        self.group_chains = {}

    def add_hintfile(self, path_to_hintfile):
        """
            Read hintfile
        """
        # read hintfile
        hintfile = Hintfile(path_to_hintfile)
        self.src = self.src.union(hintfile.src)
        for chr in hintfile.hints.keys():
            if chr not in self.hint_keys.keys():
                self.hint_keys.update({chr : {}})
            for hint in hintfile.hints[chr]:
                new_key = '{}_{}_{}_{}'.format(hint.start, hint.end, \
                    hint.type, hint.strand)
                if not new_key in self.hint_keys[chr].keys():
                    self.hint_keys[chr].update({new_key : {}})
                if not hint.src in self.hint_keys[chr][new_key].keys():
                    self.hint_keys[chr][new_key].update({hint.src : 0})
                self.hint_keys[chr][new_key][hint.src] += float(hint.mult)
                if hint.src == 'C':
                    new_key = chr + '_' + new_key
                    if not hint.type in self.coord2group: self.coord2group[type] = {}
                    if new_key not in self.coord2group[type]:
                        self.coord2group[type][new_key] = []
                    if not hint.grp in self.coord2group[type][new_key]:
                        self.coord2group[type][new_key].append(hint.grp)
                    if not hint.grp in self.group_chains:
                        self.group_chains.update({hint.grp :
                            {'start' : [], 'stop' : [],
                            'intron' : [], 'CDSpart' : []}})
                    if hint.type == 'CDSpart':
                        new_key = [hint.start, hint.end]
                    if not new_key in self.group_chains[hint.grp][hint.type]:
                        self.group_chains[hint.grp][hint.type].append(new_key)

    def get_hint(self, chr, start, end, type, strand):
        if type == 'start_codon':
            type = 'start'
        elif type == 'stop_codon':
            type = 'stop'
        key = '{}_{}_{}_{}'.format(start, end, type, strand)
        if chr in self.hint_keys.keys():
            if key in self.hint_keys[chr].keys():
                return self.hint_keys[chr][key]
        return {}

    def get_matching_chains(self, chr, coords, strand):
        #chain is matching if eiter start/stop-codon or at least one intron of tx matches
        matches = []
        for type in ['intron', 'start_codon', 'stop_codon']:
            for c in coords[type]:
                key = f"{chr}_{c[0]}_{c[1]}_{type}_{strand}"
                if key in self.coord2group[type]:
                    matches += self.coord2group[type]
        return list(set(matches))

    def get_best_chain(self, chr, coords, type, strand):
        groups = set()
        keys = []
        for c in coords['intron']:
            key = f"{chr}_{c[0]}_{c[1]}_intron_{strand}"
            keys.append(key)
            if key in self.coord2group['intron']:
                groups.update(set(self.coord2group['intron'][key]))
        #if groups:
            #print(keys)
            #print(groups)
        best_chain_fraction = 0
        best_self_fraction = 0
        if type == 'intron':
            for g in groups:
                chain_match = len(set(self.group_chains[g][type]).intersection(keys))
                best_chain_fraction = max(best_chain_fraction,  chain_match / len(self.group_chains[g][type]))
                best_self_fraction = max(best_self_fraction,  chain_match / len(keys))
                #print(chain_match)
                #print(best_chain_fraction, best_self_fraction)
        elif type == 'CDSpart':
            for g in groups:
                chain = sorted(self.group_chains[g][type])
                chain_match = 0
                total_length_coords = 0
                total_length_chain = sum([c[1]-c[0]+1 for c in chain])
                j=0
                for c1 in coords['CDS']:
                    total_length_coords += c1[1] - c1[0] + 1
                    for i in range(j, len(chain)):
                        if chain[i][1] > c1[1]:
                            j = i
                            break
                        chain_match += max(0, min(chain[i][1], c1[1]) - max(chain[i][0], c1[0]) + 1)
                best_chain_fraction = max(best_chain_fraction,  chain_match /total_length_chain)
                best_self_fraction = max(best_self_fraction,  chain_match /total_length_coords)

        return best_chain_fraction, best_self_fraction

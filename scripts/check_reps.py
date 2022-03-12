#!/bin/env python

import sys
import os, glob
import re

fname_pattern = '*.pkl'
rep_pattern = 'rep(\d+)'

def condensed(reps):

    def process_range(start, stop=None):
        if stop is None or start == stop:
            return start
        else:
            return f'{start}-{stop}'

    reps = list(sorted(reps))
    max_rep = max(reps)

    condensed_reps = []
    range_start = None
    prev_rep = float('-inf')
    for rep in reps:

        # start of a new range
        if rep != prev_rep + 1:

            # process previous range
            if prev_rep >= 0:
                condensed_reps.append(process_range(range_start, prev_rep))
            # handle last rep
            if rep == max_rep:
                condensed_reps.append(rep)

            # initialize new range
            range_start = rep

        # if last rep is part of a range
        elif rep == max_rep:
            condensed_reps.append(process_range(range_start, rep))

        prev_rep = rep

    return condensed_reps

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} dpath_results n_reps')
        sys.exit(1)

    dpath_results = sys.argv[1]
    n_reps = int(sys.argv[2])
    
    expected_reps = {i+1 for i in range(n_reps)}

    print('--------------------')
    print(f'Searching for files in {dpath_results} matching pattern {fname_pattern}')
    filenames = glob.glob(os.path.join(dpath_results, fname_pattern))
    
    print(f'Found {len(filenames)} files out of expected {n_reps}')

    re_rep = re.compile(rep_pattern)

    actual_reps = set()
    for filename in filenames:

        match = re_rep.search(filename)

        if not match:
            print(f'ERROR: no match for file {filename}')
            sys.exit(1)

        if len(match.groups()) > 1:
            print(f'ERROR: multiple matched groups for file {filename} ({match.groups()})')
            sys.exit(1)

        actual_reps.add(int(match.groups()[0]))
        
    missing_reps = list(expected_reps - actual_reps)

    if len(missing_reps) > 0:
        print(f'{len(missing_reps)} missing reps')
        print('--------------------')
        print(','.join([str(rep) for rep in condensed(missing_reps)]))
        print('--------------------')
    else:
        print('No missing rep')

    sys.exit(0)

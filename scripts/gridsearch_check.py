#!/bin/env python

import sys, os
import re
import pickle
import itertools

i_split = 6 # make sure this is the same as in slurm script

n_components1_min = 10 #  10, 140, 270
n_components1_max = 380 # 130, 260, 380
n_components1_step = 10

n_components2_min = 10
n_components2_max = 380
n_components2_step = 10

fpath_slurm_script = '/home/mwang8/ukbb-cca/scripts/slurm_job_python.sh'
max_array_jobs = 10

split_pattern = f'split{i_split}'
dname_pattern = 'cv_(\d+)_(\d+)'
rep_pattern = 'rep(\d+)'
fix_flag = '--fix'

load_pickle = False

def print_usage_and_exit(exit_code=1):
    print(f'Usage: {sys.argv[0]} dpath_gridsearch n_reps [{fix_flag}]')
    sys.exit(exit_code)

if __name__ == '__main__':

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print_usage_and_exit()

    dpath_gridsearch = sys.argv[1]
    n_reps = int(sys.argv[2])
    try:
        if sys.argv[3] != fix_flag:
            print_usage_and_exit()
        fix = True
    except IndexError:
        fix = False

    re_split = re.compile(split_pattern)
    re_dname = re.compile(dname_pattern)
    re_rep = re.compile(rep_pattern)

    dnames_params = os.listdir(dpath_gridsearch)

    expected_components_pairs = set()
    for n_components1, n_components2 in itertools.product(
                range(n_components1_min, n_components1_max+1, n_components1_step),
                range(n_components2_min, n_components2_max+1, n_components2_step),
            ):
        expected_components_pairs.add(f'{n_components1} {n_components2}')

    found_components_pairs = set()

    rerun_info = []
    for dname_params in dnames_params:

        # parse n_components
        match = re_dname.search(dname_params)
        if len(match.groups()) != 2:
            print(f'ERROR: {dname_params} did not get 2 matched groups')
            sys.exit(1)
        n_components1, n_components2 = [int(g) for g in match.groups()]
        str_components = f'{n_components1} {n_components2}'

        if not re_split.search(dname_params):
            continue

        if (n_components1 < n_components1_min 
                or n_components1 > n_components1_max 
                or n_components2 < n_components2_min 
                or n_components2 > n_components2_max):
            continue

        found_components_pairs.add(str_components)

        # get rep (pickle) files
        fnames_reps = os.listdir(os.path.join(dpath_gridsearch, dname_params))

        # if number of reps is incorrect
        # if len(fnames_reps) != n_reps:

        # figure out which reps are missing
        expected_reps = set([i+1 for i in range(n_reps)])
        found_reps = set()
        for fname_reps in fnames_reps:

            try:
                with open(os.path.join(dpath_gridsearch, dname_params, fname_reps), 'rb') as file_rep:
                    # try loading the file (slow)
                    if load_pickle:
                        pickle.load(file_rep)
            except:
                continue

            match = re_rep.search(fname_reps)
            if len(match.groups()) > 1:
                print(f'ERROR: {fname_reps} matched more than 1 group')
                sys.exit(1)
            found_rep = int(match.groups()[0])
            found_reps.add(found_rep)

        missing_reps = expected_reps - found_reps
        if len(missing_reps) > 0:
            str_missing_reps = ','.join([str(i) for i in missing_reps])

            rerun_info.append((str_components, str_missing_reps))
            print(f'{str_components}: {str_missing_reps}')

    missing_component_pairs = expected_components_pairs - found_components_pairs
    for str_components in missing_component_pairs:
        str_missing_reps = ','.join([str(i+1) for i in range(n_reps)])
        rerun_info.append((str_components, str_missing_reps))
        print(f'{str_components}: {str_missing_reps}')

    if fix:
        # pass
        for str_components, str_missing_reps in rerun_info:
            # export N_PCS=str_components
            os.environ['N_PCS'] = str_components
            # submit job array
            os.system(f'mysbatcharray {fpath_slurm_script} {str_missing_reps} {max_array_jobs}')

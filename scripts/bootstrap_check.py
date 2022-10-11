#!/bin/env python

import sys, os
import re
import pickle

n_components1 = 300
n_components2 = 300

fpath_slurm_script = '/home/mwang8/ukbb-cca/scripts/slurm_job_python.sh'
max_array_jobs = 100

bootstrap_rep_pattern = 'bootstrap(\d+)'
dname_pattern = f'cv_{n_components1}_{n_components2}_{bootstrap_rep_pattern}'

fix_flag = '--fix'

def print_usage_and_exit(exit_code=1):
    print(f'Usage: {sys.argv[0]} dpath_bootstrap n_splits n_reps [{fix_flag}]')
    sys.exit(exit_code)

if __name__ == '__main__':

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print_usage_and_exit()

    dpath_bootstrap = sys.argv[1]
    n_splits = int(sys.argv[2])
    n_reps = int(sys.argv[3])
    try:
        if sys.argv[4] != fix_flag:
            print_usage_and_exit()
        fix = True
    except IndexError:
        fix = False

    str_components = f'{n_components1} {n_components2}'

    re_bootstrap = re.compile(bootstrap_rep_pattern)

    dnames_params = os.listdir(dpath_bootstrap)

    rerun_info = []
    for i_split in range(n_splits):

        dname_pattern_with_split = f'{dname_pattern}*_split{i_split}'
        re_dname = re.compile(dname_pattern_with_split)

        missing_bootstrap_reps = [] # bootstrap cv sub-directories with missing reps 

        for dname_params in dnames_params:

            # only get the directories with the correct number of components
            dname_match = re_dname.search(dname_params)
            if dname_match:
                
                # get rep (pickle) files
                fnames_reps = os.listdir(os.path.join(dpath_bootstrap, dname_params))

                # if number of reps is incorrect
                if len(fnames_reps) != n_reps:

                    # get i_bootstrap
                    bootstrap_match = re_bootstrap.search(dname_params)
                    if len(bootstrap_match.groups()) != 1:
                        print(f'ERROR: bad i_bootstrap matches for {dname_params} ({bootstrap_match.groups()})')
                    i_bootstrap = bootstrap_match.groups()[0]

                    missing_bootstrap_reps.append(i_bootstrap)

        if len(missing_bootstrap_reps) > 0:  
            str_missing_bootstrap_reps = ','.join([str(i) for i in missing_bootstrap_reps])
            rerun_info.append((i_split, str_missing_bootstrap_reps))
            print(f'Split {i_split}: {str_missing_bootstrap_reps}')

    if fix:
        # pass
        print('--------------------')
        for i_rep in range(n_reps):
            print(f'Fixing bootstrap rep {i_rep}')
            print('--------------------')
            for i_split, str_missing_bootstrap_reps in rerun_info:

                # environment variables needed in slurm script
                os.environ['N_PCS'] = str_components
                os.environ['I_REP'] = str(i_rep)
                os.environ['I_SPLIT'] = str(i_split)

                # submit job array
                # IMPORTANT: make sure slurm script has correct commands (bootstrap cv)
                os.system(f'mysbatcharray {fpath_slurm_script} {str_missing_bootstrap_reps} {max_array_jobs}')
            print('--------------------')

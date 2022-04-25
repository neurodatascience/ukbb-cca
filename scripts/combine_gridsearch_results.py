
import sys, os
import re
import pickle
import pandas as pd
from paths import DPATHS

# 10-380
n_components1_min = 10
n_components1_max = 380

# 10-380
n_components2_min = 10
n_components2_max = 380

dname_pattern = 'cv_(\d+)_(\d+)'
rep_pattern = 'rep(\d+)'

fpath_out = os.path.join(DPATHS['cca'], 'gridsearch_results.pkl')

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} dpath_gridsearch')
        sys.exit(1)

    dpath_gridsearch = sys.argv[1]

    re_dname = re.compile(dname_pattern)
    re_rep = re.compile(rep_pattern)

    dnames_params = os.listdir(dpath_gridsearch)

    data_for_df = []
    for dname_params in dnames_params:

        # parse n_components
        match = re_dname.search(dname_params)
        if len(match.groups()) != 2:
            print(f'ERROR: {dname_params} did not get 2 matched groups')
            sys.exit(1)
        n_components1, n_components2 = [int(g) for g in match.groups()]

        if (n_components1 < n_components1_min 
                or n_components1 > n_components1_max 
                or n_components2 < n_components2_min 
                or n_components2 > n_components2_max):
            continue

        # get rep (pickle) files
        fnames_reps = os.listdir(os.path.join(dpath_gridsearch, dname_params))

        for fname_reps in fnames_reps:
            match = re_rep.search(fname_reps)
            if len(match.groups()) > 1:
                print(f'ERROR: {fname_reps} matched more than 1 group')
                sys.exit(1)
            i_rep = int(match.groups()[0])

            with open(os.path.join(dpath_gridsearch, dname_params, fname_reps), 'rb') as file_results:
                try:
                    results = pickle.load(file_results)
                except EOFError:
                    print(f'ERROR: EOFError for file {os.path.join(dname_params, fname_reps)}')
                    continue
                    # sys.exit(1)
                except pickle.UnpicklingError: 
                    print(f'ERROR: pickle.UnpicklingError for file {os.path.join(dname_params, fname_reps)}')
                    continue

            n_folds = results['n_folds']
            dataset_names = results['dataset_names']
            correlations_val = results['correlations_val']
            i_split = results['i_split']

            # sanity check
            if results['n_components_all'] != [n_components1, n_components2]:
                print(f'ERROR: mismatch between n_components_all and parsed n_components variables')

            for i_fold in range(n_folds):
                data_for_df.append({
                    'n_components1': n_components1,
                    'n_components2': n_components2,
                    'i_rep': i_rep,
                    'i_fold': i_fold,
                    'corr_CA1': correlations_val[i_fold, 0],
                    'corrs': correlations_val[i_fold, :],
                    'i_split': i_split,
                })

    # df_gridsearch = pd.DataFrame(data_for_df)
    # print(df_gridsearch)

    with open(fpath_out, 'wb') as file_out:
        pickle.dump({'data_for_df': data_for_df, 'dataset_names': dataset_names}, file_out)
        print(f'Saved pickle file: {fpath_out}')

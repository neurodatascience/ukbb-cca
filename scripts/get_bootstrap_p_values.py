
import pickle, os, sys
import numpy as np
from src.cca_utils import cca_score
from src.utils import add_suffix
from paths import DPATHS

np.set_printoptions(precision=4, suppress=True, linewidth=100, sign=' ')

# TODO compute using averaged data for all splits
# combine data from all splits (CAs, PCs): compute mean and std
# maybe keep only 1 ensemble method (nanmean)
# check bootstrap 

# dpath_results = DPATHS['central_tendencies']
dpath_out = DPATHS['central_tendencies']

keys_corrs = ['corrs_learn', 'corrs_test']
keys_loadings = ['loadings_learn', 'loadings_test']
alpha = 0.05 # for bootstrap confidence interval

n_CAs_to_print = 10
n_features_to_print = 10

ensemble_method = 'nanmean'

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} fpath_cca dpath_bootstrap')
        sys.exit(1)
    fpath_cca = sys.argv[1]
    dpath_bootstrap = sys.argv[2]
    # fname_prefix = sys.argv[1]
    # fname_results = f'{fname_prefix}_central_tendencies.pkl'
    # fpath_results = os.path.join(dpath_cv, fname_results)

    # load CCA results
    with open(fpath_cca, 'rb') as file_cca:
        cca_results = pickle.load(file_cca)
        n_datasets = cca_results['n_datasets']
        dataset_names = cca_results['dataset_names']

        # compute correlations
        cca_results['corrs_learn'] = cca_score(cca_results['CAs_learn'][ensemble_method])
        cca_results['corrs_test'] = cca_score(cca_results['CAs_test'][ensemble_method])

    # load/merge bootstrap results
    bootstrap_results = {}
    for key_corrs in keys_corrs:
        bootstrap_results[key_corrs] = []
    for key_loadings in keys_loadings:
        bootstrap_results[key_loadings] = [[] for _ in range(n_datasets)]
    # loop over all files
    fnames_bootstrap = os.listdir(dpath_bootstrap)
    for fname_bootstrap in fnames_bootstrap:

        # load file
        with open(os.path.join(dpath_bootstrap, fname_bootstrap), 'rb') as file_bootstrap:
            bootstrap_results_partial = pickle.load(file_bootstrap)
        # append
        for key_corrs in keys_corrs:
            bootstrap_results[key_corrs].append(bootstrap_results_partial[key_corrs][ensemble_method])
        for key_loadings in keys_loadings:
            for i_dataset in range(n_datasets):
                bootstrap_results[key_loadings][i_dataset].append(bootstrap_results_partial[key_loadings][ensemble_method][i_dataset])

        # for key in bootstrap_results.keys():
        #     bootstrap_results[key].append(bootstrap_results_partial[key][ensemble_method])

    # convert to np array
    for key_corrs in keys_corrs:
        bootstrap_results[key_corrs] = np.array(bootstrap_results[key_corrs])
    for key_loadings in keys_loadings:
        for i_dataset in range(n_datasets):
            bootstrap_results[key_loadings][i_dataset] = np.array(bootstrap_results[key_loadings][i_dataset])

    print('--------------------')
    for key_corrs in keys_corrs:
        key_p = f'{key_corrs}_p_values'
        key_ci = f'{key_corrs}_ci'
        corrs_bootstrap = bootstrap_results[key_corrs]
        corrs_cca = cca_results[key_corrs][np.newaxis, :]
        p_values = np.mean(corrs_bootstrap > corrs_cca, axis=0)

        print(f'p_values {key_corrs}')
        print(f'{p_values[:n_CAs_to_print]}')
        print('--------------------')

        cca_results[key_p] = p_values

        # bootstrap null distribution confidence interval
        ci = np.quantile(corrs_bootstrap, [alpha, 1-alpha], axis=0)
        cca_results[key_ci] = ci

    for key_loadings in keys_loadings:
        key_p = f'{key_loadings}_p_values'
        key_ci = f'{key_loadings}_ci'
        cca_results[key_p] = []
        cca_results[key_ci] = []
        for i_dataset in range(n_datasets):
            loadings_bootstrap = bootstrap_results[key_loadings][i_dataset]
            loadings_cca = cca_results[key_loadings][ensemble_method][i_dataset][np.newaxis, :]
            p_values_tail1 = np.mean(loadings_bootstrap > loadings_cca, axis=0)
            p_values_tail2 = np.mean(loadings_bootstrap < loadings_cca, axis=0)
            p_values = np.min([p_values_tail1, p_values_tail2], axis=0) * 2

            print(f'p_values {key_loadings} ({dataset_names[i_dataset]})')
            print(f'{p_values[:n_features_to_print, :n_CAs_to_print]}')
            print('--------------------')

            cca_results[key_p].append(p_values)

            # bootstrap null distribution confidence interval
            ci = np.quantile(loadings_bootstrap, [alpha, 1-alpha], axis=0)
            cca_results[key_ci] = ci

    cca_results['bootstrap_ci_alpha'] = alpha

    fpath_out = add_suffix(fpath_cca, 'p_values')
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(cca_results, file_out)
    print(f'CCA results with p-values saved to {fpath_out}')

    # # add 'n_merged' key (overwrites if already exists)
    # merged_results['n_merged'] = len(fnames)

    # fpath_out = os.path.join(dpath_out, 'merged_results.pkl')
    # with open(fpath_out, 'wb') as file_out:
    #     pickle.dump(merged_results, file_out)

    # print(f'Merged results saved to {fpath_out}')

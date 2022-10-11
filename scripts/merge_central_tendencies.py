
import pickle, os, sys
import numpy as np
from src.cca_utils import cca_score
from src.utils import add_suffix, sublist_append, sublist_mean, sublist_std, traverse_nested_list, traverse_nested_dict, rotate_to_match
from paths import DPATHS

np.set_printoptions(precision=4, suppress=True, linewidth=100, sign=' ')

dpath_out = DPATHS['central_tendencies']
n_splits = 10

keys_loadings = ['loadings_learn', 'loadings_test']
keys_PCs = ['PCs_learn', 'PCs_test']
keys_CAs = ['CAs_learn', 'CAs_test']
keys_corrs = ['corrs_learn', 'corrs_test']
keys_holdouts = ['holdout_learn', 'holdout_test']

alpha = 0.05 # for bootstrap confidence interval

n_CAs_to_print = 10
n_features_to_print = 10

ensemble_method = 'nanmean'

# color_learn = '#009193'
# color_test = '#ED7D31'
# label_holdout = f'$\mathregular{{r_{{age}}}}$'

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} cv_prefix_without_split dpath_bootstrap')
        sys.exit(1)
    cv_prefix_without_split = sys.argv[1]
    dpath_bootstrap = sys.argv[2]

    for i_split in range(n_splits):

        # load CCA results
        fpath_split = f'{cv_prefix_without_split}_split{i_split}_central_tendencies.pkl'
        with open(fpath_split, 'rb') as file_cca:

            cca_results_partial = pickle.load(file_cca)

            if i_split == 0:
                n_datasets = cca_results_partial['n_datasets']
                dataset_names = cca_results_partial['dataset_names']

                cca_results = {'n_datasets': n_datasets, 'dataset_names': dataset_names}
                for key in keys_CAs + keys_PCs + keys_loadings:
                    cca_results[key] = [[] for _ in range(n_datasets)]
                for key in keys_corrs + keys_holdouts:
                    cca_results[key] = []

                # TODO REMOVE
                #continue

                # CAs_learn = [[] for _ in range(n_datasets)]
                # CAs_test = [[] for _ in range(n_datasets)]
                # PCs_learn = [[] for _ in range(n_datasets)]
                # PCs_test = [[] for _ in range(n_datasets)]
                # loadings_learn = [[] for _ in range(n_datasets)]
                # loadings_test = [[] for _ in range(n_datasets)]
                # corrs_learn = []
                # corrs_test = []

            print(f'----- split {i_split} -----')
            print(f'all keys:{cca_results_partial.keys()}')
            for key in keys_CAs + keys_PCs + keys_loadings:
                # print(f'key: {key}, second-level keys: {cca_results_partial[key].keys()}')
                #to_append = cca_results_partial[key][ensemble_method]
                #to_append_rotated = traverse_nested_list(to_append, fn=(lambda x: rotate_to_match(x, weights_ref)))
                sublist_append(cca_results[key], cca_results_partial[key][ensemble_method])
            for key_corrs, key_CAs in zip(keys_corrs, keys_CAs):
                cca_results[key_corrs].append(cca_score(cca_results_partial[key_CAs][ensemble_method]))
            for key in keys_holdouts:
                cca_results[key].append(cca_results_partial[key])

            # sublist_append(CAs_learn, cca_results['CAs_learn'][ensemble_method])
            # sublist_append(CAs_test, cca_results['CAs_test'][ensemble_method])
            # sublist_append(PCs_learn, cca_results['PCs_learn'][ensemble_method])
            # sublist_append(PCs_test, cca_results['PCs_test'][ensemble_method])
            # sublist_append(loadings_learn, cca_results['loadings_learn'][ensemble_method])
            # sublist_append(loadings_test, cca_results['loadings_test'][ensemble_method])

            # corrs_learn.append(cca_score(cca_results['CAs_learn'][ensemble_method]))
            # corrs_test.append(cca_score(cca_results['CAs_test'][ensemble_method]))

    # compute mean/std
    cca_results_summary = traverse_nested_dict(cca_results, fn=(lambda val: {}))
    for key in cca_results_summary.keys():
        if key in keys_loadings:#keys_CAs + keys_PCs + keys_loadings:
            for stat_name, stat_fn in {'mean': sublist_mean, 'std': sublist_std}.items():
                # rotate first
                rotated = traverse_nested_list(cca_results[key], (lambda l: np.array([rotate_to_match(li, l[0]) for li in l])))
                cca_results_summary[key][stat_name] = stat_fn(rotated)
        elif key in keys_corrs:
            for stat_name, stat_fn in {'mean': (lambda x: np.nanmean(x, axis=0)), 'std': (lambda x: np.nanstd(x, axis=0))}.items():
                cca_results_summary[key][stat_name] = stat_fn(cca_results[key])
        else:
            # save dataset info and CAs/PCs as-is
            cca_results_summary[key] = cca_results[key]

    # CAs_learn_mean = sublist_mean(CAs_learn)
    # CAs_learn_std = sublist_std(CAs_learn)
    # CAs_test_mean = sublist_mean(CAs_test)
    # CAs_test_std = sublist_std(CAs_test)
    # PCs_learn_mean = sublist_mean(PCs_learn)
    # PCs_learn_std = sublist_std(PCs_learn)
    # PCs_test_mean = sublist_mean(PCs_test)
    # PCs_test_std = sublist_std(PCs_test)
    # loadings_learn_mean = sublist_mean(loadings_learn)
    # loadings_learn_std = sublist_std(loadings_learn)
    # loadings_test_mean = sublist_mean(loadings_test)
    # loadings_test_std = sublist_std(loadings_test)

    # corrs_learn_mean = np.nanmean(corrs_learn)
    # corrs_learn_std = np.nanstd(corrs_learn)
    # corrs_test_mean = np.nanmean(corrs_test)
    # corrs_test_std = np.nanstd(corrs_test)

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
        # key_p = f'{key_corrs}_p_values'
        # key_ci = f'{key_corrs}_ci'
        corrs_bootstrap = bootstrap_results[key_corrs]
        corrs_cca = cca_results_summary[key_corrs]['mean'][np.newaxis, :]
        p_values = np.mean(corrs_bootstrap > corrs_cca, axis=0)

        print(f'p_values {key_corrs}')
        print(f'{p_values[:n_CAs_to_print]}')
        print('--------------------')

        cca_results_summary[key_corrs]['p_values'] = p_values

        # bootstrap null distribution confidence interval
        ci = np.quantile(corrs_bootstrap, [alpha, 1-alpha], axis=0)
        cca_results_summary[key_corrs]['ci'] = ci

    for key_loadings in keys_loadings:
        # key_p = f'{key_loadings}_p_values'
        # key_ci = f'{key_loadings}_ci'
        # cca_results[key_p] = []
        # cca_results[key_ci] = []
        cca_results_summary[key_loadings]['p_values'] = []
        cca_results_summary[key_loadings]['ci'] = []
        for i_dataset in range(n_datasets):
            loadings_bootstrap = bootstrap_results[key_loadings][i_dataset]
            loadings_cca = cca_results_summary[key_loadings]['mean'][i_dataset][np.newaxis, :]
            p_values_tail1 = np.mean(loadings_bootstrap > loadings_cca, axis=0)
            p_values_tail2 = np.mean(loadings_bootstrap < loadings_cca, axis=0)
            p_values = np.min([p_values_tail1, p_values_tail2], axis=0) * 2

            print(f'p_values {key_loadings} ({dataset_names[i_dataset]})')
            print(f'{p_values[:n_features_to_print, :n_CAs_to_print]}')
            print('--------------------')

            cca_results_summary[key_loadings]['p_values'].append(p_values)

            # bootstrap null distribution confidence interval
            ci = np.quantile(loadings_bootstrap, [alpha, 1-alpha], axis=0)
            cca_results_summary[key_loadings]['ci'] = ci

    cca_results_summary['bootstrap_ci_alpha'] = alpha

    fpath_out = f'{cv_prefix_without_split}_merged_with_bootstrap.pkl'
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(cca_results_summary, file_out)
    print(f'Merged CCA results with bootstrap p-values/CIs saved to {fpath_out}')

    # # add 'n_merged' key (overwrites if already exists)
    # merged_results['n_merged'] = len(fnames)

    # fpath_out = os.path.join(dpath_out, 'merged_results.pkl')
    # with open(fpath_out, 'wb') as file_out:
    #     pickle.dump(merged_results, file_out)

    # print(f'Merged results saved to {fpath_out}')

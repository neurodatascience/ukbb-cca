import sys
import os
import pickle
import warnings
from copy import deepcopy

import numpy as np

from pipeline_definitions import build_cca_pipeline
from src.cca import cca_without_cv, cca_repeated_cv
from src.utils import make_parent_dir
from paths import FPATHS, DPATHS

np.set_printoptions(precision=4, linewidth=100, suppress=True, sign=' ')

fpath_data = FPATHS['data_Xy']

suppress_warnings = True
debug = False#True # if True, save models

# for repeated CV
cv_n_repetitions = 10
cv_n_folds = 5
cv_seed = 3791
cv_shuffle = True

# for ensemble models (repeated-CV CCA)
model_transform_cca = None
model_transform_deconfounder = (lambda model: 
    deepcopy(model)['preprocessor'].set_params(
        data_pipelines__behavioural__pca='passthrough', 
        data_pipelines__brain__pca='passthrough'
    )
)

if __name__ == '__main__':

    # process command line arguments
    if len(sys.argv) < 5:
        print(f'Usage: {sys.argv[0]} n_sample_sizes n_bootstrap_repetitions i_sample_size i_bootstrap_repetition n_PCs1 n_PCs2 [etc.]')
    n_sample_sizes, n_bootstrap_repetitions, i_sample_size, i_bootstrap_repetition = sys.argv[1:5]
    i_sample_size = int(i_sample_size) - 1 # zero-indexing
    i_bootstrap_repetition = int(i_bootstrap_repetition) - 1 # zero-indexing
    # n_PCs_all = [int(n) if n != 'None' else None for n in sys.argv[5:]] # number of PCA components
    n_PCs_all = []
    for n_PCs_str in sys.argv[5:]:
        if n_PCs_str == 'None':
            n_PCs = None
        else:
            try:
                n_PCs = int(n_PCs_str)
            except ValueError:
                n_PCs = float(n_PCs_str)
        n_PCs_all.append(n_PCs)
    fpath_bootstrap_samples = os.path.join(DPATHS['clean'], f'bootstrap_samples_{n_sample_sizes}steps_{n_bootstrap_repetitions}times.pkl')

    # load X/y data
    with open(fpath_data, 'rb') as file_data:
        data = pickle.load(file_data)
        X = data['X']
        dataset_names = data['dataset_names']
        n_datasets = len(dataset_names)
        conf_name = data['conf_name']
        udis_datasets = data['udis_datasets']
        udis_conf = data['udis_conf']
        n_features_datasets = data['n_features_datasets']
        n_features_conf = data['n_features_conf']

    # load bootstrap sample indices
    with open(fpath_bootstrap_samples, 'rb') as file_bootstrap_samples:
        bootstrap_samples = pickle.load(file_bootstrap_samples)
        sample_size = bootstrap_samples['sample_sizes'][i_sample_size]
        i_learn = bootstrap_samples['i_samples_learn_all'][i_bootstrap_repetition][sample_size]
        i_val = bootstrap_samples['i_samples_val_all'][i_bootstrap_repetition]

    # process PCA n_components
    if len(n_PCs_all) != n_datasets:
        raise ValueError(f'Mismatch between n_PCs_all (size {len(n_PCs_all)}) and data ({n_datasets} datasets)')
    for i_dataset, dataset_name in enumerate(dataset_names):
        if n_PCs_all[i_dataset] is None:
            n_PCs_all[i_dataset] = n_features_datasets[i_dataset]

    # print parameters
    print('----- Parameters -----')
    print(f'i_sample_size:\t{i_sample_size}')
    print(f'sample_size:\t{sample_size}')
    print(f'i_bootstrap_repetition:\t{i_bootstrap_repetition}')
    print(f'n_PCs_all:\t{n_PCs_all}')
    print(f'cv_n_repetitions:\t{cv_n_repetitions}')
    print(f'cv_n_folds:\t{cv_n_folds}')
    print(f'cv_seed:\t{cv_seed}')
    print(f'cv_shuffle:\t{cv_shuffle}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_bootstrap_samples:\t{fpath_bootstrap_samples}')
    print('----------------------')

    # figure out the number of latent dimensions in CCA
    # n_CAs = 10#min(n_PCs_all)
    n_CAs = min(n_PCs_all)
    print(f'Using {n_CAs} latent dimensions')

    # build pipeline/model
    preprocessing_params = {
        f'data_pipelines__{dataset_name}__pca__n_components': n_components 
        for dataset_name, n_components in zip(dataset_names, n_PCs_all)
    }
    cca_params = {
        'latent_dims': n_CAs,
    }
    cca_pipeline = build_cca_pipeline(
        dataset_names=dataset_names,
        verbosity=1,
        preprocessing_params=preprocessing_params,
        cca_params=cca_params,
    )
    print('------------------------------------------------------------------')
    print(cca_pipeline)
    print('------------------------------------------------------------------')

    # rng
    random_state = np.random.RandomState(cv_seed)

    if suppress_warnings:
        # this warnings occurs when n_samples < 1000 (default n_quantiles in sklearn QuantileTransformer)
        warnings.filterwarnings('ignore', '.*n_quantiles is set to n_samples')
    
    results_cca_without_cv = cca_without_cv(
        X, i_learn, i_val, cca_pipeline, 
        preprocess=True, normalize_loadings=True,
        debug=debug,
    )
    results_cca_repeated_cv = cca_repeated_cv(
        X, i_learn, i_val, cca_pipeline,
        cv_n_repetitions, cv_n_folds, 
        model_transform_deconfounder=model_transform_deconfounder,
        normalize_loadings=True,
        random_state=random_state,
        debug=debug,
    )

    # TODO combine rotate with no_rotate to avoid rerunning CCA
    results_cca_repeated_cv_no_rotate = cca_repeated_cv(
        X, i_learn, i_val, cca_pipeline,
        cv_n_repetitions, cv_n_folds,
        model_transform_deconfounder=model_transform_deconfounder,
        rotate_CAs=False,
        normalize_loadings=True,
        random_state=random_state,
        debug=debug,
    )

    for label, results in {'CCA without CV': results_cca_without_cv, 'CCA with repeated CV': results_cca_repeated_cv, 'CCA with repeated CV (no rotate)': results_cca_repeated_cv_no_rotate}.items():
        for set_name, set_results in results['corrs'].items():
            print(f'Corrs for {label} ({set_name}):\t{set_results[:10]}')

            # TODO remove
            print(f'\tNaNs in CAs: {[np.sum(np.isnan(result)) for result in results["CAs"][set_name]]}')
            print(f'\tNaNs in loadings: {[np.sum(np.isnan(result)) for result in results["loadings"][set_name]]}')

    results_all = {
        'sample_size': sample_size,
        'i_bootstrap_repetition': i_bootstrap_repetition,
        'i_learn': i_learn, 
        'i_val': i_val,
        'cca_without_cv': results_cca_without_cv,
        'cca_repeated_cv': results_cca_repeated_cv,
        'cca_repeated_cv_no_rotate': results_cca_repeated_cv_no_rotate,
        'dataset_names': dataset_names,
        'n_datasets': n_datasets,
        'conf_name': conf_name,
        'udis_datasets': udis_datasets,
        'udis_conf': udis_conf,
        'n_features_datasets': n_features_datasets,
        'n_features_conf': n_features_conf,
    }

    # save
    if not debug:
        dpath_cca = DPATHS['cca_sample_size']
    else:
        dpath_cca = f'{DPATHS["cca_sample_size"]}_debug'
    fpath_out = os.path.join(
        dpath_cca, f'PCs_{"_".join([str(n) for n in n_PCs_all])}', f'sample_size_{sample_size}', 
        f'sample_size_{sample_size}_rep{i_bootstrap_repetition+1}.pkl'
    )
    make_parent_dir(fpath_out)
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')


import os, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from scripts.pipeline_definitions import build_cca_pipeline

from paths import DPATHS, FPATHS

# settings
save_models = True

# model parameters: number of PCA components
n_components1 = 25
n_components2 = 25

# cross-validation parameters
verbose=True
n_folds = 5 # at least 2
shuffle = False
seed = None

# paths to data files
fpath_data = FPATHS['data_Xy_train']

# output path
dpath_out = DPATHS['cv']
fname_out_prefix = 'cca_cv_results'

if __name__ == '__main__':

    n_views = 2
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}.pkl')

    print('----- Parameters -----')
    print(f'n_components1:\t{n_components1}')
    print(f'n_components2:\t{n_components2}')
    print(f'save_models:\t{save_models}')
    print(f'verbose:\t{verbose}')
    print(f'n_folds:\t{n_folds}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'fpath_data:\t{fpath_data}')
    print('----------------------')

    # load data
    with open(fpath_data, 'rb') as file_in:
        X, y = pickle.load(file_in)

    subjects = X.index

    # random state
    if shuffle:
        random_state = np.random.RandomState(seed=seed)
    else:
        random_state = None

    # process PCA n_components
    if n_components1 is None:
        n_components1 = X['data1'].shape[1]
    if n_components2 is None:
        n_components2 = X['data2'].shape[1]

    # figure out the number of latent dimensions in CCA
    n_latent_dims = min(n_components1, n_components2)
    print(f'Using {n_latent_dims} latent dimensions')
    latent_dims_names = [f'CA{i+1}' for i in range(n_latent_dims)]

    # build pipeline/model
    cca_pipeline = build_cca_pipeline(
        n_pca_components1=n_components1, 
        n_pca_components2=n_components2,
        cca__latent_dims=n_latent_dims,
        verbose=verbose,
    )
    print('------------------------------------------------------------------')
    print(cca_pipeline)
    print('------------------------------------------------------------------')

    # cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # cross-validation loop
    cv_results = []
    projections_val_all = [[] for _ in range(2)] # estimated 'canonical factor scores' (data x weights) of validation sets
    for index_train, index_val in cv_splitter.split(X, y):

        subjects_train = subjects[index_train]
        subjects_val = subjects[index_val]

        X_train = X.loc[subjects_train]
        X_val = X.loc[subjects_val]

        # clone model and get pipeline components
        cca_pipeline_clone = clone(cca_pipeline)
        preprocessor = cca_pipeline_clone['preprocessor']
        cca = cca_pipeline_clone['cca']

        # fit pipeline in 2 steps 
        # (keeping preprocessed train data for scoring later)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        cca.fit(X_train_preprocessed)

        # get train metrics
        loadings_train = cca.get_loadings(X_train_preprocessed)
        correlations_train = cca.score(X_train_preprocessed)

        # get validation metrics
        X_val_preprocessed = preprocessor.transform(X_val)
        loadings_val = cca.get_loadings(X_val_preprocessed)
        projections_val = cca.transform(X_val_preprocessed)
        correlations_val = cca.score(X_val_preprocessed)
        
        # put all projections in a single list, to be transformed in a big dataframe later
        for i_view in range(n_views):
            projections_val_all[i_view].append(pd.DataFrame(projections_val[i_view], subjects_val, latent_dims_names))

        fold_results = {
            'subjects_train': subjects_train.tolist(),
            'subjects_val': subjects_val.tolist(),
            'loadings_train': loadings_train,
            'loadings_val': loadings_val,
            'correlations_train': correlations_train,
            'correlations_val': correlations_val,
        }

        cv_results.append(fold_results)

    # get full set of factors from all CV folds combined
    dfs_projections = []
    for i_view in range(n_views):
        dfs_projections.append(pd.concat(projections_val_all[i_view], axis='index'))
    
    # to be pickled
    results_all = {
        'cv_results': cv_results,
        'dfs_projections': dfs_projections, 
        'subjects': subjects,
        'latent_dims_names': latent_dims_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

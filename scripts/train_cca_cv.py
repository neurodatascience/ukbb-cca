
import os, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from scripts.pipeline_definitions import build_cca_pipeline

from paths import DPATHS, FPATHS

# settings
save_models = True

# model hyperparameters: number of PCA components
n_components_all = [100, 100]

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

    suffix = '_'.join([str(n) for n in n_components_all])
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{suffix}.pkl')

    print('----- Parameters -----')
    print(f'n_components_all:\t{n_components_all}')
    print(f'save_models:\t{save_models}')
    print(f'verbose:\t{verbose}')
    print(f'n_folds:\t{n_folds}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'fpath_data:\t{fpath_data}')
    print('----------------------')

    # load data
    with open(fpath_data, 'rb') as file_in:
        data = pickle.load(file_in)
        X = data['X']
        y = data['y']
        dataset_names = data['dataset_names']
        conf_name = data['conf_name']
        n_datasets = len(dataset_names)

    subjects = X.index

    # random state
    if shuffle:
        random_state = np.random.RandomState(seed=seed)
    else:
        random_state = None

    # process PCA n_components
    if len(n_components_all) != n_datasets:
        raise ValueError(f'Mismatch between n_components_all (size {len(n_components_all)}) and data ({n_datasets} datasets)')
    for i_dataset, dataset_name in enumerate(dataset_names):
        if n_components_all[i_dataset] is None:
            n_components_all[i_dataset] = X[dataset_name].shape[1]

    # figure out the number of latent dimensions in CCA
    n_latent_dims = min(n_components_all)
    print(f'Using {n_latent_dims} latent dimensions')
    latent_dims_names = [f'CA{i+1}' for i in range(n_latent_dims)]

    # build pipeline/model
    cca_pipeline = build_cca_pipeline(
        dataset_names=dataset_names,
        n_pca_components_all=n_components_all,
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
        pca_loadings_train = cca.get_loadings(X_train_preprocessed)
        correlations_train = cca.score(X_train_preprocessed)

        # get validation metrics
        X_val_preprocessed = preprocessor.transform(X_val)
        pca_loadings_val = cca.get_loadings(X_val_preprocessed)
        projections_val = cca.transform(X_val_preprocessed)
        correlations_val = cca.score(X_val_preprocessed)

        # transform loadings from PCA space to feature space
        feature_loadings_train = []
        feature_loadings_val = []
        for i_dataset in range(n_datasets):
            pca = preprocessor.data_pipelines[i_dataset]['pca']
            feature_loadings_train.append(pca.inverse_transform(pca_loadings_train[i_dataset].T).T)
            feature_loadings_val.append(pca.inverse_transform(pca_loadings_val[i_dataset].T).T)
        
        # put all projections in a single list, to be transformed in a big dataframe later
        for i_view in range(n_datasets):
            projections_val_all[i_view].append(pd.DataFrame(projections_val[i_view], subjects_val, latent_dims_names))

        fold_results = {
            'subjects_train': subjects_train.tolist(),
            'subjects_val': subjects_val.tolist(),
            'pca_loadings_train': pca_loadings_train,
            'pca_loadings_val': pca_loadings_val,
            'feature_loadings_train': feature_loadings_train,
            'feature_loadings_val': feature_loadings_val,
            'correlations_train': correlations_train,
            'correlations_val': correlations_val,
        }

        cv_results.append(fold_results)

    # get full set of factors from all CV folds combined
    dfs_projections = []
    for i_view in range(n_datasets):
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


import os, sys, pickle
from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from pipeline_definitions import build_cca_pipeline
from src.utils import load_data_df

from paths import DPATHS, FPATHS

# cross-validation parameters
verbose=True
n_folds = 5 # at least 2
shuffle = True
seed = None

# paths to data files
fpath_data = FPATHS['data_Xy_train']

# holdout variable (prediction target)
udi_holdout = '21003-2.0'
fpath_holdout = FPATHS['data_holdout_clean']

# output path
dpath_cv = DPATHS['cv']
fname_out_prefix = 'cv_cca'

if __name__ == '__main__':

    # process user inputs
    if len(sys.argv) < 5:
        raise ValueError(f'Usage: {sys.argv[0]} <CV ID> <i_repetition> <n_components1> <n_components2> [etc.]')
    dpath_out_suffix = sys.argv[1]
    i_repetition = int(sys.argv[2])
    n_components_all = [int(n) for n in sys.argv[3:]] # number of PCA components

    str_components = '_'.join([str(n) for n in n_components_all])

    # create output directory if necessary
    dpath_out = os.path.join(dpath_cv, f'cv_{str_components}_{dpath_out_suffix}')
    Path(dpath_out).mkdir(parents=True, exist_ok=True)

    fname_out_suffix = f'{str_components}_rep{i_repetition}'
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{fname_out_suffix}.pkl')

    print('----- Parameters -----')
    print(f'n_components_all:\t{n_components_all}')
    print(f'verbose:\t{verbose}')
    print(f'n_folds:\t{n_folds}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'udi_holdout\t{udi_holdout}')
    print(f'fpath_holdout\t{fpath_holdout}')
    print('----------------------')

    # load train dataset
    with open(fpath_data, 'rb') as file_in:
        data = pickle.load(file_in)

    X = data['X']
    y = data['y']
    subjects = X.index
    dataset_names = data['dataset_names']
    conf_name = data['conf_name']
    udis = data['udis']
    n_datasets = len(dataset_names)

    # load holdout variables
    df_holdout = load_data_df(fpath_holdout)

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

    # initialization
    subjects_train_all = []
    subjects_val_all = []
    weights_train_all = [[] for _ in range(n_datasets)]
    projections_val_all = [[] for _ in range(n_datasets)] # estimated 'canonical factor scores' (data x weights) of validation sets
    pca_loadings_train_all = [[] for _ in range(n_datasets)]
    pca_loadings_val_all = [[] for _ in range(n_datasets)] 
    loadings_train_all = [[] for _ in range(n_datasets)]
    loadings_val_all = [[] for _ in range(n_datasets)] # variable loadings estimated on validation sets
    correlations_train_all = []
    correlations_val_all = []
    R2_PC_reg_train_all = []
    R2_PC_reg_val_all = []

    # cross-validation loop
    for index_train, index_val in cv_splitter.split(X, y):

        subjects_train = subjects[index_train]
        subjects_val = subjects[index_val]

        X_train = X.loc[subjects_train]
        X_val = X.loc[subjects_val]

        holdout_train = df_holdout.loc[subjects_train, udi_holdout]
        holdout_val =  df_holdout.loc[subjects_val, udi_holdout]

        # clone model and get pipeline components
        cca_pipeline_clone = clone(cca_pipeline)
        preprocessor = cca_pipeline_clone['preprocessor']
        cca = cca_pipeline_clone['cca']

        # fit pipeline in 2 steps 
        # (keeping preprocessed train data for scoring later)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        cca.fit(X_train_preprocessed)

        # get CCA train metrics
        pca_loadings_train = cca.get_loadings(X_train_preprocessed, normalize=True)
        correlations_train_all.append(cca.score(X_train_preprocessed))

        # get CCA validation metrics
        X_val_preprocessed = preprocessor.transform(X_val)
        pca_loadings_val = cca.get_loadings(X_val_preprocessed, normalize=True)
        projections_val = cca.transform(X_val_preprocessed)
        correlations_val_all.append(cca.score(X_val_preprocessed))

        # get linear regression on PCs for reference
        X_train_preprocessed_concat = pd.concat([pd.DataFrame(x, index=subjects_train).iloc[:, :n_latent_dims] for x in X_train_preprocessed], axis='columns')
        X_val_preprocessed_concat = pd.concat([pd.DataFrame(x, index=subjects_val).iloc[:, :n_latent_dims] for x in X_val_preprocessed], axis='columns')
        lr = LinearRegression()
        lr.fit(X_train_preprocessed_concat, holdout_train)
        R2_PC_reg_train_all.append(lr.score(X_train_preprocessed_concat, holdout_train))
        R2_PC_reg_val_all.append(lr.score(X_val_preprocessed_concat, holdout_val))

        for i_dataset in range(n_datasets):

            # put all validation loadings in a single list, to be summed up later
            pca_loadings_train_all[i_dataset].append(pca_loadings_train[i_dataset])
            pca_loadings_val_all[i_dataset].append(pca_loadings_val[i_dataset])

            # transform loadings from PCA space to feature space
            pca = preprocessor.data_pipelines[i_dataset]['pca']
            loadings_train_all[i_dataset].append(pca.inverse_transform(pca_loadings_train[i_dataset].T).T)
            loadings_val_all[i_dataset].append(pca.inverse_transform(pca_loadings_val[i_dataset].T).T)
        
            # put all projections in a single list, to be transformed in a big dataframe later
            # using dataframes to keep track of subject IDs
            projections_val_all[i_dataset].append(pd.DataFrame(projections_val[i_dataset], subjects_val, latent_dims_names))
            
            weights_train_all[i_dataset].append(cca.weights[i_dataset])

        # save some other information about this fold
        subjects_train_all.append(subjects_train)
        subjects_val_all.append(subjects_val)

    # combine main validation set results
    projections_val_combined = []
    pca_loadings_val_combined = []
    loadings_val_combined = []
    for i_dataset in range(n_datasets):

        # concatenate all projections
        projections_val_combined.append(pd.concat(projections_val_all[i_dataset], axis='index').loc[subjects].values)

        # sum up loadings
        loadings_val_combined.append(np.array(loadings_val_all[i_dataset]).sum(axis=0))
        pca_loadings_val_combined.append(np.array(pca_loadings_val_all[i_dataset]).sum(axis=0))

        # convert train set results (weights/loadings) to numpy arrays
        weights_train_all[i_dataset] = np.array(weights_train_all[i_dataset])
        pca_loadings_train_all[i_dataset] = np.array(pca_loadings_train_all[i_dataset])
        loadings_train_all[i_dataset] = np.array(loadings_train_all[i_dataset])

    # to be pickled
    results_all = {
        'projections_val': projections_val_combined,
        'weights_train': weights_train_all,
        'loadings_val': loadings_val_combined,
        'correlations_train': np.array(correlations_train_all),
        'correlations_val': np.array(correlations_val_all),
        'R2_PC_reg_train': np.array(R2_PC_reg_train_all),
        'R2_PC_reg_val': np.array(R2_PC_reg_val_all),
        'subjects_train': subjects_train_all,
        'subjects_val': subjects_val_all,
        'subjects': subjects,
        'latent_dims_names': latent_dims_names,
        'n_latent_dims': n_latent_dims,
        'PC_names': [[f'PC{i+1}' for i in range(n_components_all[i_dataset])] for i_dataset in range(n_datasets)],
        'n_components_all': n_components_all,
        'n_datasets': n_datasets, 
        'dataset_names': dataset_names, 
        'udis_datasets': [udis[dataset_name] for dataset_name in dataset_names], 
        'udis_conf': udis[conf_name],
        'n_folds': n_folds,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

import os, sys, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from pipeline_definitions import build_cca_pipeline
from src.utils import make_dir, load_data_df, rotate_to_match

from paths import DPATHS, FPATHS

# cross-validation parameters
verbose=True
n_folds = 5 # at least 2
shuffle = True
seed = None

# only used if preprocessed is True
dpath_data = DPATHS['cca_preprocessed'] # directory containing preprocessed files
dpath_preprocessor = DPATHS['cca_preprocessor'] # directory containing fitted preprocessor (needed for inverse PCA transform)

# holdout variable (prediction target)
udi_holdout = '21003-2.0'
fpath_holdout = FPATHS['data_holdout_clean']

# output path
dpath_cv = os.path.join(DPATHS['scratch'], os.path.basename(DPATHS['cca'])) # use same folder name
fname_out_prefix = 'cv_cca'

if __name__ == '__main__':

    # process user inputs
    if len(sys.argv) < 6:
        print(f'Usage: {sys.argv[0]} <CV ID> <i_repetition> <use_preprocessed> <n_components1> <n_components2> [etc.]')
        sys.exit(1)
    dpath_out_suffix = sys.argv[1]
    i_repetition = int(sys.argv[2])
    # if 0: do PCA for each fold (using e.g. 80% of train data)
    # else: use existing PCs (computed on all train data)
    use_preprocessed = bool(int(sys.argv[3]))
    n_components_all = [int(n) for n in sys.argv[4:]] # number of PCA components
    str_components = '_'.join([str(n) for n in n_components_all])

    # generate paths to data and preprocessor objects
    if use_preprocessed:
        fpath_data = os.path.join(dpath_data, f'X_train_{str_components}.pkl')
        fpath_preprocessor = os.path.join(dpath_preprocessor, f'preprocessor_{str_components}.pkl')
    else:
        fpath_data = FPATHS['data_Xy_train']

    # create output directory if necessary
    dpath_out = os.path.join(dpath_cv, f'cv_{str_components}_{dpath_out_suffix}')
    make_dir(dpath_out)

    fname_out_suffix = f'{str_components}_rep{i_repetition}'
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{fname_out_suffix}.pkl')

    print('----- Parameters -----')
    print(f'use_preprocessed:\t{use_preprocessed}')
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

    if use_preprocessed:
        subjects = data['subjects']
    else:
        subjects = X.index # TODO save subjects as separate key in split_data.py, for consistency
    dataset_names = data['dataset_names']
    conf_name = data['conf_name']
    udis = data['udis']
    n_datasets = len(dataset_names)

    # load preprocessor
    if use_preprocessed:
        with open(fpath_preprocessor, 'rb') as file_in:
            preprocessor = pickle.load(file_in)['preprocessor']

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
            # TODO save n_features list in split_data.py and process_data.py
            if use_preprocessed:
                n_components_all[i_dataset] = X[i_dataset].shape[1]
            else:
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
    if use_preprocessed:
        print(cca_pipeline['cca']) # only the CCA part of the pipeline will be used
    else:
        print(cca_pipeline) # entire pipeline will be used
    print('------------------------------------------------------------------')

    # cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # initialization
    i_train_all = []
    i_val_all = []
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
    for i_fold, (i_train, i_val) in enumerate(cv_splitter.split(subjects, y)):

        subjects_train = subjects[i_train]
        subjects_val = subjects[i_val]

        # clone model and get pipeline components
        cca_pipeline_clone = clone(cca_pipeline)
        cca = cca_pipeline_clone['cca']

        if not use_preprocessed:
            preprocessor = cca_pipeline_clone['preprocessor']
            X_train = X.loc[subjects_train]
            X_val = X.loc[subjects_val]
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_val_preprocessed = preprocessor.transform(X_val)
        else:
            X_train_preprocessed = [X[i_dataset][i_train] for i_dataset in range(n_datasets)]
            X_val_preprocessed = [X[i_dataset][i_val] for i_dataset in range(n_datasets)]

        holdout_train = df_holdout.loc[subjects_train, udi_holdout]
        holdout_val =  df_holdout.loc[subjects_val, udi_holdout]

        # fit CCA
        cca.fit(X_train_preprocessed)

        # align weights
        # otherwise some folds may have flipped signs or switched columns
        if i_fold == 0:
            ref_weights = cca.weights
        else:
            rotated_weights = [rotate_to_match(cca.weights[i], ref_weights[i]) for i in range(n_datasets)]
            cca.weights = rotated_weights

        # get CCA train metrics
        pca_loadings_train = cca.get_loadings(X_train_preprocessed, normalize=True)
        correlations_train_all.append(cca.score(X_train_preprocessed))

        # get CCA validation metrics
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
        i_train_all.append(i_train)
        i_val_all.append(i_val)

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
        'i_train': i_train_all,
        'i_val': i_val_all,
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
        'use_preprocessed': use_preprocessed,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

import os, sys, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from pipeline_definitions import build_cca_pipeline
from src.utils import make_dir, load_data_df

from paths import DPATHS, FPATHS

# bootstrap
bootstrap = False
bootstrap_seed = None # for random number generator

# linear regression
n_CAs_to_use_in_LR = 1

# cross-validation parameters
verbosity = 1
verbose = verbosity > 0
n_folds = 5 # at least 2
shuffle = True
seed = None

fpath_data = FPATHS['data_Xy']

# output path
dpath_cv = os.path.join(DPATHS['scratch'], os.path.basename(DPATHS['cca'])) # use same folder name
fname_out_prefix = 'cv_cca'

if __name__ == '__main__':

    # process user inputs
    if len(sys.argv) < 6:
        print(f'Usage: {sys.argv[0]} <CV ID> <i_repetition> <i_split> <n_components1> <n_components2> [etc.]')
        sys.exit(1)
    dpath_out_suffix = sys.argv[1]
    i_repetition = int(sys.argv[2])
    i_split = int(sys.argv[3]) # integer from 0 to n_splits-1
    dpath_out_suffix = f'{dpath_out_suffix}_split{i_split}'
    n_components_all = [int(n) for n in sys.argv[4:]] # number of PCA components
    str_components = '_'.join([str(n) for n in n_components_all])

    # create output directory if necessary
    dpath_out = os.path.join(dpath_cv, f'cv_{str_components}_{dpath_out_suffix}')
    make_dir(dpath_out)

    fname_out_suffix = f'{str_components}_rep{i_repetition}'
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{fname_out_suffix}.pkl')

    print('----- Parameters -----')
    print(f'n_components_all:\t{n_components_all}')
    print(f'i_split:\t{i_split}')
    print(f'verbosity:\t{verbosity}')
    print(f'n_folds:\t{n_folds}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'bootstrap:\t{bootstrap}')
    print(f'bootstrap_seed:\t{bootstrap_seed}')
    print('----------------------')

    # random number generator for bootstrapping
    if bootstrap:
        rng = np.random.default_rng(bootstrap_seed)

    # load train dataset
    with open(fpath_data, 'rb') as file_in:
        data = pickle.load(file_in)
    i_learning = data['i_train_all'][i_split]
    subjects = data['subjects'][i_learning]
    X = data['X'].loc[subjects]
    y = data['y'].loc[subjects]
    holdout = data['holdout']
    dataset_names = data['dataset_names']
    conf_name = data['conf_name']
    udis_datasets = data['udis_datasets']
    udis_conf = data['udis_conf']
    n_features_datasets = data['n_features_datasets']
    n_features_conf = data['n_features_conf']
    n_datasets = len(dataset_names)

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
            n_components_all[i_dataset] = n_features_datasets[i_dataset]

    # figure out the number of latent dimensions in CCA
    n_latent_dims = min(n_components_all)
    print(f'Using {n_latent_dims} latent dimensions')
    latent_dims_names = [f'CA{i+1}' for i in range(n_latent_dims)]

    # build pipeline/model
    preprocessing_params = {
        f'data_pipelines__{dataset_name}__pca__n_components': n_components 
        for dataset_name, n_components in zip(dataset_names, n_components_all)
    }
    cca_params = {
        'latent_dims': n_latent_dims,
    }
    cca_pipeline = build_cca_pipeline(
        dataset_names=dataset_names,
        verbosity=verbosity-1,
        preprocessing_params=preprocessing_params,
        cca_params=cca_params,
    )
    print('------------------------------------------------------------------')
    print(cca_pipeline)
    print('------------------------------------------------------------------')

    # cross-validation splitter
    cv_splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # initialization
    models = []
    i_train_all = []
    i_val_all = []
    correlations_train_all = []
    correlations_val_all = []
    R2_train_all = []
    R2_val_all = []

    # cross-validation loop
    for i_fold, (i_train, i_val) in enumerate(cv_splitter.split(subjects, y)):

        if verbose:
            print(f'---------- Fold {i_fold} ----------')

        subjects_train = subjects[i_train]
        subjects_val = subjects[i_val]

        X_train = X.loc[subjects_train]
        X_val = X.loc[subjects_val]

        holdout_train = holdout[subjects_train]
        holdout_val = holdout[subjects_val]

        # clone model and get pipeline components
        model = clone(cca_pipeline)
        preprocessor = model['preprocessor']
        cca = model['cca']

        # fit model (preprocessing/PCA + CCA)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        CAs_train = cca.fit_transform(X_train_preprocessed)

        # bootstrap mode: replace fitted weights by bootstrapped weights
        if bootstrap:
            n_subjects_train = len(subjects_train)

            # compute bootstrap weights (one dataset at a time)
            weights_bootstrap = []
            for i_dataset_fix in range(n_datasets):

                # shuffle all but one dataset
                X_train_shuffled = []
                for i_dataset in range(n_datasets):
                    if i_dataset == i_dataset_fix:
                        X_train_shuffled.append(X_train_preprocessed[i_dataset]) # not shuffled
                    else:
                        # randomly sample subjects with replacement
                        i_shuffled = rng.choice(np.arange(n_subjects_train), size=n_subjects_train, replace=True)
                        X_train_shuffled.append(X_train_preprocessed[i_dataset][i_shuffled])

                cca_tmp = clone(cca)
                cca_tmp.fit(X_train_shuffled)

                weights_bootstrap.append(cca_tmp.weights[i_dataset_fix])

            model['cca'].weights = weights_bootstrap

        # get validation set scores
        X_val_preprocessed = preprocessor.transform(X_val)
        CAs_val = cca.transform(X_val_preprocessed)
        correlations_val_all.append(cca.score(X_val_preprocessed))
        correlations_train_all.append(cca.score(X_train_preprocessed)) # also get train scores

        # save model
        models.append(model)

        # save some other information about this fold
        # use np.uint16 to save space
        i_train_all.append(np.array(i_train, dtype=np.uint16))
        i_val_all.append(np.array(i_val, dtype=np.uint16))

        # regression with holdout variable
        lr = LinearRegression()
        lr_input_train = np.concatenate([CAs[:, :n_CAs_to_use_in_LR] for CAs in CAs_train], axis=1)
        lr_input_val = np.concatenate([CAs[:, :n_CAs_to_use_in_LR] for CAs in CAs_val], axis=1)
        lr.fit(lr_input_train, holdout_train)
        R2_train_all.append(lr.score(lr_input_train, holdout_train))
        R2_val_all.append(lr.score(lr_input_val, holdout_val))
  
    # to be pickled
    results_all = {
        'i_split': i_split,
        'subjects': subjects,
        'n_folds': n_folds,
        'i_train': i_train_all,
        'i_val': i_val_all,
        'models': models,
        'correlations_val': np.array(correlations_val_all),
        'correlations_train': np.array(correlations_train_all),
        'R2_val': np.array(R2_val_all),
        'R2_train': np.array(R2_train_all),
        'latent_dims_names': latent_dims_names,
        'n_latent_dims': n_latent_dims,
        'PC_names': [[f'PC{i+1}' for i in range(n_components_all[i_dataset])] for i_dataset in range(n_datasets)],
        'n_components_all': n_components_all,
        'dataset_names': dataset_names, 
        'n_datasets': n_datasets, 
        'conf_name': conf_name,
        'udis_datasets': udis_datasets, 
        'udis_conf': udis_conf,
        'n_features_datasets': n_features_datasets,
        'n_features_conf': n_features_conf,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    if verbose:
        print(f'-----------------------------------')
    print(f'Saved to {fpath_out}')

import os, sys, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from pipeline_definitions import build_cca_pipeline
from src.utils import make_dir, load_data_df

from paths import DPATHS, FPATHS

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
    print('----------------------')

    # load train dataset
    with open(fpath_data, 'rb') as file_in:
        data = pickle.load(file_in)
    i_learning = data['i_train_all'][i_split]
    subjects = data['subjects'][i_learning]
    X = data['X'].loc[subjects]
    y = data['y'].loc[subjects]
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
    correlations_val_all = []

    # cross-validation loop
    for i_fold, (i_train, i_val) in enumerate(cv_splitter.split(subjects, y)):

        if verbose:
            print(f'---------- Fold {i_fold} ----------')

        subjects_train = subjects[i_train]
        subjects_val = subjects[i_val]

        X_train = X.loc[subjects_train]
        X_val = X.loc[subjects_val]

        # clone model and get pipeline components
        model = clone(cca_pipeline)

        # fit model (preprocessing/PCA + CCA)
        model.fit(X_train)

        # get validation set scores
        preprocessor = model['preprocessor']
        cca = model['cca']
        X_val_preprocessed = preprocessor.transform(X_val)
        correlations_val_all.append(cca.score(X_val_preprocessed))

        # save model
        models.append(model)

        # save some other information about this fold
        # use np.uint16 to save space
        i_train_all.append(np.array(i_train, dtype=np.uint16))
        i_val_all.append(np.array(i_val, dtype=np.uint16))

  
    # to be pickled
    results_all = {
        'i_split': i_split,
        # 'i_learning': i_learning, # TODO remove this
        'subjects': subjects,
        'n_folds': n_folds,
        'i_train': i_train_all,
        'i_val': i_val_all,
        'models': models,
        'correlations_val': np.array(correlations_val_all),
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

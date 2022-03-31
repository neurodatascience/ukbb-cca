import os, sys, pickle
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from scripts.pipeline_definitions import build_cca_pipeline
from src.utils import load_data_df, make_dir
from paths import DPATHS, FPATHS

# parameters
n_permutations = 250
seed = None # for random number generator

# paths to data files
dpath_data_preprocessed = DPATHS['cca_preprocessed'] # directory containing preprocessed train data
fpath_data_test = FPATHS['data_Xy_test']
dpath_preprocessor = DPATHS['cca_preprocessor']
fpath_holdout = FPATHS['data_holdout_clean'] # for bootstrapped regression with holdout variable
udi_holdout = '21003-2.0'

# output path
dpath_out = DPATHS['bootstrap']
fname_out_prefix = 'bootstrap_results'

if __name__ == '__main__':

    # process user inputs
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <n_components1> <n_components2> [etc.]')
        sys.exit(1)
    n_components_all = [int(n) for n in sys.argv[1:]] # number of PCA components

    # make output directory if it doesn't exist yet
    make_dir(dpath_out)

    str_components = '_'.join([str(n) for n in n_components_all])
    fpath_data_train_preprocessed = os.path.join(dpath_data_preprocessed, f'X_train_{str_components}.pkl')
    fpath_preprocessor = os.path.join(dpath_preprocessor, f'preprocessor_{str_components}.pkl')
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{str_components}.pkl')

    print('----- Parameters -----')
    print(f'n_permutations:\t{n_permutations}')
    print(f'seed:\t{seed}')
    print(f'n_components_all:\t{n_components_all}')
    print(f'udi_holdout:\t{udi_holdout}')
    print(f'fpath_preprocessor:\t{fpath_preprocessor}')
    print(f'fpath_data_train_preprocessed:\t{fpath_data_train_preprocessed}')
    print(f'fpath_data_test:\t{fpath_data_test}')
    print(f'fpath_holdout:\t{fpath_holdout}')
    print('----------------------')

    # create random number generator
    rng = np.random.default_rng(seed)

    # load train data
    with open(fpath_data_train_preprocessed, 'rb') as file_in:
        preprocessed_data = pickle.load(file_in)
    X_train_preprocessed = preprocessed_data['X'] # already preprocessed
    dataset_names = preprocessed_data['dataset_names']
    n_datasets = preprocessed_data['n_datasets']
    conf_name = preprocessed_data['conf_name']
    subjects_train = preprocessed_data['subjects']
    n_subjects_train = len(subjects_train)

    # load preprocessor
    with open(fpath_preprocessor, 'rb') as file_in:
        fitted_preprocessor = pickle.load(file_in)['preprocessor']

    print('------------------------------------------------------------------')
    print(fitted_preprocessor)
    print('------------------------------------------------------------------')

    # load and preprocess test data
    with open(fpath_data_test, 'rb') as file_test:
        test_data = pickle.load(file_test)
    X_test = test_data['X']
    subjects_test = X_test.index
    X_test_preprocessed = fitted_preprocessor.transform(X_test)

    # load holdout data
    df_holdout = load_data_df(fpath_holdout)
    holdout_train = df_holdout.loc[subjects_train, udi_holdout]
    holdout_test = df_holdout.loc[subjects_test, udi_holdout]

    # figure out the number of latent dimensions in CCA
    n_latent_dims = min(n_components_all)
    print(f'Using {n_latent_dims} latent dimensions')
    latent_dims_names = [f'CA{i+1}' for i in range(n_latent_dims)]

    # build pipeline/model
    cca_pipeline = build_cca_pipeline(
        dataset_names=dataset_names,
        n_pca_components_all=n_components_all,
        cca__latent_dims=n_latent_dims,
    )
    cca = cca_pipeline['cca']
    print('------------------------------------------------------------------')
    print(cca)
    print('------------------------------------------------------------------')

    # 'fit' cca (weights will be overwritten later)
    cca.fit_transform(X_train_preprocessed)

    correlations_bootstrap_train = []
    correlations_bootstrap_test = []
    R2_holdout_bootstrap_train = []
    R2_holdout_bootstrap_test = []
    for i_permutation in range(n_permutations):
        weights = []
        for i_dataset_fix in range(n_datasets):

            # shuffle all but one dataset
            X_train_shuffled = []
            for i_dataset in range(n_datasets):

                if i_dataset == i_dataset_fix:
                    X_train_shuffled.append(X_train_preprocessed[i_dataset].copy())
                    continue

                i_shuffled = rng.choice(np.arange(n_subjects_train), size=n_subjects_train, replace=True)
                X_train_shuffled.append(X_train_preprocessed[i_dataset][i_shuffled])

            cca_tmp = clone(cca) # get an unfitted model
            cca_tmp.fit(X_train_shuffled)

            # fit and get weights for this dataset
            weights.append(cca_tmp.weights[i_dataset_fix])

        # set model weights (from bootstrap)
        cca.weights = weights

        # get bootstrap CCA correlations (for train and test data)
        correlations_bootstrap_train.append(cca.score(X_train_preprocessed))
        correlations_bootstrap_test.append(cca.score(X_test_preprocessed))

        # get correlations with holdout variable
        projections_train = cca.transform(X_train_preprocessed)
        projections_test = cca.transform(X_test_preprocessed)
        R2_train = []
        R2_test = []
        for i_dim in range(n_latent_dims):

            # extract canonical scores for this latent dimension
            projections_dim_train = [projections[:, i_dim] for projections in projections_train]
            projections_dim_train = np.vstack(projections_dim_train).T
            # test set
            projections_dim_test = [projections[:, i_dim] for projections in projections_test]
            projections_dim_test = np.vstack(projections_dim_test).T

            # get R2 score
            lr = LinearRegression()
            lr.fit(projections_dim_train, holdout_train)
            R2_train.append(lr.score(projections_dim_train, holdout_train))
            R2_test.append(lr.score(projections_dim_test, holdout_test))

        R2_holdout_bootstrap_train.append(R2_train)
        R2_holdout_bootstrap_test.append(R2_test)

    # to be pickled
    results_all = {
        'correlations_bootstrap_train': np.array(correlations_bootstrap_train).T,
        'correlations_bootstrap_test': np.array(correlations_bootstrap_test).T,
        'R2_holdout_bootstrap_train': np.array(R2_holdout_bootstrap_train).T,
        'r_holdout_bootstrap_train': np.sqrt(R2_holdout_bootstrap_train).T,
        'R2_holdout_bootstrap_test': np.array(R2_holdout_bootstrap_test).T,
        'r_holdout_bootstrap_test': np.sqrt(R2_holdout_bootstrap_test).T,
        'subjects_train': subjects_train.tolist(),
        'subjects_test': subjects_test.tolist(),
        'latent_dims_names': latent_dims_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

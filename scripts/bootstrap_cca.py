
import os, pickle
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from scripts.pipeline_definitions import build_cca_pipeline
from src.utils import load_data_df
from paths import DPATHS, FPATHS

# parameters
n_permutations = 1000
seed = None # for random number generator
n_components_all = [100, 100] # number of PCA components
verbose=True

# paths to data files
fpath_data_train = FPATHS['data_Xy_train']
fpath_holdout = FPATHS['data_holdout_clean'] # for bootstrapped regression with holdout variable
holdout_udi = '21003-2.0'

# output path
dpath_out = DPATHS['cca']
fname_out_prefix = 'bootstrap_results'

if __name__ == '__main__':

    suffix = '_'.join([str(n) for n in n_components_all])
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{suffix}.pkl')

    print('----- Parameters -----')
    print(f'n_permutations:\t{n_permutations}')
    print(f'seed:\t{seed}')
    print(f'n_components_all:\t{n_components_all}')
    print(f'verbose:\t{verbose}')
    print(f'fpath_data_train:\t{fpath_data_train}')
    print('----------------------')

    # create random number generator
    rng = np.random.default_rng(seed)

    # load train data
    with open(fpath_data_train, 'rb') as file_train:
        train_data = pickle.load(file_train)
        X_train = train_data['X']
        dataset_names = train_data['dataset_names']
        n_datasets = len(dataset_names)
        conf_name = train_data['conf_name']

    subjects_train = X_train.index
    n_subjects_train = len(subjects_train)

    # load holdout data
    df_holdout = load_data_df(fpath_holdout)
    holdout_train = df_holdout.loc[subjects_train, holdout_udi]

    # process PCA n_components
    if len(n_components_all) != n_datasets:
        raise ValueError(f'Mismatch between n_components_all (size {len(n_components_all)}) and dataset_names (size {len(dataset_names)})')
    for i_dataset, dataset_name in enumerate(dataset_names):
        if n_components_all[i_dataset] is None:
            n_components_all[i_dataset] = X_train[dataset_name].shape[1]

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

    preprocessor = cca_pipeline['preprocessor']
    cca = cca_pipeline['cca']

    # preprocess data
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # 'fit' cca (weights will be overwritten later)
    cca.fit_transform(X_train_preprocessed)

    correlations_bootstrap_train = []
    R2_holdout_bootstrap_train = []
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
            weights.append(cca_tmp.weights[i_dataset])

        # set model weights (from bootstrap)
        cca.weights = weights

        # get bootstrap CCA correlations
        correlations_bootstrap_train.append(cca.score(X_train_preprocessed))

        # get correlations with holdout variable
        projections_train = cca.transform(X_train_preprocessed)
        R2 = []
        for i_dim in range(n_latent_dims):

            # extract canonical scores for this latent dimension
            projections_dim = [projections[:, i_dim] for projections in projections_train]
            projections_dim = np.vstack(projections_dim).T

            # get R2 score
            lr = LinearRegression()
            lr.fit(projections_dim, holdout_train)
            R2.append(lr.score(projections_dim, holdout_train))
        R2_holdout_bootstrap_train.append(R2)

    # to be pickled
    results_all = {
        'model': cca_pipeline, # fitted model
        'correlations_bootstrap_train': np.array(correlations_bootstrap_train).T,
        'R2_holdout_bootstrap_train': np.array(R2_holdout_bootstrap_train).T,
        'r_holdout_bootstrap_train': np.sqrt(R2_holdout_bootstrap_train).T,
        'subjects_train': subjects_train.tolist(),
        'latent_dims_names': latent_dims_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

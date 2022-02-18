
import os, pickle
from scripts.pipeline_definitions import build_cca_pipeline
from paths import DPATHS, FPATHS

# model parameters: number of PCA components
n_components1 = 100
n_components2 = 100

# other pipeline parameters
verbose=True

# paths to data files
fpath_data_train = FPATHS['data_Xy_train']
fpath_data_test = FPATHS['data_Xy_test']

# output path
dpath_out = DPATHS['cca']
fname_out_prefix = 'cca_results'

if __name__ == '__main__':

    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{n_components1}_{n_components2}.pkl')

    print('----- Parameters -----')
    print(f'n_components1:\t{n_components1}')
    print(f'n_components2:\t{n_components2}')
    print(f'verbose:\t{verbose}')
    print(f'fpath_data_train:\t{fpath_data_train}')
    print(f'fpath_data_test:\t{fpath_data_test}')
    print('----------------------')

    # load train data
    with open(fpath_data_train, 'rb') as file_train:
        X_train, _ = pickle.load(file_train)

    # load test data
    with open(fpath_data_test, 'rb') as file_test:
        X_test, _ = pickle.load(file_test)

    subjects_train = X_train.index
    subjects_test = X_test.index

    # process PCA n_components
    if n_components1 is None:
        n_components1 = X_train['data1'].shape[1]
    if n_components2 is None:
        n_components2 = X_train['data2'].shape[1]

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

    preprocessor = cca_pipeline['preprocessor']
    cca = cca_pipeline['cca']

    # fit pipeline in 2 steps 
    # (keeping preprocessed train data for scoring later)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    projections_train = cca.fit_transform(X_train_preprocessed)

    # get train metrics
    loadings_train = cca.get_loadings(X_train_preprocessed)
    correlations_train = cca.score(X_train_preprocessed)

    # get test metrics
    X_test_preprocessed = preprocessor.transform(X_test)
    projections_test = cca.transform(X_test_preprocessed)
    loadings_test = cca.get_loadings(X_test_preprocessed)
    correlations_test = cca.score(X_test_preprocessed)

    # to be pickled
    results_all = {
        'model': cca_pipeline, # fitted model
        'projections_train': projections_train,
        'projections_test': projections_test,
        'loadings_train': loadings_train,
        'loadings_test': loadings_test,
        'correlations_train': correlations_train,
        'correlations_test': correlations_test,
        'subjects_train': subjects_train.tolist(),
        'subjects_test': subjects_test.tolist(),
        'latent_dims_names': latent_dims_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

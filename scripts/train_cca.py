
import os, pickle
from scripts.pipeline_definitions import build_cca_pipeline
from paths import DPATHS, FPATHS

# model hyperparameters: number of PCA components
n_components_all = [100, 100]

# other pipeline parameters
verbose=True

# paths to data files
fpath_data_train = FPATHS['data_Xy_train']
fpath_data_test = FPATHS['data_Xy_test']

# output path
dpath_out = DPATHS['cca']
fname_out_prefix = 'cca_results'

if __name__ == '__main__':

    suffix = '_'.join([str(n) for n in n_components_all])
    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}_{suffix}.pkl')

    print('----- Parameters -----')
    print(f'n_components_all:\t{n_components_all}')
    print(f'verbose:\t{verbose}')
    print(f'fpath_data_train:\t{fpath_data_train}')
    print(f'fpath_data_test:\t{fpath_data_test}')
    print('----------------------')

    # load train data
    with open(fpath_data_train, 'rb') as file_train:
        train_data = pickle.load(file_train)
        X_train = train_data['X']
        dataset_names = train_data['dataset_names']
        n_datasets = len(dataset_names)
        conf_name = train_data['conf_name']
        udis = train_data['udis']

    # load test data
    with open(fpath_data_test, 'rb') as file_test:
        test_data = pickle.load(file_test)
        X_test = test_data['X']

        if test_data['dataset_names'] != dataset_names:
            raise ValueError('Train/test sets do not have the same dataset names')
        if test_data['conf_name'] != conf_name:
            raise ValueError('Train/test sets do not have the same conf name')

    subjects_train = X_train.index
    subjects_test = X_test.index

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

    # fit pipeline in 2 steps 
    # (keeping preprocessed train data for scoring later)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    projections_train = cca.fit_transform(X_train_preprocessed)

    # get train metrics
    pca_loadings_train = cca.get_loadings(X_train_preprocessed)
    correlations_train = cca.score(X_train_preprocessed)

    # get test metrics
    X_test_preprocessed = preprocessor.transform(X_test)
    projections_test = cca.transform(X_test_preprocessed)
    pca_loadings_test = cca.get_loadings(X_test_preprocessed)
    correlations_test = cca.score(X_test_preprocessed)

    # transform loadings from PCA space to feature space
    feature_loadings_train = []
    feature_loadings_test = []
    for i_dataset in range(n_datasets):
        pca = preprocessor.data_pipelines[i_dataset]['pca']
        feature_loadings_train.append(pca.inverse_transform(pca_loadings_train[i_dataset].T).T)
        feature_loadings_test.append(pca.inverse_transform(pca_loadings_test[i_dataset].T).T)

    # to be pickled
    results_all = {
        'model': cca_pipeline, # fitted model
        'projections_train': projections_train,
        'projections_test': projections_test,
        'pca_loadings_train': pca_loadings_train,
        'pca_loadings_test': pca_loadings_test,
        'feature_loadings_train': feature_loadings_train,
        'feature_loadings_test': feature_loadings_test,
        'correlations_train': correlations_train,
        'correlations_test': correlations_test,
        'subjects_train': subjects_train.tolist(),
        'subjects_test': subjects_test.tolist(),
        'latent_dims_names': latent_dims_names,
        'udis': udis,
        'dataset_names': dataset_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

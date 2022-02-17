
import os, pickle
from scripts.pipeline_definitions import build_cca_pipeline
from paths import DPATHS, FPATHS

# settings
save_models = True

# model parameters: number of PCA components
n_components1 = 100
n_components2 = 100

# cross-validation parameters
verbose=True

# paths to data files
fpath_data = FPATHS['data_Xy_train']

# output path
dpath_out = DPATHS['cca']
fname_out_prefix = 'cca_results'

if __name__ == '__main__':

    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}.pkl')

    print('----- Parameters -----')
    print(f'n_components1:\t{n_components1}')
    print(f'n_components2:\t{n_components2}')
    print(f'save_models:\t{save_models}')
    print(f'verbose:\t{verbose}')
    print(f'fpath_data:\t{fpath_data}')
    print('----------------------')

    # load data
    with open(fpath_data, 'rb') as file_in:
        X, y = pickle.load(file_in)

    subjects = X.index

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

    preprocessor = cca_pipeline['preprocessor']
    cca = cca_pipeline['cca']

    # train on all data
    X_train = X

    # fit pipeline in 2 steps 
    # (keeping preprocessed train data for scoring later)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    cca.fit(X_train_preprocessed)

    # get train metrics
    loadings_train = cca.get_loadings(X_train_preprocessed)
    correlations_train = cca.score(X_train_preprocessed)

    # to be pickled
    results_all = {
        'model': cca_pipeline, # fitted model
        'loadings_train': loadings_train,
        'correlations_train': correlations_train,
        'subjects': subjects.tolist(),
        'latent_dims_names': latent_dims_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

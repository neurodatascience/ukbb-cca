
# TODO delete this script

import os, sys, pickle
from scripts.pipeline_definitions import build_cca_pipeline
from paths import DPATHS, FPATHS

# paths to data files
fpath_data = FPATHS['data_Xy_train']

# output paths
dpath_out_data = DPATHS['cca_preprocessed']
dpath_out_preprocessor = DPATHS['cca_preprocessor']

if __name__ == '__main__':

    # process user inputs
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <n_components1> <n_components2> [etc.]')
        sys.exit(1)
    n_components_all = [int(n) for n in sys.argv[1:]] # number of PCA components
    str_components = '_'.join([str(n) for n in n_components_all])

    # create output directories if necessary
    for dpath in [dpath_out_data, dpath_out_preprocessor]:
        Path(dpath).mkdir(parents=True, exist_ok=True)

    suffix = '_'.join([str(n) for n in n_components_all])
    fpath_out = os.path.join(dpath_out_data, f'X_train_{str_components}.pkl')
    fpath_out_preprocessor = os.path.join(dpath_out_preprocessor, f'preprocessor_{str_components}.pkl')

    print('----- Parameters -----')
    print(f'n_components_all:\t{n_components_all}')
    print(f'fpath_data:\t{fpath_data}')
    print('----------------------')

    # load data
    with open(fpath_data, 'rb') as file_data:
        data = pickle.load(file_data)
    X = data['X']
    y = data['y']
    dataset_names = data['dataset_names']
    n_datasets = len(dataset_names)
    conf_name = data['conf_name']
    udis = data['udis']

    subjects = X.index

    # process PCA n_components
    if len(n_components_all) != n_datasets:
        raise ValueError(f'Mismatch between n_components_all (size {len(n_components_all)}) and dataset_names (size {len(dataset_names)})')
    for i_dataset, dataset_name in enumerate(dataset_names):
        if n_components_all[i_dataset] is None:
            n_components_all[i_dataset] = X[dataset_name].shape[1]

    # build pipeline/model
    cca_pipeline = build_cca_pipeline(
        dataset_names=dataset_names,
        n_pca_components_all=n_components_all,
    )
    preprocessor = cca_pipeline['preprocessor']

    print('------------------------------------------------------------------')
    print(preprocessor)
    print('------------------------------------------------------------------') 

    # fit pipeline in 2 steps 
    X_preprocessed = preprocessor.fit_transform(X)

    # save results
    to_dump = {
        'X': X_preprocessed,
        'y': y,
        'subjects': subjects,
        'dataset_names': dataset_names,
        'n_datasets': n_datasets,
        'conf_name': conf_name,
        'udis': udis,
    }
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(to_dump, file_out)
    print(f'Saved data of "shape" {[dataset.shape for dataset in X_preprocessed]} to {fpath_out}')

    # also save fitted preprocessor
    with open(fpath_out_preprocessor, 'wb') as file_out:
        pickle.dump({'preprocessor': preprocessor}, file_out)
    print(f'Saved preprocessor to {fpath_out_preprocessor}')

import pickle
with open('results/cca/Xy_split.pkl', 'rb') as f: data = pickle.load(f); X = data['X'].iloc[:1000]; dataset_names=data['dataset_names']; n_components_all = [10, 20]; n_latent_dims = 2; verbosity = 1
from scripts.pipeline_definitions import build_cca_pipeline

preprocessing_params = {f'data_pipelines__{dataset_name}__pca__n_components': n_components for dataset_name, n_components in zip(dataset_names, n_components_all)}
cca_params = {'latent_dims': n_latent_dims}

cca_pipeline = build_cca_pipeline(dataset_names=dataset_names, verbosity=verbosity, preprocessing_params=preprocessing_params, cca_params=cca_params)

for keys, name in keys_and_names:
    tmp = average_nested_parameter([cca_pipeline, cca_pipeline, cca_pipeline], keys, name)
    if tmp is not None:
        if name == 'weights':
            print(f'{keys} {name} --> {[w.shape for w in tmp]}')
        elif len(tmp.shape) == 0:
            print(f'{keys} {name} --> {tmp}')

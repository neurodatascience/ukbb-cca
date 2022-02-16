
import os, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from sklearn.pipeline import Pipeline
from src import PreprocessingPipeline, PipelineXY

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from src import NanDeconfounder, NanPCA
from cca_zoo.models import CCA

from paths import DPATHS, FPATHS
from src.utils import load_data_df
from src.cca_utils import score_projections

# settings
save_models = True

# model parameters: number of PCA components
n_components1 = None
n_components2 = None

# cross-validation parameters
verbose=True
n_folds = 1 # set to 1 for no CV
shuffle = False
seed = None # TODO get seed from input argument

# paths to data files
fpath_data = FPATHS['data_Xy_train']

# output path
dpath_out = DPATHS['cca']
fname_out_prefix = 'cca_results'

def build_data_pipeline(**kwargs):
    steps = [
        ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
        ('deconfounder', NanDeconfounder()),
        ('pca', NanPCA()),
    ]
    pipeline = PipelineXY(steps, verbose=False)
    pipeline.set_params(**kwargs)
    return pipeline

def build_conf_pipeline(**kwargs):
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
    ]
    pipeline = Pipeline(steps, verbose=False)
    pipeline.set_params(**kwargs)
    return pipeline

if __name__ == '__main__':

    fpath_out = os.path.join(dpath_out, f'{fname_out_prefix}.pkl')

    print('----- Parameters -----')
    print(f'n_components1:\t{n_components1}')
    print(f'n_components2:\t{n_components2}')
    print(f'save_models:\t{save_models}')
    print(f'verbose:\t{verbose}')
    print(f'n_folds:\t{n_folds}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'fpath_data:\t{fpath_data}')
    print('----------------------')

    with open(fpath_data, 'rb') as file_in:
        X, y = pickle.load(file_in)

    subjects = np.array(X.index)

    # random state
    if shuffle:
        random_state = np.random.RandomState(seed=seed)
    else:
        random_state = None

    # process PCA n_components
    if n_components1 is None:
        n_components1 = X['data1'].shape[1]
    if n_components2 is None:
        n_components2 = X['data2'].shape[1]

    n_latent_dims = min(n_components1, n_components2)
    print(f'Using {n_latent_dims} latent dimensions')
    latent_dims_names = [f'CF{i+1}' for i in range(n_latent_dims)]

    # build final pipeline/model
    cca_pipeline = Pipeline([
        ('preprocessing', PreprocessingPipeline(
            data1_pipeline=build_data_pipeline(pca__n_components=n_components1),
            data2_pipeline=build_data_pipeline(pca__n_components=n_components2),
            conf_pipeline=build_conf_pipeline(),
            verbose=verbose,
        )),
        ('cca', CCA(latent_dims=n_latent_dims, random_state=random_state)),
    ], verbose=verbose)
    print('------------------------------------------------------------------')
    print(cca_pipeline)
    print('------------------------------------------------------------------')

    # cross-validation splitter
    if n_folds != 1:
        cv_split = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state).split
    else:
        cv_split = (lambda X, y: [(np.arange(X.shape[0]), np.asarray([]))])

    # cross-validation loop
    cv_results = []
    val_projections1_all = [] # estimated 'canonical factor scores' (data x weights) of validation sets
    val_projections2_all = []
    for index_train, index_val in cv_split(X, y):

        subjects_train = subjects[index_train]
        subjects_val = subjects[index_val]

        X_train = X.loc[subjects_train]
        X_val = X.loc[subjects_val]

        # fit model
        cca_pipeline_clone = clone(cca_pipeline)
        train_projections1, train_projections2 = cca_pipeline_clone.fit_transform(X_train)
        correlations_train = score_projections(train_projections1, train_projections2)
        
        # get predicted factor scores (shape: (n_subjects, n_latent_dims))
        # Note: some columns might have all NaNs (too many dimensions?)
        if n_folds != 1:
            val_projections1, val_projections2 = cca_pipeline_clone.transform(X_val)
            correlations_val = score_projections(val_projections1, val_projections2)
        # 1-fold case (no CV)
        else:
            val_projections1, val_projections2, correlations_val = [], [], []
        val_projections1_all.append(pd.DataFrame(val_projections1, subjects_val, latent_dims_names))
        val_projections2_all.append(pd.DataFrame(val_projections2, subjects_val, latent_dims_names))

        fold_results = {
            'subjects_train': subjects_train,
            'subjects_val': subjects_val,
            'correlations_train': correlations_train,
            'correlations_val': correlations_val,
        }

        # save entire model
        if save_models:
            fold_results['model'] = cca_pipeline_clone

        cv_results.append(fold_results)

    # get full set of factors from all CV folds combined
    df_projections1 = pd.concat(val_projections1_all, axis='index')
    df_projections2 = pd.concat(val_projections2_all, axis='index')
    if n_folds != 1:
        df_projections1 = df_projections1.loc[subjects] # use original subject order
        df_projections2 = df_projections2.loc[subjects]
    
    # to be pickled
    results_all = {
        'cv_results': cv_results,
        'df_projections1': df_projections1, 
        'df_projections2': df_projections2, 
        'subjects': subjects,
        'latent_dims_names': latent_dims_names,
    }

    # save results
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_all, file_out)
    print(f'Saved to {fpath_out}')

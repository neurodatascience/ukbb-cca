import warnings
import numpy as np

from sklearn.model_selection import KFold
from sklearn.base import clone

from .cca_utils import cca_score, cca_get_loadings
from .ensemble_model import EnsembleCCA

def cca_without_cv(X, i_train, i_test, model, preprocess=True, normalize_loadings=True, return_fitted_model=False):

    model = clone(model)

    X_train = _select_rows(X, i_train)
    X_test = _select_rows(X, i_test)

    if preprocess:

        # TODO remove
        nan_cols = X.columns[X.isna().all()]
        if len(nan_cols) > 0:
            warnings.warn(f'nan columns: {list(nan_cols)}')
        constant_cols = X.columns[(X == X.iloc[0]).all()]
        if len(constant_cols) > 0:
            warnings.warn(f'constant columns: {list(constant_cols)}')

        preprocessor = model['preprocessor']
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

    else:
        X_train_preprocessed = X_train
        X_test_preprocessed = X_test

    # print([np.sum(np.logical_not(np.isfinite(X))) for X in X_train_preprocessed])

    cca = model['cca']
    cca.fit(X_train_preprocessed)

    if return_fitted_model:
        return model
    else:
        results = {'corrs': {}, 'CAs': {}, 'loadings': {}}
        for set_name, X_preprocessed in {'learn': X_train_preprocessed, 'val': X_test_preprocessed}.items():
            CAs = cca.transform(X_preprocessed)
            results['CAs'][set_name] = CAs
            results['corrs'][set_name] = cca_score(CAs)

            np.set_printoptions(precision=4, linewidth=100, suppress=True, sign=' ')
            print(f'cca.score: {cca.score(X_preprocessed)[:10]}')
            print(f'cca_score: {results["corrs"][set_name][:10]}')

            results['loadings'][set_name] = cca_get_loadings(X_preprocessed, CAs, normalize=normalize_loadings)
        return results
    
def cca_cv(X, model, n_folds, preprocess=True, random_state=None, shuffle=True):

    model = clone(model)

    if not shuffle:
        random_state = None
    
    n_subjects = X.shape[0]
    cv_splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    
    fitted_models = []
    for i_train, i_test in cv_splitter.split(np.arange(n_subjects)):

        fitted_model = cca_without_cv(
            X, i_train, i_test, model,
            preprocess=preprocess, return_fitted_model=True)
        fitted_models.append(fitted_model)

    return fitted_models

def cca_repeated_cv(X, i_learn, i_val, model, n_repetitions, n_folds, 
        preprocess_before_cv=False, rotate_CAs=True, rotate_deconfs=False, 
        model_transform_cca=None, model_transform_deconfounder=None,
        ensemble_method='nanmean', normalize_loadings=True, random_state=None):

    def apply_ensemble_CCA(rotate, model_transform):
        ensemble_model = EnsembleCCA(fitted_models, rotate=rotate, model_transform=model_transform)
        result_learn = ensemble_model.fit_transform(X_learn, apply_ensemble_method=True, ensemble_method=ensemble_method)
        result_test = ensemble_model.transform(X_val, apply_ensemble_method=True, ensemble_method=ensemble_method)
        return result_learn, result_test

    model = clone(model)

    if random_state is None:
        random_state = np.random.RandomState()

    fitted_models = []

    if preprocess_before_cv:

        # TODO remove
        nan_cols = X.columns[X.isna().all()]
        if len(nan_cols) > 0:
            warnings.warn(f'nan columns: {list(nan_cols)}')
        constant_cols = X.columns[(X == X.iloc[0]).all()]
        if len(constant_cols) > 0:
            warnings.warn(f'constant columns: {list(constant_cols)}')

        preprocessor = model['preprocessor']
        X = preprocess.fit_transform(X)

    X_learn = _select_rows(X, i_learn)
    X_val = _select_rows(X, i_val)

    while (len(fitted_models) != (n_repetitions * n_folds)):

        try:
            fitted_models.extend(
                cca_cv(X_learn, model, n_folds, 
                preprocess=(not preprocess_before_cv), random_state=random_state))

        # try again if non-convergence error
        except np.linalg.LinAlgError as error:
            print(f'LinAlgError: {error}')
            continue

    # ensemble model
    CAs_learn, CAs_val = apply_ensemble_CCA(rotate=rotate_CAs, model_transform=model_transform_cca)
    CAs = {'learn': CAs_learn, 'val': CAs_val}
    
    # correlations
    corrs = {}
    for set_name in CAs.keys():
        corrs[set_name] = cca_score(CAs[set_name])

    # loadings
    deconfs_learn, deconfs_val = apply_ensemble_CCA(rotate=rotate_deconfs, model_transform=model_transform_deconfounder)
    deconfs = {'learn': deconfs_learn, 'val': deconfs_val}
    loadings = {}
    for set_name in deconfs.keys():
        loadings[set_name] = cca_get_loadings(CAs[set_name], deconfs[set_name], normalize=normalize_loadings)

    results = {
        'CAs': CAs,
        'corrs': corrs,
        'loadings': loadings,
    }

    return results

def _select_rows(data, indices):

    # pandas dataframe
    if hasattr(data, 'iloc'):
        return data.iloc[indices]

    # numpy array
    elif hasattr(data, 'shape'):
        return data[indices]

    # list of numpy arrays
    elif isinstance(data, list) and hasattr(data[0], 'shape'):
        return [_select_rows(d) for d in data]

    # error
    else:
        raise ValueError(f"Data type not handled: {data}")

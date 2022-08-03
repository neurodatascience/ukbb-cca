
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class EnsembleCCA(BaseEstimator, TransformerMixin):

    def __init__(self, models_orig, rotate=True, model_transform=None):

        self.n_models = len(models_orig)
        # self.models_orig = models_orig
        if model_transform is not None:
            self.models = [model_transform(model) for model in models_orig]
        else:
            self.models = models_orig
        self.rotate = rotate

    def fit(self, X, y=None):
        '''Fit the rotation matrices.'''
        self._fit(X)
        return self

    def _fit(self, X):
        '''Fit the rotation matrices.'''

        for i_model, model in enumerate(self.models):
            X_transformed = model.transform(X)

            if i_model == 0:
                n_datasets = len(X_transformed)
                Qs = [[] for _ in range(n_datasets)]
                X_transformed_all = [[] for _ in range(n_datasets)]
                X_transformed_ref = X_transformed

            for i_dataset in range(n_datasets):
                X_transformed_all[i_dataset].append(X_transformed[i_dataset])
                if self.rotate:
                    Q = self._find_rotation_matrix(X_transformed[i_dataset], X_transformed_ref[i_dataset])
                    Qs[i_dataset].append(Q)

        for i_dataset in range(n_datasets):
            X_transformed_all[i_dataset] = np.array(X_transformed_all[i_dataset])
            Qs[i_dataset] = np.array(Qs[i_dataset])

        self.n_datasets_ = n_datasets
        self.Qs_ = Qs

        return X_transformed_all

    def transform(self, X, y=None, pretransformed=False, apply_ensemble_method=False, ensemble_method='mean'):
        
        check_is_fitted(self)

        if pretransformed:
            X_transformed_all = X
        else:
            X_transformed_all = [[] for _ in range(self.n_datasets_)]
            for model in self.models:
                X_transformed = model.transform(X)
                for i_dataset in range(self.n_datasets_):
                    X_transformed_all[i_dataset].append(X_transformed[i_dataset])

        if self.rotate:
            X_transformed_all = self._rotate_X_transformed(X_transformed_all)
        
        if apply_ensemble_method:
            print(f'apply_method:\t{[np.array(X).shape for X in X_transformed_all]}')
            print(f'\tall nan:\t{[np.sum(np.all(np.isnan(X), axis=0)) for X in X_transformed_all]}')
            X_transformed_all = apply_method(X_transformed_all, ensemble_method=ensemble_method)

        for i_dataset in range(self.n_datasets_):
            X_transformed_all[i_dataset] = np.array(X_transformed_all[i_dataset])

        return X_transformed_all

    def fit_transform(self, X, y=None, apply_ensemble_method=False, ensemble_method='mean'):
        X_transformed_all = self._fit(X)
        X_transformed_all = self.transform(
            X_transformed_all, pretransformed=True, 
            apply_ensemble_method=apply_ensemble_method,
            ensemble_method=ensemble_method,
        )
        return X_transformed_all

    def _find_rotation_matrix(self, X_transformed, X_transformed_ref):
        '''Orthogonal Procrustes method.'''
        U, _, Vt = np.linalg.svd(X_transformed.T @ X_transformed_ref)
        Q = U @ Vt
        return Q

    def _rotate_X_transformed(self, X_transformed_all):
        
        X_transformed_all_rotated = [[] for _ in range(self.n_datasets_)]

        for i_dataset in range(self.n_datasets_):
            X_transformed_all_rotated[i_dataset] = X_transformed_all[i_dataset] @ self.Qs_[i_dataset]

        return X_transformed_all_rotated

def apply_method(X_transformed_all, ensemble_method='mean'):

    valid_ensemble_methods = {
        'mean': (lambda x: np.mean(x, axis=0)),
        'median': (lambda x: np.median(x, axis=0)),
        'nanmean': (lambda x: np.nanmean(x, axis=0)),
        'nanmedian': (lambda x: np.nanmedian(x, axis=0)),
    }
    if callable(ensemble_method):
        ensemble_function = ensemble_method
    else:
        if ensemble_method not in valid_ensemble_methods.keys():
            raise ValueError(f'Invalid ensemble method. Valid ones are: {valid_ensemble_methods.keys()}')
        ensemble_function = valid_ensemble_methods[ensemble_method]

    n_datasets = len(X_transformed_all)

    X_transformed_ensemble = [[] for _ in range(n_datasets)]
    for i_dataset in range(n_datasets):
        X_transformed_ensemble[i_dataset] = ensemble_function(X_transformed_all[i_dataset])
    return X_transformed_ensemble

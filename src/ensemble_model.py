
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class EnsembleCCA(BaseEstimator, TransformerMixin):

    def __init__(self, models_orig, rotate=True, model_transform=None, use_scipy_procrustes=False):

        self.n_models = len(models_orig)
        # self.models_orig = models_orig
        if model_transform is not None:
            self.models = [model_transform(model) for model in models_orig]
        else:
            self.models = models_orig
        self.rotate = rotate
        self.use_scipy_procrustes = use_scipy_procrustes

    def fit(self, X, y=None):
        '''Fit the rotation matrices.'''
        self._fit(X)
        return self

    def _fit(self, X):
        '''Fit the rotation matrices.'''

        for i_model, model in enumerate(self.models):
            X_transformed = model.transform(X)
            if self.use_scipy_procrustes:
                X_transformed = [self._scipy_procrustes_standardize(X) for X in X_transformed] # to match scipy.spatial.procrustes

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
            # Qs[i_dataset] = np.array(Qs[i_dataset])

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
            # print(f'apply_method:\t{[np.array(X).shape for X in X_transformed_all]}')
            # print(f'\tall nan:\t{[np.sum(np.all(np.isnan(X), axis=0)) for X in X_transformed_all]}')
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
    
    def _scipy_procrustes_standardize(self, matrix):

        matrix = np.array(matrix, dtype=np.double, copy=True)

        # # translate all the data to the origin
        # matrix -= np.mean(matrix, 0)

        # # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        # norm = np.linalg.norm(matrix)
        # if norm == 0:
        #     raise ValueError("Input matrices must contain >1 unique points")
        
        # matrix /= norm

        return matrix

    def _find_rotation_matrix(self, X_transformed, X_transformed_ref):
        '''Orthogonal Procrustes method.'''
        if self.use_scipy_procrustes:

            # # transform X_transformed to minimize disparity with X_transformed_ref
            # R, s = orthogonal_procrustes(X_transformed_ref, X_transformed)
            # disparity = np.sum(np.square(X_transformed_ref - self._apply_scipy_procrustes(R, s, X_transformed)))
            # print(f'Disparity: {disparity}')
            # rotation_info = (R, s)

            Q, _ = orthogonal_procrustes(X_transformed_ref, X_transformed)

        else:
            U, _, Vt = np.linalg.svd(X_transformed.T @ X_transformed_ref)
            Q = U @ Vt
            # rotation_info = Q

        return Q
        # return rotation_info

    def _rotate_X_transformed(self, X_transformed_all):
        
        X_transformed_all_rotated = [[] for _ in range(self.n_datasets_)]

        for i_dataset in range(self.n_datasets_):
            # if self.use_scipy_procrustes:
            #     for X_transformed, (R, s) in zip(X_transformed_all[i_dataset], self.Qs_[i_dataset]):
            #         X_transformed_all_rotated[i_dataset].append(self._apply_scipy_procrustes(R, s, X_transformed))
            #     X_transformed_all_rotated[i_dataset] = np.array(X_transformed_all_rotated[i_dataset])
            # else:
            #     X_transformed_all_rotated[i_dataset] = X_transformed_all[i_dataset] @ np.array(self.Qs_[i_dataset])
            X_transformed_all_rotated[i_dataset] = X_transformed_all[i_dataset] @ np.array(self.Qs_[i_dataset])

        return X_transformed_all_rotated
    
    def _apply_scipy_procrustes(self, R, s, matrix):
        return np.dot(matrix, R.T) * s

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

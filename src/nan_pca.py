
# TODO use pandas cov
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import stable_cumsum
from .utils import nearest_spd, eig_flip

class NanPCA(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def format_transformed(self, X):
        if hasattr(self, 'feature_names_in_'):
            return pd.DataFrame(
                data=X, 
                columns=self.feature_names_in_,
            )
        return X

    def fit(self, X, y=None):

        n_samples, n_features = X.shape

        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components

        # check if features have at least 1 finite value
        if np.sum(np.all(np.isnan(X), axis=0)) > 0:
            warnings.warn('Some columns only contain NaNs')
        
        # mask NaNs/infinite values
        X = np.ma.masked_invalid(X)

        # demean
        self.mean_ = X.mean(axis=0).filled()
        X = X - self.mean_

        # compute covariance matrix (n_features x n_features)
        Xt_X = np.ma.dot(X.T, X).filled()

        # project to nearest symmetric positive definite matrix
        Xt_X = nearest_spd(Xt_X)

        # eigendecomposition
        eigenvals, eigenvecs = np.linalg.eig(Xt_X)

        # sort components by decreasing eigenvalue
        i_sort = np.flip(np.argsort(eigenvals))
        eigenvals = np.real(eigenvals[i_sort])
        eigenvecs = np.real(eigenvecs[:, i_sort])

        # eigenvector sign convention for consistent signs
        eigenvecs = eig_flip(eigenvecs)

        components_ = eigenvecs.T # transpose for consistency with sklearn PCA

        explained_variance_ = eigenvals / (n_samples - 1)
        explained_variance_ratio_ = explained_variance_ / explained_variance_.sum()

        # process number of components required
        if 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components, side='right') + 1

        # compute noise covariance using Probabilistic PCA model
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        # store fitted parameters
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.eigenvalues_ = eigenvals[:n_components]

        return self

    def transform(self, X):
        
        check_is_fitted(self)

        X = np.ma.masked_invalid(X)
        X = X - self.mean_

        X_transformed = np.ma.dot(X, self.components_.T).filled()

        if np.sum(np.isnan(X_transformed)) != 0:
            raise Exception('PCA transform is returning NaNs')

        return self.format_transformed(X_transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt):
        return np.dot(Xt, self.components_) + self.mean_

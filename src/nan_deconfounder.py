
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO accept DataFrame type for conf

class NanDeconfounder(TransformerMixin, BaseEstimator):

    def fit(self, X, conf):

        if conf is None:
            return self

        self._check_conf(conf)
        X = np.ma.masked_invalid(X)
        weights = np.ma.dot(np.linalg.pinv(conf), X)
        self.weights_ = weights

        return self

    def format_transformed(self, X):
        if hasattr(self, 'feature_names_in_'):
            return pd.DataFrame(
                data=X, 
                columns=self.feature_names_in_,
            )
        return X

    def transform(self, X, conf):

        if conf is None:
            return X

        check_is_fitted(self)
        self._check_conf(conf)

        X = np.ma.masked_invalid(X)
        X_transformed = X - np.ma.dot(conf, self.weights_)
        return self.format_transformed(X_transformed.filled(np.nan))

    def fit_transform(self, X, conf):
        return self.fit(X, conf).transform(X, conf)

    def _check_conf(self, conf):
        if hasattr(conf, 'to_numpy'):
            conf = conf.to_numpy()
        if np.isnan(conf).sum() != 0:
            raise ValueError('conf cannot contain NaNs')
        return conf

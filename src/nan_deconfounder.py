
import numpy as np
from sklearn.utils.validation import check_is_fitted

# TODO accept DataFrame type for conf

class NanDeconfounder():

    def fit(self, X, conf):

        self._check_conf(conf)
        
        X = np.ma.masked_invalid(X)
        weights = np.ma.dot(np.linalg.pinv(conf), X)
        self.weights_ = weights

        return self

    def transform(self, X, conf):

        check_is_fitted(self)
        self._check_conf(conf)

        X = np.ma.masked_invalid(X)
        X_transformed = X - np.ma.dot(conf, self.weights_)
        return X_transformed.filled(np.nan)

    def fit_transform(self, X, conf):
        return self.fit(X, conf).transform(X, conf)

    def _check_conf(self, conf):
        try:
            if np.isnan(conf).sum() != 0:
                raise ValueError('conf cannot contain NaNs')
        except ValueError:
            raise ValueError('check if conf is pd.DataFrame (not handled yet)')

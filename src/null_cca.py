import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import select_rows

class NullCCA(BaseEstimator, TransformerMixin):
    """Wrapper class for null CCA model."""

    def __init__(self, base_model: BaseEstimator, n_views, random_state: np.random.RandomState) -> None:
        super().__init__()
        self.base_model = base_model
        self.n_views = n_views
        self.random_state = random_state
        self.models = [clone(base_model) for _ in range(n_views)]

    def _check_X(self, X):

        # X is a list of views (like in cca_zoo)
        if not isinstance(X, list):
            raise RuntimeError(f'Expected X to be a list, got: {type(X)}')
        if len(X) != len(self.models):
            raise RuntimeError(f'Expected X to have {len(self.models)} views, got: {len(X)}')

    def fit(self, X, y=None):

        self._check_X(X)
        
        random_idx = self.random_state.permutation(len(X[0]))
        X_shuffled = [
            select_rows(view, random_idx)
            for view in X
        ]
        for i_view, model in enumerate(self.models):
            X_tmp = X_shuffled[:]  # shallow copy
            X_tmp[i_view] = X[i_view]  # keep one view unshuffled
            model.fit(X_tmp)

    def transform(self, X, y=None):

        self._check_X(X)

        X_transformed = []
        for i_view, null_cca_model in enumerate(self.models):
            X_transformed.append(null_cca_model.transform(X)[i_view])
        
        return X_transformed
    
    def set_params(self, **params):
        self.base_model.set_params(**params)
        return self
    
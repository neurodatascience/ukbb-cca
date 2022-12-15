import warnings
from sklearn.cross_decomposition._pls import CCA, _PLS

class _SklearnPLS(_PLS):

    @property
    def sklearn_class(self) -> _PLS:
        raise NotImplementedError

    @staticmethod
    def _check_views(views):
        if len(views) != 2:
            raise ValueError(
                f'Views must have exactly 2 datasets (got {len(views)})'
            )
        return views

    def _sklearn_function(self, func_name, views, with_X=True, with_Y=True, **kwargs):
        func = getattr(self.sklearn_class, func_name)
        X, Y = self._check_views(views)
        if with_X:
            kwargs['X'] = X
        if with_Y:
            kwargs['Y'] = Y
        return func(self, **kwargs)

    def fit(self, views):
        return self._sklearn_function('fit', views)

    def transform(self, views, copy=True):
        return self._sklearn_function('transform', views, copy=copy)

    def inverse_transform(self, views):
        return self._sklearn_function('inverse_transform', views)

    def predict(self, views, copy=True):
        return self._sklearn_function('predict', views, with_Y=False, copy=copy)

    def fit_transform(self, views):
        return self._sklearn_function('fit_transform', views)

class SklearnCCA(_SklearnPLS, CCA):

    sklearn_class = CCA

    def __init__(self, latent_dims=None, n_components=2, *, scale=True,
                 max_iter=1000, tol=0.000001, copy=True, **kwargs):

        if len(kwargs) != 0:
            warnings.warn(
                f'{type(self).__name__} will ignore these parameters: {kwargs}', 
                UserWarning,
            )
        
        if latent_dims is not None:
            n_components = latent_dims

        self.sklearn_class.__init__(
            self,
            n_components=n_components, 
            scale=scale, 
            max_iter=max_iter, 
            tol=tol, 
            copy=copy,
        )
        self.latent_dims = n_components

    @property
    def latent_dims(self):
        return self.n_components

    @latent_dims.setter
    def latent_dims(self, value):
        self.set_params(n_components=value)

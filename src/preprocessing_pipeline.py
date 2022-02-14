
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src import NanDeconfounder, NanPCA

class PreprocessingPipeline():

    class PipelineXY(Pipeline):
        def transform(self, X, y=None):
            '''Same as transform() in Pipeline class but accepts y variable (needed in custom transformer classes).'''
            Xt = X
            for _, _, transform in self._iter():
                try:
                    Xt = transform.transform(Xt)
                except TypeError: # missing positional argument
                    Xt = transform.transform(Xt, y)
            return Xt

    def __init__(self, dataset_names={'data1', 'data2'}, conf_name='conf', verbose=False):

        self.dataset_names = dataset_names
        self.conf_name = conf_name
        self.verbose = verbose

        conf_steps = [
            ('conf_imputer', SimpleImputer(strategy='median')),
            ('conf_inv_norm', QuantileTransformer(output_distribution='normal')),
            ('conf_scaler', StandardScaler()),
        ]

        self.conf_pipeline = Pipeline(conf_steps, verbose=verbose)

        self.data_pipelines = {}
        for dataset_name in dataset_names:
            data_steps = [
                (f'{dataset_name}_inv_norm', QuantileTransformer(output_distribution='normal')),
                (f'{dataset_name}_scaler', StandardScaler()),
                (f'{dataset_name}_deconfounder', NanDeconfounder()),
                (f'{dataset_name}_pca', NanPCA()),
            ]
            self.data_pipelines[dataset_name] = self.PipelineXY(data_steps, verbose=verbose)

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _fit(self, X):

        self._check_X(X)

        # preprocess confounds, if any
        conf_preprocessed = self._preprocess_confounds(X, fit=True)

        # preprocess each dataset separately
        for dataset_name in self.dataset_names:

            if self.verbose:
             print(f'[Preprocessing pipeline] Processing {dataset_name}')

            self.data_pipelines[dataset_name].fit(X[dataset_name], conf_preprocessed)

        return conf_preprocessed

    def transform(self, X, y=None, conf_preprocessed=None):

        self._check_X(X)

        # preprocess confounds if necessary
        if conf_preprocessed is None:
            conf_preprocessed = self._preprocess_confounds(X)

        views = []
        for dataset_name in self.dataset_names:
            view = self.data_pipelines[dataset_name].transform(X[dataset_name], y=conf_preprocessed)
            views.append(view)

        # return a list of preprocessed datasets ('views' in cca-zoo)
        return views

    def fit_transform(self, X, y=None):
        # avoids preprocessing conf twice
        conf_preprocessed = self._fit(X)
        return self.transform(X, conf_preprocessed=conf_preprocessed)

    def _check_X(self, X):
        # top level unique keys (except for conf_name)
        dataset_names = set(X.drop(columns=self.conf_name, errors='ignore').columns.get_level_values(0))
        if dataset_names != self.dataset_names:
            raise ValueError(f'Mismatch between dataset names. Expected {self.dataset_names} (conf: {self.conf_name}), got {dataset_names})')

    def _preprocess_confounds(self, X, fit=False):

        try:
            conf = X[self.conf_name]
        except KeyError:
            return None

        if self.verbose:
            print(f'[Preprocessing pipeline] Processing {self.conf_name}')

        if fit:
            return self.conf_pipeline.fit_transform(conf)
        else:
            return self.conf_pipeline.transform(conf)

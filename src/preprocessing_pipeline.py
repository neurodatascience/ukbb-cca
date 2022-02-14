
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src import NanDeconfounder, NanPCA

class PreprocessingPipeline(_BaseComposition):

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

    def __init__(self, 
        data1_pipeline=None, data2_pipeline=None, conf_pipeline=None, 
        data1_name='data1', data2_name='data2', conf_name='conf', verbose=False
    ):

        def build_data_pipeline():
            data_steps = [
                (f'inv_norm', QuantileTransformer(output_distribution='normal')),
                (f'scaler', StandardScaler()),
                (f'deconfounder', NanDeconfounder()),
                (f'pca', NanPCA()),
            ]
            return self.PipelineXY(data_steps, verbose=verbose)

        self.data1_name = data1_name
        self.data2_name = data2_name
        self.conf_name = conf_name
        self.verbose = verbose

        # TODO move this to main CV script
        conf_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('inv_norm', QuantileTransformer(output_distribution='normal')),
            ('scaler', StandardScaler()),
        ]
        self.conf_pipeline = Pipeline(conf_steps, verbose=verbose)
        self.data1_pipeline = build_data_pipeline()
        self.data2_pipeline = build_data_pipeline()

        self.dataset_names = [data1_name, data2_name]
        self.data_pipelines = {data1_name: self.data1_pipeline, data2_name: self.data2_pipeline}

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
        if len(dataset_names) != 2:
            raise ValueError(f'X must contain exactly 2 datasets (excluding confounders)')
        if dataset_names != set(self.dataset_names):
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


from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import Bunch
from sklearn.pipeline import Pipeline

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

class PipelineList(_BaseComposition):
    def __init__(self, pipelines):
        self.pipelines = pipelines
        self._validate_names([name for name, _ in pipelines])

    def get_params(self, deep=True):
        return self._get_params('pipelines', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('pipelines', **kwargs)
        return self

    def __len__(self):
        '''Returns the length of the list.'''
        return len(self.pipelines)

    def __getitem__(self, ind):
        '''
        Returns a pipeline from the list. 
        Index can be either an int or the name of the pipeline.
        '''
        if isinstance(ind, slice):
            raise ValueError('PipelineList does not support indexing by slice')
        try:
            name, pipeline = self.pipelines[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_pipelines[ind]
        return pipeline

    @property
    def named_pipelines(self):
        '''
        Access the pipelines by name.
        Read-only attribute to access any pipeline in the list by given name.
        Keys are pipeline names and values are the pipeline objects.
        '''
        
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.pipelines))

class PreprocessingPipeline(_BaseComposition):

    def __init__(self, data_pipelines, conf_pipeline=None,
        dataset_names=None, conf_name='conf', verbose=False):

        # input validation
        if (dataset_names is not None) and (len(data_pipelines) != len(dataset_names)):
            print(f'Mismatch between number of data pipelines ({len(data_pipelines)}) and dataset names ({len(dataset_names)})')

        if dataset_names is None:
            dataset_names = [f'data{i+1}' for i in range(len(data_pipelines))]

        self.data_pipelines = data_pipelines
        self.dataset_names = dataset_names
        self.conf_pipeline = conf_pipeline
        self.conf_name = conf_name
        self.verbose = verbose

        # to allow accessing the data/conf pipelines by name
        self.pipelines = [(name, pipeline) for (name, pipeline) in vars(self).items() if 'pipeline' in name]

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

            data_pipeline = self.data_pipelines[dataset_name]
            data_pipeline.fit(X[dataset_name], conf_preprocessed)

        return conf_preprocessed

    def transform(self, X, y=None, conf_preprocessed=None):

        self._check_X(X)

        # preprocess confounds if necessary
        if conf_preprocessed is None:
            conf_preprocessed = self._preprocess_confounds(X)

        views = []
        for dataset_name in self.dataset_names:
            data_pipeline = self.data_pipelines[dataset_name]
            view = data_pipeline.transform(X[dataset_name], y=conf_preprocessed)
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
        if dataset_names != set(self.dataset_names):
            raise ValueError(f'Mismatch between dataset names. Expected {self.dataset_names} (conf: {self.conf_name}), got {dataset_names})')

    def _preprocess_confounds(self, X, fit=False):

        if self.conf_pipeline is None:
            return None

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

    def __getitem__(self, ind):
        '''
        Returns a data/conf Pipeline or PipelineList. 
        Index can be either an int or the name of the pipeline.
        '''
        if isinstance(ind, slice):
            raise ValueError('PipelineList does not support indexing by slice')
        try:
            name, pipeline = self.pipelines[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_pipelines[ind]
        return pipeline

    @property
    def named_pipelines(self):
        '''
        Access the pipelines by name.
        Read-only attribute to access any pipeline in the list by given name.
        Keys are pipeline names and values are the pipeline objects.
        '''
        
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.pipelines))


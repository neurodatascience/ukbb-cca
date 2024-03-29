from sklearn.pipeline import Pipeline
from src import PreprocessingPipeline, PipelineXY, PipelineList, UkbbSquarer

from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from . import NanDeconfounder, NanPCA#, NullCCA
from . import FilteringQuantileTransformer
from . import FeatureSelectorMissing, FeatureSelectorHighFreq, FeatureSelectorOutlier
from .sklearn_cca import SklearnCCA

from cca_zoo.models import CCA

def process_verbosity(verbosity):
    return verbosity > 0

def build_data_pipeline(verbosity=0, **kwargs):
    steps = [
        # ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('inv_norm', FilteringQuantileTransformer(output_distribution='normal')),
        # ('scaler', StandardScaler()),
        ('scaler', RobustScaler()),
        ('deconfounder', NanDeconfounder()),
        ('pca', NanPCA()),
    ]
    # steps = [
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('inv_norm', QuantileTransformer(output_distribution='normal')),
    #     ('scaler', StandardScaler()),
    #     ('deconfounder', NanDeconfounder()),
    #     ('pca', PCA()),
    # ]
    pipeline = PipelineXY(steps, verbose=process_verbosity(verbosity))
    pipeline.set_params(**kwargs)
    return pipeline

def build_conf_pipeline(verbosity=0, **kwargs):
    steps = [
        # ('imputer', SimpleImputer(strategy='median')),
        ('imputer', IterativeImputer(max_iter=100)),
        # ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('inv_norm', FilteringQuantileTransformer(output_distribution='normal')),
        # ('scaler', StandardScaler()),
        ('scaler', RobustScaler()),
        ('squarer', UkbbSquarer()),
        # ('pca', NanPCA()),
    ]
    pipeline = Pipeline(steps, verbose=process_verbosity(verbosity))
    params_to_remove = []
    for param in kwargs:
        step_name = param.split('__')[0]
        if step_name not in [step[0] for step in steps]:
            print(f'Ignoring conf pipeline parameter: {param}')
            params_to_remove.append(param)
    for param in params_to_remove:
        kwargs.pop(param)
    pipeline.set_params(**kwargs)
    return pipeline

def build_cca_pipeline(dataset_names, conf_name='conf', n_PCs_all=None, n_CAs=1, verbosity=0, kwargs_conf=None, null_model=False, null_model_random_state=None, use_specialized_null_cca=True):

    if n_PCs_all is None:
        n_PCs_all = [1 for _ in range(len(dataset_names))]
    if kwargs_conf is None:
        kwargs_conf = {}

    data_pipelines = [build_data_pipeline(verbosity=verbosity-2) for _ in dataset_names]

    # if use_specialized_null_cca and null_model:
    #     cca_model = NullCCA(
    #         base_model=CCA(),
    #         n_views=len(dataset_names),
    #         random_state=null_model_random_state,
    #     )
    # else:
    cca_model = CCA()

    steps = [
        ('preprocessor', PreprocessingPipeline(
            dataset_names=dataset_names,
            conf_name=conf_name,
            data_pipelines=PipelineList(list(zip(dataset_names, data_pipelines))),
            conf_pipeline=build_conf_pipeline(verbosity=verbosity-2, **kwargs_conf),
            verbose=process_verbosity(verbosity-1),
        )),
        ('cca', cca_model),
        # ('cca', SklearnCCA()),
    ]
    pipeline = Pipeline(steps, verbose=process_verbosity(verbosity))
    
    preprocessing_params = {
        f'data_pipelines__{dataset_name}__pca__n_components': n_components 
        for dataset_name, n_components in zip(dataset_names, n_PCs_all)
    }
    pipeline['preprocessor'].set_params(**preprocessing_params)
    
    cca_params = {
        'latent_dims': n_CAs,
    }
    pipeline['cca'].set_params(**cca_params)

    return pipeline

def build_feature_selector(verbosity=0, dropped_features_dict=None, **kwargs):
    steps = [
        ('remove_missing', FeatureSelectorMissing(threshold=0.5, dropped_features_dict=dropped_features_dict)),
        ('remove_high_freq', FeatureSelectorHighFreq(threshold=0.80, dropped_features_dict=dropped_features_dict)),
        # ('remove_outlier', FeatureSelectorOutlier(threshold=100, with_scaler=True, with_inv_norm=True, dropped_features_dict=dropped_features_dict)),
    ]
    pipeline = Pipeline(steps, verbose=process_verbosity(verbosity))
    pipeline.set_params(**kwargs)
    return pipeline

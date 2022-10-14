from sklearn.pipeline import Pipeline
from src import PreprocessingPipeline, PipelineXY, PipelineList

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import NanDeconfounder, NanPCA
from cca_zoo.models import CCA

def process_verbosity(verbosity):
    return verbosity > 0

def build_data_pipeline(verbosity=0, **kwargs):
    steps = [
        # ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
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
        ('imputer', SimpleImputer(strategy='median')),
        # ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
        # ('pca', NanPCA()),
    ]
    pipeline = Pipeline(steps, verbose=process_verbosity(verbosity))
    pipeline.set_params(**kwargs)
    return pipeline

def build_cca_pipeline(dataset_names=None, conf_name='conf', n_PCs_all=None, n_CAs=1, verbosity=0):

    if dataset_names is None:
        dataset_names = [f'data{i+1}' for i in range(len(dataset_names))]
    if n_PCs_all is None:
        n_PCs_all = [1 for _ in range(len(dataset_names))]

    data_pipelines = [build_data_pipeline(verbosity=verbosity-2) for _ in dataset_names]

    steps = [
        ('preprocessor', PreprocessingPipeline(
            dataset_names=dataset_names,
            conf_name=conf_name,
            data_pipelines=PipelineList(list(zip(dataset_names, data_pipelines))),
            conf_pipeline=build_conf_pipeline(verbosity=verbosity-2),
            verbose=process_verbosity(verbosity-1),
        )),
        ('cca', CCA()),
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

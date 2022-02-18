
from sklearn.pipeline import Pipeline
from src import PreprocessingPipeline, PipelineXY

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from src import NanDeconfounder, NanPCA
from cca_zoo.models import CCA

def build_data_pipeline(**kwargs):
    steps = [
        ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
        ('deconfounder', NanDeconfounder()),
        ('pca', NanPCA()),
    ]
    pipeline = PipelineXY(steps)
    pipeline.set_params(**kwargs)
    return pipeline

def build_conf_pipeline(**kwargs):
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
    ]
    pipeline = Pipeline(steps)
    pipeline.set_params(**kwargs)
    return pipeline

def build_cca_pipeline(n_pca_components_all, dataset_names=None, conf_name='conf', verbose=False, **kwargs):

    if dataset_names is None:
        dataset_names = [f'data{i+1}' for i in range(len(n_pca_components_all))]

    data_pipelines = []
    for n_pca_components in n_pca_components_all:
        data_pipelines.append(build_data_pipeline(pca__n_components=n_pca_components))

    steps = [
        ('preprocessor', PreprocessingPipeline(
            dataset_names=dataset_names,
            data_pipelines=data_pipelines,
            conf_pipeline=build_conf_pipeline(),
        )),
        ('cca', CCA()),
    ]
    pipeline = Pipeline(steps, verbose=verbose)
    pipeline.set_params(**kwargs)
    return pipeline

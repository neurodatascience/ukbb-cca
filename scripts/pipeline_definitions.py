
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

def build_cca_pipeline(n_pca_components1=None, n_pca_components2=None, verbose=False, **kwargs):
    steps = [
        ('preprocessor', PreprocessingPipeline(
            data1_pipeline=build_data_pipeline(pca__n_components=n_pca_components1),
            data2_pipeline=build_data_pipeline(pca__n_components=n_pca_components2),
            conf_pipeline=build_conf_pipeline(),
        )),
        ('cca', CCA()),
    ]
    pipeline = Pipeline(steps, verbose=verbose)
    pipeline.set_params(**kwargs)
    return pipeline

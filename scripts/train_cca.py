
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from src import NanDeconfounder, NanPCA
from cca_zoo.models import CCA

from src import PreprocessingPipeline, PipelineXY

from paths import FPATHS
from src.utils import load_data_df

verbose=True

# cross-validation parameters
n_folds = 5
shuffle = False
# seed = None # TODO use RandomState

# paths to data files
fpath_data1 = FPATHS['data_behavioural_clean']
fpath_data2 = FPATHS['data_brain_clean']
fpath_conf = FPATHS['data_demographic_clean']
fpath_train_subjects = FPATHS['subjects_train'] # subject IDs
fpath_groups = FPATHS['data_groups_clean']

def build_data_pipeline():
    steps = [
        ('inv_norm', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler()),
        ('deconfounder', NanDeconfounder()),
        ('pca', NanPCA()),
    ]
    return PipelineXY(steps, verbose=verbose)

def build_conf_pipeline():
    steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('inv_norm', QuantileTransformer(output_distribution='normal')),
            ('scaler', StandardScaler()),
    ]
    return Pipeline(steps, verbose=verbose)

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'verbose:\t{verbose}')
    print(f'n_folds:\t{n_folds}')
    print(f'shuffle:\t{shuffle}')
    # print(f'seed:\t{seed}')
    print(f'fpath_data1:\t{fpath_data1}')
    print(f'fpath_data2:\t{fpath_data2}')
    print(f'fpath_conf:\t{fpath_conf}')
    print(f'fpath_train_subjects:\t{fpath_train_subjects}')
    print(f'fpath_groups:\t{fpath_groups}')
    print('----------------------')

    # extract subjects IDs
    train_subjects = pd.read_csv(fpath_train_subjects).squeeze('columns')[:2000]

    # load all datasets and extract subjects
    df_data1 = load_data_df(fpath_data1, encoded=True).loc[train_subjects]
    df_data2 = load_data_df(fpath_data2, encoded=True).loc[train_subjects]
    df_conf = load_data_df(fpath_conf, encoded=True).loc[train_subjects]
    dfs_dict = {'data1': df_data1, 'data2': df_data2, 'conf': df_conf}

    # grouping variable for stratification
    groups = load_data_df(fpath_groups, encoded=False).squeeze('columns').loc[train_subjects]

    # drop one of the multiindex levels (after storing it)
    # (some sklearn classes cannot handle multiindex columns)
    udis = {}
    level_to_drop = 'udi'
    for dataset_name, df in dfs_dict.items():
        udis[dataset_name] = df.columns.get_level_values(level_to_drop)
        dfs_dict[dataset_name] = dfs_dict[dataset_name].droplevel(level_to_drop, axis='columns')

    # combine into a single big dataframe
    # for compatibility with sklearn Pipeline
    train_data = pd.concat(dfs_dict, axis='columns').loc[train_subjects]

    # make CV splitter object
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)

    # make preprocessing pipeline
    cca_pipeline = Pipeline([
        ('preprocessing', PreprocessingPipeline(
            data1_pipeline=build_data_pipeline(),
            data2_pipeline=build_data_pipeline(),
            conf_pipeline=build_conf_pipeline(),
            verbose=verbose,
        )),
        ('cca', CCA(latent_dims=1)),
    ])

    # run cross-validation
    cv_results = cross_validate(
        cca_pipeline, train_data, groups, 
        cv=cv_splitter, return_estimator=True, verbose=verbose,
    )

    # add info to results dict
    cv_results['udis'] = udis

    # temporary, for testing
    with open('test.pkl', 'wb') as file_out:
        pickle.dump(cv_results, file_out)

    # TODO
    # get score (R2)
    # extract coefficients from saved estimators (?)

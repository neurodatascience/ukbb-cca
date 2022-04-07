
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from paths import FPATHS
from src.utils import load_data_df, make_parent_dir

# parameters
shuffle = True
seed = 3791
column_level_to_drop = 'udi'
dataset_names = ['behavioural', 'brain']
conf_name = 'conf'

# input paths
fpaths_data = [FPATHS['data_behavioural_clean'], FPATHS['data_brain_clean']]
fpath_conf = FPATHS['data_demographic_clean']
fpath_groups = FPATHS['data_groups_clean']
fpath_holdout = FPATHS['data_holdout_clean']

udi_holdout = '21003-2.0'

# output
fpath_out = FPATHS['data_Xy']

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} n_splits')
        sys.exit(1)

    n_splits = int(sys.argv[1])

    print('----- Parameters -----')
    print(f'n_splits:\t{n_splits}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'column_level_to_drop:\t{column_level_to_drop}')
    print(f'dataset_names:\t{dataset_names}')
    print(f'conf_name:\t{conf_name}')
    print(f'fpaths_data:\t{fpaths_data}')
    print(f'fpath_conf:\t{fpath_conf}')
    print(f'fpath_groups:\t{fpath_groups}')
    print(f'fpath_holdout\t{fpath_holdout}')
    print(f'udi_holdout\t{udi_holdout}')
    print('----------------------')

    # input validation
    if len(dataset_names) != len(fpaths_data):
        raise ValueError('dataset_names and fpaths_data must have the same length')

    # build dict of dataframes
    print('Loading data...')
    dfs_dict = {}
    udis_datasets = []
    udis_conf = None
    n_features_datasets = []
    n_features_conf = None
    subjects = None
    for name, fpath in zip(dataset_names + [conf_name], fpaths_data + [fpath_conf]):

        # load data
        df = load_data_df(fpath, encoded=True)
        dfs_dict[name] = df
        print(f'\t{name}:\t{df.shape}')

        # get UDIs and number of features
        udis = df.columns
        n_features = df.shape[1]

        # make sure all datasets contain the same subjects
        subjects_new = set(df.index)
        if subjects is not None:
            if subjects_new != subjects:
                raise ValueError(f'Datasets {list(df.keys())} do not have exactly the same subjects')
        subjects = subjects_new

        if name in dataset_names:
            udis_datasets.append(udis)
            n_features_datasets.append(n_features)
        else:
            udis_conf = udis
            n_features_conf = n_features
    
    # make sure all subjects are in the same order
    subjects_sorted = np.sort(list(subjects))
    for name in dfs_dict.keys():
        dfs_dict[name] = dfs_dict[name].loc[subjects_sorted]

    # load group data (for stratification)
    groups = load_data_df(fpath_groups, encoded=False).squeeze('columns')[subjects_sorted]

    # load holdout data (for prediction)
    holdout = load_data_df(fpath_holdout).loc[subjects_sorted, udi_holdout]

    # combine into a single big dataframe
    # for compatibility with sklearn Pipeline
    X = pd.concat(dfs_dict, axis='columns')
    X = X.droplevel(column_level_to_drop, axis='columns') # some sklearn classes cannot handle multiindex columns
    X = X.loc[subjects_sorted]
    print(f'X shape: {X.shape}')

    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    # splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    i_train_all = []
    i_test_all = []
    print('----------')
    for i_split, (i_train, i_test) in enumerate(splitter.split(subjects_sorted)):

        print(f'Split {i_split+1}')
        print(f'\ti_train: {i_train.shape}')
        print(f'\ti_test: {i_test.shape}')

        i_train_all.append(i_train)
        i_test_all.append(i_test)

    # save single large X dataframe and train/test indices for each split
    # instead of many X_train and X_test
    data = {
        'X': X,
        'y': groups,
        'holdout': holdout,
        'i_train_all': i_train_all,
        'i_test_all': i_test_all,
        'dataset_names': dataset_names,
        'conf_name': conf_name,
        'udis_datasets': udis_datasets,
        'udis_conf': udis_conf,
        'n_features_datasets': n_features_datasets,
        'n_features_conf': n_features_conf,
        'subjects': subjects_sorted,
    }

    make_parent_dir(fpath_out)
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(data, file_out)
    print(f'Saved data (X: {data["X"].shape}, y: {data["y"].shape}) and split indices  to {fpath_out}')

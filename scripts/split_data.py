
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import FPATHS
from src.utils import load_data_df

# parameters
test_size = 0.1
shuffle = True
seed = 3791
column_level_to_drop = 'udi'

# input paths
fpath_data1 = FPATHS['data_behavioural_clean']
fpath_data2 = FPATHS['data_brain_clean']
fpath_conf = FPATHS['data_demographic_clean']
fpath_groups = FPATHS['data_groups_clean']

# output paths
fpath_train = FPATHS['data_Xy_train']
fpath_test = FPATHS['data_Xy_test']

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'test_size:\t{test_size}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'column_level_to_drop:\t{column_level_to_drop}')
    print(f'fpath_data1:\t{fpath_data1}')
    print(f'fpath_data2:\t{fpath_data2}')
    print(f'fpath_conf:\t{fpath_conf}')
    print(f'fpath_groups:\t{fpath_groups}')
    print('----------------------')

    # load datasets
    print('Loading data...')
    df_data1 = load_data_df(fpath_data1, encoded=True)
    print(f'\tdata1:\t{df_data1.shape}')
    df_data2 = load_data_df(fpath_data2, encoded=True)
    print(f'\tdata2:\t{df_data2.shape}')
    df_conf = load_data_df(fpath_conf, encoded=True)
    print(f'\tconf:\t{df_conf.shape}')

    # make sure all datasets contain the same subjects
    print('Checking that all datasets have the same subjects... ', end='')
    subjects = set(df_data1.index)
    if (subjects != set(df_data2.index)) or (subjects != set(df_conf.index)):
        raise ValueError('Datasets (data1/data2/conf) must have exactly the same subjects')
    print('OK!')
    
    subjects_sorted = sorted(list(subjects))

    # load group data (for stratification)
    groups = load_data_df(fpath_groups, encoded=False).squeeze('columns')[subjects_sorted]

    # load all datasets and extract subjects
    dfs_dict = {
        'data1': df_data1.loc[subjects_sorted],
        'data2': df_data2.loc[subjects_sorted],
        'conf': df_conf.loc[subjects_sorted],
    }

    # combine into a single big dataframe
    # for compatibility with sklearn Pipeline
    X = pd.concat(dfs_dict, axis='columns')
    X = X.droplevel(column_level_to_drop, axis='columns') # some sklearn classes cannot handle multiindex columns
    print(f'X shape before split: {X.shape}')

    # split
    subjects_train, subjects_test = train_test_split(
        subjects_sorted, stratify=groups.loc[subjects_sorted],
        test_size=test_size, shuffle=shuffle, random_state=seed,
    )
    print(f'Train subjects: {len(subjects_train)}')
    print(f'Test subjects: {len(subjects_test)}')

    data_train = (X.loc[subjects_train], groups.loc[subjects_train])
    data_test = (X.loc[subjects_test], groups.loc[subjects_test])

    # save in train/test files
    for fpath_out, data in zip([fpath_train, fpath_test], [data_train, data_test]):
        with open(fpath_out, 'wb') as file_out:
            pickle.dump(data, file_out)
        print(f'Saved data (X: {data[0].shape}, y: {data[1].shape})  to {fpath_out}')

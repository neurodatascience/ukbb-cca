
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
dataset_names = ['behavioural', 'brain']
conf_name = 'conf'

# input paths
fpaths_data = [FPATHS['data_behavioural_clean'], FPATHS['data_brain_clean']]
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
    print(f'dataset_names:\t{dataset_names}')
    print(f'conf_name:\t{conf_name}')
    print(f'fpaths_data:\t{fpaths_data}')
    print(f'fpath_conf:\t{fpath_conf}')
    print(f'fpath_groups:\t{fpath_groups}')
    print('----------------------')

    # input validation
    if len(dataset_names) != len(fpaths_data):
        raise ValueError('dataset_names and fpaths_data must have the same length')

    # build dict of dataframes
    print('Loading data...')
    dfs_dict = {}
    udis = {}
    subjects = None
    for name, fpath in zip(dataset_names + [conf_name], fpaths_data + [fpath_conf]):

        # load data
        df = load_data_df(fpath, encoded=True)
        dfs_dict[name] = df
        print(f'\t{name}:\t{df.shape}')

        # get UDIs
        udis[name] = df.columns

        # make sure all datasets contain the same subjects
        subjects_new = set(df.index)
        if subjects is not None:
            if subjects_new != subjects:
                raise ValueError(f'Datasets {list(df.keys())} do not have exactly the same subjects')
        subjects = subjects_new
    
    # make sure all subjects are in the same order
    subjects_sorted = sorted(list(subjects))
    for name in dfs_dict.keys():
        dfs_dict[name] = dfs_dict[name].loc[subjects_sorted]

    # load group data (for stratification)
    groups = load_data_df(fpath_groups, encoded=False).squeeze('columns')[subjects_sorted]

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
    print(f'\tTrain subjects: {len(subjects_train)}')
    print(f'\tTest subjects: {len(subjects_test)}')

    data_train = {
        'X': X.loc[subjects_train],
        'y': groups.loc[subjects_train],
    }
    data_test = {
        'X': X.loc[subjects_test],
        'y': groups.loc[subjects_test],
    }

    # save in train/test files
    for fpath_out, data in zip([fpath_train, fpath_test], [data_train, data_test]):

        data['dataset_names'] = dataset_names
        data['conf_name'] = conf_name
        data['udis'] = udis

        with open(fpath_out, 'wb') as file_out:
            pickle.dump(data, file_out)
        print(f'Saved data (X: {data["X"].shape}, y: {data["y"].shape})  to {fpath_out}')

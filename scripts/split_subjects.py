
import pandas as pd
from sklearn.model_selection import train_test_split

from paths import FPATHS
from src.utils import load_data_df

# parameters
test_size = 0.1
shuffle = True
seed = 3791

# input paths
fpath_data1 = FPATHS['data_behavioural_clean']
fpath_data2 = FPATHS['data_brain_clean']
fpath_conf = FPATHS['data_demographic_clean']
fpath_stratification = FPATHS['data_stratification_clean']

# output paths
fpath_train = FPATHS['subjects_train']
fpath_test = FPATHS['subjects_test']

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'test_size:\t{test_size}')
    print(f'shuffle:\t{shuffle}')
    print(f'seed:\t{seed}')
    print(f'fpath_data1:\t{fpath_data1}')
    print(f'fpath_data2:\t{fpath_data2}')
    print(f'fpath_conf:\t{fpath_conf}')
    print(f'fpath_stratification:\t{fpath_stratification}')
    print('----------------------')

    # load data
    print('Loading data...')
    df_data1 = load_data_df(fpath_data1, encoded=True)
    print(f'\t{df_data1.shape}')
    df_data2 = load_data_df(fpath_data2, encoded=True)
    print(f'\t{df_data2.shape}')
    df_conf = load_data_df(fpath_conf, encoded=True)
    print(f'\t{df_conf.shape}')
    stratification_var = load_data_df(fpath_stratification, encoded=False).squeeze('columns')

    # make sure all datasets contain the same subjects
    print('Checking that all datasets have the same subjects... ', end='')
    subjects = set(df_data1.index)
    if (subjects != set(df_data2.index)) or (subjects != set(df_conf.index)):
        raise ValueError('Datasets (data1/data2/conf) must have exactly the same subjects')
    print('OK!')
    
    subjects_sorted = sorted(list(subjects))

    # split (order is data1_train, data1_test, data2_train, etc.)
    subjects_train, subjects_test = train_test_split(
        subjects_sorted, stratify=stratification_var.loc[subjects_sorted],
        test_size=test_size, shuffle=shuffle, random_state=seed,
    )

    # save in train/test files
    for fpath_out, subjects_subset in zip([fpath_train, fpath_test], [subjects_train, subjects_test]):
        pd.Series(subjects_subset).to_csv(fpath_out, header=False, index=False)
        print(f'Saved {len(subjects_subset)} subjects to {fpath_out}')

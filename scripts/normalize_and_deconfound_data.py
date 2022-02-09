
import sys

from src.data_processing import inv_norm_df, deconfound_df
from src.utils import load_data_df, zscore_df
from paths import FPATHS

# parameters
configs = {
    'behavioural': {
        'fpath_data': FPATHS['data_behavioural_clean'],
        'fpath_conf': FPATHS['data_demographic_clean'],
        'fpath_out': FPATHS['data_behavioural_norm'],
        'apply_inverse': True,
    },
    'brain': {
        'fpath_data': FPATHS['data_brain_clean'],
        'fpath_conf': FPATHS['data_demographic_clean'],
        'fpath_out': FPATHS['data_brain_norm'],
        'apply_inverse': True,
    },
}

def normalize_df(df, apply_inverse=True, fill_na=True):

    # rank-based inverse normal transformation
    if apply_inverse:
        print(f'\tApplying inverse normal transformation...')
        df = inv_norm_df(df)

    # normalize/impute confounders dataset
    print('\tZ-scoring...')
    df = zscore_df(df)

    if fill_na:
        print('\tFilling NaNs with zeros...')
        df = df.fillna(0)

    return df

if __name__ == '__main__':

    # input validation
    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <domain>')

    domain = sys.argv[1]

    try:
        config = configs[domain]
    except KeyError:
        raise ValueError(f'Invalid domain "{domain}". Accepted domains are: {configs.keys()}')

    fpath_data = config['fpath_data']
    fpath_conf = config['fpath_conf']
    fpath_out = config['fpath_out']
    apply_inverse = config['apply_inverse']

    # print parameters
    print('----- Parameters -----')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_conf:\t{fpath_conf}')
    print(f'fpath_out:\t{fpath_out}')
    print(f'apply_inverse:\t{apply_inverse}')
    print('----------------------')

    # load data and confounds
    df_data = load_data_df(fpath_data)
    df_conf = load_data_df(fpath_conf)

    # preprocess main data (drop bad columns, normalize, fill NaNs)
    print(f'Normalizing main dataset (shape: {df_data.shape})')
    df_data = normalize_df(df_data, apply_inverse=apply_inverse, fill_na=True)

    # preprocess confounds (drop bad columns, normalize, fill NaNs)
    print(f'Normalizing confounds dataset (shape: {df_conf.shape})')
    df_conf = normalize_df(df_conf, apply_inverse=apply_inverse, fill_na=True)

    # remove confounds from main dataset
    print('Deconfounding data...')
    df_deconfounded = deconfound_df(df_data, df_conf)
    df_deconfounded = zscore_df(df_deconfounded) # normalize deconfounded data

    # save csv file
    print(f'Saving final dataframe (shape: {df_deconfounded.shape})...')
    df_deconfounded.to_csv(fpath_out, header=True, index=True)

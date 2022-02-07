
import sys
import pickle
import numpy as np
import pandas as pd
from src.database_helpers import DatabaseHelper
from src.utils import fill_df_with_mean
from paths import DPATHS, FPATHS

dpath_schema = DPATHS['schema']
fpath_udis = FPATHS['udis_tabular_raw']

configs = {
    'behavioural': {
        'fpath_data': FPATHS['data_behavioural_clean'],
        'fpath_out': FPATHS['res_behavioural_svd'],
    },
    'brain': {
        'fpath_data': FPATHS['data_brain_clean'],
        'fpath_out': FPATHS['res_brain_svd'],
    },
    'demographic': {
        'fpath_data': FPATHS['data_demographic_clean'],
        'fpath_out': FPATHS['res_demographic_svd'],
    },
}

def svd(data, full_matrices=False, compute_uv=True):
    return np.linalg.svd(data, full_matrices=full_matrices, compute_uv=compute_uv)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <domain>')

    domain = sys.argv[1]

    try:
        config = configs[domain]
    except KeyError:
        raise ValueError(f'Invalid domain "{domain}". Accepted domains are: {configs.keys()}')

    fpath_data = config['fpath_data']
    fpath_out = config['fpath_out']

    print('----- Parameters -----')
    print(f'domain:\t{domain}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print('----------------------')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    # load data
    df_data = pd.read_csv(fpath_data, index_col='eid')

    # fill missing data
    na_count = df_data.isna().values.sum()
    df_data = fill_df_with_mean(df_data)
    print(f'Filled {na_count} NaNs with mean')

    # get category (main_category) of each field
    udis = df_data.columns
    main_categories = db_helper.get_categories_from_udis(udis)
    main_categories.append('all')

    results = {}
    for category in main_categories:

        print(f'Category: {category}')

        if category == 'all':
            df_data_subset = df_data
        else:
            udis_subset = db_helper.filter_udis_by_category(udis, [category])
            df_data_subset = df_data.loc[:, udis_subset]

        data = df_data_subset.values

        print(f'\tData array shape: {data.shape}')

        U, s, VT = svd(data, full_matrices=False, compute_uv=True)

        print(f'\tSVD results: U {U.shape}, s {s.shape}, VT {VT.shape}')

        results[category] = {
            'subjects': df_data_subset.index.tolist(),
            'udis': df_data_subset.columns.tolist(),
            'U': U,
            's': s,
            'VT': VT,
        }

    # save everything in a single pickle file
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results, file_out)

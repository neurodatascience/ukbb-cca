
import sys
import pickle
import numpy as np
import pandas as pd
from src.data_selection import FieldHelper, UDIHelper
from paths import DPATHS, FPATHS

def svd(data, full_matrices=False, compute_uv=True):
    return np.linalg.svd(data, full_matrices=full_matrices, compute_uv=compute_uv)

if __name__ == '__main__':

    dpath_schema = DPATHS['schema']
    fpath_udis = FPATHS['udis_tabular_raw']

    configs = {
        'behavioural': {
            'fpath_data': FPATHS['data_behavioural_qc'],
            'fpath_out': FPATHS['res_behavioural_svd'],
        },
        # 'brain': {
        #     'fpath_data': FPATHS['data_brain_qc'],
        #     'fpath_out': FPATHS['res_brain_svd'],
        # },
        # 'demographic': {
        #     'fpath_data': FPATHS['data_demographic_qc'],
        #     'fpath_out': FPATHS['res_demographic_svd'],
        # },
    }

    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <type>')

    type = sys.argv[1]

    try:
        config = configs[type]
    except KeyError:
        raise ValueError(f'Invalid type "{type}". Accepted types are: {configs.keys()}')

    fpath_data = config['fpath_data']
    fpath_out = config['fpath_out']

    print('----- Parameters -----')
    print(f'type:\t{type}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print('----------------------')

    field_helper = FieldHelper(dpath_schema)
    udi_helper = UDIHelper(fpath_udis)

    # load data
    df_data = pd.read_csv(fpath_data, index_col='eid')

    # get category of each field (main_category)
    udis = df_data.columns
    fields = udi_helper.get_info(udis, colnames='field_id').drop_duplicates()
    main_categories = field_helper.get_info(fields, colnames='main_category').drop_duplicates().tolist()

    main_categories.append('all')

    results = {}
    for category in main_categories:

        print(f'Category: {category}')

        if category == 'all':
            df_data_subset = df_data
        else:
            fields_subset = field_helper.filter_by_value(fields, category, colname='main_category')
            udis_subset = udi_helper.filter_by_field(udis, fields_subset)
            df_data_subset = df_data.loc[:, udis_subset]

        data = df_data_subset.values.copy()

        print(f'\tData array shape: {data.shape}')

        u, s, vh = svd(data, full_matrices=False, compute_uv=True)

        print(f'\tSVD results: u {u.shape}, s {s.shape}, vh {vh.shape}')

        results[category] = {
            'df_data': df_data_subset,
            'u': u,
            's': s,
            'vh': vh,
        }

    # save everything in a single pickle file
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results, file_out)


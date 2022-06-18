
import sys
from src.database_helpers import DatabaseHelper
from src.data_processing import write_subset
from paths import DPATHS, FPATHS

value_types = 'numeric' # integer, categorical (single/multiple), continuous

dpath_schema = DPATHS['schema']
fpath_udis = FPATHS['udis_tabular_raw']
fpath_data = FPATHS['data_tabular_mri_subjects_filtered']

chunksize = 10000

configs = {
    'behavioural': {
        'categories': [100026], # [100026, 116]
        'title_substring': None,
        'title_substrings_reject': [],
        'instances': [2],
        'keep_instance': 'all',
        'fpath_out': FPATHS['data_behavioural'],
    },
    'brain': {
        'categories': [135],
        # 'title_substring': 'Mean FA', # dMRI measure
        'title_substring': None,
        'title_substrings_reject': ['L1', 'L2', 'L3'],
        'instances': [2],
        'keep_instance': 'all',
        'fpath_out': FPATHS['data_brain'],
    },
    'demographic': {
        'categories': [1001, 1002, 1006],
        'title_substring': None,
        'title_substrings_reject': [],
        'instances': [0, 2],
        'keep_instance': 'max',
        'fpath_out': FPATHS['data_demographic'],
    },
}

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <domain>')

    domain = sys.argv[1]

    try:
        config = configs[domain]
    except KeyError:
        raise ValueError(f'Invalid domain "{domain}". Accepted domains are: {configs.keys()}')

    categories = config['categories']
    title_substring = config['title_substring']
    title_substrings_reject = config['title_substrings_reject']
    instances = config['instances']
    keep_instance = config['keep_instance']
    fpath_out = config['fpath_out']

    print('----- Parameters -----')
    print(f'domain:\t{domain}')
    print(f'categories:\t{categories}')
    print(f'value_types:\t{value_types}')
    print(f'title_substring:\t{title_substring}')
    print(f'title_substrings_reject:\t{title_substrings_reject}')
    print(f'instances:\t{instances}')
    print(f'keep_instance:\t{keep_instance}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print(f'chunksize:\t{chunksize}')
    print('----------------------')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    udis = db_helper.get_udis_from_categories(categories, value_types=value_types, 
        title_substring=title_substring, title_substrings_reject=title_substrings_reject,
        instances=instances, keep_instance=keep_instance)

    print(f'Selected {len(udis)} UDIs')

    n_rows, n_cols = write_subset(fpath_data, fpath_out, colnames=udis, chunksize=chunksize)
    print(f'Wrote {n_rows} rows and {n_cols} columns')

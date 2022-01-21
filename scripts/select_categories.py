
import sys
from src.data_selection import FieldHelper, UDIHelper
from src.data_processing import write_subset
from paths import DPATHS, FPATHS

if __name__ == '__main__':

    chunksize = 10000

    dpath_schema = DPATHS['schema']
    fpath_udis = FPATHS['udis_tabular_raw']
    fpath_data = FPATHS['data_tabular_mri_subjects']

    configs = {
        'behavioural': {
            'categories': [100026, 116],
            'fpath_out': FPATHS['data_behavioural'],
            'filter_value': None,
        },
        'brain': {
            'categories': [134],
            'fpath_out': FPATHS['data_brain'],
            'filter_value': 'Mean FA', # dMRI measure
        },
        'demographic': {
            'categories': [1001, 1002, 1006],
            'fpath_out': FPATHS['data_demographic'],
            'filter_value': None,
        },
    }

    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <type>')

    type = sys.argv[1]

    try:
        config = configs[type]
    except KeyError:
        raise ValueError(f'Invalid type "{type}". Accepted types are: {configs.keys()}')

    categories = config['categories']
    fpath_out = config['fpath_out']
    filter_value = config['filter_value']

    print('----- Parameters -----')
    print(f'type:\t\t{type}')
    print(f'categories:\t{categories}')
    print(f'fpath_out:\t{fpath_out}')
    print(f'filter_value:\t{filter_value}')
    print('----------------------')

    field_helper = FieldHelper(dpath_schema)
    udi_helper = UDIHelper(fpath_udis)

    fields = field_helper.get_fields_from_categories(categories)
    print(f'Selected {len(fields)} fields')

    if filter_value is not None:
        fields = field_helper.filter_by_value(fields, filter_value)
        print(f'{len(fields)} fields remaining after filtering')

    udis = udi_helper.get_udis_from_fields(fields)
    print(f'Selected {len(udis)} UDIs')

    n_rows, n_cols = write_subset(fpath_data, fpath_out, colnames=udis, chunksize=chunksize)
    print(f'Wrote {n_rows} rows and {n_cols} columns')


from src.data_selection import DatabaseHelper
from src.data_processing import write_subset
from paths import FPATHS, DPATHS

categories_brain = [1014]
instances = [2]
chunksize = 10000

fpath_data = FPATHS['data_tabular_raw']
fpath_out = FPATHS['data_tabular_mri_subjects']

fpath_udis = FPATHS['udis_tabular_raw']
dpath_schema = DPATHS['schema']

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print(f'categories_brain:\t{categories_brain}')
    print(f'instances:\t{instances}')
    print(f'chunksize:\t{chunksize}')
    print('----------------------')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)
    udis_brain = db_helper.get_udis_from_categories(categories_brain, instances=instances)

    n_rows, n_cols = write_subset(
        fpath_data, fpath_out, colnames=None, chunksize=chunksize,
        fn_to_apply=(lambda df: df.dropna(axis='index', how='all', subset=udis_brain))
    )

    print(f'Wrote {n_rows} rows and {n_cols} columns')

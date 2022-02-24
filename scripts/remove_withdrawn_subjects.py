
import pandas as pd
from src.data_processing import write_subset
from paths import FPATHS

chunksize=10000

fpath_data = FPATHS['data_tabular_mri_subjects']
fpath_subjects = FPATHS['subjects_to_remove']
fpath_out = FPATHS['data_tabular_mri_subjects_filtered']

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'chunksize:\t{chunksize}')
    print(f'fpath_subjects:\t{fpath_subjects}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print('----------------------')

    subjects_to_remove = pd.read_csv(fpath_subjects)['eid']
    print(f'Number of subjects to remove from dataset: {len(subjects_to_remove)}')

    n_rows, n_cols = write_subset(fpath_data, fpath_out, colnames=None, chunksize=chunksize,
        fn_to_apply=(lambda df: df.drop(index=subjects_to_remove, errors='ignore'))
    )
    print(f'Wrote {n_rows} rows and {n_cols} columns')


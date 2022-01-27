
from src.data_selection import UDIHelper
from src.data_processing import write_subset
from paths import FPATHS

def filter_df_by_content(df, colname, value=1):
    return df.loc[df[colname] == value]

if __name__ == '__main__':

    field_completed_mri = 12188 # 1: yes, 0: no, -1: unknown
    instances = [2]
    chunksize = 10000

    fpath_data = FPATHS['data_tabular_raw']
    fpath_out = FPATHS['data_tabular_mri_subjects']

    fpath_udis = FPATHS['udis_tabular_raw']

    print('----- Parameters -----')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print(f'fpath_udis:\t{fpath_udis}')
    print(f'field_completed_mri:\t{field_completed_mri}')
    print(f'instances:\t{instances}')
    print(f'chunksize:\t{chunksize}')
    print('----------------------')

    udi_helper = UDIHelper(fpath_udis)
    udi_completed_mri = udi_helper.get_udis_from_fields([field_completed_mri], instances=instances)[0]

    n_rows, n_cols = write_subset(
        fpath_data, fpath_out, colnames=None, chunksize=chunksize,
        fn_to_apply=(lambda df: filter_df_by_content(df, udi_completed_mri))
    )

    print(f'Wrote {n_rows} rows and {n_cols} columns')

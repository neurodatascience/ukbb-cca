
from src.data_selection import UDIHelper, FieldHelper
from src.data_processing import write_subset
from paths import FPATHS, DPATHS

# def filter_df_by_content(df, colnames, value=1):
#     df_out = df
#     for colname in colnames:
#         df_out = df_out.loc[df_out[colname] == value]
#     return df_out

if __name__ == '__main__':

    # field_completed_mri = 12188 # 1: yes, 0: no, -1: unknown

    categories_brain = [1014]
    instances = [2]
    chunksize = 10000

    fpath_data = FPATHS['data_tabular_raw']
    fpath_out = FPATHS['data_tabular_mri_subjects']

    fpath_udis = FPATHS['udis_tabular_raw']
    dpath_schema = DPATHS['schema']

    print('----- Parameters -----')
    print(f'fpath_data:\t{fpath_data}')
    print(f'fpath_out:\t{fpath_out}')
    print(f'fpath_udis:\t{fpath_udis}')
    print(f'dpath_schema:\t{dpath_schema}')
    # print(f'field_completed_mri:\t{field_completed_mri}')
    print(f'categories_brain:\t{categories_brain}')
    print(f'instances:\t{instances}')
    print(f'chunksize:\t{chunksize}')
    print('----------------------')

    field_helper = FieldHelper(dpath_schema)
    udi_helper = UDIHelper(fpath_udis)

    fields_brain = field_helper.get_fields_from_categories(categories_brain)
    udis_brain = udi_helper.get_udis_from_fields(fields_brain, instances=instances)

    # udis_completed_mri = udi_helper.get_udis_from_fields([field_completed_mri], instances=instances)

    n_rows, n_cols = write_subset(
        fpath_data, fpath_out, colnames=None, chunksize=chunksize,
        fn_to_apply=(lambda df: df.dropna(axis='index', how='all', subset=udis_brain))
    )

    print(f'Wrote {n_rows} rows and {n_cols} columns')

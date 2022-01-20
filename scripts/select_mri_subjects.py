
from src.data_selection import UDIHelper
from src.data_processing import write_subset
from paths import FPATHS

def filter_df_by_content(df, colname, value=1):
    return df.loc[df[colname] == value]

if __name__ == '__main__':

    field_completed_mri = 12188 # 1: yes, 0: no, -1: unknown

    fpath_udis = FPATHS['udis_tabular_raw']
    fpath_data = FPATHS['data_tabular_raw']

    udi_helper = UDIHelper(fpath_udis)

    udi_completed_mri = udi_helper.get_udis_from_fields([field_completed_mri], instances=[2])[0]

    fpath_out = FPATHS['data_tabular_mri_subjects']
    write_subset(
        fpath_data, fpath_out, colnames=None, chunksize=10,
        fn_to_apply=(lambda df: filter_df_by_content(df, udi_completed_mri))
    )



import re
import pandas as pd

from paths import FPATHS
from src.utils import load_data_df

def parse_udis(fpath_data):

    def parse_udi(udi):
        match = re_udi.match(udi)
        if not match:
            raise ValueError(f'Could not parse UDI {udi}')
        else:
            return match.groups()

    re_udi = re.compile('(\d+)-(\d+)\\.(\d+)')

    # read UDIs (column names) from data
    # faster to read 1 row then drop it than reading 0 rows
    df_tabular = load_data_df(fpath_data, nrows=1)
    df_udis = df_tabular.iloc[:0].transpose()
    
    df_udis.index.name = 'udi'
    df_udis.columns.name = None

    df_udis = df_udis.reset_index()
    df_udis['field_id'], df_udis['instance'], df_udis['array_index'] =  zip(*df_udis['udi'].map(parse_udi))

    return df_udis

if __name__ == '__main__':

    df_udis = parse_udis(FPATHS['data_tabular_raw'])
    df_udis.to_csv(FPATHS['udis_tabular_raw'], header=True, index=False)

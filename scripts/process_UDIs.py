
import re
import pandas as pd
from paths import FPATHS

def parse_UDIs(fpath_data):

    def parse_UDI(udi):
        match = re_udi.match(udi)
        if not match:
            raise ValueError(f'Could not parse UDI {udi}')
        else:
            return match.groups()

    re_udi = re.compile('(\d+)-(\d+)\\.(\d+)')

    # read UDIs (column names) from data
    # faster to read 1 row then drop it than reading 0 rows
    df_tabular = pd.read_csv(fpath_data, header=0, index_col='eid', nrows=1)
    df_UDIs = df_tabular.iloc[:0].transpose()
    
    df_UDIs.index.name = 'udi'
    df_UDIs.columns.name = None

    df_UDIs = df_UDIs.reset_index()
    df_UDIs['field_id'], df_UDIs['instance_id'], df_UDIs['array_index'] =  zip(*df_UDIs['udi'].map(parse_UDI))

    return df_UDIs

if __name__ == '__main__':

    df_UDIs = parse_UDIs(FPATHS['data_tabular_raw'])
    df_UDIs.to_csv(FPATHS['UDIs_tabular_raw'], header=True, index=False)

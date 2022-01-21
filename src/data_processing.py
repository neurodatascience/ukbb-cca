
import pandas as pd

def write_subset(fpath_data, fpath_out, colnames=None, fn_to_apply=None, header=0, index_col='eid', chunksize=10000):
    
    if (colnames is not None) and (index_col not in colnames):
        colnames.insert(0, index_col) # passing index_col to pd.read_csv() directly causes KeyError when writing

    df_chunks = pd.read_csv(
        fpath_data, usecols=colnames, header=header, chunksize=chunksize, low_memory=False)

    mode = 'w' # first chunk
    write_header=True

    n_rows = 0
    n_cols = 0

    for df_chunk in df_chunks:

        if not (fn_to_apply is None):
            df_chunk = fn_to_apply(df_chunk)

        n_rows += df_chunk.shape[0]
        n_cols = df_chunk.shape[1]

        df_chunk.to_csv(fpath_out, mode=mode, header=write_header, index=False)
        mode = 'a' # append subsequent chunks
        write_header=False

    return n_rows, n_cols

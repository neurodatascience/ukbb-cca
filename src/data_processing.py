import re
import numpy as np
import pandas as pd
from scipy import stats
from .utils import make_parent_dir, zscore_df, load_data_df

def parse_udis(fpath_in) -> pd.DataFrame:

    def parse_udi(udi):
        match = re_udi.match(udi)
        if not match:
            raise ValueError(f'Could not parse UDI {udi}')
        else:
            return match.groups()

    re_udi = re.compile('(\d+)-(\d+)\\.(\d+)')

    # read UDIs (column names) from data
    # faster to read 1 row then drop it than reading 0 rows
    df_tabular = load_data_df(fpath_in, nrows=1)
    df_udis = df_tabular.iloc[:0].transpose()
    
    df_udis.index.name = 'udi'
    df_udis.columns.name = None

    df_udis = df_udis.reset_index()
    df_udis['field_id'], df_udis['instance'], df_udis['array_index'] =  zip(*df_udis['udi'].map(parse_udi))

    return df_udis

def write_subset(fpath_data, fpath_out, colnames=None, fn_to_apply=None, header=0, index_col='eid', chunksize=10000):
    
    if (colnames is not None) and (index_col not in colnames):
        colnames.insert(0, index_col) # passing index_col to pd.read_csv() directly causes KeyError when writing

    df_chunks = pd.read_csv(
        fpath_data, usecols=colnames, header=header, chunksize=chunksize, low_memory=False)

    mode = 'w' # first chunk
    write_header=True

    n_rows = 0
    n_cols = 0

    make_parent_dir(fpath_out)

    for df_chunk in df_chunks:

        df_chunk = df_chunk.set_index(index_col)

        if not (fn_to_apply is None):
            df_chunk = fn_to_apply(df_chunk)

        n_rows += df_chunk.shape[0]
        n_cols = df_chunk.shape[1]

        df_chunk.to_csv(fpath_out, mode=mode, header=write_header, index=True)
        mode = 'a' # append subsequent chunks
        write_header=False

    return n_rows, n_cols

def find_cols_with_missing(df, threshold=0.5):
    na_freqs = df.isna().mean(axis='index')
    return df.columns[na_freqs >= threshold].tolist()

def find_cols_with_high_freq(df, threshold=0.95):
    n_rows_nonan = (~df.isna()).sum(axis='index').values
    _, mode_counts = np.squeeze(stats.mode(df, axis=0))
    return df.columns[(mode_counts / n_rows_nonan) >= threshold].tolist() 

def find_cols_with_outliers(df, threshold=100):

    def has_outliers(s):
        # outlier if max((X - median(X))**2) > 100*mean((X - median(X))**2)
        squared_distances_from_median = (s - s.median()) ** 2
        ratio = squared_distances_from_median.max() / squared_distances_from_median.mean()
        return ratio >= threshold

    df = inv_norm_df(df)
    df = zscore_df(df)

    return df.columns[df.apply(has_outliers, axis='index')].tolist()

def deconfound_df(df_data, df_conf):

    # TODO add reference for deconfounding method

    # df_conf cannot have missing data because pinv() would fail
    if df_conf.isna().values.sum() != 0:
        raise ValueError('confound data cannot contain NaNs')

    # remember columns/index of original dataframe
    data_cols = df_data.columns
    data_index = df_data.index

    conf = df_conf.values

    # mask NaNs in df_data
    data_masked = np.ma.masked_invalid(df_data.values)

    # deconfound
    conf_weights = np.ma.dot(np.linalg.pinv(conf), data_masked)
    conf_effects = np.ma.dot(conf, conf_weights)
    data_deconfounded = data_masked - conf_effects

    # put back NaNs and rebuild dataframe
    return pd.DataFrame(data=data_deconfounded.filled(np.nan), index=data_index, columns=data_cols)

def inv_norm(X, method='blom', rank_method='average'):
    '''Rank-based inverse normal transformation.'''

    constants = {
        'blom': 3/8, 
        'tukey': 1/3, 
        'bliss': 1/2,
        'waerden': 0,
    }

    try:
        c = constants[method]
    except KeyError:
        raise KeyError(f'Invalid method "{method}". Valid methods are: {constants.keys()}')

    if len(X.shape) > 2:
        raise ValueError(f'Matrix must have at most 2 dimensions (got shape {X.shape})')

    n_rows = X.shape[0]
    try:
        n_cols = X.shape[1]
    except IndexError:
        n_cols = 1

    if np.isnan(X).sum() == 0:

        # get within-column ranks
        ranks = stats.rankdata(X, axis=0, method=rank_method)

        # inverse normal transform
        quantiles = (ranks - c) / (n_rows - (2*c) + 1)
        X_transformed = stats.norm.ppf(quantiles)

    else:

        # initialize matrix of NaNs
        # to be filled with transformed data in the correct positions
        # so that elements that were NaN are still NaN
        X_transformed = np.empty((n_rows, n_cols))
        X_transformed.fill(np.nan)

        for i_col in range(n_cols):

            # find where non-NaN elements
            not_na = ~(np.isnan(X[:, i_col]))

            # call inv_norm() on clean data and fill spots in NaN matrix
            X_transformed[not_na, i_col] = inv_norm(X[not_na, i_col], method=method, rank_method=rank_method)

    return X_transformed

def inv_norm_df(df, method='blom', rank_method='average'):

    # deconstruct dataframe
    cols = df.columns
    index = df.index
    data = df.values

    # reconstruct dataframe
    data_transformed = inv_norm(data, method=method, rank_method=rank_method)
    return pd.DataFrame(data=data_transformed, index=index, columns=cols)

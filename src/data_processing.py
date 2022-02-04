
import numpy as np
import pandas as pd
from scipy import stats
from src.utils import zscore_df

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

def find_cols_with_missing(df, threshold=0.5):
    na_freqs = df.isna().mean(axis='index')
    return df.columns[na_freqs >= threshold].tolist()

def find_cols_with_high_freq(df, threshold=0.95):
    n_rows_nonan = (~df.isna()).sum(axis='index').values
    _, mode_counts = np.squeeze(stats.mode(df, axis=0))
    return df.columns[(mode_counts / n_rows_nonan) > threshold].tolist() 

def find_cols_with_outliers(df, threshold=100):

    def has_outliers(s):
        # outlier if max((X - median(X))**2) > 100*mean((X - median(X))**2)
        squared_distances_from_median = (s - s.median()) ** 2
        ratio = squared_distances_from_median.max() / squared_distances_from_median.mean()
        return ratio > threshold

    df = inv_norm_df(df)
    df = zscore_df(df)

    return df.columns[df.apply(has_outliers, axis='index')].tolist()

def deconfound_df(df_data, df_conf):

    # TODO add reference for deconfounding method

    # remember columns/index of original dataframe
    data_cols = df_data.columns
    data_index = df_data.index

    # deconfound (on numpy arrays)
    data_deconfounded = df_data.values - (df_conf.values @ (np.linalg.pinv(df_conf.values) @ df_data.values))

    # rebuild dataframe
    return pd.DataFrame(data=data_deconfounded, index=data_index, columns=data_cols)

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

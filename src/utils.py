
import warnings
import pandas as pd

def load_df(fpath, index_col='eid'):
    return pd.read_csv(fpath, index_col=index_col)

def demean_df(df, axis='index'):
    means = df.mean(axis=axis, skipna=True)
    return df - means

def find_constant_cols(df):
    '''Returns list of column names where the non-NaN values are all the same or where everything is NaN.'''
    nunique = df.nunique()
    return nunique[nunique <= 1].index.tolist()

def zscore_df(df, axis='index'):
    if len(find_constant_cols(df)) != 0:
        warnings.warn(f'Constant column(s) detected. Z-scored dataframe may have NaNs', UserWarning)
    df_demeaned = demean_df(df)
    stds = df.std(axis=axis, ddof=1, skipna=True)
    return df_demeaned / stds

def fill_df_with_mean(df, axis='index'):
    means = df.mean(axis=axis, skipna=True)
    return df.fillna(means)

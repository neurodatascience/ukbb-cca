import warnings
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

def print_params(parameters: dict[str, str]):
    max_len = max([len(p) for p in parameters]) # for alignment
    print('----- Parameters -----')
    for name, value in parameters.items():
        name = f'{name}:' # add colon
        print(f'{name.ljust(max_len+1)}\t{value}')
    print('----------------------')

def make_dir(dpath):
    Path(dpath).mkdir(parents=True, exist_ok=True)

def make_parent_dir(fpath):
    Path(fpath).parents[0].mkdir(parents=True, exist_ok=True)

# def add_suffix(path, suffix, sep='_'):
#     root, ext = os.path.splitext(path)
#     return f'{root}{sep}{suffix}{ext}'

def add_suffix(path: Union[Path, str], suffix: str, sep='_') -> Path:
    path = Path(path)
    return Path(path.parent, f'{path.stem}{sep}{suffix}{path.suffix}')

def load_data_df(fpath, index_col=0, nrows=None, encoded=False) -> pd.DataFrame:
    header = [0, 1] if encoded else 0
    return pd.read_csv(fpath, index_col=index_col, nrows=nrows, header=header)

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

def nearest_spd(A):
    """
    Finds nearest symmetric positive definite matrix.
    Adapted from https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194.
    """

    def is_symmetric(A, rtol=1e-05, atol=1e-08):
        return np.allclose(A, A.T, rtol=rtol, atol=atol)

    def is_pd(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    # input validation
    if len(A.shape) != 2:
        raise ValueError('matrix must be 2D')

    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        raise ValueError('matrix must be square')

    if is_symmetric(A) and is_pd(A):
        return A

    if n_rows == 1:
        return np.spacing(1)

    # symmetrize A into B
    B = (A + A.T) / 2

    # H is the symmetric polar factor of B
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt

    A_hat = (B + H) / 2
    A_hat = (A_hat + A_hat.T) / 2 # make it symmetric

    k = 1
    I = np.eye(n_rows)
    spacing = np.spacing(np.linalg.norm(A_hat))
    while not is_pd(A_hat):

        min_eig = np.min(np.real(np.linalg.eigvals(A_hat)))
        A_hat += I * (-min_eig * k**2 + spacing)
        k += 1

    return A_hat

def eig_flip(V):
    '''
    Similar to sklearn's svd_flip() but for a single matrix (columns are eigenvectors).
    Makes the largest coefficient in each eigenvector positive.
    '''
    
    max_abs_cols = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[max_abs_cols, range(V.shape[1])])
    V *= signs
    
    return V

def rotate_to_match(weights, weights_ref):
    U, _, Vt = np.linalg.svd(weights.T @ weights_ref)
    Q = U @ Vt
    return weights @ Q

def traverse_nested_dict(d, fn=None, with_keys=False, keys=[]):
    if not isinstance(d, dict):
        if fn is None:
            val = None
        elif not with_keys:
            val = fn(d)
        else:
            val = fn(d, keys)
        return val
    return {k: traverse_nested_dict(v, fn=fn, with_keys=with_keys, keys=keys+[k]) for k, v in d.items()}

def traverse_nested_list(l, fn, pass_index=False):
    out_l = []
    for i, sub_l in enumerate(l):
        if pass_index:
            out = fn(sub_l, i)
        else:
            out = fn(sub_l)
        out_l.append(out)
    return out_l

def sublist_append(l1, l2):
    traverse_nested_list(l1, (lambda l, i: l.append(l2[i])), pass_index=True)

def sublist_mean(l, axis=0):
    return traverse_nested_list(l, (lambda x: np.nanmean(x, axis=axis)))

def sublist_std(l, axis=0):
    return traverse_nested_list(l, (lambda x: np.nanstd(x, axis=axis)))
    

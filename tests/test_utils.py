
import pytest
from pandas.testing import assert_frame_equal

import numpy as np
import pandas as pd
from src.utils import demean_df, find_constant_cols, zscore_df, fill_df_with_mean

data = dict(
    col1=[0, 1, 2, 3, 4, 5], 
    col2=[-20, -2, 0, 0, 2, 2], 
    col3=[-2, -2, 0, 0, 2, 20],
)
data_with_const_cols = dict(data, 
    col4=[1, 1, 1, 1, 1, 1], 
    col5=[-5, -5, -5, -5, -5, -5], 
    col6=[0, 0, 0, 0, 0, 0],
)

data_demeaned = dict(
    col1=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], 
    col2=[-17, 1, 3, 3, 5, 5], 
    col3=[-5, -5, -3, -3, -1, 17],
    )
data_with_const_cols_demeaned = dict(data_demeaned, col4=[0, 0, 0, 0, 0, 0], col5=[0, 0, 0, 0, 0, 0], col6=[0, 0, 0, 0, 0, 0])

data_zscored = dict(
    col1=[-1.33630621, -0.80178373, -0.26726124, 0.26726124, 0.80178373, 1.33630621], 
    col2=[-2.0090577, 0.11817986, 0.35453959, 0.35453959, 0.59089932, 0.59089932], 
    col3=[-0.59089932, -0.59089932, -0.35453959, -0.35453959, -0.11817986, 2.0090577],
)
data_with_const_cols_zscored = dict(data_zscored, 
    col4=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 
    col5=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    col6=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
)

data_with_na = dict(
    col1=[0, 1, np.nan, 3, 4, np.nan], 
    col2=[-20, -2, 0, 0, np.nan, 2], 
    col3=[-2, np.nan, 0, 0, 2, 20],
)

data_with_na_filled = dict(
    col1=[0, 1, 2, 3, 4, 2], 
    col2=[-20, -2, 0, 0, -4, 2], 
    col3=[-2, 4, 0, 0, 2, 20],
)

@pytest.mark.parametrize('data, expected', [
    [data, data_demeaned],
    [data_with_const_cols, data_with_const_cols_demeaned]
])
def test_demean_df(data, expected):
    assert_frame_equal(
        demean_df(pd.DataFrame(data)), pd.DataFrame(expected),
        check_dtype=False, 
    )

@pytest.mark.parametrize('data, expected', [
    [data, []],
    [data_with_const_cols, ['col4', 'col5', 'col6']],
])
def test_find_constant_cols(data, expected):
    assert set(find_constant_cols(pd.DataFrame(data))) == set(expected)

@pytest.mark.parametrize('data, expected', [
    [data, data_zscored],
    [data_with_const_cols, data_with_const_cols_zscored]
])
def test_zscore_df(data, expected):
    assert_frame_equal(
        zscore_df(pd.DataFrame(data)), pd.DataFrame(expected),
        check_dtype=False, 
    )

@pytest.mark.parametrize('data, expected', [
    [data_with_na, data_with_na_filled]
])
def test_fill_df_with_mean(data, expected):
    assert_frame_equal(
        fill_df_with_mean(pd.DataFrame(data)), pd.DataFrame(expected),
        check_dtype=False, 
    )

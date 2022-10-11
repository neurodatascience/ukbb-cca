
import pytest

import pandas as pd
from src.data_processing import find_bad_cols

data = dict(col1=[0, 1, 2, 3, 4, 5], col2=[-20, -2, 0, 0, 2, 2], col3=[-2, -2, 0, 0, 2, 20])
data_with_const_col = dict(data, col4=[1, 1, 1, 1, 1, 1])

@pytest.mark.parametrize('data, k, expected', [
    [data, 1, ['col1', 'col2', 'col3']],
    [data, 2, ['col2', 'col3']],
    [data, 3, []],
    [data, 0, ValueError],
    [data_with_const_col, 3, ValueError],
])
def test_find_bad_cols(data, k, expected):
    if type(expected) == type and issubclass(expected, Exception):
        with pytest.raises(expected):
            find_bad_cols(pd.DataFrame(data), k)
    else:
        assert find_bad_cols(pd.DataFrame(data), k) == expected

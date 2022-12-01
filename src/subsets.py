from functools import wraps
import numpy as np
from .data_processing import XyData

DATASET_DISEASE = 'disease'

def check_indices(func):
    @wraps(func)
    def _check_indices(data: XyData, *args, **kwargs):
        data.check_index_order()
        return func(data, *args, **kwargs)
    return _check_indices

def all(data: XyData):
    return np.arange(len(data.X))

@check_indices
def healthy(data: XyData):
    df_disease = data.extra[DATASET_DISEASE].reset_index(drop=True)
    to_keep = df_disease.isna().all(axis='columns')
    return df_disease[to_keep].index

@check_indices
def _disease_subset_helper(data: XyData, fn_filter):
    df_disease = data.extra[DATASET_DISEASE].reset_index(drop=True)
    to_keep = df_disease.applymap(fn_filter).any(axis='columns')
    return df_disease[to_keep].index

def psychoactive(data: XyData):
    return _disease_subset_helper(
        data,
        lambda x: (x == 'Z864') or (str(x)[:3] in [f'F{i}' for i in range(10,20)]),
    )

def hypertension(data: XyData):
    return _disease_subset_helper(
        data,
        lambda x: (x == 'I10'),
    )

SUBSET_FN_MAP = {
    fn.__name__: fn
    for fn in [all, psychoactive, hypertension, healthy]
}

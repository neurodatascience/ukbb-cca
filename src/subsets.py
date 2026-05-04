from functools import wraps
import numpy as np
import pandas as pd
from .data_processing import XyData

DATASET_DISEASE = "disease"


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
    to_keep = df_disease.isna().all(axis="columns")
    return df_disease[to_keep].index


@check_indices
def _disease_subset_helper(data: XyData, fn_filter):
    df_disease = data.extra[DATASET_DISEASE].reset_index(drop=True)
    to_keep = df_disease.applymap(fn_filter).any(axis="columns")
    return df_disease[to_keep].index


def psychoactive(data: XyData):
    return _disease_subset_helper(
        data,
        lambda x: (x == "Z864") or (str(x)[:3] in [f"F{i}" for i in range(10, 20)]),
    )


def hypertension(data: XyData):
    return _disease_subset_helper(
        data,
        lambda x: (x == "I10"),
    )


def psychoactive_no_overlap(data: XyData):
    idx_psychoactive = psychoactive(data)
    idx_hypertension = hypertension(data)
    return pd.Index(
        set(idx_psychoactive) - set(idx_hypertension), dtype=idx_psychoactive.dtype
    )


def hypertension_no_overlap(data: XyData):
    idx_hypertension = hypertension(data)
    idx_psychoactive = psychoactive(data)
    return pd.Index(
        set(idx_hypertension) - set(idx_psychoactive), dtype=idx_hypertension.dtype
    )


def hyperpsycho(data: XyData):
    idx_hypertension = hypertension(data)
    idx_psychoactive = psychoactive(data)
    return pd.Index(
        set(idx_hypertension) & set(idx_psychoactive), dtype=idx_hypertension.dtype
    )


SUBSET_FN_MAP = {
    fn.__name__: fn
    for fn in [
        all,
        psychoactive,
        hypertension,
        healthy,
        psychoactive_no_overlap,
        hypertension_no_overlap,
        hyperpsycho,
    ]
}

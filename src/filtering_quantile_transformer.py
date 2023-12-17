from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from .database_helpers import DatabaseHelper

class FilteringQuantileTransformer(QuantileTransformer):

    categories_to_transform = [11, 31]

    def __init__(self, db_helper: DatabaseHelper=None, expected_udis=None, encoded=True, output_distribution='uniform', **kwargs) -> None:
        
        super().__init__(**kwargs)

        self.output_distribution = output_distribution
        self.db_helper = db_helper
        self.expected_udis = expected_udis
        self.encoded = encoded

    @staticmethod
    def _unencode_udis(udis):
        return [udi.split('_')[0] for udi in udis]

    def _filter_udis(self, udis):
        return self.db_helper.filter_udis_by_value_type(udis, self.categories_to_transform)
    
    def _to_df(self, X):
        if hasattr(X, 'columns'):
            udis = np.array(X.columns)
            if (self.expected_udis is not None) and (len(udis) != len(self.expected_udis)):
                raise RuntimeError(
                    'Invalid number of columns (expected '
                    f'{len(self.expected_udis)}, got {len(udis)})'
                )
            return X
        else:
            return pd.DataFrame(X, columns=self.expected_udis)

    def _get_filtered_udis(self, X: pd.DataFrame):
        udis = np.array(X.columns)

        if self.encoded:
            is_original_udi = np.array([
                (not '_mean' in udi and not '_std' in udi)
                for udi in udis
            ])
            udis[is_original_udi] = self._unencode_udis(udis[is_original_udi])
            X.columns = udis

        non_categorical_udis = np.concatenate([
            self._filter_udis(udis[is_original_udi]),
            udis[~is_original_udi],
        ])

        if len(non_categorical_udis) != len(set(non_categorical_udis)):
            existing_udis = set()
            for udi in non_categorical_udis:
                if udi in existing_udis:
                    print(f'{udi} duplicated')
                else:
                    existing_udis.add(udi)
            raise RuntimeError('Non-unique UDIs found')
        
        return non_categorical_udis

    def fit(self, X, y=None):
        X = self._to_df(X)
        return super().fit(X.loc[:, self._get_filtered_udis(X)], y)

    def transform(self, X):
        X = deepcopy(X)
        X = self._to_df(X)
        non_categorical_udis = self._get_filtered_udis(X)
        X.loc[:, non_categorical_udis] = super().transform(X.loc[:, non_categorical_udis])
        return X

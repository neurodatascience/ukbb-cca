import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .database_helpers import DatabaseHelper

class UkbbSquarer(BaseEstimator, TransformerMixin):

    categories_to_square = [11, 31]

    def __init__(self, db_helper: DatabaseHelper=None, expected_udis=None, encoded=True) -> None:
        
        if expected_udis is None:
            expected_udis = []

        self.db_helper = db_helper
        self.expected_udis = expected_udis
        self.encoded = encoded

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _unencode_udis(udis):
        return [udi.split('_')[0] for udi in udis]

    def _filter_udis(self, udis):
        return self.db_helper.filter_udis_by_value_type(udis, self.categories_to_square)

    def transform(self, X):
        if hasattr(X, 'columns'):
            udis = X.columns
            if (self.expected_udis is not None) and (len(udis) != len(self.expected_udis)):
                raise RuntimeError(
                    'Invalid number of columns (expected '
                    f'{len(self.expected_udis)}, got {len(udis)})'
                )
        else:
            udis = self.expected_udis
            X = pd.DataFrame(X, columns=udis)

        if self.encoded:
            udis = self._unencode_udis(udis)
            X.columns = udis

        non_categorical_udis = self._filter_udis(udis)

        if len(non_categorical_udis) != len(set(non_categorical_udis)):
            raise RuntimeError('Non-unique UDIs found:')

        squared = np.square(X.loc[:, non_categorical_udis])
        squared.columns = [f'{udi}_squared' for udi in non_categorical_udis]
        X_transformed_df = pd.concat([X, squared], axis='columns')
        return X_transformed_df.to_numpy()

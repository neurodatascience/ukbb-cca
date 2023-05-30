from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

class NanFeatureSelectorMixin(SelectorMixin, BaseEstimator, ABC):
    
    def __init__(self, dropped_features_dict=None) -> None:
        if dropped_features_dict is None:
            dropped_features_dict = {}
        self.dropped_features_dict = dropped_features_dict
        self.reason = NotImplemented

    def _validate_data(self, X="no_validation", y="no_validation", reset=True, validate_separately=False, **check_params):
        check_params['force_all_finite'] = 'allow-nan'
        return super()._validate_data(X, y, reset, validate_separately, **check_params)

    def fit(self, X, y=None):
        self._fit(X, y=y)
        self.dropped_features_dict[self.reason] = self.get_feature_names_dropped()
        # print(f'Dropped {len(self.dropped_features_dict[self.reason])} features')
        # print(f'Dropped: {self.dropped_features_dict[self.reason]}')
        # print(f'reason: {self.reason}')
        return self

    @abstractmethod
    def _fit(self, X, y=None):
        raise NotImplementedError

    def get_feature_names_dropped(self):
        input_features = getattr(self, 'feature_names_in_')
        return input_features[~self.get_support()]

    def format_transformed(self, X):
        if hasattr(self, 'feature_names_in_'):
            return pd.DataFrame(
                data=X, 
                columns=self.get_feature_names_out(self.feature_names_in_),
            )
        else:
            return X

    def transform(self, X):
        return self.format_transformed(super().transform(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.format_transformed(super().fit_transform(X, y, **fit_params))

class FeatureSelectorMissing(NanFeatureSelectorMixin):

    def __init__(self, threshold=0.5, dropped_features_dict=None) -> None:
        super().__init__(dropped_features_dict=dropped_features_dict)
        if threshold < 0 or threshold > 1:
            raise RuntimeError(f'Invalid threshold: {threshold}')
        self.threshold = threshold
        self.reason = f'NAs >= {self.threshold}'

    def _fit(self, X, y=None):
        X = np.asarray(self._validate_data(X))
        self.freqs_ = np.isnan(X).mean(axis=0)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.freqs_ < self.threshold

class FeatureSelectorHighFreq(NanFeatureSelectorMixin):

    def __init__(self, threshold=0.95, dropped_features_dict=None) -> None:
        super().__init__(dropped_features_dict=dropped_features_dict)
        if threshold < 0 or threshold > 1:
            raise RuntimeError(f'Invalid threshold: {threshold}')
        self.threshold = threshold
        self.reason = f'Highest frequency >= {self.threshold}'

    def _fit(self, X, y=None):
        X = np.asarray(self._validate_data(X))
        # n_rows = sum(~np.any(np.isnan(X), axis=1))
        n_rows = np.sum(~np.isnan(X), axis=0)
        _, mode_counts = np.squeeze(stats.mode(X, axis=0))
        self.freqs_ = mode_counts / n_rows
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.freqs_ < self.threshold

class FeatureSelectorOutlier(NanFeatureSelectorMixin):

    def __init__(self, threshold=100, with_scaler=True, with_inv_norm=False, dropped_features_dict=None) -> None:
        super().__init__(dropped_features_dict=dropped_features_dict)
        if threshold < 0:
            raise RuntimeError(f'Invalid threshold: {threshold}')
        self.threshold = threshold
        self.with_scaler = with_scaler
        self.with_inv_norm = with_inv_norm
        self.reason = f'Outlier ratio >= {self.threshold}'

    def _fit(self, X, y=None):

        X = np.asarray(self._validate_data(X))

        if self.with_scaler:
            X = StandardScaler().fit_transform(X)
        if self.with_inv_norm:
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
            
        squared_distance_from_median = (X - np.nanmedian(X, axis=0))**2
        self.ratios_ = np.nanmax(squared_distance_from_median, axis=0) / np.nanmean(squared_distance_from_median, axis=0)
        return self

    def _get_support_mask(self):
        return self.ratios_ < self.threshold